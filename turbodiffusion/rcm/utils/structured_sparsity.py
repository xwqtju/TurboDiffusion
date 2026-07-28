"""Structured activation sparsity utilities.

These operators simulate sparse hardware numerics by materializing zeros. They
do not provide a sparse kernel or a latency speedup by themselves.
"""

import torch
from torch import nn


def _new_profile_stats() -> dict:
    return {
        "calls": 0,
        "elements": 0,
        "zeros_before": 0,
        "zeros_after": 0,
        "error_sq_sum": 0.0,
        "reference_sq_sum": 0.0,
    }


def configure_sparsity_profile(attention: nn.Module, enabled: bool = True) -> nn.Module:
    """Enable expensive dense-vs-sparse diagnostics on an attention module."""

    attention._sparsity_profile_enabled = enabled
    if enabled and not hasattr(attention, "_sparsity_profile_stats"):
        attention._sparsity_profile_stats = {
            "Q": _new_profile_stats(),
            "K": _new_profile_stats(),
            "attention_output": _new_profile_stats(),
        }
    if enabled and not getattr(attention, "_sparsity_profile_output_hook_enabled", False):
        attention.register_forward_hook(_profile_attention_output)
        attention._sparsity_profile_output_hook_enabled = True
    return attention


def _accumulate_error(stats: dict, reference: torch.Tensor, result: torch.Tensor) -> None:
    difference = result.float() - reference.float()
    reference_float = reference.float()
    stats["calls"] += 1
    stats["elements"] += reference.numel()
    stats["zeros_before"] += torch.count_nonzero(reference == 0).item()
    stats["zeros_after"] += torch.count_nonzero(result == 0).item()
    stats["error_sq_sum"] += difference.square().sum().item()
    stats["reference_sq_sum"] += reference_float.square().sum().item()


def _profiled_sparsify(
    module: nn.Module,
    args: tuple,
    argument_index: int,
    label: str,
    sparsify,
) -> tuple:
    if getattr(module, "_sparsity_profile_dense_replay", False):
        return args
    reference = args[argument_index]
    result = sparsify(reference)
    if getattr(module, "_sparsity_profile_enabled", False):
        # Preserve the inputs before the first sparsity hook. This also makes a
        # simultaneous Q+K profile compare against a fully dense baseline.
        if not hasattr(module, "_sparsity_profile_dense_args"):
            module._sparsity_profile_dense_args = args
        _accumulate_error(module._sparsity_profile_stats[label], reference, result)
    return (*args[:argument_index], result, *args[argument_index + 1:])


def _profile_attention_output(module: nn.Module, _args: tuple, sparse_output):
    if getattr(module, "_sparsity_profile_dense_replay", False):
        return sparse_output
    dense_args = getattr(module, "_sparsity_profile_dense_args", None)
    if dense_args is None:
        return sparse_output
    del module._sparsity_profile_dense_args
    module._sparsity_profile_dense_replay = True
    try:
        with torch.no_grad():
            dense_output = module(*dense_args)
    finally:
        module._sparsity_profile_dense_replay = False
    sparse_tensor = sparse_output[0] if isinstance(sparse_output, tuple) else sparse_output
    dense_tensor = dense_output[0] if isinstance(dense_output, tuple) else dense_output
    _accumulate_error(module._sparsity_profile_stats["attention_output"], dense_tensor, sparse_tensor)
    return sparse_output


def get_sparsity_profile(attention: nn.Module) -> dict:
    """Return finalized zero-rate and relative-L2 diagnostics."""

    result = {}
    for label, raw in getattr(attention, "_sparsity_profile_stats", {}).items():
        elements = raw["elements"]
        reference_sq_sum = raw["reference_sq_sum"]
        result[label] = {
            "calls": raw["calls"],
            "elements": elements,
            "zeros_before": raw["zeros_before"],
            "zeros_after": raw["zeros_after"],
            "zero_rate_before": raw["zeros_before"] / elements if elements else None,
            "zero_rate_after": raw["zeros_after"] / elements if elements else None,
            "relative_l2_error": (
                (raw["error_sq_sum"] / reference_sq_sum) ** 0.5
                if reference_sq_sum > 0
                else None
            ),
            "error_sq_sum": raw["error_sq_sum"],
            "reference_sq_sum": reference_sq_sum,
        }
    return result


def sparsify_2_to_4(x: torch.Tensor) -> torch.Tensor:
    """Keep the two largest-magnitude values in every contiguous group of four.

    Grouping is performed along the last dimension. For attention queries with
    shape ``[batch, sequence, heads, head_dim]``, this is the per-head feature
    dimension consumed by the QK matrix multiplication.
    """

    if x.shape[-1] % 4 != 0:
        raise ValueError(
            "2:4 activation sparsity requires the last dimension to be "
            f"divisible by 4, got {x.shape[-1]}"
        )

    grouped = x.reshape(*x.shape[:-1], -1, 4)
    keep_indices = grouped.abs().topk(k=2, dim=-1, largest=True, sorted=False).indices
    keep_mask = torch.zeros_like(grouped, dtype=torch.bool)
    keep_mask.scatter_(-1, keep_indices, True)
    return grouped.masked_fill(~keep_mask, 0).reshape_as(x)


def sparsify_4_to_8_pairwise(x: torch.Tensor) -> torch.Tensor:
    """Keep two of four adjacent pairs in every contiguous group of eight.

    Each pair is scored by the sum of the absolute values of its two elements.
    Both elements of the two highest-scoring pairs are retained, giving exactly
    four retained values per group of eight.
    """

    if x.shape[-1] % 8 != 0:
        raise ValueError(
            "4:8 pairwise activation sparsity requires the last dimension to be "
            f"divisible by 8, got {x.shape[-1]}"
        )

    grouped = x.reshape(*x.shape[:-1], -1, 4, 2)
    pair_scores = grouped.abs().sum(dim=-1)
    keep_pairs = pair_scores.topk(k=2, dim=-1, largest=True, sorted=False).indices
    keep_mask = torch.zeros_like(pair_scores, dtype=torch.bool)
    keep_mask.scatter_(-1, keep_pairs, True)
    return grouped.masked_fill(~keep_mask.unsqueeze(-1), 0).reshape_as(x)


def sparsify_2_to_4_share_index_2(x: torch.Tensor) -> torch.Tensor:
    """Apply feature-wise 2:4 with one L1-selected mask per two tokens.

    Input must have shape ``[batch, sequence, heads, head_dim]``. Consecutive
    tokens are paired along ``sequence``. For each head and group of four
    features, the importance of feature ``i`` is
    ``abs(token0[i]) + abs(token1[i])``; the two highest-scoring feature
    indices are retained in both tokens. An unpaired final token uses its own
    magnitude, equivalent to ordinary per-token 2:4 selection.
    """

    if x.ndim != 4:
        raise ValueError(f"share-index=2 expects [B, S, H, D], got shape {tuple(x.shape)}")
    if x.shape[-1] % 4 != 0:
        raise ValueError(
            "2:4 share-index=2 activation sparsity requires head_dim divisible by 4, "
            f"got {x.shape[-1]}"
        )

    batch, sequence, heads, head_dim = x.shape
    paired_sequence = sequence - sequence % 2
    parts = []
    if paired_sequence:
        paired = x[:, :paired_sequence].reshape(batch, paired_sequence // 2, 2, heads, head_dim // 4, 4)
        feature_scores = paired.abs().sum(dim=2)
        keep_indices = feature_scores.topk(k=2, dim=-1, largest=True, sorted=False).indices
        keep_mask = torch.zeros_like(feature_scores, dtype=torch.bool)
        keep_mask.scatter_(-1, keep_indices, True)
        sparse_paired = paired.masked_fill(~keep_mask.unsqueeze(2), 0)
        parts.append(sparse_paired.reshape(batch, paired_sequence, heads, head_dim))
    if paired_sequence < sequence:
        parts.append(sparsify_2_to_4(x[:, paired_sequence:]))
    return torch.cat(parts, dim=1)


class QActivation2To4Attention(nn.Module):
    """Apply 2:4 activation sparsity to Q before an attention implementation."""

    def __init__(self, attention: nn.Module):
        super().__init__()
        self.attention = attention

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        sparse_query = sparsify_2_to_4(query)
        return self.attention(sparse_query, key, value, *args, **kwargs)


def enable_q_activation_2_to_4(attention: nn.Module) -> nn.Module:
    """Enable transparent Q 2:4 sparsity without changing state-dict keys."""

    if getattr(attention, "_q_activation_2to4_enabled", False):
        return attention

    def sparsify_query(_module: nn.Module, args: tuple) -> tuple:
        if not args:
            raise ValueError("Attention Q 2:4 hook expected query as the first positional argument")
        return _profiled_sparsify(_module, args, 0, "Q", sparsify_2_to_4)

    attention.register_forward_pre_hook(sparsify_query)
    attention._q_activation_2to4_enabled = True
    return attention


def enable_q_activation_4_to_8_pairwise(attention: nn.Module) -> nn.Module:
    """Enable transparent pairwise Q 4:8 sparsity without state-dict changes."""

    if getattr(attention, "_q_activation_4to8_pairwise_enabled", False):
        return attention

    def sparsify_query(_module: nn.Module, args: tuple) -> tuple:
        if not args:
            raise ValueError("Attention Q 4:8 pairwise hook expected query as the first positional argument")
        return _profiled_sparsify(_module, args, 0, "Q", sparsify_4_to_8_pairwise)

    attention.register_forward_pre_hook(sparsify_query)
    attention._q_activation_4to8_pairwise_enabled = True
    return attention


def enable_k_activation_2_to_4(attention: nn.Module) -> nn.Module:
    """Enable transparent K 2:4 sparsity without changing state-dict keys."""

    if getattr(attention, "_k_activation_2to4_enabled", False):
        return attention

    def sparsify_key(_module: nn.Module, args: tuple) -> tuple:
        if len(args) < 2:
            raise ValueError("Attention K 2:4 hook expected key as the second positional argument")
        return _profiled_sparsify(_module, args, 1, "K", sparsify_2_to_4)

    attention.register_forward_pre_hook(sparsify_key)
    attention._k_activation_2to4_enabled = True
    return attention


def enable_k_activation_4_to_8_pairwise(attention: nn.Module) -> nn.Module:
    """Enable transparent pairwise K 4:8 sparsity without state-dict changes."""

    if getattr(attention, "_k_activation_4to8_pairwise_enabled", False):
        return attention

    def sparsify_key(_module: nn.Module, args: tuple) -> tuple:
        if len(args) < 2:
            raise ValueError("Attention K 4:8 pairwise hook expected key as the second positional argument")
        return _profiled_sparsify(_module, args, 1, "K", sparsify_4_to_8_pairwise)

    attention.register_forward_pre_hook(sparsify_key)
    attention._k_activation_4to8_pairwise_enabled = True
    return attention


def enable_q_activation_2_to_4_share_index_2(attention: nn.Module) -> nn.Module:
    """Enable Q 2:4 whose feature indices are shared by pairs of tokens."""

    if getattr(attention, "_q_activation_2to4_share2_enabled", False):
        return attention

    def sparsify_query(_module: nn.Module, args: tuple) -> tuple:
        if not args:
            raise ValueError("Attention Q 2:4 share-index=2 hook expected query first")
        return _profiled_sparsify(_module, args, 0, "Q", sparsify_2_to_4_share_index_2)

    attention.register_forward_pre_hook(sparsify_query)
    attention._q_activation_2to4_share2_enabled = True
    return attention


def enable_k_activation_2_to_4_share_index_2(attention: nn.Module) -> nn.Module:
    """Enable K 2:4 whose feature indices are shared by pairs of tokens."""

    if getattr(attention, "_k_activation_2to4_share2_enabled", False):
        return attention

    def sparsify_key(_module: nn.Module, args: tuple) -> tuple:
        if len(args) < 2:
            raise ValueError("Attention K 2:4 share-index=2 hook expected key second")
        return _profiled_sparsify(_module, args, 1, "K", sparsify_2_to_4_share_index_2)

    attention.register_forward_pre_hook(sparsify_key)
    attention._k_activation_2to4_share2_enabled = True
    return attention
