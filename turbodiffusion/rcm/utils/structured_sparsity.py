"""Structured activation sparsity utilities.

These operators simulate sparse hardware numerics by materializing zeros. They
do not provide a sparse kernel or a latency speedup by themselves.
"""

import torch
from torch import nn


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
        return (sparsify_2_to_4(args[0]), *args[1:])

    attention.register_forward_pre_hook(sparsify_query)
    attention._q_activation_2to4_enabled = True
    return attention
