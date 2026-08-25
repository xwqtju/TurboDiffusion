""" 
Copyright (c) 2025 by SLA team.

Licensed under the Apache License, Version 2.0 (the "License");

Citation (please cite if you use this code):

@article{zhang2025sla,
  title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention}, 
  author={Jintao Zhang and Haoxu Wang and Kai Jiang and Shuo Yang and Kaiwen Zheng and Haocheng Xi and Ziteng Wang and Hongzhou Zhu and Min Zhao and Ion Stoica and Joseph E. Gonzalez and Jun Zhu and Jianfei Chen},
  journal={arXiv preprint arXiv:2509.24006},
  year={2025}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SAGESLA_ENABLED = True
try:
    import spas_sage_attn._qattn as qattn
    import spas_sage_attn._fused as fused
    from spas_sage_attn.utils import get_vanilla_qk_quant, block_map_lut_triton
except ImportError:
    SAGESLA_ENABLED = False

SAGE2PP_ENABLED = True
try:
    from spas_sage_attn._qattn import qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
except ImportError:
    SAGE2PP_ENABLED = False

from .kernel import _attention, rubin_2to4_attention_forward
from .hif4 import hif4_qdq
from .utils import get_block_map, get_cuda_arch
from rcm.utils.structured_sparsity import (
    begin_internal_sparsity_profile,
    record_internal_sparsity,
)


def apply_2_to_4_sparsity(x, return_mask=False, norm_compensate=True):
    """Apply a deterministic 2-of-4 index mask on the last dimension.

    Numeric zeros remain zero. Equal magnitudes are dropped in increasing
    index order, so an all-zero quartet retains the last two indices.

    When ``norm_compensate=True`` (default), applies RT-Lynx norm compensation
    (Section 4.1, Algorithm 1): rescales the sparse activation so its L2 norm
    matches the original dense activation.

        X̃ ← TopK(X)
        s ← sqrt(||X||²₂ / (||X̃||²₂ + ε))
        S(X) ← s · X̃

    This eliminates norm attenuation induced by sparsification.
    """
    dense_dim = x.shape[-1]
    sparse_dim = dense_dim // 4 * 4
    if sparse_dim == 0:
        return (x, torch.ones_like(x, dtype=torch.bool)) if return_mask else x

    x_sparse_part = x[..., :sparse_dim]
    x_tail = x[..., sparse_dim:]

    grouped = x_sparse_part.reshape(*x_sparse_part.shape[:-1], -1, 4)
    # RT-Lynx Eq. 4 defines X as one 1x4 structured group. Each quartet
    # therefore owns an independent compensation factor.
    if norm_compensate:
        orig_norm_sq = grouped.float().square().sum(dim=-1, keepdim=True)
    keep_idx = torch.argsort(grouped.abs(), dim=-1, stable=True)[..., -2:]
    keep_mask = torch.zeros_like(grouped, dtype=torch.bool)
    keep_mask.scatter_(-1, keep_idx, True)
    x_sparse_part = grouped.masked_fill(~keep_mask, 0).reshape_as(x_sparse_part)
    flat_mask = keep_mask.reshape_as(x_sparse_part)

    if norm_compensate:
        # RT-Lynx Eq. 4: s = sqrt(||X||²₂ / (||X̃||²₂ + ε)),  S(X) = s · X̃
        eps = 1e-8
        sparse_grouped = x_sparse_part.reshape_as(grouped)
        sparse_norm_sq = sparse_grouped.float().square().sum(dim=-1, keepdim=True)
        scale = torch.sqrt(orig_norm_sq / (sparse_norm_sq + eps))
        x_sparse_part = (sparse_grouped.float() * scale).reshape_as(x_sparse_part).to(x.dtype)

    if x_tail.numel() == 0:
        result, result_mask = x_sparse_part, flat_mask
    else:
        result = torch.cat((x_sparse_part, x_tail), dim=-1)
        result_mask = torch.cat(
            (flat_mask, torch.ones_like(x_tail, dtype=torch.bool)), dim=-1
        )
    return (result, result_mask) if return_mask else result


def apply_weight_norm_2_to_4_sparsity(x, weight):
    """Select 2:4 activation entries using ``abs(x) * column_l2(weight)``.

    ``x`` is [..., M, K] and ``weight`` is [..., K, N]. This matches the
    weight-normal activation pruning rule used by ActivateSparseHW: weight
    norms affect mask selection only; retained activation values are unchanged.
    """
    if x.shape[-1] != weight.shape[-2]:
        raise ValueError(f"Incompatible activation/weight shapes: {x.shape}, {weight.shape}")
    dense_dim = x.shape[-1]
    sparse_dim = dense_dim // 4 * 4
    if sparse_dim == 0:
        return x
    channel_weight = weight.float().square().sum(dim=-1).sqrt()
    scale = channel_weight.mean(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
    channel_weight = channel_weight / scale
    grouped_score = (
        x[..., :sparse_dim].float().abs() * channel_weight[..., None, :sparse_dim]
    ).reshape(*x.shape[:-1], -1, 4)
    keep_idx = torch.argsort(grouped_score, dim=-1, stable=True)[..., -2:]
    keep_mask = torch.zeros_like(grouped_score, dtype=torch.bool)
    keep_mask.scatter_(-1, keep_idx, True)
    sparse_part = x[..., :sparse_dim].reshape_as(grouped_score).masked_fill(~keep_mask, 0).reshape(*x.shape[:-1], sparse_dim)
    return torch.cat((sparse_part, x[..., sparse_dim:]), dim=-1) if sparse_dim != dense_dim else sparse_part


def apply_weight_norm_4_to_8_pairwise_sparsity(x, weight):
    """Keep two adjacent pairs per 8 using weight-normal activation scores."""
    if x.shape[-1] != weight.shape[-2] or x.shape[-1] % 8:
        raise ValueError(f"Incompatible 4:8 activation/weight shapes: {x.shape}, {weight.shape}")
    channel_weight = weight.float().square().sum(dim=-1).sqrt()
    scale = channel_weight.mean(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
    weighted = x.float().abs() * (channel_weight / scale)[..., None, :]
    pair_score = weighted.reshape(*x.shape[:-1], -1, 4, 2).sum(dim=-1)
    keep = pair_score.topk(k=2, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros_like(pair_score, dtype=torch.bool).scatter_(-1, keep, True)
    grouped = x.reshape(*x.shape[:-1], -1, 4, 2)
    return grouped.masked_fill(~mask.unsqueeze(-1), 0).reshape_as(x)

def apply_weight_norm_2_to_4_share_index_2(x, weight):
    """Weight-normal scored 2:4 with one mask shared by each token pair."""
    if x.shape[-1] != weight.shape[-2] or x.shape[-2] % 2 or x.shape[-1] % 4:
        raise ValueError(f"Incompatible share-index shapes: {x.shape}, {weight.shape}")
    w = weight.float().square().sum(dim=-1).sqrt()
    w = w / w.mean(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
    grouped = x.reshape(*x.shape[:-2], -1, 2, x.shape[-1] // 4, 4)
    wg = w.reshape(*w.shape[:-1], x.shape[-1] // 4, 4)
    score = (grouped.float().abs() * wg[..., None, None, :, :]).sum(dim=-3)
    keep = score.topk(2, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros_like(score, dtype=torch.bool).scatter_(-1, keep, True)
    return grouped.masked_fill(~mask.unsqueeze(-3), 0).reshape_as(x)


def build_rpq_feature_permutation(score, block=4, sample_tokens=256):
    """Build a Qwen3-style balanced anti-cluster permutation from Top-K conflicts."""
    if score.shape[-1] % block:
        raise ValueError(f"RPQ dimension must be divisible by {block}, got {score.shape[-1]}")
    flat = score.detach().float().reshape(-1, score.shape[-1])
    if flat.shape[0] > sample_tokens:
        indices = torch.linspace(0, flat.shape[0] - 1, sample_tokens, device=flat.device).long()
        flat = flat.index_select(0, indices)
    topk = flat.shape[-1] // 2
    selected = flat.topk(k=topk, dim=-1).indices
    mask = torch.zeros_like(flat)
    mask.scatter_(1, selected, 1)
    conflict = mask.transpose(0, 1) @ mask
    conflict.fill_diagonal_(0)
    order = conflict.sum(dim=1).argsort(descending=True)
    groups = [[int(x)] for x in order[: flat.shape[-1] // block]]
    for channel in order[len(groups):].tolist():
        candidates = [i for i, group in enumerate(groups) if len(group) < block]
        best = min(candidates, key=lambda i: (float(conflict[channel, groups[i]].sum()), len(groups[i]), i))
        groups[best].append(channel)
    return torch.tensor([channel for group in groups for channel in group], device=score.device)


def apply_2_to_4_sparsity_along_dim(x, dim, norm_compensate=True):
    """Apply 2:4 along a selected GEMM reduction dimension."""
    dim = dim if dim >= 0 else x.ndim + dim
    if dim == x.ndim - 1:
        return apply_2_to_4_sparsity(x, norm_compensate=norm_compensate)
    order = [axis for axis in range(x.ndim) if axis != dim] + [dim]
    inverse = [0] * x.ndim
    for new_axis, old_axis in enumerate(order):
        inverse[old_axis] = new_axis
    return apply_2_to_4_sparsity(x.permute(order), norm_compensate=norm_compensate).permute(inverse).contiguous()


def apply_4_to_8_pairwise_sparsity(x):
    """Keep the two highest-L1 adjacent pairs in each last-dim group of 8."""
    if x.shape[-1] % 8:
        raise ValueError(f"4:8 pairwise sparsity requires a dimension divisible by 8, got {x.shape[-1]}")
    grouped = x.reshape(*x.shape[:-1], -1, 4, 2)
    scores = grouped.abs().sum(dim=-1)
    keep = scores.topk(k=2, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, keep, True)
    return grouped.masked_fill(~mask.unsqueeze(-1), 0).reshape_as(x)


def apply_4_to_8_pairwise_sparsity_along_dim(x, dim):
    """Apply pairwise 4:8 along a selected GEMM reduction dimension."""
    dim = dim if dim >= 0 else x.ndim + dim
    if dim == x.ndim - 1:
        return apply_4_to_8_pairwise_sparsity(x)
    order = [axis for axis in range(x.ndim) if axis != dim] + [dim]
    inverse = [0] * x.ndim
    for new_axis, old_axis in enumerate(order):
        inverse[old_axis] = new_axis
    return apply_4_to_8_pairwise_sparsity(x.permute(order)).permute(inverse).contiguous()


def apply_2_to_4_share_index_2(x, sparse_dim, share_dim, norm_compensate=True):
    """Apply 2:4 on ``sparse_dim`` with one L1 mask per pair on ``share_dim``."""
    sparse_dim = sparse_dim if sparse_dim >= 0 else x.ndim + sparse_dim
    share_dim = share_dim if share_dim >= 0 else x.ndim + share_dim
    if sparse_dim == share_dim:
        raise ValueError("sparse_dim and share_dim must be different")
    if x.shape[sparse_dim] % 4:
        raise ValueError(f"2:4 share-index requires a dimension divisible by 4, got {x.shape[sparse_dim]}")
    order = [axis for axis in range(x.ndim) if axis not in (share_dim, sparse_dim)] + [share_dim, sparse_dim]
    inverse = [0] * x.ndim
    for new_axis, old_axis in enumerate(order):
        inverse[old_axis] = new_axis
    arranged = x.permute(order)
    share, sparse = arranged.shape[-2:]
    paired_share = share - share % 2
    parts = []
    if paired_share:
        paired = arranged[..., :paired_share, :].reshape(*arranged.shape[:-2], paired_share // 2, 2, sparse // 4, 4)
        scores = paired.abs().sum(dim=-3)
        keep = scores.topk(k=2, dim=-1, largest=True, sorted=False).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(-1, keep, True)
        sparse_paired = paired.masked_fill(~mask.unsqueeze(-3), 0)
        if norm_compensate:
            # The two tokens share indices, while each token/quartet retains
            # its own dense L2 magnitude.
            eps = 1e-8
            dense_norm_sq = paired.float().square().sum(dim=-1, keepdim=True)
            sparse_norm_sq = sparse_paired.float().square().sum(dim=-1, keepdim=True)
            scale = torch.sqrt(dense_norm_sq / (sparse_norm_sq + eps))
            sparse_paired = (sparse_paired.float() * scale).to(paired.dtype)
        parts.append(sparse_paired.reshape(*arranged.shape[:-2], paired_share, sparse))
    if paired_share < share:
        parts.append(apply_2_to_4_sparsity(arranged[..., paired_share:, :], norm_compensate=norm_compensate))
    return torch.cat(parts, dim=-2).permute(inverse).contiguous()


def branch_aware_sparse_k(k, mode, norm_compensate=True):
    """Return K grouped along feature dim for sparse QK."""
    if mode == "none":
        return k
    if mode == "2to4":
        return apply_2_to_4_sparsity(k, norm_compensate=norm_compensate)
    if mode == "4to8_pairwise":
        return apply_4_to_8_pairwise_sparsity(k)
    if mode == "2to4_share2":
        return apply_2_to_4_share_index_2(k, sparse_dim=-1, share_dim=-2, norm_compensate=norm_compensate)
    raise ValueError(f"Invalid branch-aware K sparsity mode: {mode}")


def branch_aware_q(q, mode, norm_compensate=True):
    """Return feature-sparse Q for sparse QK or post-feature-map Q@KV."""
    if mode == "none":
        return q
    if mode == "2to4":
        return apply_2_to_4_sparsity(q, norm_compensate=norm_compensate)
    if mode == "4to8_pairwise":
        return apply_4_to_8_pairwise_sparsity(q)
    if mode == "2to4_share2":
        return apply_2_to_4_share_index_2(q, sparse_dim=-1, share_dim=-2, norm_compensate=norm_compensate)
    raise ValueError(f"Invalid branch-aware Q sparsity mode: {mode}")


def branch_aware_linear_k(k, mode, norm_compensate=True):
    """Return K grouped along token dim for linear K.T@V."""
    if mode == "none":
        return k
    if mode == "2to4":
        return apply_2_to_4_sparsity_along_dim(k, -2, norm_compensate=norm_compensate)
    if mode == "4to8_pairwise":
        return apply_4_to_8_pairwise_sparsity_along_dim(k, -2)
    if mode == "2to4_share2":
        return apply_2_to_4_share_index_2(k, sparse_dim=-2, share_dim=-1, norm_compensate=norm_compensate)
    raise ValueError(f"Invalid branch-aware K sparsity mode: {mode}")


def branch_aware_k_tensors(k, mode, norm_compensate=True):
    """Return both branch-specific K tensors; primarily used by diagnostics."""
    return branch_aware_sparse_k(k, mode, norm_compensate=norm_compensate), branch_aware_linear_k(k, mode, norm_compensate=norm_compensate)


def _strict_two_of_four(tensor, label):
    if tensor.shape[-1] % 4:
        raise ValueError(f"{label} reduction dimension must be divisible by 4, got {tensor.shape[-1]}")
    groups = tensor.reshape(*tensor.shape[:-1], -1, 4)
    nonzeros = torch.count_nonzero(groups, dim=-1)
    if not torch.all(nonzeros == 2).item():
        raise RuntimeError(f"{label} is not strict 2:4 immediately before GEMM")


def explicit_block_sparse_attention_rubin_2to4(q, k, v, lut, block_q, block_k):
    """Reference Rubin-style path with packed block-mask P and dense GEMMs.

    QK uses feature-2:4 K supplied by the caller. Scores in every selected key
    block are pruned 2:4 before a masked softmax; packed P then multiplies the
    corresponding packed V. Unselected blocks are implicit zeros.
    """
    batch, heads, length, head_dim = q.shape
    if head_dim % 4 or length % 4:
        raise ValueError(f"Rubin 2:4 requires head_dim and token length divisible by 4, got D={head_dim}, L={length}")
    _strict_two_of_four(k, "sparse QK K")
    output = torch.empty_like(q)
    p_elements = 0
    p_zeros = torch.zeros((), device=q.device, dtype=torch.int64)
    max_row_sum_error = torch.zeros((), device=q.device, dtype=torch.float32)
    p_group_violation = torch.zeros((), device=q.device, dtype=torch.int64)
    token_offsets = torch.arange(block_k, device=q.device)
    scale = head_dim ** -0.5
    for query_block in range(lut.shape[-2]):
        q_start = query_block * block_q
        q_stop = min(q_start + block_q, length)
        selected_blocks = lut[:, :, query_block]
        token_indices = selected_blocks[..., None] * block_k + token_offsets
        valid = token_indices < length
        safe_indices = token_indices.clamp_max(length - 1).reshape(batch, heads, -1)
        gather_index = safe_indices[..., None].expand(-1, -1, -1, head_dim)
        selected_k = torch.gather(k, -2, gather_index)
        selected_v = torch.gather(v, -2, gather_index)
        valid = valid.reshape(batch, heads, -1)
        # FP32 score accumulation is the mathematical oracle used by the
        # fused kernel; probabilities are still rounded to the model dtype
        # immediately before P@V.
        scores = (
            q[:, :, q_start:q_stop].float() @ selected_k.float().transpose(-1, -2)
        ) * scale
        scores = scores.masked_fill(~valid.unsqueeze(-2), float("-inf"))
        grouped_scores = scores.reshape(*scores.shape[:-1], -1, 4)
        # Stable left-to-right tie breaking matches the fused kernel. This is
        # important for BF16 model scores, where exact ties are common.
        keep_indices = torch.argsort(
            grouped_scores, dim=-1, descending=True, stable=True
        )[..., :2]
        keep = torch.zeros_like(grouped_scores, dtype=torch.bool)
        keep.scatter_(-1, keep_indices, True)
        grouped_valid = valid.reshape(batch, heads, -1, 4).unsqueeze(-3)
        keep = keep & grouped_valid
        sparse_scores = grouped_scores.masked_fill(~keep, float("-inf")).reshape_as(scores)
        probability_float = torch.softmax(sparse_scores.float(), dim=-1)
        probability_float = probability_float.masked_fill(~valid.unsqueeze(-2), 0)
        row_error = (probability_float.sum(dim=-1) - 1).abs().max()
        max_row_sum_error = torch.maximum(max_row_sum_error, row_error)
        probability = probability_float.to(v.dtype)
        p_groups = probability.reshape(*probability.shape[:-1], -1, 4)
        valid_groups = grouped_valid.expand(*p_groups.shape)
        fully_valid = valid_groups.all(dim=-1)
        group_nonzeros = torch.count_nonzero(p_groups, dim=-1)
        kept_per_group = torch.count_nonzero(keep, dim=-1)
        leaked_outside_mask = torch.count_nonzero(probability.masked_fill(keep.reshape_as(probability), 0))
        p_group_violation += torch.count_nonzero(fully_valid & (kept_per_group != 2))
        p_group_violation += torch.count_nonzero(group_nonzeros > 2)
        p_group_violation += leaked_outside_mask
        p_elements += probability.numel()
        p_zeros += torch.count_nonzero(probability == 0)
        output[:, :, q_start:q_stop] = probability @ selected_v
    if p_group_violation.item():
        raise RuntimeError(f"P is not strict 2:4 in {p_group_violation.item()} valid groups before P@V")
    if max_row_sum_error.item() > 2e-3:
        raise RuntimeError(f"Sparse P rows are not normalized; max error={max_row_sum_error.item()}")
    p_zeros_value = p_zeros.item()
    return output, {
        "p_elements": p_elements,
        "p_zeros": p_zeros_value,
        "p_zero_rate": p_zeros_value / p_elements,
        "p_max_row_sum_error": max_row_sum_error.item(),
        "p_group_violations": 0,
    }


def linear_attention_2to4(
    q, k, v, kv_sparse_operand="none", qkv_sparse_operand="none",
    branch_aware_q_sparsity="none", branch_aware_k_sparsity="none",
    branch_aware_q_profile=None, branch_aware_k_profile=None,
    hif4_ktv=False, hif4_qkv=False, operand_audit=None,
    norm_compensate=True,
    weight_norm_k_2to4=False,
):
    """Linear attention with structured sparsity and optional HiF4 W4A4 QDQ."""
    if kv_sparse_operand not in {"none", "k", "v"}:
        raise ValueError(f"Invalid K.T@V sparse operand: {kv_sparse_operand}")
    if qkv_sparse_operand not in {"none", "q", "kv"}:
        raise ValueError(f"Invalid Q@KV sparse operand: {qkv_sparse_operand}")
    if branch_aware_k_sparsity != "none" and kv_sparse_operand != "none":
        raise ValueError("branch-aware K sparsity conflicts with a separate K.T@V sparse operand")
    if branch_aware_q_sparsity != "none" and qkv_sparse_operand != "none":
        raise ValueError("branch-aware Q sparsity conflicts with a separate Q@KV sparse operand")

    branch_k_for_gemm = (
        apply_weight_norm_2_to_4_sparsity(k.transpose(-1, -2), v).transpose(-1, -2).contiguous()
        if weight_norm_k_2to4 else
        branch_aware_linear_k(k, branch_aware_k_sparsity, norm_compensate=norm_compensate)
    )
    if branch_aware_k_profile is not None and (branch_aware_k_sparsity != "none" or weight_norm_k_2to4):
        branch_aware_k_profile(k, branch_k_for_gemm)
    k_for_gemm = apply_2_to_4_sparsity_along_dim(k, -2, norm_compensate=norm_compensate) if kv_sparse_operand == "k" else branch_k_for_gemm
    v_for_gemm = apply_2_to_4_sparsity_along_dim(v, -2, norm_compensate=norm_compensate) if kv_sparse_operand == "v" else v
    k_sparse = kv_sparse_operand == "k" or branch_aware_k_sparsity != "none" or weight_norm_k_2to4
    v_sparse = kv_sparse_operand == "v"
    if hif4_ktv:
        k_for_gemm = (
            hif4_qdq(k_for_gemm, -2, sparse_mask=k_for_gemm != 0, protect_sparse_mask=True)
            if k_sparse else hif4_qdq(k_for_gemm, -2)
        )
        v_for_gemm = (
            hif4_qdq(v_for_gemm, -2, sparse_mask=v_for_gemm != 0, protect_sparse_mask=True)
            if v_sparse else hif4_qdq(v_for_gemm, -2)
        )
        k_for_gemm = k_for_gemm.to(v.dtype)
        v_for_gemm = v_for_gemm.to(v.dtype)
    k_for_gemm = k_for_gemm.to(v.dtype)
    v_for_gemm = v_for_gemm.to(v.dtype)
    if operand_audit is not None and hif4_ktv:
        operand_audit(
            "linear_ktv", k_for_gemm.transpose(-1, -2), v_for_gemm,
            "k" if k_sparse else "v" if v_sparse else None,
        )
    kvsum = k_for_gemm.transpose(-1, -2) @ v_for_gemm

    branch_q_for_gemm = branch_aware_q(q, branch_aware_q_sparsity, norm_compensate=norm_compensate)
    if branch_aware_q_profile is not None and branch_aware_q_sparsity != "none":
        branch_aware_q_profile(q, branch_q_for_gemm)
    q_for_gemm = apply_2_to_4_sparsity(q, norm_compensate=norm_compensate) if qkv_sparse_operand == "q" else branch_q_for_gemm
    kv_for_gemm = apply_2_to_4_sparsity_along_dim(kvsum, -2, norm_compensate=norm_compensate) if qkv_sparse_operand == "kv" else kvsum
    q_sparse = qkv_sparse_operand == "q" or branch_aware_q_sparsity != "none"
    kv_sparse = qkv_sparse_operand == "kv"
    if hif4_qkv:
        q_for_gemm = (
            hif4_qdq(q_for_gemm, -1, sparse_mask=q_for_gemm != 0, protect_sparse_mask=True)
            if q_sparse else hif4_qdq(q_for_gemm, -1)
        )
        kv_for_gemm = (
            hif4_qdq(kv_for_gemm, -2, sparse_mask=kv_for_gemm != 0, protect_sparse_mask=True)
            if kv_sparse else hif4_qdq(kv_for_gemm, -2)
        )
        q_for_gemm = q_for_gemm.to(v.dtype)
        kv_for_gemm = kv_for_gemm.to(v.dtype)
    q_for_gemm = q_for_gemm.to(v.dtype)
    kv_for_gemm = kv_for_gemm.to(v.dtype)
    if operand_audit is not None and hif4_qkv:
        operand_audit(
            "linear_qkv", q_for_gemm, kv_for_gemm,
            "q" if q_sparse else "kv" if kv_sparse else None,
        )
    numerator = q_for_gemm @ kv_for_gemm
    dense_q = q.to(v.dtype)
    dense_k = k.to(v.dtype)
    ksum = torch.sum(dense_k, dim=-2, keepdim=True)
    denominator = 1e-5 + (dense_q * ksum).sum(dim=-1, keepdim=True)
    return numerator / denominator


class SparseLinearAttention(nn.Module):
    def _audit_hif4_operands(self, label, lhs, rhs, sparse_role):
        stats = self._hif4_operand_audit.setdefault(label, {
            "calls": 0, "group_violations": 0, "checked_groups": 0,
            "lhs_shape": list(lhs.shape), "rhs_shape": list(rhs.shape),
            "dtype": str(lhs.dtype), "sparse_role": sparse_role,
        })
        stats["calls"] += 1
        if sparse_role is None:
            return
        if sparse_role == "q":
            operand = lhs
        elif sparse_role == "k":
            operand = rhs if label == "sparse_qk" else lhs
        else:
            operand = rhs
        if (label == "linear_qkv" and sparse_role == "kv") or (
            label == "linear_ktv" and sparse_role == "v"
        ):
            operand = operand.transpose(-1, -2)
        mode = "2to4"
        if label == "sparse_qk":
            mode = self.branch_aware_q_sparsity if sparse_role == "q" else (
                "2to4" if self.rubin_triple_2to4 else self.branch_aware_k_sparsity
            )
        elif label == "linear_ktv" and self.branch_aware_k_sparsity != "none":
            mode = self.branch_aware_k_sparsity
        elif label == "linear_qkv" and self.branch_aware_q_sparsity != "none":
            mode = self.branch_aware_q_sparsity
        if mode == "4to8_pairwise":
            groups = operand.reshape(*operand.shape[:-1], -1, 4, 2)
            pair_counts = torch.count_nonzero(groups, dim=-1)
            pair_full = pair_counts == 2
            violations = torch.count_nonzero(
                torch.count_nonzero(groups, dim=(-1, -2)) > 4
            )
            checked = groups.numel() // 8
        else:
            groups = operand.reshape(*operand.shape[:-1], -1, 4)
            violations = torch.count_nonzero(
                torch.count_nonzero(groups, dim=-1) > 2
            )
            checked = groups.numel() // 4
        count = int(violations.item())
        stats["group_violations"] += count
        stats["checked_groups"] += checked
        if count:
            raise RuntimeError(
                f"{label} {sparse_role} failed post-HiF4 {mode} audit: "
                f"{count} violations"
            )

    def __init__(
        self,
        head_dim,
        topk,
        feature_map='softmax',
        BLKQ=64,
        BLKK=64,
        use_bf16=True,
        tie_feature_map_qk=True,
        linear_q_2to4=False,
        linear_kv_2to4_operand="none",
        linear_qkv_2to4_operand="none",
        branch_aware_q_sparsity="none",
        branch_aware_k_sparsity="none",
        rubin_triple_2to4=False,
        rubin_sparse_engine="fused",
        rubin_validate_fused=False,
        hif4_sparse_upgrade=False,
        hif4_only_scope="none",
        operand_audit=None,
        norm_compensate=True,
        q_weight_norm_2to4=False,
        k_weight_norm_2to4=False,
        k_weight_norm_rpq=False,
        q_weight_norm_rpq=False,
    ):
        R'''
        Args:
            head_dim: dimension of each head.
            topk: ratio of keys selected for sparse attention, shared across all queries.
            feature_map: feature map for linear attention, one of ['hedgehog', 'elu', 'relu', 'softmax'].
            BLKQ: block size for query.
            BLKK: block size for key.
            use_bf16: whether to use bfloat16 (default) or float16 for computation. The conversion to bf16/fp16 is done inside the module.
            tie_feature_map_qk: whether to use the same feature map for query and key.
            linear_q_2to4: whether to simulate 2:4 activation sparsity on Q in the linear-attention branch.
        '''
        super().__init__()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk = topk
        self.BLKQ = BLKQ
        self.BLKK = BLKK
        self.linear_q_2to4 = linear_q_2to4
        self.linear_kv_2to4_operand = linear_kv_2to4_operand
        self.linear_qkv_2to4_operand = linear_qkv_2to4_operand
        self.branch_aware_q_sparsity = branch_aware_q_sparsity
        self.branch_aware_k_sparsity = branch_aware_k_sparsity
        self.rubin_triple_2to4 = rubin_triple_2to4
        self.rubin_sparse_engine = rubin_sparse_engine
        self.rubin_validate_fused = rubin_validate_fused
        self.hif4_sparse_upgrade = hif4_sparse_upgrade
        self.hif4_only_scope = hif4_only_scope
        self.operand_audit = operand_audit
        self.norm_compensate = norm_compensate
        self.q_weight_norm_2to4 = q_weight_norm_2to4
        self.k_weight_norm_2to4 = k_weight_norm_2to4
        self.k_weight_norm_rpq = k_weight_norm_rpq
        self.q_weight_norm_rpq = q_weight_norm_rpq
        self.register_buffer("k_rpq_permutation", None, persistent=False)
        self.register_buffer("q_rpq_permutation", None, persistent=False)
        self._hif4_operand_audit = {}
        if hif4_only_scope not in {"none", "q_path", "k_path", "linear", "rubin"}:
            raise ValueError(f"Invalid HiF4-only scope: {hif4_only_scope}")
        if hif4_sparse_upgrade and hif4_only_scope != "none":
            raise ValueError("HiF4 sparse-upgrade and HiF4-only are mutually exclusive")
        has_structured_mode = any((
            linear_q_2to4, linear_kv_2to4_operand != "none",
            linear_qkv_2to4_operand != "none", branch_aware_q_sparsity != "none",
            branch_aware_k_sparsity != "none", rubin_triple_2to4,
            q_weight_norm_2to4,
            k_weight_norm_2to4,
        ))
        if hif4_sparse_upgrade and not has_structured_mode:
            raise ValueError("HiF4 sparse-upgrade requires one structured-sparsity mode")
        if hif4_only_scope != "none" and has_structured_mode:
            raise ValueError("HiF4-only cannot be combined with structured sparsity")
        if rubin_sparse_engine not in {"reference", "fused"}:
            raise ValueError(f"Invalid Rubin sparse engine: {rubin_sparse_engine}")
        if rubin_triple_2to4 and rubin_sparse_engine != "fused":
            raise ValueError("Rubin triple 2:4 only supports the fused forward engine")
        if hif4_only_scope == "rubin" and rubin_sparse_engine != "fused":
            raise ValueError("Rubin-scope HiF4-only requires the fused forward engine")
        if rubin_validate_fused and (not rubin_triple_2to4 or rubin_sparse_engine != "fused"):
            raise ValueError("Rubin fused validation requires triple 2:4 with the fused engine")
        if linear_q_2to4 and linear_qkv_2to4_operand != "none":
            raise ValueError("--linear_q_2to4 conflicts with --linear_qkv_2to4_operand")
        if branch_aware_k_sparsity != "none" and linear_kv_2to4_operand != "none":
            raise ValueError("branch-aware K sparsity conflicts with --linear_kv_2to4_operand")
        if branch_aware_q_sparsity != "none" and (linear_q_2to4 or linear_qkv_2to4_operand != "none"):
            raise ValueError("branch-aware Q sparsity conflicts with a separate linear Q sparse operand")
        if q_weight_norm_2to4 and (branch_aware_q_sparsity not in {"none", "2to4_share2", "4to8_pairwise"} or linear_q_2to4 or linear_qkv_2to4_operand != "none"):
            raise ValueError("weight-normal Q 2:4 must be the only Q sparsity mode")
        if k_weight_norm_2to4 and (branch_aware_k_sparsity not in {"none", "2to4_share2"} or linear_kv_2to4_operand != "none"):
            raise ValueError("weight-normal K 2:4 must be the only K sparsity mode")
        if rubin_triple_2to4 and any((
            linear_q_2to4, linear_kv_2to4_operand != "none", linear_qkv_2to4_operand != "none",
            branch_aware_q_sparsity != "none", branch_aware_k_sparsity != "none",
        )):
            raise ValueError("Rubin triple 2:4 must be the only activation-sparsity mode")
        if rubin_triple_2to4:
            self._rubin_triple_audit = {
                "calls": 0,
                "qk_k_zero_rate_min": 1.0, "qk_k_zero_rate_max": 0.0,
                "pv_p_zero_rate_min": 1.0, "pv_p_zero_rate_max": 0.0,
                "linear_k_zero_rate_min": 1.0, "linear_k_zero_rate_max": 0.0,
                "p_max_row_sum_error": 0.0, "violations": 0,
                "fused_reference_rel_l2_max": 0.0,
                "fused_p_operand_checked_elements": 0,
                "fused_p_operand_group_violations": 0,
            }
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=torch.float32)

        if feature_map == 'elu':
            def elu_feature_map(x):
                return F.elu(x) + 1
            self.feature_map_q = elu_feature_map
            self.feature_map_k = elu_feature_map
        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
        elif feature_map == 'softmax':
            def softmax_feature_map(x):
                return F.softmax(x, dim=-1)
            self.feature_map_q = softmax_feature_map
            self.feature_map_k = softmax_feature_map
        else:
            raise NotImplementedError(f'Not supported feature map {feature_map}.')

        if tie_feature_map_qk:
            self.feature_map_k = self.feature_map_q

        self.init_weights_()

    def init_weights_(self):
        with torch.no_grad():
            nn.init.zeros_(self.proj_l.weight)
            nn.init.zeros_(self.proj_l.bias)

    def forward(self, q, k, v, return_sparsity=False):
        R'''
        Args:
            q: queries of shape (B, H, L, D).
            k: keys of shape (B, H, L, D).
            v: values of shape (B, H, L, D).
            return_sparsity: whether to return the actual sparsity.
        '''
        dtype = q.dtype
        original_args = (q, k, v)
        profile_branch = (
            (self.branch_aware_q_sparsity != "none" or self.branch_aware_k_sparsity != "none" or self.q_weight_norm_2to4 or self.k_weight_norm_2to4)
            and begin_internal_sparsity_profile(self, original_args)
        )
        effective_weight_norm_q = self.q_weight_norm_2to4 and not getattr(self, "_sparsity_profile_dense_replay", False)
        effective_weight_norm_k = self.k_weight_norm_2to4 and not getattr(self, "_sparsity_profile_dense_replay", False)
        effective_q_mode = (
            self.branch_aware_q_sparsity
            if not getattr(self, "_sparsity_profile_dense_replay", False)
            else "none"
        )
        effective_branch_mode = (
            self.branch_aware_k_sparsity
            if not getattr(self, "_sparsity_profile_dense_replay", False)
            else "none"
        )
        effective_rubin = self.rubin_triple_2to4 and not getattr(self, "_sparsity_profile_dense_replay", False)
        qk_hif4 = (
            self.hif4_only_scope in {"q_path", "k_path", "rubin"}
            or self.hif4_sparse_upgrade and (
                effective_q_mode != "none" or effective_branch_mode != "none" or effective_rubin
            )
        )
        linear_hif4_ktv = (
            self.hif4_only_scope in {"k_path", "linear", "rubin"}
            or self.hif4_sparse_upgrade and any((
                effective_branch_mode != "none", effective_rubin,
                self.linear_kv_2to4_operand != "none",
            ))
        )
        linear_hif4_qkv = (
            self.hif4_only_scope in {"q_path", "linear"}
            or self.hif4_sparse_upgrade and any((
                effective_q_mode != "none", self.linear_q_2to4,
                self.linear_qkv_2to4_operand != "none",
            ))
        )
        
        def audit_operands(label, lhs, rhs, sparse_role):
            self._audit_hif4_operands(label, lhs, rhs, sparse_role)
            if self.operand_audit is not None:
                self.operand_audit(label, lhs, rhs, sparse_role)

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        linear_q, linear_k = q, k

        pairwise_weight_norm_k = effective_branch_mode == "4to8_pairwise" and self.k_weight_norm_rpq
        share2_weight_norm_k = effective_branch_mode == "2to4_share2" and self.k_weight_norm_rpq
        if (effective_weight_norm_k or pairwise_weight_norm_k) and self.k_weight_norm_rpq:
            if self.k_rpq_permutation is None:
                q_weight = q.float().square().sum(dim=-2).sqrt()
                score = k.float().abs() * q_weight[..., None, :]
                self.k_rpq_permutation = build_rpq_feature_permutation(
                    score, block=8 if pairwise_weight_norm_k else 4
                )
            permutation = self.k_rpq_permutation
            q = q.index_select(-1, permutation)
            k = k.index_select(-1, permutation)

        q_rpq_enabled = self.q_weight_norm_rpq and effective_q_mode in {"2to4", "4to8_pairwise", "2to4_share2"}
        if q_rpq_enabled:
            if self.q_rpq_permutation is None:
                score_q = q.float().abs() * k.float().square().sum(dim=-2).sqrt()[..., None, :]
                self.q_rpq_permutation = build_rpq_feature_permutation(score_q, block=8 if effective_q_mode == "4to8_pairwise" else 4)
            q = q.index_select(-1, self.q_rpq_permutation)
            k = k.index_select(-1, self.q_rpq_permutation)

        sparse_q = (
            apply_weight_norm_2_to_4_share_index_2(q, k.transpose(-1, -2))
            if effective_weight_norm_q and effective_q_mode == "2to4_share2" else
            apply_weight_norm_4_to_8_pairwise_sparsity(q, k.transpose(-1, -2))
            if effective_weight_norm_q and effective_q_mode == "4to8_pairwise" else
            apply_weight_norm_2_to_4_sparsity(q, k.transpose(-1, -2))
            if effective_weight_norm_q else
            branch_aware_q(q, effective_q_mode, norm_compensate=self.norm_compensate)
        )
        if profile_branch and (effective_q_mode != "none" or effective_weight_norm_q):
            record_internal_sparsity(self, "Q_sparse_branch_operand", q, sparse_q)
        dense_sparse_branch_k = k
        sparse_k = (
            apply_weight_norm_2_to_4_share_index_2(k, q.transpose(-1, -2))
            if share2_weight_norm_k else
            apply_weight_norm_4_to_8_pairwise_sparsity(k, q.transpose(-1, -2))
            if pairwise_weight_norm_k else
            apply_weight_norm_2_to_4_sparsity(k, q.transpose(-1, -2))
            if effective_weight_norm_k else
            branch_aware_sparse_k(
                k, "2to4" if effective_rubin else effective_branch_mode,
                norm_compensate=self.norm_compensate,
            )
        )
        if profile_branch and (effective_branch_mode != "none" or effective_weight_norm_k):
            record_internal_sparsity(self, "K_sparse_branch_operand", dense_sparse_branch_k, sparse_k)
        sparse_map, lut, real_topk = get_block_map(
            sparse_q, sparse_k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK
        )

        q = q.to(self.dtype)
        sparse_q = sparse_q.to(self.dtype)
        k = k.to(self.dtype)
        sparse_k = sparse_k.to(self.dtype)
        v = v.to(self.dtype)
        if qk_hif4:
            sparse_q = (
                hif4_qdq(sparse_q, -1, sparse_mask=sparse_q != 0, protect_sparse_mask=True)
                if effective_q_mode != "none" else hif4_qdq(sparse_q, -1)
            )
            sparse_k = (
                hif4_qdq(sparse_k, -1, sparse_mask=sparse_k != 0, protect_sparse_mask=True)
                if effective_branch_mode != "none" or effective_rubin else hif4_qdq(sparse_k, -1)
            )
            audit_operands("sparse_qk", sparse_q, sparse_k, "q" if effective_q_mode != "none" else "k" if effective_branch_mode != "none" or effective_rubin else None)
        p_audit = None
        fused_reference_sparse_output = None
        if effective_rubin or self.hif4_only_scope == "rubin":
            if self.rubin_sparse_engine == "fused":
                _strict_two_of_four(sparse_k, "sparse QK K")
                fused_result = rubin_2to4_attention_forward(
                    sparse_q, sparse_k, v, lut, real_topk, self.BLKQ, self.BLKK,
                    audit=self.rubin_validate_fused,
                    hif4=qk_hif4,
                    structured_p=effective_rubin,
                )
                if self.rubin_validate_fused:
                    o_s, fused_p_audit = fused_result
                    self._rubin_triple_audit["fused_p_operand_checked_elements"] += (
                        fused_p_audit["p_operand_checked_elements"]
                    )
                    self._rubin_triple_audit["fused_p_operand_group_violations"] += (
                        fused_p_audit["p_operand_group_violations"]
                    )
                else:
                    o_s = fused_result
                p_audit = None
                fused_reference_sparse_output = None
                if self.rubin_validate_fused:
                    fused_reference_sparse_output, p_audit = explicit_block_sparse_attention_rubin_2to4(
                        sparse_q, sparse_k, v, lut, self.BLKQ, self.BLKK
                    )
            else:
                o_s, p_audit = explicit_block_sparse_attention_rubin_2to4(
                    sparse_q, sparse_k, v, lut, self.BLKQ, self.BLKK
                )
                fused_reference_sparse_output = None
        else:
            o_s = _attention.apply(sparse_q, sparse_k, v, sparse_map, lut, real_topk, self.BLKQ, self.BLKK)

        q, k = linear_q, linear_k
        feature_dtype = torch.float32 if (linear_hif4_ktv or linear_hif4_qkv) else self.dtype
        q = self.feature_map_q(q.to(feature_dtype)).contiguous() # c_q
        k = self.feature_map_k(k.to(feature_dtype)).contiguous() # c_k
        if self.linear_q_2to4:
            q = apply_2_to_4_sparsity(q, norm_compensate=self.norm_compensate).contiguous()
        linear_k_audit = {}
        def audit_linear_k(dense_k, sparse_linear_k):
            _strict_two_of_four(sparse_linear_k.transpose(-1, -2), "linear K.T@V K")
            linear_k_audit["zero_rate"] = torch.count_nonzero(sparse_linear_k == 0).item() / sparse_linear_k.numel()

        if effective_weight_norm_q:
            kv_weight = k.transpose(-1, -2) @ v.to(k.dtype)
            q = apply_weight_norm_2_to_4_sparsity(q, kv_weight).contiguous()
        o_l = linear_attention_2to4(
            q, k, v,
            kv_sparse_operand=self.linear_kv_2to4_operand,
            qkv_sparse_operand=self.linear_qkv_2to4_operand,
            branch_aware_q_sparsity=effective_q_mode,
            branch_aware_k_sparsity="2to4" if effective_rubin else effective_branch_mode,
            norm_compensate=self.norm_compensate,
            weight_norm_k_2to4=effective_weight_norm_k,
            branch_aware_q_profile=(
                lambda dense_q, sparse_linear_q: record_internal_sparsity(
                    self, "Q_linear_QKV_operand", dense_q, sparse_linear_q
                )
            ) if profile_branch and effective_q_mode != "none" else None,
            branch_aware_k_profile=audit_linear_k if effective_rubin else (
                lambda dense_k, sparse_linear_k: record_internal_sparsity(
                    self, "K_linear_KtV_operand", dense_k, sparse_linear_k
                )
            ) if profile_branch and (effective_branch_mode != "none" or effective_weight_norm_k) else None,
            hif4_ktv=linear_hif4_ktv,
            hif4_qkv=linear_hif4_qkv,
            operand_audit=audit_operands if (linear_hif4_ktv or linear_hif4_qkv) else None,
        )

        if effective_rubin:
            qk_zero_rate = torch.count_nonzero(sparse_k == 0).item() / sparse_k.numel()
            stats = self._rubin_triple_audit
            stats["calls"] += 1
            stats["qk_k_zero_rate_min"] = min(stats["qk_k_zero_rate_min"], qk_zero_rate)
            stats["qk_k_zero_rate_max"] = max(stats["qk_k_zero_rate_max"], qk_zero_rate)
            # The fused kernel enforces score 2:4 by construction but does not
            # materialize P. Detailed P statistics are available in validation.
            if p_audit is not None:
                stats["pv_p_zero_rate_min"] = min(stats["pv_p_zero_rate_min"], p_audit["p_zero_rate"])
                stats["pv_p_zero_rate_max"] = max(stats["pv_p_zero_rate_max"], p_audit["p_zero_rate"])
            stats["linear_k_zero_rate_min"] = min(stats["linear_k_zero_rate_min"], linear_k_audit["zero_rate"])
            stats["linear_k_zero_rate_max"] = max(stats["linear_k_zero_rate_max"], linear_k_audit["zero_rate"])
            if p_audit is not None:
                stats["p_max_row_sum_error"] = max(stats["p_max_row_sum_error"], p_audit["p_max_row_sum_error"])
                stats["violations"] += p_audit["p_group_violations"]

        with torch.amp.autocast('cuda', dtype=self.dtype):
            o_l = self.proj_l(o_l)
        if fused_reference_sparse_output is not None:
            fused_layer = o_s.float() + o_l.float()
            reference_layer = fused_reference_sparse_output.float() + o_l.float()
            rel_l2 = (
                torch.linalg.vector_norm((fused_layer - reference_layer).reshape(-1))
                / torch.linalg.vector_norm(reference_layer.reshape(-1)).clamp_min(1e-12)
            ).item()
            if rel_l2 > 1e-3:
                raise RuntimeError(f"Rubin fused/reference layer relative L2 {rel_l2:.6g} exceeds 1e-3")
            self._rubin_triple_audit["fused_reference_rel_l2_max"] = max(
                self._rubin_triple_audit["fused_reference_rel_l2_max"], rel_l2
            )
        o = (o_s + o_l).to(dtype).transpose(1, 2)

        if return_sparsity:
            return o, real_topk / sparse_map.shape[-1]
        else:
            return o


class SageSparseLinearAttention(nn.Module):
    def __init__(
        self,
        head_dim,
        topk,
        feature_map='softmax',
        use_bf16=True,
        tie_feature_map_qk=True,
        linear_q_2to4=False,
        linear_kv_2to4_operand="none",
        linear_qkv_2to4_operand="none",
        branch_aware_q_sparsity="none",
        branch_aware_k_sparsity="none",
        norm_compensate=True,
    ):
        R'''
        Args:
            head_dim: dimension of each head.
            topk: ratio of keys selected for sparse attention, shared across all queries.
            feature_map: feature map for linear attention, one of ['hedgehog', 'elu', 'relu', 'softmax'].
            BLKQ: block size for query.
            BLKK: block size for key.
            use_bf16: whether to use bfloat16 (default) or float16 for computation. The conversion to bf16/fp16 is done inside the module.
            tie_feature_map_qk: whether to use the same feature map for query and key.
            timestep_adaptive_topk: whether to adaptively adjust topk during diffusion.
            linear_q_2to4: whether to simulate 2:4 activation sparsity on Q in the linear-attention branch.
        '''
        assert SAGESLA_ENABLED, "Install SpargeAttn first to enable SageSLA."

        super().__init__()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk = topk
        self.linear_q_2to4 = linear_q_2to4
        self.linear_kv_2to4_operand = linear_kv_2to4_operand
        self.linear_qkv_2to4_operand = linear_qkv_2to4_operand
        self.branch_aware_q_sparsity = branch_aware_q_sparsity
        self.branch_aware_k_sparsity = branch_aware_k_sparsity
        self.norm_compensate = norm_compensate
        if linear_q_2to4 and linear_qkv_2to4_operand != "none":
            raise ValueError("--linear_q_2to4 conflicts with --linear_qkv_2to4_operand")
        if branch_aware_k_sparsity != "none" and linear_kv_2to4_operand != "none":
            raise ValueError("branch-aware K sparsity conflicts with --linear_kv_2to4_operand")
        if branch_aware_q_sparsity != "none" and (linear_q_2to4 or linear_qkv_2to4_operand != "none"):
            raise ValueError("branch-aware Q sparsity conflicts with a separate linear Q sparse operand")
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=torch.float32)

        if feature_map == 'elu':
            def elu_feature_map(x):
                return F.elu(x) + 1
            self.feature_map_q = elu_feature_map
            self.feature_map_k = elu_feature_map
        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
        elif feature_map == 'softmax':
            def softmax_feature_map(x):
                return F.softmax(x, dim=-1)
            self.feature_map_q = softmax_feature_map
            self.feature_map_k = softmax_feature_map
        else:
            raise NotImplementedError(f'Not supported feature map {feature_map}.')

        if tie_feature_map_qk:
            self.feature_map_k = self.feature_map_q

        self.init_weights_()

    def init_weights_(self):
        with torch.no_grad():
            nn.init.zeros_(self.proj_l.weight)
            nn.init.zeros_(self.proj_l.bias)
        
    def forward(self, q, k, v, return_sparsity=False):
        R'''
        Args:
            q: queries of shape (B, H, L, D).
            k: keys of shape (B, H, L, D).
            v: values of shape (B, H, L, D).
            return_sparsity: whether to return the actual sparsity.
            timestep: current timestep for diffusion models.
            total_timesteps: total timesteps for diffusion models.
        '''
        
        dtype = q.dtype
        original_args = (q, k, v)
        profile_branch = (
            (self.branch_aware_q_sparsity != "none" or self.branch_aware_k_sparsity != "none")
            and begin_internal_sparsity_profile(self, original_args)
        )
        effective_q_mode = (
            self.branch_aware_q_sparsity
            if not getattr(self, "_sparsity_profile_dense_replay", False)
            else "none"
        )
        effective_branch_mode = (
            self.branch_aware_k_sparsity
            if not getattr(self, "_sparsity_profile_dense_replay", False)
            else "none"
        )
        
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        sparse_q = branch_aware_q(q, effective_q_mode, norm_compensate=self.norm_compensate)
        if profile_branch and effective_q_mode != "none":
            record_internal_sparsity(self, "Q_sparse_branch_operand", q, sparse_q)
        dense_sparse_branch_k = k
        sparse_k = branch_aware_sparse_k(k, effective_branch_mode, norm_compensate=self.norm_compensate)
        if profile_branch and effective_branch_mode != "none":
            record_internal_sparsity(self, "K_sparse_branch_operand", dense_sparse_branch_k, sparse_k)
        arch = get_cuda_arch(q.device.index)
        if arch == "sm90":
            sparse_map, lut, real_topk = get_block_map(sparse_q, sparse_k, topk_ratio=self.topk, BLKQ=64, BLKK=128)
        else:
            sparse_map, lut, real_topk = get_block_map(sparse_q, sparse_k, topk_ratio=self.topk, BLKQ=128, BLKK=64)

        q = q.to(self.dtype)
        sparse_q = sparse_q.to(self.dtype)
        k = k.to(self.dtype)
        sparse_k = sparse_k.to(self.dtype)
        v = v.to(self.dtype)

        ########## SPARGE BEGIN ##########

        km = sparse_k.mean(dim=-2, keepdim=True)
        headdim = q.size(-1)
        
        if arch == "sm90":
            q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(sparse_q, sparse_k, km, 64, 128)
        else:
            q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(sparse_q, sparse_k, km, 128, 64)
        lut, valid_block_num = block_map_lut_triton(sparse_map)
        scale = 1.0 / (headdim ** 0.5)

        assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

        o_s = torch.empty_like(q)

        if arch in ("sm80", "sm86", "sm87"):
            pvthreshold = torch.full((q.shape[-3],), 1e6, dtype=torch.float32, device=q.device)
            v_fp16 = v.to(torch.float16)
            qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
                q_int8, k_int8, v_fp16, o_s, lut, valid_block_num, pvthreshold, q_scale, k_scale, 1, False, 1, scale, 0
            )
        else:
            b, h_kv, kv_len, head_dim = v.shape
            padded_len = (kv_len + 127) // 128 * 128
            v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
            fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
            v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
            v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
            fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 2.25, 1)

            if arch == "sm90":
                qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90(
                    q_int8, k_int8, v_fp8, o_s, lut, valid_block_num, q_scale, k_scale, v_scale, 1, False, 1, scale
                )
            else:
                pvthreshold = torch.full((q.shape[-3],), 1e6, dtype=torch.float32, device=q.device)
                if SAGE2PP_ENABLED:
                    qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
                        q_int8, k_int8, v_fp8, o_s, lut, valid_block_num, pvthreshold, q_scale, k_scale, v_scale, 1, False, 1, scale, 0
                    )
                else:
                    qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
                        q_int8, k_int8, v_fp8, o_s, lut, valid_block_num, pvthreshold, q_scale, k_scale, v_scale, 1, False, 1, scale, 0
                    )

        ########## SPARGE END ##########

        q = self.feature_map_q(q).contiguous().to(self.dtype) # c_q
        k = self.feature_map_k(k).contiguous().to(self.dtype) # c_k
        if self.linear_q_2to4:
            q = apply_2_to_4_sparsity(q, norm_compensate=self.norm_compensate).contiguous()
        o_l = linear_attention_2to4(
            q, k, v,
            kv_sparse_operand=self.linear_kv_2to4_operand,
            qkv_sparse_operand=self.linear_qkv_2to4_operand,
            branch_aware_q_sparsity=effective_q_mode,
            branch_aware_k_sparsity=effective_branch_mode,
            norm_compensate=self.norm_compensate,
            branch_aware_q_profile=(
                lambda dense_q, sparse_linear_q: record_internal_sparsity(
                    self, "Q_linear_QKV_operand", dense_q, sparse_linear_q
                )
            ) if profile_branch and effective_q_mode != "none" else None,
            branch_aware_k_profile=(
                lambda dense_k, sparse_linear_k: record_internal_sparsity(
                    self, "K_linear_KtV_operand", dense_k, sparse_linear_k
                )
            ) if profile_branch and effective_branch_mode != "none" else None,
        )

        with torch.amp.autocast('cuda', dtype=self.dtype):
            o_l = self.proj_l(o_l)
        o = (o_s + o_l).to(dtype).transpose(1, 2)

        if return_sparsity:
            return o, real_topk / sparse_map.shape[-1]
        else:
            return o
