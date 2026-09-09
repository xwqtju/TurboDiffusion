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
import triton
import triton.language as tl
from triton.language.extra import libdevice



@triton.jit
def _hif4_qdq_128_last(x, support, OUTER: tl.constexpr, PROTECT: tl.constexpr):
    """HiF4 [16,2,4] QDQ for OUTER independent 128-wide rows."""
    x_shaped = tl.reshape(x, (OUTER, 16, 2, 4))
    shaped = tl.abs(x_shaped)
    max_lv3 = tl.max(shaped, axis=3, keep_dims=True)
    max_lv2 = tl.max(max_lv3, axis=2, keep_dims=True)
    max_lv1 = tl.max(max_lv2, axis=1, keep_dims=True)

    sf = (max_lv1 / 7.0).to(tl.bfloat16).to(tl.float32)
    sf = tl.minimum(tl.maximum(sf, 2.0 ** -48), 49152.0)
    exponent = tl.floor(tl.log2(sf))
    sf = tl.floor(sf / tl.exp2(exponent) * 128.0 + 0.5) / 128.0 * tl.exp2(exponent)
    sf = tl.floor(sf * tl.exp2(2.0 - exponent) + 0.5) * tl.exp2(exponent - 2.0)

    reciprocal_sf = (1.0 / sf).to(tl.bfloat16).to(tl.float32)
    scale_lv2 = tl.exp2(tl.floor(tl.minimum(tl.maximum(max_lv2 * reciprocal_sf, 0.0), 4.0) / 4.0))
    scale_lv3 = tl.exp2(tl.floor(tl.minimum(tl.maximum(max_lv3 * reciprocal_sf / scale_lv2, 0.0), 2.0) / 2.0))
    scale = sf * scale_lv2 * scale_lv3
    mantissa = tl.floor(shaped / scale * 4.0 + 0.5) / 4.0
    mantissa = tl.where(mantissa >= 2.0, 1.75, mantissa)
    quantized = tl.reshape(
        tl.where(x_shaped < 0, -mantissa * scale, mantissa * scale),
        (OUTER, 128),
    )
    if PROTECT:
        support_rows = tl.reshape(support, (OUTER, 128))
        quantum = tl.reshape(tl.broadcast_to(scale / 4.0, (OUTER, 16, 2, 4)), (OUTER, 128))
        quantized = tl.where(support_rows, quantized, 0.0)
        quantized = tl.where(support_rows & (quantized == 0.0), tl.where(x < 0, -quantum, quantum), quantized)
    return quantized

@triton.jit
def _rubin_2to4_attn_fwd(
    Q, K, V, LUT, OS, AUDIT,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    AUDIT_ENABLED: tl.constexpr,
    HIF4_ENABLED: tl.constexpr,
    P_SPARSE_ENABLED: tl.constexpr,
    P_MODE: tl.constexpr,  # 0=2:4, 1=4:8 pairwise, 2=2:4 shared by rows
):
    """Fused forward: QK -> score 2:4 -> online softmax -> PV.

    P is never materialized in global memory.  Every contiguous group of four
    score elements along the selected key/token dimension keeps exactly the
    two largest values before softmax.  Selected key blocks are 64-wide, so
    the groups are identical to grouping the packed reference scores.
    """
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)
    qkv_offset = idx_bh * L * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    k_ptrs = K + qkv_offset + offs_n[None, :] * D + offs_d[:, None]
    v_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    o_ptrs = OS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    lut_ptr = LUT + lut_offset

    q = tl.load(q_ptrs, mask=offs_m[:, None] < L, other=0.0)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    lane = tl.arange(0, 4)[None, None, :]

    # Pass 1: compute one global softmax normalizer across all selected blocks.
    for block_idx in tl.range(topk):
        idx_n = tl.load(lut_ptr + block_idx)
        valid_n = offs_n < L - idx_n * BLOCK_N
        k = tl.load(k_ptrs + idx_n * BLOCK_N * D, mask=valid_n[None, :], other=0.0)
        scores = tl.dot(q, k) * qk_scale
        scores = tl.where(valid_n[None, :], scores, -float("inf"))

        if P_SPARSE_ENABLED:
            if P_MODE == 0:
                groups = tl.reshape(scores, (BLOCK_M, BLOCK_N // 4, 4))
                first = tl.argmax(groups, axis=2, tie_break_left=True)
                keep_first = lane == first[:, :, None]
                second = tl.argmax(tl.where(keep_first, -float("inf"), groups), axis=2, tie_break_left=True)
                keep = keep_first | (lane == second[:, :, None])
            elif P_MODE == 1:
                # Build pair scores from the ordinary 2:4 groups.  Keeping
                # the reduction in rank-3 tensors avoids layout-dependent
                # rank-4 reshapes in Triton while preserving the logical
                # order [x0,x1],[x2,x3],[x4,x5],[x6,x7].
                groups4 = tl.reshape(scores, (BLOCK_M, BLOCK_N // 4, 4))
                pair_vals = tl.reshape(groups4, (BLOCK_M, BLOCK_N // 8, 2, 2, 2))
                pair_scores = tl.reshape(tl.sum(pair_vals, axis=4), (BLOCK_M, BLOCK_N // 8, 4))
                first = tl.argmax(pair_scores, axis=2, tie_break_left=True)
                keep_first = tl.arange(0, 4)[None, None, :] == first[:, :, None]
                second = tl.argmax(tl.where(keep_first, -float("inf"), pair_scores), axis=2, tie_break_left=True)
                keep_pair = keep_first | (tl.arange(0, 4)[None, None, :] == second[:, :, None])
                keep_pairs = tl.reshape(keep_pair, (BLOCK_M, BLOCK_N // 8, 2, 2))
                keep = tl.broadcast_to(keep_pairs[:, :, :, :, None], (BLOCK_M, BLOCK_N // 8, 2, 2, 2))
                keep = tl.reshape(keep, (BLOCK_M, BLOCK_N // 4, 4))
            else:
                groups = tl.reshape(scores, (BLOCK_M, BLOCK_N // 4, 4))
                shared = tl.reshape(groups, (BLOCK_M // 2, 2, BLOCK_N // 4, 4))
                shared_score = tl.sum(shared, axis=1)
                # shared_score has shape [BLOCK_M//2, groups, 4]; Triton
                # reductions are performed over the final (pair-choice)
                # axis.  The previous axis=3 was out of range and made the
                # share-index kernel fail at compile time.
                first = tl.argmax(shared_score, axis=2, tie_break_left=True)
                keep_first = tl.arange(0, 4)[None, None, :] == first[:, :, None]
                second = tl.argmax(tl.where(keep_first, -float("inf"), shared_score), axis=2, tie_break_left=True)
                keep_shared = keep_first | (tl.arange(0, 4)[None, None, :] == second[:, :, None])
                keep = tl.broadcast_to(keep_shared[:, None, :, :, :], (BLOCK_M // 2, 2, BLOCK_N // 4, 4))
                keep = tl.reshape(keep, (BLOCK_M, BLOCK_N // 4, 4))
            scores = tl.reshape(tl.where(keep, tl.reshape(scores, (BLOCK_M, BLOCK_N // 4, 4)), -float("inf")), (BLOCK_M, BLOCK_N))

        local_m = tl.max(scores, axis=1)
        new_m = tl.maximum(m_i, local_m)
        alpha = libdevice.exp(m_i - new_m)
        l_i = l_i * alpha + tl.sum(libdevice.exp(scores - new_m[:, None]), axis=1)
        m_i = new_m

    # Pass 2: HiF4 consumes LUT blocks in pairs so each QDQ reduction block is
    # exactly [selected block 2i (64), selected block 2i+1 (64)].  An odd tail
    # uses a masked all-zero second half.
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    if HIF4_ENABLED:
        offs_r = tl.arange(0, 128)
        local_r = offs_r % BLOCK_N
        for pair_idx in tl.range(0, topk, 2):
            slot = offs_r // BLOCK_N
            packed_pos = pair_idx + slot
            slot_valid = packed_pos < topk
            idx_n = tl.load(lut_ptr + packed_pos, mask=slot_valid, other=0)
            token = idx_n * BLOCK_N + local_r
            valid_r = slot_valid & (token < L)
            k_pair = tl.load(
                K + qkv_offset + token[None, :] * D + offs_d[:, None],
                mask=valid_r[None, :], other=0.0,
            )
            scores = tl.dot(q, k_pair) * qk_scale
            scores = tl.where(valid_r[None, :], scores, -float("inf"))
            if P_SPARSE_ENABLED:
                if P_MODE == 0:
                    groups = tl.reshape(scores, (BLOCK_M, 32, 4))
                    first = tl.argmax(groups, axis=2, tie_break_left=True)
                    keep_first = lane == first[:, :, None]
                    second = tl.argmax(tl.where(keep_first, -float("inf"), groups), axis=2, tie_break_left=True)
                    keep = keep_first | (lane == second[:, :, None])
                elif P_MODE == 1:
                    groups4 = tl.reshape(scores, (BLOCK_M, 32, 4))
                    pair_vals = tl.reshape(groups4, (BLOCK_M, 16, 2, 2, 2))
                    pair_scores = tl.reshape(tl.sum(pair_vals, axis=4), (BLOCK_M, 16, 4))
                    first = tl.argmax(pair_scores, axis=2, tie_break_left=True)
                    keep_first = tl.arange(0, 4)[None, None, :] == first[:, :, None]
                    second = tl.argmax(tl.where(keep_first, -float("inf"), pair_scores), axis=2, tie_break_left=True)
                    keep_pair = keep_first | (tl.arange(0, 4)[None, None, :] == second[:, :, None])
                    keep_pairs = tl.reshape(keep_pair, (BLOCK_M, 16, 2, 2))
                    keep = tl.broadcast_to(keep_pairs[:, :, :, :, None], (BLOCK_M, 16, 2, 2, 2))
                    keep = tl.reshape(keep, (BLOCK_M, 32, 4))
                else:
                    groups = tl.reshape(scores, (BLOCK_M, 32, 4))
                    shared = tl.reshape(groups, (BLOCK_M // 2, 2, 32, 4))
                    shared_score = tl.sum(shared, axis=1)
                    first = tl.argmax(shared_score, axis=2, tie_break_left=True)
                    keep_first = tl.arange(0, 4)[None, None, :] == first[:, :, None]
                    second = tl.argmax(tl.where(keep_first, -float("inf"), shared_score), axis=2, tie_break_left=True)
                    keep_shared = keep_first | (tl.arange(0, 4)[None, None, :] == second[:, :, None])
                    keep = tl.broadcast_to(keep_shared[:, None, :, :, :], (BLOCK_M // 2, 2, 32, 4))
                    keep = tl.reshape(keep, (BLOCK_M, 32, 4))
                keep_flat = tl.reshape(keep, (BLOCK_M, 128))
                valid_support = keep_flat & valid_r[None, :] & (offs_m[:, None] < L)
            else:
                valid_support = valid_r[None, :] & (offs_m[:, None] < L)
            sparse_scores = tl.where(valid_support, scores, -float("inf"))
            p = libdevice.exp(sparse_scores - m_i[:, None]) / l_i[:, None]
            p = tl.where(valid_support, p, 0.0)
            # The structured-P production path materializes softmax output
            # in the model dtype before it becomes the P@V activation.  HiF4
            # QDQ must consume that BF16/FP16 value, not the FP32 online-
            # softmax accumulator, otherwise it uses different hierarchy
            # scales from the explicit structured reference.  Keep the
            # legacy dense-P HiF4 path in FP32 for compatibility with its
            # established packed-128 oracle.
            if P_SPARSE_ENABLED:
                p = p.to(Q.type.element_ty)
            p = _hif4_qdq_128_last(
                # A selected probability may legitimately encode as the HiF4
                # zero value.  Its 2:4/4:8 *index* remains in the sparse
                # metadata, while reviving it to the hierarchy minimum would
                # add artificial probability mass and break normalization.
                p, valid_support, OUTER=BLOCK_M, PROTECT=False,
            )
            if Q.type.element_ty == tl.float16:
                p = tl.where(
                    valid_support & (tl.abs(p) < 2.0 ** -24),
                    tl.where(p < 0, -(2.0 ** -24), 2.0 ** -24),
                    p,
                )

            v_transposed = tl.load(
                V + qkv_offset + token[None, :] * D + offs_d[:, None],
                mask=valid_r[None, :], other=0.0,
            )
            v_transposed = _hif4_qdq_128_last(
                v_transposed, valid_r[None, :], OUTER=D, PROTECT=False,
            )
            p_operand = p.to(Q.type.element_ty)
            v_operand = tl.trans(v_transposed).to(V.type.element_ty)
            if AUDIT_ENABLED:
                if P_MODE == 1:
                    # Pairwise mode is a 2-of-4 pair pattern: audit groups
                    # of eight key tokens and require exactly four nonzeros.
                    p_groups8 = tl.reshape(p_operand, (BLOCK_M, 16, 8))
                    valid_groups8 = tl.reshape(valid_r, (16, 8))
                    fully_valid8 = tl.sum(valid_groups8.to(tl.int32), axis=1) == 8
                    nonzeros8 = tl.sum((p_groups8 != 0.0).to(tl.int32), axis=2)
                    checked8 = (offs_m[:, None] < L) & fully_valid8[None, :]
                    tl.atomic_add(AUDIT + 0, tl.sum(tl.where(checked8, nonzeros8 > 4, False).to(tl.int32)))
                    tl.atomic_add(AUDIT + 1, tl.sum(checked8.to(tl.int32)) * 8)
                    tl.atomic_add(AUDIT + 2, tl.sum(tl.where(checked8, nonzeros8, 0)))
                else:
                    p_groups = tl.reshape(p_operand, (BLOCK_M, 32, 4))
                    valid_groups = tl.reshape(valid_r, (32, 4))
                    fully_valid = tl.sum(valid_groups.to(tl.int32), axis=1) == 4
                    nonzeros = tl.sum((p_groups != 0.0).to(tl.int32), axis=2)
                    checked = (offs_m[:, None] < L) & fully_valid[None, :]
                    tl.atomic_add(AUDIT + 0, tl.sum(tl.where(checked, nonzeros > 2, False).to(tl.int32)))
                    tl.atomic_add(AUDIT + 1, tl.sum(checked.to(tl.int32)) * 4)
                    tl.atomic_add(AUDIT + 2, tl.sum(tl.where(checked, nonzeros, 0)))
            acc += tl.dot(p_operand, v_operand)
    else:
        for block_idx in tl.range(topk):
            idx_n = tl.load(lut_ptr + block_idx)
            valid_n = offs_n < L - idx_n * BLOCK_N
            k = tl.load(k_ptrs + idx_n * BLOCK_N * D, mask=valid_n[None, :], other=0.0)
            scores = tl.dot(q, k) * qk_scale
            scores = tl.where(valid_n[None, :], scores, -float("inf"))
            groups = tl.reshape(scores, (BLOCK_M, BLOCK_N // 4, 4))
            if P_SPARSE_ENABLED and P_MODE == 1:
                pair_vals = tl.reshape(groups, (BLOCK_M, BLOCK_N // 8, 2, 2, 2))
                pair_scores = tl.reshape(tl.sum(pair_vals, axis=4), (BLOCK_M, BLOCK_N // 8, 4))
                first = tl.argmax(pair_scores, axis=2, tie_break_left=True)
                keep_first = tl.arange(0, 4)[None, None, :] == first[:, :, None]
                second = tl.argmax(tl.where(keep_first, -float("inf"), pair_scores), axis=2, tie_break_left=True)
                keep_pair = keep_first | (tl.arange(0, 4)[None, None, :] == second[:, :, None])
                keep_pairs = tl.reshape(keep_pair, (BLOCK_M, BLOCK_N // 8, 2, 2))
                keep = tl.broadcast_to(keep_pairs[:, :, :, :, None], (BLOCK_M, BLOCK_N // 8, 2, 2, 2))
                keep = tl.reshape(keep, (BLOCK_M, BLOCK_N // 4, 4))
            elif P_SPARSE_ENABLED and P_MODE == 2:
                shared = tl.reshape(groups, (BLOCK_M // 2, 2, BLOCK_N // 4, 4))
                shared_score = tl.sum(shared, axis=1)
                first = tl.argmax(shared_score, axis=2, tie_break_left=True)
                keep_first = tl.arange(0, 4)[None, None, :] == first[:, :, None]
                second = tl.argmax(tl.where(keep_first, -float("inf"), shared_score), axis=2, tie_break_left=True)
                keep_shared = keep_first | (tl.arange(0, 4)[None, None, :] == second[:, :, None])
                keep = tl.broadcast_to(keep_shared[:, None, :, :], (BLOCK_M // 2, 2, BLOCK_N // 4, 4))
                keep = tl.reshape(keep, (BLOCK_M, BLOCK_N // 4, 4))
            else:
                first = tl.argmax(groups, axis=2, tie_break_left=True)
                keep_first = lane == first[:, :, None]
                second = tl.argmax(
                    tl.where(keep_first, -float("inf"), groups),
                    axis=2, tie_break_left=True,
                )
                keep = keep_first | (lane == second[:, :, None])
            scores = tl.reshape(tl.where(keep, groups, -float("inf")), (BLOCK_M, BLOCK_N))
            p = libdevice.exp(scores - m_i[:, None]) / l_i[:, None]
            v = tl.load(v_ptrs + idx_n * BLOCK_N * D, mask=valid_n[:, None], other=0.0)
            p_operand = p.to(v.dtype)
            if AUDIT_ENABLED:
                p_groups = tl.reshape(p_operand, (BLOCK_M, BLOCK_N // 4, 4))
                valid_groups = tl.reshape(valid_n, (BLOCK_N // 4, 4))
                fully_valid = tl.sum(valid_groups.to(tl.int32), axis=1) == 4
                nonzeros = tl.sum((p_groups != 0.0).to(tl.int32), axis=2)
                checked = (offs_m[:, None] < L) & fully_valid[None, :]
                tl.atomic_add(AUDIT + 0, tl.sum(tl.where(checked, nonzeros > 2, False).to(tl.int32)))
                tl.atomic_add(AUDIT + 1, tl.sum(checked.to(tl.int32)) * 4)
                tl.atomic_add(AUDIT + 2, tl.sum(tl.where(checked, nonzeros, 0)))
            acc += tl.dot(p_operand, v)

    tl.store(o_ptrs, acc.to(OS.type.element_ty), mask=offs_m[:, None] < L)


def rubin_2to4_attention_forward(
    q, k, v, lut, topk, block_m, block_n, qk_scale=None, audit=False, hif4=False, structured_p=True, p_mode="2to4",
):
    """RTX 5090-only, forward-only launcher for fused Rubin score sparsity."""
    tensors = {"q": q, "k": k, "v": v}
    if any(not tensor.is_cuda for tensor in tensors.values()):
        raise RuntimeError("Rubin fused attention requires CUDA tensors")
    if torch.cuda.get_device_capability(q.device) != (12, 0):
        raise RuntimeError("Rubin fused attention is validated only on the current sm_120 GPU")
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise RuntimeError("Rubin fused attention is inference-forward-only; gradients are unsupported")
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError(f"q/k/v must have one identical [B,H,L,D] shape, got {q.shape}, {k.shape}, {v.shape}")
    if q.dtype not in (torch.bfloat16, torch.float16) or any(t.dtype != q.dtype for t in tensors.values()):
        raise TypeError("Rubin fused attention supports matching BF16 or FP16 q/k/v only")
    if any(not tensor.is_contiguous() for tensor in tensors.values()) or not lut.is_contiguous():
        raise ValueError("Rubin fused attention requires contiguous q/k/v/lut")
    if lut.device != q.device or lut.dtype not in (torch.int32, torch.int64):
        raise ValueError("lut must be an int32/int64 tensor on the same CUDA device")
    if block_m not in (64, 128) or block_n != 64:
        raise ValueError(f"Rubin fused attention supports BLOCK_M=64/128 and BLOCK_N=64, got {block_m}/{block_n}")
    batch, heads, length, head_dim = q.shape
    if head_dim not in (64, 128) or length % 4:
        raise ValueError(f"Rubin fused attention supports D=64/128 and L divisible by 4, got D={head_dim}, L={length}")
    m_blocks = triton.cdiv(length, block_m)
    if lut.shape != (batch, heads, m_blocks, topk):
        raise ValueError(f"lut must have shape {(batch, heads, m_blocks, topk)}, got {tuple(lut.shape)}")
    if qk_scale is None:
        qk_scale = head_dim ** -0.5
    output = torch.empty_like(q)
    audit_counts = torch.zeros(3, device=q.device, dtype=torch.int32)
    p_mode_id = {"2to4": 0, "4to8_pairwise": 1, "2to4_share2": 2}[p_mode]
    _rubin_2to4_attn_fwd[(m_blocks, batch * heads)](
        q, k, v, lut, output, audit_counts, qk_scale, topk, length, m_blocks,
        head_dim, block_m, block_n,
        AUDIT_ENABLED=audit,
        HIF4_ENABLED=hif4,
        P_SPARSE_ENABLED=structured_p,
        P_MODE=p_mode_id,
        num_warps=4 if hif4 or head_dim == 64 else 8,
        num_stages=1 if hif4 or audit else 3,
    )
    if not audit:
        return output
    violations, checked_elements, nonzero_elements = audit_counts.tolist()
    if structured_p and violations:
        raise RuntimeError(
            f"Fused P@V operand exceeds the 2-of-4 index-mask budget in {violations} valid groups"
        )
    return output, {
        "p_operand_group_violations": violations,
        "p_operand_checked_elements": checked_elements,
        "p_operand_nonzero_elements": nonzero_elements,
        "p_operand_zero_rate_inside_valid_groups": (
            1.0 - nonzero_elements / checked_elements if checked_elements else 0.0
        ),
    }


@triton.jit
def _attn_fwd(
    Q, K, V,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT, LSE, OS,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    qkv_offset = idx_bh * L * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    lse_offset = idx_bh * L
    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + qkv_offset + offs_n[None, :] * D + offs_d[:, None]
    V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    OS_ptrs = OS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    LUT_ptr = LUT + lut_offset
    LSE_ptrs = LSE + lse_offset + offs_m
    
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L)
    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT_ptr + block_idx)
        n_mask = offs_n < L - idx_n * BLOCK_N
        
        k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[None, :])
        qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)  # = 1 / ln(2)
        if L - idx_n * BLOCK_N < BLOCK_N:
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

        v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        o_s += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < L)
    
    m_i += tl.math.log2(l_i)
    tl.store(LSE_ptrs, m_i, mask=offs_m < L)


@triton.jit
def _attn_bwd_preprocess(
    OS, DOS, DELTAS,
    L,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    OS += idx_bh * L * D
    DOS += idx_bh * L * D
    DELTAS += idx_bh * L

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    o_s = tl.load(OS + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < L)
    do_s = tl.load(DOS + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < L)
    
    delta_s = tl.sum(o_s * do_s, axis=1).to(DELTAS.type.element_ty)
    tl.store(DELTAS + offs_m, delta_s, mask=offs_m < L)


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    Q, K, V, LSE, DELTAS,
    DOS, DQ, LUT,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    qkv_offset = idx_bh * L * D
    lse_offset = idx_bh * L
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk

    Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    DQ_ptrs = DQ + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    DOS_ptrs = DOS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    LSE_ptrs = LSE + lse_offset + offs_m
    DELTAS_ptrs = DELTAS + lse_offset + offs_m
    LUT_ptr = LUT + lut_offset

    # load Q, DOS, DOL, LSE, DELTA, S: they stay in SRAM throughout the inner loop.
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L)
    do_s = tl.load(DOS_ptrs, mask=offs_m[:, None] < L)
    delta_s = tl.load(DELTAS_ptrs, mask=offs_m < L)
    lse = tl.load(LSE_ptrs, mask=offs_m < L, other=float("inf"))
    
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    for block_idx in tl.range(topk, num_stages=2):
        idx_n = tl.load(LUT_ptr + block_idx)
        n_mask = offs_n < L - idx_n * BLOCK_N
        
        k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        qk = tl.dot(q, k.T) * (qk_scale * 1.4426950408889634)  # = 1 / ln(2)
        p = tl.math.exp2(qk - lse[:, None])
        p = tl.where(n_mask[None, :], p, 0.0)
        
        # Compute dP and dS.
        dp = tl.dot(do_s, v.T).to(tl.float32)
        ds = p * (dp - delta_s[:, None])
        # Compute dQ.
        dq += tl.dot(ds.to(k.dtype), k)
    tl.store(DQ_ptrs, dq * qk_scale, mask=offs_m[:, None] < L)
    

@triton.jit
def _attn_bwd_dkdv(
    Q, K, V, DOS, DK, DV,
    qk_scale, KBID, LSE, DELTAS,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SLICE_FACTOR: tl.constexpr,
):
    BLOCK_M2: tl.constexpr = BLOCK_M // BLOCK_SLICE_FACTOR

    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M2)
    offs_d = tl.arange(0, D)

    qkv_offset = idx_bh * L * D
    kbid_offset = idx_bh * M_BLOCKS * N_BLOCKS
    lse_offset = idx_bh * L

    Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    DOS_ptrs = DOS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    DK_ptrs = DK + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    DV_ptrs = DV + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    LSE_ptrs = LSE + lse_offset + offs_m
    DELTAS_ptrs = DELTAS + lse_offset + offs_m
    KBID_ptr = KBID + kbid_offset + idx_n

    # load K, V and CK: they stay in SRAM throughout the inner loop.
    k = tl.load(K_ptrs, mask=offs_n[:, None] < L)
    v = tl.load(V_ptrs, mask=offs_n[:, None] < L)
        
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    for idx_m in tl.range(0, L, BLOCK_M2):
        kbid = tl.load(KBID_ptr)
        if kbid == 1:
            m_mask = offs_m < L - idx_m
            q = tl.load(Q_ptrs, mask=m_mask[:, None])
            lse = tl.load(LSE_ptrs, mask=m_mask, other=float("inf"))
            qkT = tl.dot(k, q.T) * (qk_scale * 1.4426950408889634)  # = 1 / ln(2)
            pT = tl.math.exp2(qkT - lse[None, :])
            pT = tl.where(offs_n[:, None] < L, pT, 0.0)

            do = tl.load(DOS_ptrs, mask=m_mask[:, None])
            # Compute dV.
            dv += tl.dot(pT.to(do.dtype), do)
            delta = tl.load(DELTAS_ptrs, mask=m_mask)
            # Compute dP and dS.
            dpT = tl.dot(v, tl.trans(do))
            dsT = pT * (dpT - delta[None, :])
            dk += tl.dot(dsT.to(q.dtype), q)
        
        # Increment pointers
        Q_ptrs += BLOCK_M2 * D
        DOS_ptrs += BLOCK_M2 * D
        LSE_ptrs += BLOCK_M2
        DELTAS_ptrs += BLOCK_M2
        if (idx_m + BLOCK_M2) % BLOCK_M == 0:
            KBID_ptr += N_BLOCKS

    # Write back dK, dV and dCK
    tl.store(DK_ptrs, dk * qk_scale, mask=offs_n[:, None] < L)
    tl.store(DV_ptrs, dv, mask=offs_n[:, None] < L)
    

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, k_block_id, lut, topk, BLOCK_M, BLOCK_N, qk_scale=None):
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert k_block_id.is_contiguous() and lut.is_contiguous()

        # We recommend the following two settings
        assert BLOCK_M == 64 or BLOCK_M == 128
        assert BLOCK_N == 64

        B, H, L, D = q.shape
        if qk_scale is None:
            qk_scale = D**-0.5

        M_BLOCKS = triton.cdiv(L, BLOCK_M)

        o_s = torch.empty_like(v)
        lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)

        grid = (M_BLOCKS, B * H)
        _attn_fwd[grid](
            q, k, v, qk_scale, topk,
            lut, lse, o_s,
            L, M_BLOCKS,
            D, BLOCK_M, BLOCK_N,
            num_warps=4 if q.shape[-1] == 64 else 8,
            num_stages=3
        )
        
        ctx.save_for_backward(q, k, v, k_block_id, lut, lse, o_s)
        ctx.qk_scale = qk_scale
        ctx.topk = topk
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        return o_s

    @staticmethod
    def backward(ctx, do_s):
        q, k, v, k_block_id, lut, lse, o_s = ctx.saved_tensors
        do_s = do_s.contiguous()

        BLOCK_M, BLOCK_N = ctx.BLOCK_M, ctx.BLOCK_N
        B, H, L, D = q.shape

        M_BLOCKS = triton.cdiv(L, BLOCK_M)
        N_BLOCKS = triton.cdiv(L, BLOCK_N)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta_s = torch.empty_like(lse)

        grid = (M_BLOCKS, B * H)
        _attn_bwd_preprocess[grid](
            o_s, do_s, delta_s,
            L, D, BLOCK_M,
        )

        grid = (M_BLOCKS, B * H)
        _attn_bwd_dq[grid](
            q, k, v, lse, delta_s,
            do_s, dq, lut,
            ctx.qk_scale, ctx.topk,
            L, M_BLOCKS,
            D, BLOCK_M, BLOCK_N,
            num_warps=4 if q.shape[-1] == 64 else 8,
            num_stages=4 if q.shape[-1] == 64 else 5
        )

        grid = (N_BLOCKS, B * H)
        _attn_bwd_dkdv[grid](
            q, k, v, do_s, dk, dv,
            ctx.qk_scale, k_block_id, lse, delta_s,
            L, M_BLOCKS, N_BLOCKS,
            D, BLOCK_M, BLOCK_N,
            BLOCK_SLICE_FACTOR=BLOCK_M // 64,
            num_warps=4 if q.shape[-1] == 64 else 8,
            num_stages=4 if q.shape[-1] == 64 else 5
        )

        return dq, dk, dv, None, None, None, None, None, None
