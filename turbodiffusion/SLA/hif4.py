"""Numerical HiF4 W4/A4 fake quantization for attention GEMM operands.

This module deliberately returns a dense-layout floating tensor.  It models
the quantize/dequantize numerics only; it does not claim packed 4-bit storage
or sparse Tensor-Core execution.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


HIF4_BLOCK_SIZE = 128


def _round_half_up(x: torch.Tensor) -> torch.Tensor:
    return torch.floor(x + 0.5)


def hif4_qdq(
    x: torch.Tensor,
    reduction_dim: int = -1,
    *,
    sparse_mask: torch.Tensor | None = None,
    protect_sparse_mask: bool = False,
    _max_chunk_elements: int = 4 * 1024 * 1024,
) -> torch.Tensor:
    """Apply block-128 HiF4 fake quantization along a GEMM K dimension.

    Each consecutive reduction block is arranged as ``[16, 2, 4]``.  The
    four-element group owns the local scale, two adjacent groups own the
    level-2 scale, and all 128 elements own the top scale.  A short final
    block is zero padded and trimmed after QDQ.

    When ``sparse_mask`` is supplied, zeros outside the pre-quantization mask
    are restored exactly.  With ``protect_sparse_mask=True``, a retained value
    that rounds to zero is restored to the smallest representable nonzero at
    that position, preserving strict structured sparsity at the GEMM input.
    """
    if not x.is_floating_point():
        raise TypeError(f"HiF4 expects a floating tensor, got {x.dtype}")
    if x.numel() == 0:
        return x.clone()
    dim = reduction_dim if reduction_dim >= 0 else x.ndim + reduction_dim
    if dim < 0 or dim >= x.ndim:
        raise IndexError(f"reduction_dim {reduction_dim} is invalid for rank {x.ndim}")
    if sparse_mask is not None and sparse_mask.shape != x.shape:
        raise ValueError("sparse_mask must have the same shape as x")
    if sparse_mask is not None and sparse_mask.dtype is not torch.bool:
        raise TypeError("sparse_mask must be boolean")
    if protect_sparse_mask and sparse_mask is None:
        raise ValueError("protect_sparse_mask requires sparse_mask")
    if protect_sparse_mask and torch.any(sparse_mask & (x == 0)).item():
        raise ValueError(
            "sparse_mask selects an original exact zero; HiF4 support protection "
            "may only restore retained values that were nonzero before QDQ"
        )

    arranged = x.movedim(dim, -1).contiguous()
    length = arranged.shape[-1]
    rows = arranged.numel() // length
    rows_per_chunk = max(1, _max_chunk_elements // length)
    if rows > rows_per_chunk:
        flat = arranged.reshape(rows, length)
        flat_mask = None
        if sparse_mask is not None:
            flat_mask = sparse_mask.movedim(dim, -1).contiguous().reshape(rows, length)
        chunked = torch.empty_like(flat)
        for start in range(0, rows, rows_per_chunk):
            stop = min(start + rows_per_chunk, rows)
            chunked[start:stop] = hif4_qdq(
                flat[start:stop],
                -1,
                sparse_mask=None if flat_mask is None else flat_mask[start:stop],
                protect_sparse_mask=protect_sparse_mask,
                _max_chunk_elements=_max_chunk_elements,
            )
        result = chunked.reshape_as(arranged)
        return result.movedim(-1, dim).contiguous()

    padded_length = math.ceil(length / HIF4_BLOCK_SIZE) * HIF4_BLOCK_SIZE
    work = F.pad(arranged.float(), (0, padded_length - length))
    blocks = work.reshape(-1, padded_length // HIF4_BLOCK_SIZE, 16, 2, 4)

    magnitude = blocks.abs()
    max_lv3 = magnitude.amax(dim=-1, keepdim=True)
    max_lv2 = max_lv3.amax(dim=-2, keepdim=True)
    max_lv1 = max_lv2.amax(dim=-3, keepdim=True)

    # Match the local HiF4 reference: BF16-rounded E6M2-like top scale.
    sf = (max_lv1 / 7.0).to(torch.bfloat16).float().clamp(2.0 ** -48, 49152.0)
    exponent = torch.floor(torch.log2(sf))
    sf = _round_half_up(sf / torch.exp2(exponent) * 128.0) / 128.0 * torch.exp2(exponent)
    sf = _round_half_up(sf * torch.exp2(2.0 - exponent)) * torch.exp2(exponent - 2.0)

    reciprocal_sf = (1.0 / sf).to(torch.bfloat16).float()
    shift_lv2 = torch.floor((max_lv2 * reciprocal_sf).clamp(0.0, 4.0) / 4.0)
    scale_lv2 = torch.exp2(shift_lv2)
    shift_lv3 = torch.floor((max_lv3 * reciprocal_sf / scale_lv2).clamp(0.0, 2.0) / 2.0)
    scale_lv3 = torch.exp2(shift_lv3)

    quantum = sf * scale_lv2 * scale_lv3 / 4.0
    mantissa = _round_half_up(magnitude / (sf * scale_lv2 * scale_lv3) * 4.0) / 4.0
    mantissa = torch.where(mantissa >= 2.0, torch.full_like(mantissa, 1.75), mantissa)
    qdq = torch.copysign(mantissa * sf * scale_lv2 * scale_lv3, blocks)

    if sparse_mask is not None:
        arranged_mask = sparse_mask.movedim(dim, -1).contiguous()
        mask = F.pad(arranged_mask, (0, padded_length - length), value=False)
        mask = mask.reshape_as(blocks)
        qdq = qdq.masked_fill(~mask, 0.0)
        if protect_sparse_mask:
            sign = torch.where(blocks < 0, -torch.ones_like(blocks), torch.ones_like(blocks))
            qdq = torch.where(mask & (qdq == 0), sign * quantum, qdq)

    result = qdq.reshape(*arranged.shape[:-1], padded_length)[..., :length]
    result = result.to(x.dtype)
    if sparse_mask is not None:
        arranged_mask = sparse_mask.movedim(dim, -1).contiguous()
        result = result.masked_fill(~arranged_mask, 0)
        if protect_sparse_mask:
            underflow = arranged_mask & (result == 0)
            if underflow.any():
                original = arranged[..., :length]
                signs = torch.where(
                    original < 0, -torch.ones_like(result), torch.ones_like(result)
                )
                hierarchy_floor = quantum.expand_as(blocks).reshape(
                    *arranged.shape[:-1], padded_length
                )[..., :length].to(x.dtype)
                if torch.any(underflow & (hierarchy_floor == 0)).item():
                    raise RuntimeError(
                        "HiF4 hierarchy minimum is not representable in the output dtype"
                    )
                result = torch.where(underflow, signs * hierarchy_floor, result)
    return result.movedim(-1, dim).contiguous()


def hif4_sparse_qdq(x: torch.Tensor, reduction_dim: int) -> torch.Tensor:
    """QDQ a pre-sparsified operand while preserving its exact support."""
    return hif4_qdq(
        x,
        reduction_dim,
        sparse_mask=x != 0,
        protect_sparse_mask=True,
    )
