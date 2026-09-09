"""Checks for the SLA score-mask and P@V structured operand paths."""

import pytest
import torch

from SLA.core import explicit_block_sparse_attention_pv_structured


@pytest.mark.parametrize("mode", ["2to4", "4to8_pairwise", "2to4_share2"])
def test_pv_reference_modes_have_expected_structured_density(mode):
    torch.manual_seed(123)
    q = torch.randn(1, 1, 8, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    lut = torch.tensor([[[[0, 1]]]], dtype=torch.int64)

    output, audit = explicit_block_sparse_attention_pv_structured(
        q, k, v, lut, block_q=8, block_k=4, mode=mode,
    )

    assert output.shape == q.shape
    assert audit["p_group_violations"] == 0
    assert audit["p_zero_rate"] == pytest.approx(0.5)
    assert audit["p_max_row_sum_error"] < 1e-6


@pytest.mark.parametrize("mode", ["2to4", "4to8_pairwise", "2to4_share2"])
def test_pv_fused_modes_enforce_mask_on_sm120(mode):
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)):
        pytest.skip("Rubin fused kernel requires sm_120")
    from SLA.kernel import rubin_2to4_attention_forward

    torch.manual_seed(321)
    q = torch.randn(1, 1, 128, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    lut = torch.tensor([[[[0, 1]]]], device="cuda", dtype=torch.int32)

    output, audit = rubin_2to4_attention_forward(
        q, k, v, lut, topk=2, block_m=128, block_n=64,
        hif4=True, structured_p=True, p_mode=mode, audit=True,
    )

    assert output.shape == q.shape
    assert torch.isfinite(output).all()
    assert audit["p_operand_group_violations"] == 0
    # HiF4 may encode a selected tiny probability as zero.  The selected
    # 2:4/4:8 indices are still valid sparse metadata, so numeric density can
    # only be lower than (never higher than) the nominal 50%.
    assert audit["p_operand_zero_rate_inside_valid_groups"] >= 0.5


@pytest.mark.parametrize("mode", ["2to4", "4to8_pairwise", "2to4_share2"])
def test_pv_fused_hif4_matches_structured_bf16_oracle(mode):
    """The fused path must QDQ the same BF16 P/V operands as SLA does."""
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)):
        pytest.skip("Rubin fused kernel requires sm_120")
    from SLA.kernel import rubin_2to4_attention_forward

    torch.manual_seed(707)
    q = torch.randn(1, 1, 256, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    # Two query blocks and an odd top-k exercise both packed-128 and padded
    # HiF4 reductions.
    lut = torch.tensor([[[[0, 2, 3], [1, 2, 3]]]], device="cuda", dtype=torch.int32)
    reference, _ = explicit_block_sparse_attention_pv_structured(
        q, k, v, lut, block_q=128, block_k=64, mode=mode, hif4_pv=True,
    )
    fused = rubin_2to4_attention_forward(
        q, k, v, lut, topk=3, block_m=128, block_n=64,
        hif4=True, structured_p=True, p_mode=mode,
    )
    relative_l2 = (fused.float() - reference.float()).norm() / reference.float().norm()
    assert relative_l2.item() < 1e-3
