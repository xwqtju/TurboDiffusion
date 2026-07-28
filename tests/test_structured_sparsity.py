import unittest

import torch
from torch import nn

from rcm.utils.structured_sparsity import (
    QActivation2To4Attention,
    enable_q_activation_2_to_4,
    enable_q_activation_4_to_8_pairwise,
    enable_k_activation_2_to_4,
    enable_k_activation_4_to_8_pairwise,
    sparsify_2_to_4,
    sparsify_4_to_8_pairwise,
    sparsify_2_to_4_share_index_2,
    enable_q_activation_2_to_4_share_index_2,
    enable_k_activation_2_to_4_share_index_2,
)


class _CaptureAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = None

    def forward(self, query, key, value):
        self.query = query
        return value


class StructuredSparsityTest(unittest.TestCase):
    def test_keeps_two_largest_magnitudes_per_group(self):
        x = torch.tensor([[[[1.0, -4.0, 3.0, 2.0, -8.0, 5.0, 7.0, 6.0]]]])

        result = sparsify_2_to_4(x)

        expected = torch.tensor([[[[0.0, -4.0, 3.0, 0.0, -8.0, 0.0, 7.0, 0.0]]]])
        torch.testing.assert_close(result, expected)
        self.assertEqual(torch.count_nonzero(result.reshape(-1, 4), dim=-1).tolist(), [2, 2])

    def test_rejects_incompatible_head_dimension(self):
        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            sparsify_2_to_4(torch.ones(1, 1, 1, 6))

    def test_attention_wrapper_sparsifies_query(self):
        attention = _CaptureAttention()
        wrapped = QActivation2To4Attention(attention)
        query = torch.tensor([[[[1.0, 4.0, 3.0, 2.0]]]])
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        result = wrapped(query, key, value)

        torch.testing.assert_close(attention.query, torch.tensor([[[[0.0, 4.0, 3.0, 0.0]]]]))
        torch.testing.assert_close(result, value)

    def test_hook_preserves_state_dict_keys(self):
        class ParameterizedAttention(_CaptureAttention):
            def __init__(self):
                super().__init__()
                self.proj_l = nn.Linear(4, 4)

        attention = ParameterizedAttention()
        keys_before = set(attention.state_dict())
        enable_q_activation_2_to_4(attention)
        result = attention(
            torch.tensor([[[[1.0, 4.0, 3.0, 2.0]]]]),
            torch.zeros(1, 1, 1, 4),
            torch.ones(1, 1, 1, 4),
        )

        self.assertEqual(set(attention.state_dict()), keys_before)
        torch.testing.assert_close(attention.query, torch.tensor([[[[0.0, 4.0, 3.0, 0.0]]]]))
        torch.testing.assert_close(result, torch.ones(1, 1, 1, 4))

    def test_pairwise_4_to_8_keeps_two_highest_scoring_pairs(self):
        x = torch.tensor([[[[8.0, 1.0, 4.0, 4.0, -7.0, 3.0, 2.0, -1.0]]]])

        result = sparsify_4_to_8_pairwise(x)

        expected = torch.tensor([[[[8.0, 1.0, 0.0, 0.0, -7.0, 3.0, 0.0, 0.0]]]])
        torch.testing.assert_close(result, expected)
        pairs = result.reshape(-1, 4, 2)
        self.assertEqual(torch.count_nonzero(pairs, dim=-1).gt(0).sum(dim=-1).tolist(), [2])

    def test_pairwise_4_to_8_rejects_incompatible_head_dimension(self):
        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            sparsify_4_to_8_pairwise(torch.ones(1, 1, 1, 12))

    def test_pairwise_4_to_8_hook_preserves_state_dict_keys(self):
        attention = _CaptureAttention()
        keys_before = set(attention.state_dict())
        enable_q_activation_4_to_8_pairwise(attention)
        query = torch.tensor([[[[8.0, 1.0, 4.0, 4.0, -7.0, 3.0, 2.0, -1.0]]]])
        value = torch.ones_like(query)

        result = attention(query, torch.zeros_like(query), value)

        self.assertEqual(set(attention.state_dict()), keys_before)
        expected = torch.tensor([[[[8.0, 1.0, 0.0, 0.0, -7.0, 3.0, 0.0, 0.0]]]])
        torch.testing.assert_close(attention.query, expected)
        torch.testing.assert_close(result, value)

    def test_k_2_to_4_hook_sparsifies_only_key(self):
        attention = _CaptureAttention()
        query = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        key = torch.tensor([[[[1.0, -4.0, 3.0, 2.0]]]])
        value = torch.ones_like(query)
        captured = {}

        def capture_key(_module, args):
            captured["key"] = args[1]

        enable_k_activation_2_to_4(attention)
        attention.register_forward_pre_hook(capture_key)
        attention(query, key, value)

        torch.testing.assert_close(attention.query, query)
        torch.testing.assert_close(captured["key"], torch.tensor([[[[0.0, -4.0, 3.0, 0.0]]]]))

    def test_k_4_to_8_pairwise_hook_sparsifies_only_key(self):
        attention = _CaptureAttention()
        query = torch.arange(8.0).reshape(1, 1, 1, 8)
        key = torch.tensor([[[[8.0, 1.0, 4.0, 4.0, -7.0, 3.0, 2.0, -1.0]]]])
        value = torch.ones_like(query)
        captured = {}

        def capture_key(_module, args):
            captured["key"] = args[1]

        enable_k_activation_4_to_8_pairwise(attention)
        attention.register_forward_pre_hook(capture_key)
        attention(query, key, value)

        torch.testing.assert_close(attention.query, query)
        expected = torch.tensor([[[[8.0, 1.0, 0.0, 0.0, -7.0, 3.0, 0.0, 0.0]]]])
        torch.testing.assert_close(captured["key"], expected)

    def test_share_index_2_uses_l1_across_two_tokens(self):
        x = torch.tensor([[
            [[10.0, 6.0, 9.0, 0.0]],
            [[0.1, 6.0, 0.0, 0.0]],
        ]])

        result = sparsify_2_to_4_share_index_2(x)

        expected = torch.tensor([[
            [[10.0, 6.0, 0.0, 0.0]],
            [[0.1, 6.0, 0.0, 0.0]],
        ]])
        torch.testing.assert_close(result, expected)
        masks = result.ne(0)
        self.assertEqual(masks[0, 0, 0].tolist(), [True, True, False, False])
        self.assertEqual(masks[0, 1, 0].tolist(), [True, True, False, False])

    def test_share_index_2_handles_odd_tail_independently(self):
        x = torch.tensor([[
            [[10.0, 6.0, 9.0, 0.0]],
            [[0.1, 6.0, 0.0, 0.0]],
            [[1.0, 4.0, 3.0, 2.0]],
        ]])
        result = sparsify_2_to_4_share_index_2(x)
        torch.testing.assert_close(result[0, 2], torch.tensor([[0.0, 4.0, 3.0, 0.0]]))

    def test_q_and_k_share_index_hooks_target_correct_arguments(self):
        query = torch.tensor([[[[10.0, 6.0, 9.0, 0.0]], [[0.1, 6.0, 0.0, 0.0]]]])
        key = query.flip(-1)
        value = torch.ones_like(query)

        q_attention = _CaptureAttention()
        enable_q_activation_2_to_4_share_index_2(q_attention)
        q_attention(query, key, value)
        torch.testing.assert_close(q_attention.query, sparsify_2_to_4_share_index_2(query))

        k_attention = _CaptureAttention()
        captured = {}
        enable_k_activation_2_to_4_share_index_2(k_attention)
        def capture_key(_module, args):
            captured["key"] = args[1]
        k_attention.register_forward_pre_hook(capture_key)
        k_attention(query, key, value)
        torch.testing.assert_close(k_attention.query, query)
        torch.testing.assert_close(captured["key"], sparsify_2_to_4_share_index_2(key))


if __name__ == "__main__":
    unittest.main()
