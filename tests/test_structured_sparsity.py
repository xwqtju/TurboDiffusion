import unittest

import torch
from torch import nn

from rcm.utils.structured_sparsity import (
    QActivation2To4Attention,
    enable_q_activation_2_to_4,
    sparsify_2_to_4,
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


if __name__ == "__main__":
    unittest.main()
