"""Milestone 1: scaled dot-product attention vs a hand-worked example and a reference."""
import math

import torch
import torch.nn.functional as F

from src.attention import causal_mask, scaled_dot_product_attention

# The 3-token example used in the report (experiments/hand_calculation.py)
Q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
K = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
V = torch.tensor([[1., 0.], [0., 2.], [3., 1.]])


def test_scores_match_hand_calculation():
    scores = Q @ K.T / math.sqrt(2)
    expected = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 2]]) / math.sqrt(2)
    assert torch.allclose(scores, expected.float(), atol=1e-6)


def test_weights_rows_sum_to_one_and_output_is_weighted_sum():
    out, w = scaled_dot_product_attention(Q, K, V)
    assert torch.allclose(w.sum(-1), torch.ones(3), atol=1e-6)
    assert torch.allclose(out, w @ V, atol=1e-6)


def test_matches_pytorch_reference_implementation():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 7, 8) for _ in range(3))
    ours, _ = scaled_dot_product_attention(q, k, v, causal_mask(7))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # used ONLY as a test oracle
    assert torch.allclose(ours, ref, atol=1e-5)


def test_unmasked_matches_reference():
    torch.manual_seed(1)
    q, k, v = (torch.randn(3, 5, 6) for _ in range(3))
    ours, _ = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(ours, F.scaled_dot_product_attention(q, k, v), atol=1e-5)


def test_scaling_by_sqrt_dk_matters():
    """Large d_k without scaling would give much peakier softmax rows."""
    torch.manual_seed(2)
    d = 256
    q, k = torch.randn(1, 8, d), torch.randn(1, 8, d)
    scaled = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d), -1)
    unscaled = torch.softmax(q @ k.transpose(-2, -1), -1)
    assert scaled.max(-1).values.mean() < unscaled.max(-1).values.mean()
