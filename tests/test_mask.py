"""Causal masking: future positions must receive exactly zero attention / influence."""
import torch

from src.attention import MultiHeadAttention, causal_mask, scaled_dot_product_attention
from src.gpt import GPT, GPTConfig


def test_mask_is_strictly_upper_triangular():
    m = causal_mask(5)
    assert m.dtype == torch.bool and m.shape == (5, 5)
    assert torch.equal(m, torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1))
    assert not m.diagonal().any()          # a token may attend to itself


def test_future_positions_get_zero_weight():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 6, 4) for _ in range(3))
    _, w = scaled_dot_product_attention(q, k, v, causal_mask(6))
    assert torch.all(torch.triu(w, diagonal=1) == 0)
    assert torch.allclose(w.sum(-1), torch.ones(2, 3, 6), atol=1e-6)


def test_first_token_attends_only_to_itself():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 4, 4) for _ in range(3))
    out, w = scaled_dot_product_attention(q, k, v, causal_mask(4))
    assert torch.allclose(w[0, 0, 0], torch.tensor([1., 0., 0., 0.]))
    assert torch.allclose(out[0, 0, 0], v[0, 0, 0])


def test_mha_output_independent_of_future_inputs():
    torch.manual_seed(0)
    mha = MultiHeadAttention(16, 4).eval()
    x = torch.randn(1, 8, 16)
    y1 = mha(x, causal_mask(8))
    x2 = x.clone(); x2[:, 5:] = torch.randn(1, 3, 16)         # change positions 5..7
    y2 = mha(x2, causal_mask(8))
    assert torch.allclose(y1[:, :5], y2[:, :5], atol=1e-6)    # positions <5 unchanged
    assert not torch.allclose(y1[:, 5:], y2[:, 5:])


def test_full_model_is_causal():
    """Acceptance test 'causal correctness': logits at t depend only on tokens <= t."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=30, context_length=16, d_model=32, n_heads=4, n_layers=3,
                    ffn_dim=64, dropout=0.0)
    m = GPT(cfg).eval()
    a = torch.randint(0, 30, (2, 16))
    b = a.clone(); b[:, 9:] = torch.randint(0, 30, (2, 7))    # change the future of position 8
    la, _ = m(a); lb, _ = m(b)
    assert torch.allclose(la[:, :9], lb[:, :9], atol=1e-5)
    assert not torch.allclose(la[:, 9:], lb[:, 9:])


def test_bidirectional_model_is_NOT_causal():
    """Sanity check of the MLM-vs-CLM story: without the mask the future leaks in."""
    cfg = GPTConfig(vocab_size=30, context_length=16, d_model=32, n_heads=4, n_layers=2,
                    ffn_dim=64, dropout=0.0, causal=False)
    torch.manual_seed(0)
    m = GPT(cfg).eval()
    a = torch.randint(0, 30, (1, 16)); b = a.clone(); b[:, 9:] = (b[:, 9:] + 1) % 30
    la, _ = m(a); lb, _ = m(b)
    assert not torch.allclose(la[:, :9], lb[:, :9], atol=1e-5)
