"""Milestones 2-3: tensor shapes through every transformation + positional encoding."""
import math

import pytest
import torch

from src.attention import MultiHeadAttention
from src.gpt import GPT, GPTConfig
from src.positional_encoding import LearnedPositionalEncoding, SinusoidalPositionalEncoding, sinusoidal_table
from src.transformer_block import TransformerBlock


def test_split_merge_roundtrip():
    mha = MultiHeadAttention(32, 4)
    x = torch.randn(2, 10, 32)
    h = mha.split_heads(x)
    assert h.shape == (2, 4, 10, 8)
    assert torch.equal(mha.merge_heads(h), x)


def test_split_heads_puts_contiguous_feature_chunks_in_each_head():
    mha = MultiHeadAttention(8, 2)
    x = torch.arange(8.).view(1, 1, 8)
    h = mha.split_heads(x)
    assert torch.equal(h[0, 0, 0], torch.tensor([0., 1., 2., 3.]))
    assert torch.equal(h[0, 1, 0], torch.tensor([4., 5., 6., 7.]))


def test_mha_shape_trace():
    mha = MultiHeadAttention(32, 4)
    mha.trace = []
    mha(torch.randn(2, 10, 32))
    shapes = dict(mha.trace)
    assert shapes["Q split into heads"] == (2, 4, 10, 8)
    assert shapes["attention weights (per head)"] == (2, 4, 10, 10)
    assert shapes["heads concatenated"] == (2, 10, 32)
    assert shapes["output projection W_o"] == (2, 10, 32)


def test_d_model_must_divide_by_heads():
    with pytest.raises(ValueError):
        MultiHeadAttention(30, 4)


def test_sinusoidal_values_and_shape():
    pe = sinusoidal_table(10, 8)
    assert pe.shape == (10, 8)
    assert torch.allclose(pe[0, 0::2], torch.zeros(4)) and torch.allclose(pe[0, 1::2], torch.ones(4))
    assert math.isclose(pe[3, 0].item(), math.sin(3.0), rel_tol=1e-5)
    assert math.isclose(pe[3, 1].item(), math.cos(3.0), rel_tol=1e-5)
    assert math.isclose(pe[3, 2].item(), math.sin(3.0 / 10000 ** (2 / 8)), rel_tol=1e-5)


def test_positions_are_distinct_and_added():
    enc = SinusoidalPositionalEncoding(16, 32)
    x = torch.zeros(1, 6, 16)
    y = enc(x)
    assert y.shape == x.shape
    assert len({tuple(r.tolist()) for r in y[0]}) == 6           # all positions differ
    assert len(list(enc.parameters())) == 0                      # not trainable
    assert len(list(LearnedPositionalEncoding(16, 32).parameters())) == 1


def test_block_preserves_shape():
    blk = TransformerBlock(32, 4, 128, 0.1)
    assert blk(torch.randn(3, 9, 32)).shape == (3, 9, 32)


@pytest.mark.parametrize("pos", ["sinusoidal", "learned"])
@pytest.mark.parametrize("tie", [False, True])
def test_gpt_shapes_and_initial_loss(pos, tie):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=50, context_length=24, d_model=32, n_heads=4, n_layers=2,
                    ffn_dim=128, dropout=0.0, pos_encoding=pos, tie_weights=tie)
    m = GPT(cfg)
    x = torch.randint(0, 50, (4, 24)); y = torch.randint(0, 50, (4, 24))
    logits, loss = m(x, y)
    assert logits.shape == (4, 24, 50)
    # An untrained model should be close to uniform guessing: loss ~ ln(V).
    # With tied weights the residual stream still contains the input token's own embedding,
    # and logits = x . E^T score that same token highly, so the untrained model "echoes" its
    # input and starts above ln(V). This is expected (optional experiment only).
    tol = 2.0 if tie else 0.3
    assert abs(loss.item() - math.log(50)) < tol


def test_parameter_count_formula():
    cfg = GPTConfig(vocab_size=66, context_length=128, d_model=128, n_heads=4, n_layers=4,
                    ffn_dim=512, tie_weights=False)
    d, f, V, L = 128, 512, 66, 4
    per_block = (4 * (d * d + d)) + (d * f + f) + (f * d + d) + 2 * (2 * d)   # attn + ffn + 2 LN
    expected = V * d + L * per_block + 2 * d + d * V                          # emb + blocks + ln_f + head
    assert GPT(cfg).num_parameters() == expected


def test_sequence_longer_than_context_raises():
    m = GPT(GPTConfig(vocab_size=10, context_length=8, d_model=16, n_heads=2, n_layers=1, ffn_dim=32))
    with pytest.raises(ValueError):
        m(torch.zeros(1, 9, dtype=torch.long))
