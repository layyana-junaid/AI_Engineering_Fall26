"""Milestone 4 debugging gate: loss must collapse on a tiny repeated batch.

This proves gradients flow and that the target shifting is correct (a model can only
memorise a batch if input and target are aligned consistently).
"""
import torch

from src.dataset import get_batch, make_shifted
from src.gpt import GPT, GPTConfig


def test_targets_are_inputs_shifted_by_one():
    data = torch.arange(1000)
    x, y = get_batch(data, 4, 16, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(y[:, :-1], x[:, 1:])          # y[t] == x[t+1]
    assert torch.equal(y, x + 1)                     # on arange data, next token = token + 1
    w = torch.arange(6)
    xi, yi = make_shifted(w)
    assert xi.tolist() == [0, 1, 2, 3, 4] and yi.tolist() == [1, 2, 3, 4, 5]


def test_tiny_overfit():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=66, context_length=32, d_model=64, n_heads=4, n_layers=2,
                    ffn_dim=256, dropout=0.0)
    model = GPT(cfg)
    data = torch.randint(0, 66, (2000,))
    x, y = get_batch(data, 8, 32, torch.Generator().manual_seed(0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(200):
        _, loss = model(x, y)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert first > 3.5
    assert loss.item() < 0.1 * first, f"loss only fell {first:.3f} -> {loss.item():.3f}"
