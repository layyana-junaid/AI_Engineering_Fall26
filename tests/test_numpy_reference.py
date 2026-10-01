"""Cross-check: the hand-written NumPy backward pass (reference/numpy_llm/core.py)
must agree with PyTorch autograd on the same maths.

This ties the original from-scratch NumPy implementation (manual chain rule) to the
PyTorch port and is evidence that the attention gradients are derived correctly.
"""
import sys

import numpy as np
import pytest
import torch

from src.utils import ROOT

sys.path.insert(0, str(ROOT / "reference" / "numpy_llm"))
core = pytest.importorskip("core")


def t(a, grad=True):
    return torch.tensor(a, dtype=torch.float64, requires_grad=grad)


def test_attention_forward_and_backward_match_autograd():
    rng = np.random.default_rng(0)
    B, T, D, H = 2, 5, 8, 2
    X = rng.standard_normal((B, T, D)) * 0.5
    params = {}
    for n in ("q", "k", "v", "o"):
        params["W" + n] = rng.standard_normal((D, D)) * 0.3
        params["b" + n] = rng.standard_normal((1, D)) * 0.1
    mask = core.causal_mask(T)
    out, cache = core.attention_forward(X, params, H, mask)
    dOut = rng.standard_normal(out.shape)
    dX, grads = core.attention_backward(dOut, cache, params)

    # same computation in PyTorch autograd (float64)
    Xt = t(X); P = {k: t(v) for k, v in params.items()}
    q, k, v = (Xt @ P["W" + n] + P["b" + n] for n in "qkv")
    sp = lambda z: z.view(B, T, H, D // H).transpose(1, 2)
    s = sp(q) @ sp(k).transpose(-1, -2) / np.sqrt(D // H)
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    o = (torch.softmax(s, -1) @ sp(v)).transpose(1, 2).reshape(B, T, D)
    ref = o @ P["Wo"] + P["bo"]
    ref.backward(torch.tensor(dOut))

    assert np.allclose(out, ref.detach().numpy(), atol=1e-6)
    assert np.allclose(dX, Xt.grad.numpy(), atol=1e-6)
    for n in ("q", "k", "v", "o"):
        assert np.allclose(grads["dW" + n], P["W" + n].grad.numpy(), atol=1e-6), n
        assert np.allclose(grads["db" + n], P["b" + n].grad.numpy(), atol=1e-6), n


def test_layer_norm_backward_matches_autograd():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2, 3, 6)); g = rng.standard_normal((1, 1, 6)); b = rng.standard_normal((1, 1, 6))
    out, cache = core.layer_norm_forward(X, g, b)
    dOut = rng.standard_normal(out.shape)
    dX, dg, db = core.layer_norm_backward(dOut, cache)
    Xt, gt, bt = t(X), t(g), t(b)
    ref = torch.nn.functional.layer_norm(Xt, (6,), eps=1e-5) * gt + bt
    ref.backward(torch.tensor(dOut))
    assert np.allclose(out, ref.detach().numpy(), atol=1e-7)
    assert np.allclose(dX, Xt.grad.numpy(), atol=1e-6)
    assert np.allclose(dg, gt.grad.numpy(), atol=1e-6) and np.allclose(db, bt.grad.numpy(), atol=1e-6)


def test_softmax_cross_entropy_gradient_matches_autograd():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((3, 4, 7)); targets = rng.integers(0, 7, (3, 4))
    loss, dlog = core.softmax_cross_entropy(logits, targets)
    lt = t(logits)
    ref = torch.nn.functional.cross_entropy(lt.reshape(-1, 7), torch.tensor(targets).reshape(-1))
    ref.backward()
    assert np.isclose(loss, ref.item(), atol=1e-6)
    assert np.allclose(dlog, lt.grad.numpy(), atol=1e-7)


def test_sinusoidal_encoding_matches_pytorch_port():
    from src.positional_encoding import sinusoidal_table
    assert np.allclose(core.positional_encoding(20, 16), sinusoidal_table(20, 16).numpy(), atol=1e-6)
