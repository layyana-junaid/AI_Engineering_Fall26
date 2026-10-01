"""Scaled dot-product attention and multi-head attention, written from scratch.

Only matrix multiplication, softmax and ``nn.Linear`` from PyTorch are used
(explicitly allowed by handout section 8). ``nn.MultiheadAttention`` and
``F.scaled_dot_product_attention`` are NOT used by the model.

Notation: B batch, T sequence length, D = d_model, H heads, d_k = D / H.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """Upper-triangular future-token mask, shape (T, T), dtype bool.

    mask[i, j] is True when query position i is FORBIDDEN from attending to key
    position j, i.e. when j > i (a strictly-upper-triangular matrix).
    """
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None,
                                 dropout: Optional[nn.Module] = None):
    """softmax(Q K^T / sqrt(d_k) [+ mask]) V

    q, k, v : (..., T, d_k).  mask : broadcastable to (..., T, T), True = forbidden.
    dropout : optional module applied to the attention weights (training only).
    Returns (output (..., T, d_k), weights (..., T, T)); the returned weights are the
    un-dropped ones so they can be inspected/plotted.
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)          # (..., T, T)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))        # future -> -inf -> weight 0
    weights = torch.softmax(scores, dim=-1)                     # row-wise softmax
    used = dropout(weights) if dropout is not None else weights
    return used @ v, weights


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with explicit split / attend / merge / project steps."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)   # output projection after concat
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # Inspection hooks (used by experiments/, off by default):
        self.store_weights = False
        self.last_weights: Optional[torch.Tensor] = None
        self.trace: Optional[list] = None     # set to [] to record (name, shape) pairs

    # ----------------------------------------------------------- reshape logic
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, H, T, d_k)"""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, d_k) -> (B, T, D)   (concatenation of all heads)"""
        B, H, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * d_k)

    def _t(self, name, t):
        if self.trace is not None:
            self.trace.append((name, tuple(t.shape)))

    # ----------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._t("input x", x)
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        self._t("Q = x W_q", q)
        qh, kh, vh = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        self._t("Q split into heads", qh)
        self._t("K split into heads", kh)
        self._t("V split into heads", vh)

        out, weights = scaled_dot_product_attention(qh, kh, vh, mask, self.attn_dropout)
        self._t("attention weights (per head)", weights)
        self._t("per-head output = weights @ V", out)
        if self.store_weights:
            self.last_weights = weights.detach()


        merged = self.merge_heads(out)
        self._t("heads concatenated", merged)
        y = self.resid_dropout(self.w_o(merged))
        self._t("output projection W_o", y)
        return y
