"""Transformer decoder block (GPT style, pre-LayerNorm).

    x -> x + Dropout(MHA(LN1(x), causal_mask))      # attention sublayer + residual
      -> x + FFN(LN2(x))                            # feed-forward sublayer + residual

Pre-LN (normalise BEFORE each sublayer, keep the residual path clean) is what GPT-2
uses. It trains stably without a long warm-up because the identity path from
input to output is never passed through a normalisation layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise MLP: Linear(d -> ffn) -> GELU -> Linear(ffn -> d) -> Dropout."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)   # residual around attention
        x = x + self.ffn(self.ln2(x))          # residual around feed-forward
        return x
