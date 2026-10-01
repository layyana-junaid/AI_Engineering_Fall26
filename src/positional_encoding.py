"""Positional encodings: sinusoidal (required) and learned (optional experiment).

Self-attention is permutation-equivariant: without position information, the
sentence "dog bites man" and "man bites dog" would be indistinguishable to the
attention layers. We therefore ADD a position-dependent vector to every token embedding.

Sinusoidal formulas (Vaswani et al., 2017):
    PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    """Returns the (max_len, d_model) sinusoidal table."""
    if d_model % 2 != 0:
        raise ValueError("sinusoidal encoding needs an even d_model")
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)          # (L, 1)
    # 1 / 10000^(2i/d) computed in log-space for numerical stability
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                         * (-math.log(10000.0) / d_model))                      # (d/2,)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed (non-trainable) encoding added to the token embeddings."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.register_buffer("pe", sinusoidal_table(max_len, d_model), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (B, T, D)
        return x + self.pe[: x.size(1)].unsqueeze(0)


class LearnedPositionalEncoding(nn.Module):
    """Trainable position embedding table (alternative used in the optional ablation)."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)
        # token embeddings have ~unit scale after the sqrt(d_model) factor in GPT.forward,
        # so the position table starts at the same scale (GPT-2 also uses equal scales).
        nn.init.normal_(self.emb.weight, mean=0.0, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device)
        return x + self.emb(pos).unsqueeze(0)


def build_positional_encoding(kind: str, d_model: int, max_len: int) -> nn.Module:
    if kind == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_len)
    if kind == "learned":
        return LearnedPositionalEncoding(d_model, max_len)
    raise ValueError(f"unknown pos_encoding '{kind}' (use 'sinusoidal' or 'learned')")
