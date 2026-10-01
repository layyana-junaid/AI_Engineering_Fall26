"""MiniGPT: a decoder-only causal language model built from the pieces in this package."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import causal_mask
from .positional_encoding import build_positional_encoding
from .transformer_block import TransformerBlock


@dataclass
class GPTConfig:
    vocab_size: int = 66
    context_length: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    pos_encoding: str = "sinusoidal"   # sinusoidal | learned
    tie_weights: bool = False
    causal: bool = True                # False only for the MLM-vs-CLM leakage experiment

    def to_dict(self) -> dict:
        return asdict(self)


class GPT(nn.Module):
    """token ids (B,T) -> embeddings (B,T,D) -> +position -> N blocks -> LN -> logits (B,T,V)"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = build_positional_encoding(cfg.pos_encoding, cfg.d_model, cfg.context_length)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ffn_dim, cfg.dropout)
             for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.register_buffer("mask", causal_mask(cfg.context_length), persistent=False)
        self._init_weights()

    # ------------------------------------------------------------ initialisation
    def _init_weights(self):
        cfg = self.cfg
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Token embeddings: variance 1/d, then multiplied by sqrt(d) in forward() so the
        # embedding has unit scale and is not drowned out by the (+-1 valued) position code.
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=cfg.d_model ** -0.5)
        # GPT-2 trick: shrink the layers that write into the residual stream so its
        # variance does not grow with depth.
        scale = 0.02 / math.sqrt(2 * cfg.n_layers)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.w_o.weight, mean=0.0, std=scale)
            nn.init.normal_(blk.ffn.fc2.weight, mean=0.0, std=scale)

    # ------------------------------------------------------------------ forward
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        if T > self.cfg.context_length:
            raise ValueError(f"sequence length {T} exceeds context_length {self.cfg.context_length}")
        x = self.tok_emb(idx) * math.sqrt(self.cfg.d_model)     # (B, T, D)
        x = self.emb_drop(self.pos_enc(x))                      # add position information
        mask = self.mask[:T, :T] if self.cfg.causal else None
        for blk in self.blocks:
            x = blk(x, mask)
        logits = self.lm_head(self.ln_f(x))                     # (B, T, V)
        loss = None
        if targets is not None:
            # targets[b, t] is the token that follows idx[b, t]  (already shifted by the dataset)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # ---------------------------------------------------------------- utilities
    def num_parameters(self, trainable_only: bool = True) -> int:
        # tied weights are counted once because .parameters() de-duplicates
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def set_attention_recording(self, flag: bool = True):
        for blk in self.blocks:
            blk.attn.store_weights = flag
