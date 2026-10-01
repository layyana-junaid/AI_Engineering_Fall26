"""Next-token-prediction dataset construction (handout milestone 4).

The corpus is one long stream of token ids. A training example is a window of
``context_length`` tokens, and the target is the SAME window shifted left by one:

    tokens : t0  t1  t2  t3  t4  ...
    x      : t0  t1  t2  t3          (input at position i)
    y      : t1  t2  t3  t4          (target at position i = token i+1)

so the target at position t is the next token after the input token at position t.
"""
from __future__ import annotations

import os
import urllib.request
from typing import Tuple

import torch

# Hugging Face card: https://huggingface.co/datasets/karpathy/tiny_shakespeare
# The same text is mirrored in karpathy/char-rnn, which is a plain-text download.
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Downloading Tiny Shakespeare to {path} ...")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)


def load_text(path: str) -> str:
    if not os.path.exists(path):
        download_tiny_shakespeare(path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_ids(ids, val_fraction: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Contiguous split (no shuffling) so validation text is never seen in training."""
    data = torch.tensor(ids, dtype=torch.long)
    n_val = int(len(data) * val_fraction)
    return data[: len(data) - n_val], data[len(data) - n_val:]


def make_shifted(window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """window has length T+1 (last dim) -> (inputs, targets), each length T."""
    return window[..., :-1], window[..., 1:]


def get_batch(data: torch.Tensor, batch_size: int, context_length: int,
              generator: torch.Generator | None = None, device="cpu"):
    """Sample ``batch_size`` random windows. Returns x, y of shape (B, T)."""
    max_start = len(data) - context_length - 1
    starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator)
    offsets = torch.arange(context_length + 1)
    windows = data[starts[:, None] + offsets[None, :]]  # (B, T+1)
    x, y = make_shifted(windows)
    return x.to(device), y.to(device)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/input.txt")
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    if a.download or not os.path.exists(a.path):
        download_tiny_shakespeare(a.path)
    t = load_text(a.path)
    print(f"{a.path}: {len(t):,} characters, {t.count(chr(10)):,} lines")
