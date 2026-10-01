"""Small shared helpers: seeding, device, config loading, checkpoint IO."""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from .gpt import GPT, GPTConfig
from .tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """Load YAML and apply dotted overrides such as ``model.n_layers=6``."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key '{key}'")
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, model: GPT, tokenizer: CharTokenizer, train_cfg: dict, extra: dict | None = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_config": model.cfg.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "train_config": train_cfg,
        "extra": extra or {},
    }, path)


def load_checkpoint(path, device="cpu"):
    """Rebuild model + tokenizer from a checkpoint file. Returns (model, tokenizer, ckpt)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, CharTokenizer.from_dict(ckpt["tokenizer"]), ckpt
