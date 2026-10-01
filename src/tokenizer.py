"""Deterministic character-level tokenizer.

Vocabulary policy (documented as required by the handout, section 7):
  * The vocabulary is every distinct character of the training corpus, sorted by
    Unicode code point, so the same corpus always yields the same ids.
  * id 0 is reserved for the single special token ``<unk>``. A character that was
    not in the training corpus is encoded as ``<unk>`` and decoded as ``?``.
  * There are no BOS/EOS/PAD tokens: training windows are fixed-length slices of one
    long token stream, so padding is never needed.

For Tiny Shakespeare this gives 65 characters + 1 ``<unk>`` = vocab_size 66.
"""
from __future__ import annotations

import json
from typing import Dict, List

UNK_TOKEN = "<unk>"
UNK_ID = 0
UNK_DECODE = "?"


class CharTokenizer:
    def __init__(self, chars: List[str]):
        self.chars = list(chars)
        self.stoi: Dict[str, int] = {c: i + 1 for i, c in enumerate(self.chars)}
        self.itos: Dict[int, str] = {i + 1: c for i, c in enumerate(self.chars)}

    # ------------------------------------------------------------------ build
    @classmethod
    def fit(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + 1  # + <unk>

    # ----------------------------------------------------------------- coding
    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(c, UNK_ID) for c in text]

    def decode(self, ids) -> str:
        return "".join(self.itos.get(int(i), UNK_DECODE) for i in ids)

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        return {"type": "char", "chars": self.chars, "unk_token": UNK_TOKEN, "unk_id": UNK_ID}

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        return cls(d["chars"])

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
