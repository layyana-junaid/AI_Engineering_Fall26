"""Optional experiment: greedy vs temperature / top-k / top-p sampling (qualitative diversity).

Metrics (computed over 5 prompts x 400 generated characters):
  valid-word rate : share of generated words that occur in the training corpus
  distinct-2      : unique word bigrams / total word bigrams  (diversity)
  repeat-4gram    : share of character 4-grams that already occurred earlier in the same sample
"""
import argparse
import json
import re

import torch

from experiments._common import LOGS, ROOT
from src.dataset import load_text
from src.evaluate import FIXED_PROMPTS
from src.generate import generate_text
from src.utils import load_checkpoint

STRATS = [("greedy (T=0)", dict(temperature=0)), ("T=0.5", dict(temperature=0.5, top_k=None)),
          ("T=0.8 top-k 40", dict(temperature=0.8, top_k=40)), ("T=0.8 top-p 0.9", dict(temperature=0.8, top_k=None, top_p=0.9)),
          ("T=1.2", dict(temperature=1.2, top_k=None))]


def metrics(samples, vocab):
    words = [w for s in samples for w in re.findall(r"[A-Za-z']+", s)]
    valid = sum(w.lower() in vocab for w in words) / max(1, len(words))
    bg = [(a, b) for s in samples for a, b in zip(*(lambda ws: (ws, ws[1:]))(re.findall(r"[A-Za-z']+", s)))]
    d2 = len(set(bg)) / max(1, len(bg))
    rep = tot = 0
    for s in samples:
        seen = set()
        for i in range(len(s) - 3):
            g = s[i:i + 4]; tot += 1; rep += g in seen; seen.add(g)
    return valid, d2, rep / max(1, tot)


def main(ckpt):
    model, tok, c = load_checkpoint(ROOT / ckpt)
    corpus = load_text(str(ROOT / c["train_config"]["data"]["path"]))
    vocab = set(w.lower() for w in re.findall(r"[A-Za-z']+", corpus))
    rows = ["| strategy | valid-word rate | distinct-2 | repeated 4-gram share |", "|---|---:|---:|---:|"]
    example = {}
    for name, kw in STRATS:
        samples = [generate_text(model, tok, p, 400, seed=7 + i, **kw) for i, p in enumerate(FIXED_PROMPTS)]
        v, d, r = metrics(samples, vocab)
        rows.append(f"| {name} | {v:.3f} | {d:.3f} | {r:.3f} |"); example[name] = samples[0][:300]
    txt = "\n".join(rows)
    (LOGS / "generation_strategy.md").write_text(txt)
    (LOGS / "generation_strategy_examples.json").write_text(json.dumps(example, indent=1))
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", default="checkpoints/minigpt.pt"); main(ap.parse_args().checkpoint)
