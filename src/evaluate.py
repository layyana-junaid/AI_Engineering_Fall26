"""Quantitative + qualitative evaluation of a trained checkpoint (Figure 8).

    python -m src.evaluate --checkpoint checkpoints/minigpt.pt

* validation loss / perplexity over the WHOLE held-out split (non-overlapping windows)
* generations from fixed prompts (seeded), saved to outputs/samples/eval_samples.txt
"""
import argparse
import json
import math

import torch

from .dataset import load_text, split_ids
from .generate import generate_text
from .utils import ROOT, get_device, load_checkpoint, set_seed

FIXED_PROMPTS = ["ROMEO:", "To be, or not to be", "KING HENRY:\n", "First Citizen:\nBefore we proceed", "O Juliet, "]


@torch.no_grad()
def full_split_loss(model, data, device, batch_size=64):
    T = model.cfg.context_length
    n = (len(data) - 1) // T
    x = torch.stack([data[i * T:(i + 1) * T] for i in range(n)])
    y = torch.stack([data[i * T + 1:(i + 1) * T + 1] for i in range(n)])
    tot, cnt = 0.0, 0
    for i in range(0, n, batch_size):
        xb, yb = x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        _, loss = model(xb, yb)
        tot += loss.item() * xb.numel(); cnt += xb.numel()
    return tot / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/minigpt.pt")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=250)
    a = ap.parse_args()
    set_seed(0)
    device = get_device("cpu")
    model, tok, ckpt = load_checkpoint(ROOT / a.checkpoint, device)
    cfgd = ckpt["train_config"]
    text = load_text(str(ROOT / cfgd["data"]["path"]))
    train, val = split_ids(tok.encode(text), cfgd["data"]["val_fraction"])
    vl, tl = full_split_loss(model, val, device), full_split_loss(model, train, device)
    res = {"params": model.num_parameters(), "val_loss": vl, "val_ppl": math.exp(vl),
           "train_loss": tl, "train_ppl": math.exp(tl), "val_bits_per_char": vl / math.log(2)}
    print(json.dumps(res, indent=2))
    out = [f"checkpoint: {a.checkpoint}   params: {model.num_parameters():,}   temperature: {a.temperature}   top_k: {a.top_k}",
           f"full-split val loss {vl:.4f} (ppl {math.exp(vl):.2f}, {vl / math.log(2):.3f} bits/char)   train loss {tl:.4f}", ""]
    for i, p in enumerate(FIXED_PROMPTS):
        s = generate_text(model, tok, p, a.max_new_tokens, a.temperature, a.top_k, seed=100 + i, device=device)
        out += [f"##### sample {i + 1}   prompt = {p!r}", s, ""]
    (ROOT / "outputs" / "samples").mkdir(exist_ok=True)
    (ROOT / "outputs" / "samples" / "eval_samples.txt").write_text("\n".join(out), encoding="utf-8")
    (ROOT / "outputs" / "logs" / "eval_metrics.json").write_text(json.dumps(res, indent=2))
    print("\n".join(out))


if __name__ == "__main__":
    main()
