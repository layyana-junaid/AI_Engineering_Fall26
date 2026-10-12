"""Training pipeline for MiniGPT (handout milestones 4 and 5).

    python -m src.train --config configs/tiny_gpt.yaml --run-name minigpt
    python -m src.train --run-name depth2 --set model.n_layers=2 --set train.max_steps=1500
    python -m src.train --run-name overfit --overfit-batch --set train.max_steps=300

Writes
    checkpoints/<run>.pt            best-validation checkpoint (weights + config + tokenizer)
    outputs/logs/<run>/metrics.jsonl  train/val loss, lr, step, elapsed (one json per line)
    outputs/logs/<run>/summary.json   final numbers used in the report
    outputs/samples/<run>.txt         generations from fixed prompts during/after training
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

try:  # Unix only; on Windows peak RAM is not measured and is logged as 0
    import resource
except ImportError:
    resource = None

import torch

from .dataset import get_batch, load_text, split_ids
from .generate import generate_text
from .gpt import GPT, GPTConfig
from .tokenizer import CharTokenizer
from .utils import ROOT, count_parameters, get_device, load_config, save_checkpoint, set_seed

SAMPLE_PROMPTS = ["ROMEO:", "To be, or not to be", "KING HENRY:\n", "First Citizen:\n", "O Juliet, "]


def lr_at(step: int, tc: dict) -> float:
    """Linear warm-up then cosine decay to lr * min_lr_ratio."""
    lr, warm, total = tc["lr"], tc["warmup_steps"], tc["max_steps"]
    if step <= warm:
        return lr * step / max(1, warm)
    progress = (step - warm) / max(1, total - warm)
    floor = lr * tc["min_lr_ratio"]
    return floor + 0.5 * (lr - floor) * (1 + math.cos(math.pi * min(1.0, progress)))


def build_optimizer(model: GPT, tc: dict):
    """AdamW; weight decay only on matrices (not biases / LayerNorm gains)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": tc["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=tc["lr"], betas=tuple(tc["betas"]))


@torch.no_grad()
def estimate_loss(model, data, tc, ctx, device, seed=0) -> float:
    """Mean loss over ``eval_iters`` fixed (seeded) random batches, dropout disabled."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(tc["eval_iters"]):
        x, y = get_batch(data, tc["batch_size"], ctx, g, device)
        losses.append(model(x, y)[1].item())
    model.train()
    return sum(losses) / len(losses)


def peak_memory_mb(device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if resource is None:
        return 0.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB -> MiB (Linux)


def train(cfg: dict, run_name: str, overfit_batch: bool = False, verbose: bool = True,
          ckpt_path: str | None = None) -> dict:
    tc, mc = cfg["train"], dict(cfg["model"])
    set_seed(tc["seed"])
    device = get_device(tc["device"])

    # ---- data -------------------------------------------------------------------------
    text = load_text(str(ROOT / cfg["data"]["path"]))
    tok = CharTokenizer.fit(text)
    train_data, val_data = split_ids(tok.encode(text), cfg["data"]["val_fraction"])
    ctx = mc["context_length"]

    # ---- model ------------------------------------------------------------------------
    if overfit_batch:
        mc["dropout"] = 0.0        # the overfit gate must be able to reach ~0 loss
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, **mc)).to(device)
    n_params = count_parameters(model)
    opt = build_optimizer(model, tc)
    if verbose:
        print(f"[{run_name}] device={device} params={n_params:,} vocab={tok.vocab_size} "
              f"train_tokens={len(train_data):,} val_tokens={len(val_data):,}")

    out_dir = ROOT / "outputs" / "logs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "samples").mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "metrics.jsonl", "w")
    sample_f = open(ROOT / "outputs" / "samples" / f"{run_name}.txt", "w", encoding="utf-8")
    ckpt_path = ckpt_path or str(ROOT / "checkpoints" / f"{run_name}.pt")

    gen = torch.Generator().manual_seed(tc["seed"])
    fixed = get_batch(train_data, tc["batch_size"], ctx, gen, device) if overfit_batch else None

    def log(d):
        log_f.write(json.dumps(d) + "\n"); log_f.flush()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    best_val, best_step = float("inf"), 0
    model.train()
    for step in range(1, tc["max_steps"] + 1):
        lr = lr_at(step, tc)
        for g_ in opt.param_groups:
            g_["lr"] = lr
        x, y = fixed if overfit_batch else get_batch(train_data, tc["batch_size"], ctx, gen, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()

        if step % tc["log_interval"] == 0 or step == 1:
            log({"type": "train", "step": step, "loss": loss.item(), "lr": lr,
                 "elapsed": time.time() - t0})
            if verbose and (step % (tc["log_interval"] * 2) == 0 or step == 1):
                print(f"[{run_name}] step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} "
                      f"| {time.time() - t0:6.1f}s")

        if step % tc["eval_interval"] == 0 or step == tc["max_steps"]:
            if overfit_batch:      # report loss on the ONE batch being memorised
                model.eval()
                with torch.no_grad():
                    tr = model(*fixed)[1].item()
                model.train()
                va = tr
            else:
                tr = estimate_loss(model, train_data, tc, ctx, device)
                va = estimate_loss(model, val_data, tc, ctx, device)
            log({"type": "eval", "step": step, "train_loss": tr, "val_loss": va, "lr": lr,
                 "elapsed": time.time() - t0})
            if verbose:
                print(f"[{run_name}] EVAL step {step:5d} | train {tr:.4f} | val {va:.4f} "
                      f"| ppl {math.exp(va):.2f}")
            if va < best_val and not overfit_batch:
                best_val, best_step = va, step
                save_checkpoint(ckpt_path, model, tok, cfg, {"step": step, "val_loss": va})

        if step % tc["sample_interval"] == 0 or step == tc["max_steps"]:
            s = generate_text(model, tok, SAMPLE_PROMPTS[0], 200, 0.8, 40, seed=0, device=device)
            sample_f.write(f"=== step {step} ===\n{s}\n\n"); sample_f.flush()
            if verbose:
                print(f"--- sample @ step {step} ---\n{s}\n---")

    wall = time.time() - t0
    if overfit_batch:
        model.eval()
        with torch.no_grad():
            final_train = final_val = model(*fixed)[1].item()
    else:
        final_train = estimate_loss(model, train_data, tc, ctx, device)
        final_val = estimate_loss(model, val_data, tc, ctx, device)
    if overfit_batch:
        save_checkpoint(ckpt_path, model, tok, cfg, {"step": tc["max_steps"]})
        best_val, best_step = final_val, tc["max_steps"]

    summary = {
        "run": run_name, "device": str(device), "params": n_params,
        "model_config": model.cfg.to_dict(), "train_config": tc,
        "steps": tc["max_steps"], "wall_clock_s": wall,
        "tokens_per_s": tc["max_steps"] * tc["batch_size"] * ctx / wall,
        "peak_memory_mb": peak_memory_mb(device),
        "final_train_loss": final_train, "final_val_loss": final_val,
        "final_val_ppl": math.exp(final_val), "best_val_loss": best_val, "best_step": best_step,
        "train_tokens": len(train_data), "val_tokens": len(val_data),
        "overfit_batch": overfit_batch,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log_f.close(); sample_f.close()
    if verbose:
        print(f"[{run_name}] done in {wall:.1f}s | final train {final_train:.4f} | "
              f"final val {final_val:.4f} | best val {best_val:.4f} @ {best_step}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny_gpt.yaml")
    ap.add_argument("--run-name", default="minigpt")
    ap.add_argument("--set", action="append", default=[], help="dotted override, e.g. model.n_layers=6")
    ap.add_argument("--overfit-batch", action="store_true", help="debug gate: train on ONE fixed batch")
    a = ap.parse_args()
    cfg = load_config(str(ROOT / a.config), a.set)
    train(cfg, a.run_name, overfit_batch=a.overfit_batch)


if __name__ == "__main__":
    main()
