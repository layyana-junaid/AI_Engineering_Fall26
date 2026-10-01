"""Controlled ablations (handout section 13): change ONE factor, keep everything else fixed.

    python -m experiments.ablation                       # default reduced budget (CPU friendly)
    python -m experiments.ablation --steps 3000 --batch-size 64     # full budget (Colab GPU)
    python -m experiments.ablation --suite depth heads   # subset

Every run is a separate process (so peak-memory numbers are not polluted by earlier runs).
Context-length runs keep TOKENS PER STEP constant by scaling the batch size, so they see the
same amount of data and differ only in how far back the model can look.
"""
import argparse
import csv
import json
import subprocess
import sys

import matplotlib.pyplot as plt

from experiments._common import AQUA, BLUE, FIG, MUTED, ORANGE, ROOT

OUT = ROOT / "outputs" / "ablation"

# name -> (suite, label, overrides builder)
def variants(batch):
    v = [("base", "baseline", "4 layers / 4 heads / ctx 128", {})]
    for L in (2, 6):
        v.append((f"depth{L}", "depth", f"{L} layers", {"model.n_layers": L}))
    for H in (2, 8):
        v.append((f"heads{H}", "heads", f"{H} heads", {"model.n_heads": H}))
    for C in (64, 256):
        v.append((f"ctx{C}", "context", f"ctx {C}", {"model.context_length": C, "train.batch_size": max(1, batch * 128 // C)}))
    v.append(("learnedpos", "position", "learned positions", {"model.pos_encoding": "learned"}))
    return v


def run(name, overrides, steps, batch, resume):
    summ = ROOT / "outputs" / "logs" / f"abl_{name}" / "summary.json"
    if resume and summ.exists():
        return json.loads(summ.read_text())
    cmd = [sys.executable, "-m", "src.train", "--run-name", f"abl_{name}",
           "--set", f"train.max_steps={steps}", "--set", f"train.batch_size={batch}",
           "--set", f"train.eval_interval={max(1, steps // 4)}", "--set", f"train.sample_interval={steps}",
           "--set", f"train.warmup_steps={max(10, steps // 20)}", "--set", "train.log_interval=25"]
    for k, val in overrides.items():
        cmd += ["--set", f"{k}={val}"]
    print(">>", " ".join(cmd[3:]), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return json.loads(summ.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--suite", nargs="*", default=["depth", "heads", "context", "position"])
    ap.add_argument("--resume", action="store_true", help="reuse finished runs")
    ap.add_argument("--plot-only", action="store_true", help="re-render Figure 9 from results.csv")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        rows = list(csv.DictReader(open(OUT / "results.csv")))
        for r in rows:
            for k in ("val_loss", "params", "time_s"):
                r[k] = float(r[k])
        return plot(rows, a)
    rows = []
    for name, suite, label, ov in variants(a.batch_size):
        if suite not in a.suite and suite != "baseline":
            continue
        s = run(name, ov, a.steps, a.batch_size, a.resume)
        rows.append({"run": name, "suite": suite, "setting": label, "params": s["params"],
                     "val_loss": round(s["final_val_loss"], 4), "val_ppl": round(s["final_val_ppl"], 2),
                     "train_loss": round(s["final_train_loss"], 4),
                     "time_s": round(s["wall_clock_s"], 1), "tokens_per_s": round(s["tokens_per_s"]),
                     "peak_mem_mb": round(s["peak_memory_mb"]), "batch": s["train_config"]["batch_size"],
                     "ctx": s["model_config"]["context_length"]})
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    md = ["| run | setting | params | val loss | val ppl | train loss | time (s) | tokens/s | peak mem (MB) |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    md += [f"| {r['run']} | {r['setting']} | {r['params']:,} | {r['val_loss']} | {r['val_ppl']} | {r['train_loss']} "
           f"| {r['time_s']} | {r['tokens_per_s']:,} | {r['peak_mem_mb']} |" for r in rows]
    (OUT / "results.md").write_text("\n".join(md)); print("\n".join(md))
    plot(rows, a)


def plot(rows, a):
    """Dot plot (not bars): the axis does not start at zero, and bars would exaggerate small gaps."""
    seed_run = ROOT / "outputs" / "logs" / "abl_base_seed7" / "summary.json"
    base = next(r for r in rows if r["run"] == "base")["val_loss"]
    band = None
    if seed_run.exists():
        other = json.loads(seed_run.read_text())["final_val_loss"]
        band = (min(base, other), max(base, other))
    groups = [("depth", "Depth (n_layers)", ["depth2", "base", "depth6"]),
              ("heads", "Heads (n_heads)", ["heads2", "base", "heads8"]),
              ("context", "Context length", ["ctx64", "base", "ctx256"]),
              ("position", "Position method", ["base", "learnedpos"])]
    groups = [g for g in groups if all(any(r["run"] == n for r in rows) for n in g[2])]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.5 * len(groups), 4.3), squeeze=False, sharey=True)
    allv = [r["val_loss"] for r in rows]
    lo, hi = min(allv) - 0.06, max(allv) + 0.06
    for ax, (_, title, names) in zip(axes[0], groups):
        rs = [next(r for r in rows if r["run"] == n) for n in names]
        if band:
            ax.axhspan(*band, color=MUTED, alpha=0.18, lw=0)
        ax.plot(range(len(rs)), [r["val_loss"] for r in rs], color=MUTED, lw=1, zorder=1)
        for i, r in enumerate(rs):
            ax.plot(i, r["val_loss"], "o", ms=11, color=BLUE if r["run"] == "base" else ORANGE,
                    mec="#fcfcfb", mew=2, zorder=3)
            ax.text(i, r["val_loss"] + 0.018, f"{r['val_loss']:.3f}", ha="center", fontsize=9)
            ax.text(i, lo + 0.012, f"{r['params'] / 1e6:.2f}M\n{r['time_s'] / 60:.1f} min", ha="center",
                    va="bottom", fontsize=7.5, color=MUTED)
        ax.set_xticks(range(len(rs)))
        ax.set_xticklabels([r["setting"].replace(" / ", "\n") if r["run"] == "base" else r["setting"] for r in rs], fontsize=8)
        ax.set_xlim(-0.6, len(rs) - 0.4); ax.set_ylim(lo, hi); ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("validation loss (lower is better)")
    note = "grey band = baseline re-run with another seed (noise level)" if band else ""
    fig.suptitle(f"Figure 9: ablations ({a.steps} steps, batch {a.batch_size}x128 tokens; blue = baseline; {note})",
                 y=1.03, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig9_ablations.png", dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
