"""Figure 6: training vs validation loss (+ learning rate) from outputs/logs/<run>/metrics.jsonl."""
import argparse
import json
import math

import matplotlib.pyplot as plt

from experiments._common import BLUE, FIG, LOGS, MUTED, ORANGE


def load(run):
    ev, tr = [], []
    for line in open(LOGS / run / "metrics.jsonl"):
        d = json.loads(line)
        (ev if d["type"] == "eval" else tr).append(d)
    return tr, ev


def main(run="minigpt"):
    tr, ev = load(run)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].plot([d["step"] for d in tr], [d["loss"] for d in tr], color=BLUE, alpha=0.35, lw=1, label="train (minibatch)")
    ax[0].plot([d["step"] for d in ev], [d["train_loss"] for d in ev], color=BLUE, marker="o", ms=4, label="train (eval mode)")
    ax[0].plot([d["step"] for d in ev], [d["val_loss"] for d in ev], color=ORANGE, marker="o", ms=4, label="validation")
    best = min(ev, key=lambda d: d["val_loss"])
    ax[0].annotate(f"best val {best['val_loss']:.3f}\n(ppl {math.exp(best['val_loss']):.1f})", (best["step"], best["val_loss"]),
                   (best["step"] * 0.55, best["val_loss"] + 0.35), color=MUTED, fontsize=9,
                   arrowprops=dict(arrowstyle="-", color=MUTED))
    ax[0].axhline(math.log(66), color=MUTED, lw=1, ls=":"); ax[0].text(tr[-1]["step"], math.log(66) + 0.03, "uniform guess ln(66)=4.19", ha="right", color=MUTED, fontsize=8)
    ax[0].set_ylim(1.0, 4.4); ax[0].set_xlabel("step"); ax[0].set_ylabel("cross-entropy (nats / char)"); ax[0].legend()
    ax[0].set_title("Training vs validation loss")
    ax[1].plot([d["step"] for d in tr], [d["lr"] for d in tr], color=BLUE)
    ax[1].set_xlabel("step"); ax[1].set_ylabel("learning rate"); ax[1].set_title("Learning-rate schedule (warm-up + cosine)")
    fig.suptitle(f"Figure 6: {run}", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "fig6_loss_curves.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print("saved fig6; final eval:", ev[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="minigpt"); main(ap.parse_args().run)
