"""Lesson 01: why Transformers reduce sequential computation relative to recurrent nets.

Compares, for growing sequence length T (d = 128, batch 16, forward + backward, CPU):
  * a hand-written tanh RNN layer:   h_t = tanh(x_t W + h_{t-1} U)   -> T strictly sequential steps
  * ONE transformer block (causal self-attention + FFN) from this repo -> O(1) sequential matmuls
and the memory that each holds for the backward pass (analytic + measured peak RSS delta not used:
we report the analytic activation size, which is what scales).

Run this on an otherwise idle machine - timings are meaningless under CPU contention.
"""
import csv
import time

import matplotlib.pyplot as plt
import torch

from experiments._common import BLUE, FIG, LOGS, MUTED, ORANGE
from src.attention import causal_mask
from src.transformer_block import TransformerBlock

D, B, H = 128, 16, 4


class RNNLayer(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = torch.nn.Linear(d, d)
        self.U = torch.nn.Linear(d, d, bias=False)

    def forward(self, x):                       # x: (B, T, D)
        h = x.new_zeros(x.size(0), x.size(2))
        outs = []
        for t in range(x.size(1)):              # <- T sequential dependent steps
            h = torch.tanh(self.W(x[:, t]) + self.U(h))
            outs.append(h)
        return torch.stack(outs, 1)


def bench(fn, reps=3):
    fn()                                         # warm-up
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t)
    return best


def main(plot_only=False):
    if plot_only:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(LOGS / "seq_vs_parallel.csv"))]
        return plot(rows)
    torch.manual_seed(0); torch.set_num_threads(2)
    rnn, blk = RNNLayer(D), TransformerBlock(D, H, 4 * D, 0.0)
    rows = []
    for T in (32, 64, 128, 256, 512):
        x = torch.randn(B, T, D, requires_grad=True)
        m = causal_mask(T)
        t_rnn = bench(lambda: rnn(x).sum().backward())
        t_trf = bench(lambda: blk(x, m).sum().backward())
        attn_mb = B * H * T * T * 4 / 2**20          # attention weights, fp32
        rnn_mb = B * T * D * 4 / 2**20               # hidden states, fp32
        rows.append(dict(T=T, rnn_ms=round(t_rnn * 1e3, 1), transformer_ms=round(t_trf * 1e3, 1),
                         rnn_sequential_steps=T, transformer_sequential_matmuls=6,
                         attn_scores_MB=round(attn_mb, 2), rnn_hidden_MB=round(rnn_mb, 2)))
        print(rows[-1], flush=True)
    with open(LOGS / "seq_vs_parallel.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    plot(rows)


def plot(rows):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    Ts = [r["T"] for r in rows]
    ax[0].plot(Ts, [r["rnn_ms"] for r in rows], color=ORANGE, marker="o", label="RNN layer (T sequential steps)")
    ax[0].plot(Ts, [r["transformer_ms"] for r in rows], color=BLUE, marker="o", label="Transformer block")
    ax[0].set_xlabel("sequence length T"); ax[0].set_ylabel("forward+backward time (ms)"); ax[0].legend()
    ax[0].set_title("Wall-clock on a 2-core CPU: roughly equal\n(transformer does ~6x the FLOPs per token)")
    ax[1].plot(Ts, [r["attn_scores_MB"] for r in rows], color=BLUE, marker="o", label="attention weights  O(T^2)")
    ax[1].plot(Ts, [r["rnn_hidden_MB"] for r in rows], color=ORANGE, marker="o", label="RNN hidden states  O(T)")
    ax[1].set_xlabel("sequence length T"); ax[1].set_ylabel("activation memory (MB)"); ax[1].legend()
    ax[1].set_title("Attention pays quadratic activation memory")
    fig.suptitle("Lesson 01: sequential vs parallel depth (measured, CPU, d=128, batch 16)", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "fig_seq_vs_parallel.png", dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    import sys
    main("--plot-only" in sys.argv)
