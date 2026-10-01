"""Lesson 06/07: why GPT (causal) and BERT (bidirectional) need different attention masks.

1. Mask matrices: BERT-style bidirectional attention (with [MASK]ed positions) vs GPT causal.
2. Short experiment: train the SAME small transformer on the SAME next-token objective with
   (a) a causal mask and (b) full bidirectional attention.
   (b) reaches near-zero loss because position t can simply READ token t+1 (its own target) -
   information that does not exist at generation time. Evaluating (b) with a causal mask
   (what generation actually looks like) exposes the leak.
"""
import json
import math

import matplotlib.pyplot as plt
import torch

from experiments._common import BLUE, FIG, LOGS, MUTED, ORANGE, ROOT
from src.attention import causal_mask
from src.dataset import get_batch, load_text, split_ids
from src.gpt import GPT, GPTConfig
from src.tokenizer import CharTokenizer


def masks_figure():
    T = 8
    causal = (~causal_mask(T)).float()
    bidir = torch.ones(T, T)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for a, m, t in ((ax[0], bidir, "BERT-style masked LM\nbidirectional: every token sees every token"),
                    (ax[1], causal, "GPT causal LM\ntoken i sees tokens 1..i only")):
        a.imshow(m, cmap="Blues", vmin=0, vmax=1.2); a.grid(False); a.set_title(t, fontsize=10)
        a.set_xticks(range(T)); a.set_yticks(range(T)); a.set_xticklabels(range(1, T + 1)); a.set_yticklabels(range(1, T + 1))
        a.set_xlabel("key position"); a.set_ylabel("query position")
    for pos in (2, 5):   # BERT corrupts ~15% of the INPUT tokens and predicts only those
        ax[0].text(pos, -0.75, "[MASK]", ha="center", fontsize=7, color=ORANGE, weight="bold")
    fig.suptitle("Figure: attention-mask matrices (dark = allowed)", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "fig_mlm_vs_clm_masks.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def train(causal, data, vdata, steps, device="cpu"):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=66, context_length=64, d_model=64, n_heads=4, n_layers=2, ffn_dim=256,
                    dropout=0.0, causal=causal)
    m = GPT(cfg).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(0)
    hist = []
    for s in range(1, steps + 1):
        x, y = get_batch(data, 32, 64, g)
        _, loss = m(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 25 == 0:
            hist.append((s, loss.item()))
    return m, hist


@torch.no_grad()
def evaluate(m, vdata, force_causal):
    old = m.cfg.causal; m.cfg.causal = force_causal; m.eval()
    g = torch.Generator().manual_seed(1)
    ls = [m(*get_batch(vdata, 32, 64, g))[1].item() for _ in range(20)]
    m.cfg.causal = old
    return sum(ls) / len(ls)


def main(steps=300):
    masks_figure()
    text = load_text(str(ROOT / "data/input.txt"))
    tok = CharTokenizer.fit(text)
    tr, va = split_ids(tok.encode(text), 0.1)
    out = {}
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for causal, name, col in ((True, "causal (GPT)", BLUE), (False, "bidirectional (leaky)", ORANGE)):
        m, hist = train(causal, tr, va, steps)
        ax.plot([h[0] for h in hist], [h[1] for h in hist], color=col, label=name)
        out[name] = {"train_loss_final": hist[-1][1],
                     "val_loss_same_mask_as_training": evaluate(m, va, causal),
                     "val_loss_with_causal_mask (generation-like)": evaluate(m, va, True)}
    ax.axhline(math.log(66), color=MUTED, ls=":", lw=1); ax.text(steps, math.log(66) + 0.05, "ln(66)", ha="right", color=MUTED, fontsize=8)
    ax.set_xlabel("step"); ax.set_ylabel("next-token training loss"); ax.legend()
    ax.set_title("Same objective, different mask: the bidirectional model 'cheats'")
    fig.tight_layout(); fig.savefig(FIG / "fig_mlm_vs_clm_loss.png", dpi=170); plt.close(fig)
    (LOGS / "mlm_vs_clm.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
