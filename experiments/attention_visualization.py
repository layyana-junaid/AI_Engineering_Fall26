"""Figure 7: attention maps from the TRAINED model (every layer x every head).

    python -m experiments.attention_visualization --checkpoint checkpoints/minigpt.pt
"""
import argparse

import matplotlib.pyplot as plt
import torch

from experiments._common import FIG, LOGS, ROOT
from src.utils import load_checkpoint


def main(checkpoint, text):
    model, tok, _ = load_checkpoint(ROOT / checkpoint)
    model.set_attention_recording(True)
    ids = torch.tensor([tok.encode(text)])
    with torch.no_grad():
        model(ids)
    L, H, T = model.cfg.n_layers, model.cfg.n_heads, ids.size(1)
    labels = [c if c != "\n" else "\\n" for c in text]
    fig, axes = plt.subplots(L, H, figsize=(3.1 * H, 3.2 * L), squeeze=False)
    stats = []
    for l, blk in enumerate(model.blocks):
        w = blk.attn.last_weights[0]                       # (H, T, T)
        assert torch.all(torch.triu(w, diagonal=1) == 0)   # causality holds in the trained model
        for h in range(H):
            ax = axes[l][h]
            ax.imshow(w[h], cmap="Blues", vmin=0, vmax=w[h].max().item()); ax.grid(False)
            ax.set_title(f"layer {l} head {h}", fontsize=9)
            if l == L - 1:
                ax.set_xticks(range(T)); ax.set_xticklabels(labels, fontsize=5)
            else:
                ax.set_xticks([])
            if h == 0:
                ax.set_yticks(range(T)); ax.set_yticklabels(labels, fontsize=5)
            else:
                ax.set_yticks([])
            prev = (w[h].diagonal(-1).mean().item())        # attention to the previous character
            selfw = w[h].diagonal().mean().item()
            stats.append((l, h, prev, selfw, w[h][:, 0].mean().item()))
    fig.suptitle(f'Figure 7: attention of the trained model on "{text}"\n'
                 "(rows = query token, columns = key token; the upper triangle is empty because of the causal mask)", y=1.0)
    fig.tight_layout(); fig.savefig(FIG / "fig7_attention_maps.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    lines = ["layer head  mean attn->previous char  mean attn->self  mean attn->first char"]
    lines += [f"{l:5d} {h:4d}  {p:24.3f}  {s:16.3f}  {f:20.3f}" for l, h, p, s, f in stats]
    (LOGS / "attention_stats.txt").write_text("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/minigpt.pt")
    ap.add_argument("--text", default="ROMEO:\nO, speak again, bright angel!")
    a = ap.parse_args(); main(a.checkpoint, a.text)
