"""Evidence figures that do not need a trained model.

    Fig 3  multi-head tensor-shape printout
    Fig 4  positional-encoding visualisation
    Fig 5  model summary / parameter count   (uses checkpoints/<run>.pt if present)
    +      architecture diagram
"""
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch

from experiments._common import AQUA, BLUE, FIG, GRID, INK, LOGS, MUTED, ORANGE, ROOT, text_to_png
from src.attention import MultiHeadAttention, causal_mask
from src.gpt import GPT, GPTConfig
from src.positional_encoding import sinusoidal_table
from src.utils import load_checkpoint


def fig3_shapes():
    torch.manual_seed(0)
    B, T, D, H = 2, 16, 128, 4
    mha = MultiHeadAttention(D, H).eval()
    mha.trace = []
    x = torch.randn(B, T, D)
    mha(x, causal_mask(T))
    lines = [f"MultiHeadAttention(d_model={D}, n_heads={H}) -> d_k = d_model / n_heads = {D // H}",
             f"batch B={B}, sequence T={T}", "-" * 66]
    for name, shp in mha.trace:
        lines.append(f"{name:<34s} {str(shp):<22s}")
    lines += ["-" * 66,
              "split : (B,T,D) -> view (B,T,H,d_k) -> transpose -> (B,H,T,d_k)",
              "merge : (B,H,T,d_k) -> transpose -> (B,T,H,d_k) -> view (B,T,D)   [= concatenation]",
              f"check : merge(split(x)) == x  ->  {torch.equal(mha.merge_heads(mha.split_heads(x)), x)}"]
    txt = "\n".join(lines)
    (LOGS / "shape_trace.txt").write_text(txt)
    print(txt)
    text_to_png(txt, FIG / "fig3_multihead_shapes.png", title="Figure 3: tensor shapes through multi-head attention")


def fig4_positional():
    pe = sinusoidal_table(128, 128).numpy()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    im = ax[0].imshow(pe, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1); ax[0].grid(False)
    ax[0].set_title("PE table (128 positions x 128 dims)")
    ax[0].set_xlabel("embedding dimension"); ax[0].set_ylabel("position")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    for d, c in ((0, BLUE), (8, ORANGE), (32, AQUA)):
        ax[1].plot(pe[:, d], color=c, label=f"dim {d}  (wavelength {2 * math.pi * 10000 ** ((d - d % 2) / 128):.0f})")
    ax[1].set_title("Low dims oscillate fast, high dims slowly")
    ax[1].set_xlabel("position"); ax[1].set_ylabel("PE value"); ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    sim = pe @ pe[40]
    ax[2].plot(sim, color=BLUE); ax[2].axvline(40, color=MUTED, lw=1, ls="--")
    ax[2].set_title("Similarity of position 40 to every position")
    ax[2].set_xlabel("position"); ax[2].set_ylabel("dot product PE[40] . PE[p]")
    ax[2].annotate("self (max)", (40, sim[40]), (62, sim[40] * 0.93), color=MUTED, fontsize=8,
                   arrowprops=dict(arrowstyle="-", color=MUTED))
    fig.suptitle("Figure 4: sinusoidal positional encoding - each position gets a unique, smoothly varying signature", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "fig4_positional_encoding.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    # does adding PE change identical tokens at different positions?  (printed for the report)
    torch.manual_seed(0)
    e = torch.randn(1, 1, 128).expand(1, 6, 128)
    y = e + torch.tensor(pe[:6]).unsqueeze(0)
    d = torch.cdist(y[0], y[0])
    msg = ("Same token embedding at positions 0..5 -> pairwise L2 distance after adding PE:\n"
           + np.array2string(d.numpy(), precision=2, suppress_small=True)
           + "\n(identical tokens become distinguishable only because of position; distance grows with offset)")
    (LOGS / "pe_distinctness.txt").write_text(msg); print(msg)


def fig5_summary(run):
    ckpt = ROOT / "checkpoints" / f"{run}.pt"
    if ckpt.exists():
        model, tok, c = load_checkpoint(ckpt)
        src = f"trained checkpoint {ckpt.name} (val loss {c['extra'].get('val_loss', float('nan')):.4f})"
    else:
        model = GPT(GPTConfig()); src = "untrained baseline config"
    cfg = model.cfg
    rows, total = [], 0
    for name, p in model.named_parameters():
        pass
    groups = {"token embedding": model.tok_emb, "positional encoding": model.pos_enc,
              "final LayerNorm": model.ln_f, "LM head": model.lm_head}
    lines = [f"MiniGPT summary  [{src}]", "=" * 62, "configuration:"]
    for k, v in cfg.to_dict().items():
        lines.append(f"  {k:<16s} {v}")
    lines += ["-" * 62, f"{'component':<34s}{'shape / detail':<18s}{'params':>9s}", "-" * 62]
    n_emb = sum(p.numel() for p in model.tok_emb.parameters()); total += n_emb
    lines.append(f"{'token embedding':<34s}{str((cfg.vocab_size, cfg.d_model)):<18s}{n_emb:>9,}")
    lines.append(f"{'positional encoding (sinusoidal)':<34s}{'fixed buffer':<18s}{0:>9,}" if cfg.pos_encoding == "sinusoidal"
                 else f"{'positional encoding (learned)':<34s}{'':<18s}{sum(p.numel() for p in model.pos_enc.parameters()):>9,}")
    for i, b in enumerate(model.blocks):
        a = sum(p.numel() for p in b.attn.parameters()); f = sum(p.numel() for p in b.ffn.parameters())
        n = sum(p.numel() for p in list(b.ln1.parameters()) + list(b.ln2.parameters()))
        total += a + f + n
        lines.append(f"{f'block {i}: attention (Wq,Wk,Wv,Wo)':<34s}{f'{cfg.n_heads} heads x {cfg.d_model // cfg.n_heads}':<18s}{a:>9,}")
        lines.append(f"{f'block {i}: FFN (GELU)':<34s}{f'{cfg.d_model}->{cfg.ffn_dim}->{cfg.d_model}':<18s}{f:>9,}")
        lines.append(f"{f'block {i}: 2 x LayerNorm':<34s}{'':<18s}{n:>9,}")
    n_ln = sum(p.numel() for p in model.ln_f.parameters()); total += n_ln
    lines.append(f"{'final LayerNorm':<34s}{'':<18s}{n_ln:>9,}")
    n_head = 0 if cfg.tie_weights else sum(p.numel() for p in model.lm_head.parameters()); total += n_head
    lines.append(f"{'LM head (' + ('tied' if cfg.tie_weights else 'untied') + ')':<34s}{str((cfg.d_model, cfg.vocab_size)):<18s}{n_head:>9,}")
    lines += ["-" * 62, f"{'TOTAL trainable parameters':<52s}{model.num_parameters():>9,}",
              f"  ({model.num_parameters() / 1e6:.3f} M; sum of rows = {total:,}; "
              f"fp32 size = {model.num_parameters() * 4 / 2**20:.1f} MiB)"]
    assert total == model.num_parameters()
    txt = "\n".join(lines)
    (LOGS / "model_summary.txt").write_text(txt); print(txt)
    text_to_png(txt, FIG / "fig5_model_summary.png", title="Figure 5: model summary / parameter count")


def architecture_diagram():
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(6.4, 10)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(-0.8, 16.2)
    def box(x, y, w, h, t, fc="#eef3fb", ec=BLUE, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK)
    def arrow(y0, y1, x=5):
        ax.annotate("", (x, y1), (x, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
    def plus(y):
        ax.add_patch(Circle((5, y), 0.2, fc="white", ec=AQUA, lw=1.6, zorder=3))
        ax.text(5, y, "+", ha="center", va="center", fontsize=11, color=AQUA, weight="bold", zorder=4)
    def skip(y_from, y_to, x):
        ax.plot([5, x, x], [y_from, y_from, y_to], color=AQUA, lw=1.4, ls="--", zorder=1)
        ax.annotate("", (4.78, y_to), (x, y_to), arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=1.4, ls="--"), zorder=1)
        ax.text(x - 0.25, (y_from + y_to) / 2, "residual", rotation=90, ha="center", va="center", fontsize=8, color=AQUA)
    box(2.5, 0.0, 5, 0.7, "token ids  (B, T)", "#f4f4f1", GRID); arrow(0.7, 1.3)
    box(2.5, 1.3, 5, 0.9, "Token embedding  x sqrt(d_model)\n(B, T, D)"); arrow(2.2, 2.8)
    box(2.5, 2.8, 5, 0.9, "+ Sinusoidal positional encoding\n(B, T, D)", "#fdf0ea", ORANGE); arrow(3.7, 4.4)
    ax.add_patch(FancyBboxPatch((1.0, 4.4), 8.0, 7.6, boxstyle="round,pad=0.02,rounding_size=0.2", fc="none", ec=MUTED, lw=1.2, ls="--"))
    ax.text(8.8, 11.6, "x N", ha="right", fontsize=11, weight="bold", color=MUTED)
    box(2.6, 5.0, 4.8, 0.65, "LayerNorm", "#f4f4f1", GRID); arrow(4.4, 5.0); arrow(5.65, 6.2)
    box(2.6, 6.2, 4.8, 1.5, "Masked multi-head self-attention\nQ,K,V -> split into H heads\nsoftmax(QK^T / sqrt(d_k) + mask) V\nconcat heads -> W_o", fs=7.5)
    arrow(7.7, 7.9); plus(8.1); arrow(8.3, 8.75)
    skip(4.8, 8.1, 1.9)
    box(2.6, 8.75, 4.8, 0.65, "LayerNorm", "#f4f4f1", GRID); arrow(9.4, 9.85)
    box(2.6, 9.85, 4.8, 1.1, "Feed-forward\nLinear(D->4D) -> GELU -> Linear(4D->D)", fs=8)
    arrow(10.95, 11.15); plus(11.35)
    skip(8.55, 11.35, 1.5)
    arrow(12.0, 12.6)
    box(2.5, 12.6, 5, 0.65, "Final LayerNorm", "#f4f4f1", GRID); arrow(13.25, 13.8)
    box(2.5, 13.8, 5, 0.8, "LM head  Linear(D -> V)  ->  logits (B, T, V)"); arrow(14.6, 15.1)
    box(2.5, 15.1, 5, 0.65, "softmax / cross-entropy vs. shifted targets", "#eaf6f1", AQUA, fs=8)
    ax.text(5, -0.6, "Pre-LN decoder-only transformer (GPT style)", ha="center", fontsize=8, color=MUTED)
    fig.suptitle("Model architecture", y=0.92, weight="bold")
    fig.savefig(FIG / "architecture_diagram.png", dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="minigpt"); a = ap.parse_args()
    fig3_shapes(); fig4_positional(); fig5_summary(a.run); architecture_diagram()
