"""Shared plotting helpers: consistent style, text-to-PNG 'terminal screenshots'."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
LOGS = ROOT / "outputs" / "logs"
FIG.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

# Categorical slots 1-3 of the validated default palette (fixed order, never cycled)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2dd", "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 2.0, "font.size": 10, "legend.frameon": False,
    })


style()


def text_to_png(text: str, path, title: str | None = None, fontsize: int = 9):
    """Render monospaced text as an image - used for terminal-style evidence screenshots."""
    lines = text.rstrip("\n").split("\n")
    w = max(8, min(16, 0.085 * max(len(l) for l in lines) * fontsize / 9 + 0.6))
    h = 0.2 * fontsize / 9 * (len(lines) + (2 if title else 1)) + 0.4
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_facecolor("#14141a")
    y = 1 - 0.3 / h
    if title:
        fig.text(0.02, y, title, color="#9aa0ff", family="monospace", fontsize=fontsize, weight="bold", va="top")
        y -= 0.2 * fontsize / 9 / h
    fig.text(0.02, y, "\n".join(lines), color="#e6e6e6", family="monospace", fontsize=fontsize, va="top",
             linespacing=1.25)
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
