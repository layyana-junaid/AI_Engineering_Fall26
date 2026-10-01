"""Required hand calculation (handout section 6) -> Figure 1 and Figure 2 (mask).

3 tokens, d_k = d_v = 2. Every intermediate value is printed so it can be copied
into the report, and the final result is checked against src.attention.
"""
import math

import matplotlib.pyplot as plt
import torch

from experiments._common import AQUA, BLUE, FIG, LOGS, MUTED, ORANGE, text_to_png
from src.attention import causal_mask, scaled_dot_product_attention

Q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
K = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
V = torch.tensor([[1., 0.], [0., 2.], [3., 1.]])


def fmt(m, d=3):
    return "\n".join("    [" + "  ".join(f"{v:7.{d}f}" for v in row) + " ]" for row in m.tolist())


def main():
    out = []
    P = out.append
    P("HAND CALCULATION: scaled dot-product attention, 3 tokens, d_k = d_v = 2")
    P("\n1. Matrices (rows = tokens 1..3)")
    P("  Q =\n" + fmt(Q, 0)); P("  K =\n" + fmt(K, 0)); P("  V =\n" + fmt(V, 0))
    S_raw = Q @ K.T
    P("\n2. Q K^T  (entry [i,j] = q_i . k_j)")
    P(fmt(S_raw, 0))
    P("   e.g. [3,3] = (1)(1) + (1)(1) = 2 ;  [1,2] = (1)(0) + (0)(1) = 0")
    S = S_raw / math.sqrt(2)
    P(f"\n3. Divide by sqrt(d_k) = sqrt(2) = {math.sqrt(2):.4f}")
    P(fmt(S))
    W = torch.softmax(S, -1)
    P("\n4. Row-wise softmax  (each row sums to 1)")
    P(fmt(W, 4))
    P("   row 1: exp(.707)=2.028, exp(0)=1, exp(.707)=2.028 -> sum 5.056 -> [0.4011, 0.1978, 0.4011]")
    O = W @ V
    P("\n5. Output = weights @ V  (each row = weighted average of the value vectors)")
    P(fmt(O, 4))
    P("   row 1: 0.4011*[1,0] + 0.1978*[0,2] + 0.4011*[3,1] = [1.6044, 0.7967]")

    mask = causal_mask(3)
    Sm = S.masked_fill(mask, float("-inf"))
    P("\n6. Causal mask: add -inf above the diagonal (token 1 cannot see 2,3; token 2 cannot see 3)")
    P("  masked scores =\n" + "\n".join("    [" + "  ".join("   -inf" if v == float('-inf') else f"{v:7.3f}" for v in r) + " ]" for r in Sm.tolist()))
    Wm = torch.softmax(Sm, -1)
    P("  masked softmax weights =\n" + fmt(Wm, 4))
    Om = Wm @ V
    P("  masked output =\n" + fmt(Om, 4))
    P("   row 1 = [1, 0] = v_1 exactly: with only itself visible, token 1 copies its own value.")
    P("   row 2: scores [0, 0.707] -> weights [1/(1+2.028), 2.028/(1+2.028)] = [0.3302, 0.6698]")
    P("          output = 0.3302*[1,0] + 0.6698*[0,2] = [0.3302, 1.3395]")
    P("   row 3: identical to the unmasked row, because token 3 may already see everyone.")

    P("\n7. What changes: unmasked, every token mixes information from ALL positions (including the")
    P("   future). Masked, token i receives a convex combination of tokens 1..i only, so information")
    P("   flows strictly left-to-right. Row 1 loses all context; row 3 is unchanged.")

    # ---- verify against the library implementation
    ref_o, ref_w = scaled_dot_product_attention(Q, K, V)
    ref_om, ref_wm = scaled_dot_product_attention(Q, K, V, mask)
    assert torch.allclose(ref_w, W, atol=1e-6) and torch.allclose(ref_o, O, atol=1e-6)
    assert torch.allclose(ref_wm, Wm, atol=1e-6) and torch.allclose(ref_om, Om, atol=1e-6)
    P("\nVERIFIED: matches src.attention.scaled_dot_product_attention (masked and unmasked).")

    text = "\n".join(out)
    (LOGS / "hand_calculation.txt").write_text(text)
    print(text)
    text_to_png(text, FIG / "fig1_hand_calculation.png", fontsize=8)

    # ---- Figure 2: causal mask matrix + effect on attention weights
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    big = causal_mask(8).float()
    ax[0].imshow(big, cmap="Greys", vmin=0, vmax=1.3); ax[0].grid(False)
    ax[0].set_title("Causal mask, T=8\n(dark = forbidden)")
    for name, w, a in (("Unmasked weights", W, ax[1]), ("Masked weights", Wm, ax[2])):
        a.imshow(w, cmap="Blues", vmin=0, vmax=1); a.grid(False); a.set_title(name + " (3 tokens)")
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center",
                       color="white" if w[i, j] > 0.5 else "black")
    for a, n in zip(ax, (8, 3, 3)):
        a.set_xticks(range(n)); a.set_yticks(range(n))
        a.set_xticklabels(range(1, n + 1)); a.set_yticklabels(range(1, n + 1))
        a.set_xlabel("key position (attended to)"); a.set_ylabel("query position")
    fig.suptitle("Figure 2: future tokens are blocked - upper triangle has exactly zero weight", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "fig2_causal_mask.png", dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
