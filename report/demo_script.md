# Demo script (5-7 min screen recording) - maps to handout section 17

Have two terminals open in the repo root, venv activated. Rehearse once; say what you see, not what you read.

| # | Time | Show | Command / file | Say |
|---|---|---|---|---|
| 1 | 0:00 | Repo structure, core model file | `tree -L 2` or file explorer; open `src/gpt.py` | "Everything is under src/. gpt.py assembles embedding + position + N blocks + LM head. No pretrained weights, no HF model classes." |
| 2 | 0:45 | Attention + causal mask | `src/attention.py`: `causal_mask`, `scaled_dot_product_attention`, `split_heads`/`merge_heads` | "Mask is strictly upper-triangular, filled with -inf before softmax, so future weights are exactly zero." |
| 3 | 1:45 | Tests | `python -m pytest -q` (36 pass; mention test_mask, test_overfit) | "Includes causality of the whole model and a NumPy-vs-autograd gradient cross-check." |
| 4 | 2:30 | Config + parameter count | open `outputs/figures/fig5_model_summary.png` or `python -m experiments.make_evidence` | "810,240 parameters; 4 layers, 4 heads, d_model 128, context 128." |
| 5 | 3:00 | Loss curve | `outputs/figures/fig6_loss_curves.png` | "Val loss 4.19 -> 1.645, perplexity 5.18. Gap of 0.2: capacity-limited, not badly overfit." |
| 6 | 3:30 | One generation | `python -m src.generate --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8` | "Format is right, some words are invented - char-level, 0.8M params." |
| 7 | 4:15 | Attention visualization | `outputs/figures/fig7_attention_maps.png` | "Layer 0 head 1 attends to the previous character (50%). Layer 3 head 2 parks 61% of its attention on the newline after the speaker tag. Upper triangle empty = causal." |
| 8 | 5:00 | Deployed API | open your Render URL `/docs`, POST /generate | "Same model behind FastAPI." |
| 9 | 5:45 | Ablation + one limitation | `outputs/figures/fig9_ablations.png` | "Depth and sinusoidal positions help; heads are within seed noise. Limitation: single seed, reduced budget; greedy decoding loops." |
