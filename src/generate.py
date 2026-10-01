"""Autoregressive generation + command-line interface.

    python -m src.generate --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8
"""
from __future__ import annotations

import argparse

import torch

from .utils import get_device, load_checkpoint, set_seed


@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
             top_k: int | None = None, top_p: float | None = None,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """Append ``max_new_tokens`` sampled tokens to ``idx`` (shape (B, T0)).

    temperature == 0 -> greedy argmax. Otherwise logits are divided by the temperature,
    optionally truncated by top-k / top-p, and sampled from the softmax.
    The model only ever sees the last ``context_length`` tokens (sliding window).
    """
    was_training = model.training
    model.eval()
    ctx = model.cfg.context_length
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -ctx:])
        logits = logits[:, -1, :]                              # next-token distribution
        if temperature == 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p is not None and 0 < top_p < 1:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cum - torch.softmax(sorted_logits, dim=-1) > top_p   # keep first token over p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
        idx = torch.cat([idx, nxt], dim=1)
    model.train(was_training)
    return idx


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8,
                  top_k: int | None = 40, top_p: float | None = None, seed: int | None = None,
                  device="cpu") -> str:
    ids = tokenizer.encode(prompt) if prompt else tokenizer.encode("\n")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(seed)
    out = generate(model, idx, max_new_tokens, temperature, top_k, top_p, gen)
    return tokenizer.decode(out[0].tolist())


def main():
    ap = argparse.ArgumentParser(description="Generate text with a trained MiniGPT checkpoint")
    ap.add_argument("--checkpoint", default="checkpoints/minigpt.pt")
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    # Shells pass "\\n" as two literal characters; turn common escapes into real ones so
    # `--prompt "KING HENRY:\\n"` means "KING HENRY:" followed by a newline.
    a.prompt = a.prompt.replace("\\n", "\n").replace("\\t", "\t")

    set_seed(a.seed)
    device = get_device(a.device)
    model, tok, _ = load_checkpoint(a.checkpoint, device)
    text = generate_text(model, tok, a.prompt, a.max_new_tokens, a.temperature,
                         a.top_k, a.top_p, a.seed, device)
    print(f"Prompt: {a.prompt}")
    print(f"Generated: {text}\n")
    print(f"model parameters: {model.num_parameters() / 1e6:.2f}M")
    print(f"context length: {model.cfg.context_length}")
    print(f"temperature: {a.temperature}")


if __name__ == "__main__":
    main()