"""Acceptance test 'checkpoint reload': a FRESH process loads saved weights and generates."""
import subprocess
import sys

import torch

from src.generate import generate_text
from src.gpt import GPT, GPTConfig
from src.tokenizer import CharTokenizer
from src.utils import ROOT, load_checkpoint, save_checkpoint


def test_save_load_roundtrip_in_fresh_process(tmp_path):
    torch.manual_seed(0)
    tok = CharTokenizer.fit("abcdefghij \n")
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, context_length=16, d_model=32, n_heads=2,
                          n_layers=2, ffn_dim=64, dropout=0.0)).eval()
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, tok, {})
    in_proc = generate_text(model, tok, "abc", 20, temperature=0.8, top_k=5, seed=7)

    code = ("from src.utils import load_checkpoint;from src.generate import generate_text;"
            f"m,t,_=load_checkpoint(r'{path}');print(repr(generate_text(m,t,'abc',20,0.8,5,None,7)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True)
    assert out.stdout.strip() == repr(in_proc)     # identical text => identical weights + sampling


def test_cli_prints_required_fields(tmp_path):
    tok = CharTokenizer.fit("abcdefghij \n")
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, context_length=16, d_model=32, n_heads=2,
                          n_layers=1, ffn_dim=64))
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, tok, {})
    out = subprocess.run([sys.executable, "-m", "src.generate", "--checkpoint", str(path),
                          "--prompt", "abc", "--max-new-tokens", "10"],
                         capture_output=True, text=True, cwd=ROOT, check=True).stdout
    for needle in ("Prompt: abc", "Generated:", "model parameters:", "context length: 16", "temperature: 0.8"):
        assert needle in out


def test_greedy_is_deterministic():
    tok = CharTokenizer.fit("abcdefghij \n")
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, context_length=16, d_model=32, n_heads=2,
                          n_layers=1, ffn_dim=64)).eval()
    a = generate_text(model, tok, "abc", 15, temperature=0)
    b = generate_text(model, tok, "abc", 15, temperature=0)
    assert a == b
