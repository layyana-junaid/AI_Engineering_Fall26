"""FastAPI inference service.

    uvicorn src.api:app --host 0.0.0.0 --port 8000
    curl -X POST localhost:8000/generate -H 'content-type: application/json' \
         -d '{"prompt": "ROMEO:", "max_new_tokens": 120, "temperature": 0.8}'

The checkpoint path comes from the CHECKPOINT environment variable
(default: checkpoints/minigpt.pt). No secrets or API keys are used anywhere.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .generate import generate_text
from .utils import ROOT, load_checkpoint

STATE: dict = {}
MAX_NEW_TOKENS = 1000


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.environ.get("CHECKPOINT", str(ROOT / "checkpoints" / "minigpt.pt"))
    model, tok, ckpt = load_checkpoint(path, "cpu")
    STATE.update(model=model, tok=tok, path=path, val_loss=ckpt.get("extra", {}).get("val_loss"))
    yield
    STATE.clear()


app = FastAPI(title="MiniGPT", description="Small GPT trained from scratch on Tiny Shakespeare",
              version="1.0.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field("ROMEO:", max_length=2000)
    max_new_tokens: int = Field(200, ge=1, le=MAX_NEW_TOKENS)
    temperature: float = Field(0.8, ge=0.0, le=3.0, description="0 = greedy")
    top_k: int | None = Field(40, ge=1)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    seed: int | None = None


class GenerateResponse(BaseModel):
    prompt: str
    generated: str
    model_parameters: int
    context_length: int
    temperature: float
    latency_ms: float


@app.get("/")
def root():
    return {"service": "MiniGPT", "docs": "/docs", "health": "/health", "generate": "POST /generate"}


@app.get("/health")
def health():
    m = STATE["model"]
    return {"status": "ok", "parameters": m.num_parameters(), "context_length": m.cfg.context_length,
            "checkpoint": os.path.basename(STATE["path"]), "val_loss": STATE.get("val_loss")}


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    if "model" not in STATE:
        raise HTTPException(503, "model not loaded")
    t0 = time.time()
    text = generate_text(STATE["model"], STATE["tok"], req.prompt, req.max_new_tokens, req.temperature,
                         req.top_k, req.top_p, req.seed)
    m = STATE["model"]
    return GenerateResponse(prompt=req.prompt, generated=text, model_parameters=m.num_parameters(),
                            context_length=m.cfg.context_length, temperature=req.temperature,
                            latency_ms=(time.time() - t0) * 1000)
