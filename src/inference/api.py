"""FastAPI serving interface."""

from __future__ import annotations

import subprocess

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, Field, StrictInt, field_validator

from .decode import iterative_decode
from .model_loader import MODEL_CACHE, load_model_for_size


class PredictRequest(BaseModel):
    grid: list[list[StrictInt]] = Field(validation_alias=AliasChoices("grid", "board"))

    @field_validator("grid")
    @classmethod
    def _validate_grid(cls, v: list[list[int]]) -> list[list[int]]:
        if not v or not all(isinstance(r, list) for r in v):
            raise ValueError("grid must be a 2D list")
        rows, cols = len(v), len(v[0])
        if any(len(r) != cols for r in v):
            raise ValueError("grid must be rectangular")
        N = rows * cols
        if N == 0:
            raise ValueError("grid too large")
        for r in v:
            for val in r:
                if not 0 <= val <= N:
                    raise ValueError("grid values must be between 0 and N")
        return v


class PredictResponse(BaseModel):
    rows: int
    cols: int
    grid: list[list[int]]


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:  # pragma: no cover - HTTP interface
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, object]:  # pragma: no cover - HTTP interface
    sha = "unknown"
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"]
            )  # pragma: no cover - best effort
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover - best effort
        pass
    model = next(iter(MODEL_CACHE.values()), None)
    vocab = model.embed.num_embeddings - 1 if model is not None else 0
    device = str(next(model.parameters()).device) if model is not None else "cpu"
    return {"git_sha": sha, "vocab_size": vocab, "device": device}


@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
) -> PredictResponse:  # pragma: no cover - HTTP interface
    grid = req.grid
    rows, cols = len(grid), len(grid[0])
    model = load_model_for_size(rows, cols)
    vocab = model.embed.num_embeddings - 1
    N = rows * cols
    if N > vocab:
        raise HTTPException(status_code=400, detail="grid too large for model")
    max_val = max(max(r) for r in grid)
    if max_val > vocab:
        raise HTTPException(
            status_code=400, detail="grid values exceed model vocabulary"
        )
    tokens = torch.tensor(grid, dtype=torch.long).view(1, -1)
    attn = torch.ones_like(tokens, dtype=torch.bool)
    out = iterative_decode(model, tokens, attn, N)
    return PredictResponse(rows=rows, cols=cols, grid=out.view(rows, cols).tolist())
