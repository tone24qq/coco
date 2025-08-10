"""FastAPI serving interface."""

from __future__ import annotations

import subprocess

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, Field, StrictInt, field_validator

from ..models.vocab import masked_logits_clip
from .decode import iterative_decode
from .model_loader import MODEL_CACHE, load_model_for_size
from .topk import compute_topk_positions


class PredictRequest(BaseModel):
    grid: list[list[StrictInt]] = Field(validation_alias=AliasChoices("grid", "board"))
    target: StrictInt | None = None

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
                if val < -1 or val > N:
                    raise ValueError("grid values must be between -1 and N")
        return v


class PredictResponse(BaseModel):
    rows: int
    cols: int
    grid: list[list[int]]


class TopkPrediction(BaseModel):
    row: int
    col: int
    prob: float


class TargetResponse(BaseModel):
    target: int
    predictions: list[TopkPrediction]
    log: str


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


@app.post("/predict", response_model=PredictResponse | TargetResponse)
def predict(
    req: PredictRequest,
) -> PredictResponse | TargetResponse:  # pragma: no cover - HTTP interface
    grid = req.grid
    rows, cols = len(grid), len(grid[0])
    model = load_model_for_size(rows, cols)
    vocab = model.embed.num_embeddings - 1
    N = rows * cols
    if N > vocab:
        raise HTTPException(status_code=400, detail="grid too large for model")

    # Replace non-positive values with 0 for model input
    processed_grid = [[0 if v <= 0 else v for v in r] for r in grid]
    max_val = max(max(r) for r in processed_grid)
    if max_val > vocab:
        raise HTTPException(
            status_code=400, detail="grid values exceed model vocabulary"
        )

    tokens = torch.tensor(processed_grid, dtype=torch.long).view(1, -1)
    attn = torch.ones_like(tokens, dtype=torch.bool)

    if req.target is not None:
        target = int(req.target)
        if not 1 <= target <= N:
            raise HTTPException(status_code=400, detail="target out of range")
        num_holes = int((tokens == 0).sum().item())
        log_lines = [f"[步驟1] 找到 {num_holes} 個空格。"]
        log_lines.append(f"[步驟2] 開始計算每個空格填入 {target} 的機率。")
        with torch.no_grad():
            logits = model(tokens, attn)
            logits = masked_logits_clip(logits, N)
            probs = torch.softmax(logits, dim=-1)[0]
            topk = compute_topk_positions(probs, tokens[0], target, 3, cols)
        log_lines.append("[步驟3] 計算完成，取 Top3：")
        for item in topk:
            log_lines.append(
                f"  (row={item['row']},col={item['col']}) 機率={item['prob']:.3f}"
            )
        return TargetResponse(target=target, predictions=topk, log="\n".join(log_lines))

    out = iterative_decode(model, tokens, attn, N)
    return PredictResponse(rows=rows, cols=cols, grid=out.view(rows, cols).tolist())
