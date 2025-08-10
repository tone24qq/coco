"""FastAPI serving interface."""

from __future__ import annotations

import os
import subprocess

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, Field, StrictInt, field_validator

from ..models.vocab import masked_logits_clip
from ..training.dep_bias import apply_dep_bias
from ..utils.seed import seed_all
from .decode import iterative_decode
from .model_loader import MODEL_CACHE, load_model_for_size
from .topk import compute_topk_positions

seed_all(int(os.getenv("COCO_SEED", "0")))


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

    # Replace -1 with 0 for model input but keep track of holes
    holes = [v == -1 for r in grid for v in r]
    processed_grid = [[0 if v == -1 else v for v in r] for r in grid]
    max_val = max(max(r) for r in processed_grid)
    if max_val > vocab:
        raise HTTPException(
            status_code=400, detail="grid values exceed model vocabulary"
        )

    tokens = torch.tensor(processed_grid, dtype=torch.long).view(1, -1)
    hole_mask = torch.tensor(holes, dtype=torch.bool)
    attn = torch.ones_like(tokens, dtype=torch.bool)

    if req.target is not None:
        target = int(req.target)
        if not 1 <= target <= N:
            raise HTTPException(status_code=400, detail="target out of range")
        num_holes = int(hole_mask.sum().item())
        log_lines = [f"[步驟1] 盤面共有 {num_holes} 個空格。"]
        log_lines.append("[步驟2] 剔除已開數字，保留空格作為候選點。")
        log_lines.append(f"[步驟3] 根據已開數字佈局計算各候選點填入 {target} 的機率。")
        with torch.no_grad():
            logits = model(tokens, attn)
            logits = masked_logits_clip(logits, N)
            tgt_mask = torch.zeros_like(tokens)
            tgt_mask[0, hole_mask] = 1
            apply_dep_bias(
                logits,
                tokens,
                tgt_mask,
                torch.tensor([rows]),
                torch.tensor([cols]),
                torch.tensor([N]),
                dep_alpha=0.5,
            )
            probs = torch.softmax(logits, dim=-1)[0]
            topk = compute_topk_positions(probs, hole_mask, target, 3, cols)
        log_lines.append("[步驟4] 取機率最高的 Top3：")
        for item in topk:
            log_lines.append(
                f"  (row={item['row']},col={item['col']}) 機率={item['prob']:.3f}"
            )
        return TargetResponse(target=target, predictions=topk, log="\n".join(log_lines))

    out = iterative_decode(model, tokens, attn, N)
    return PredictResponse(rows=rows, cols=cols, grid=out.view(rows, cols).tolist())
