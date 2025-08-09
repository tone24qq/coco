"""FastAPI serving interface."""

from __future__ import annotations

import torch
from fastapi import FastAPI

from .decode import iterative_decode
from .model_loader import load_model

app = FastAPI()
model = None


@app.on_event("startup")
def _load() -> None:  # pragma: no cover - server startup
    global model
    model = load_model("weights/best.ckpt")
    model.eval()


@app.post("/predict")
def predict(req: dict) -> dict:  # pragma: no cover - HTTP interface
    grid = req["grid"]
    rows, cols = len(grid), len(grid[0])
    tokens = torch.tensor(grid, dtype=torch.long).view(1, -1)
    attn = torch.ones_like(tokens, dtype=torch.bool)
    out = iterative_decode(model, tokens, attn, rows * cols)
    return {"rows": rows, "cols": cols, "grid": out.view(rows, cols).tolist()}
