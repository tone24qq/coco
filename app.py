import glob
import os
import re
from typing import Dict, Tuple

import numpy as np

try:
    import torch
except Exception:  # torch may be unavailable in minimal runtimes
    torch = None  # type: ignore[assignment]
from fastapi import FastAPI
from pydantic import BaseModel

from model import DynamicMET

app = FastAPI()


@app.get("/health")
def health():
    """Simple readiness/liveness probe.

    Returns the loaded model shapes so ops can verify startup state.
    """
    return {
        "status": "ok",
        "models": [{"rows": r, "cols": c} for (r, c) in models.keys()],
    }


class BoardInput(BaseModel):
    board: list[list[int]]
    target_value: int  # 1..N (N=rows*cols)


MODEL_GLOB = os.environ.get("MODEL_GLOB", "met_*x*.pth")
_PATTERN = re.compile(r"met_(\d+)x(\d+)\.pth$")
models: Dict[Tuple[int, int], DynamicMET] = {}


def _load_one(path: str, rows: int, cols: int) -> DynamicMET:
    n = rows * cols
    model = DynamicMET(n, n)
    if torch is not None and os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        if hasattr(model, "eval"):
            model.eval()
    return model


def _discover_models() -> None:
    found = False
    for path in glob.glob(MODEL_GLOB):
        m = _PATTERN.match(os.path.basename(path))
        if not m:
            continue
        r, c = int(m.group(1)), int(m.group(2))
        models[(r, c)] = _load_one(path, r, c)
        found = True
    if not found:
        r, c = 8, 10
        models[(r, c)] = DynamicMET(r * c, r * c)
        if hasattr(models[(r, c)], "eval"):
            models[(r, c)].eval()


@app.on_event("startup")
def load_models() -> None:
    """Search and load checkpoints following pattern met_{R}x{C}.pth."""
    _discover_models()


@app.post("/predict")
async def predict(input: BoardInput):
    board = np.array(input.board)
    rows, cols = board.shape
    n = rows * cols
    if not (1 <= input.target_value <= n):
        return {"error": f"target_value must be in [1, {n}], got {input.target_value}"}

    flat = board.flatten()
    mask_pos = np.where(flat == -1)[0]
    if mask_pos.size == 0:
        return {"error": "no blank cells (-1) to predict"}

    flat_input = np.where(flat < 0, 0, flat)

    model = models.get((rows, cols))
    if model is None:
        model = DynamicMET(n, n)
        if hasattr(model, "eval"):
            model.eval()
        models[(rows, cols)] = model

    if torch is not None:
        inp = torch.tensor(flat_input).long().unsqueeze(0)
        logits = model(inp)  # type: ignore[misc]
        probs = torch.softmax(logits, dim=-1)
        scores_all = probs[0, :, input.target_value]
        scores_np = scores_all.detach().cpu().numpy()
    else:
        inp = flat_input.reshape(1, -1)
        logits = model(inp)
        scores_np = logits[0, :, input.target_value]

    candidate_scores = scores_np[mask_pos]
    topk_local = np.argsort(candidate_scores)[-min(3, len(candidate_scores)) :][::-1]
    top_indices = mask_pos[topk_local]

    return [
        {
            "row": int(idx // cols),
            "col": int(idx % cols),
            "score": float(scores_np[idx]),
        }
        for idx in top_indices
    ]
