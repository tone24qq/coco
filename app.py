import glob
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # torch may be unavailable in minimal runtimes
    torch = None  # type: ignore[assignment]
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from model import DynamicMET

logger = logging.getLogger(__name__)

app = FastAPI(title="Matrix Factorization Service", version="0.1.0")


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "coco", "version": "0.1.0"}


@app.get("/health")
def health():
    """Simple readiness/liveness probe.

    Returns the loaded model shapes so ops can verify startup state.
    """
    return {
        "status": "ok",
        "models": [{"rows": r, "cols": c} for (r, c) in models.keys()],
    }


class PredictRequest(BaseModel):
    board: List[List[int]] = Field(..., description="2D grid, blanks use -1.")
    # 兩個欄位都接受，擇一或兩者皆送都行
    target: Optional[int] = Field(None, description="Preferred. Target number.")
    target_value: Optional[int] = Field(
        None,
        description="Alias of `target`. Accepted for backward compatibility.",
    )

    @model_validator(mode="after")
    def _normalize(cls, values: "PredictRequest"):
        if values.target is None and values.target_value is None:
            raise ValueError("One of `target` or `target_value` must be provided.")
        if values.target is None and values.target_value is not None:
            values.target = values.target_value
            logger.warning("[REQ] using `target_value` as `target`: %s", values.target)
        elif values.target is not None and values.target_value is not None:
            if values.target != values.target_value:
                raise ValueError(
                    f"Inconsistent target: target={values.target} != target_value={values.target_value}"
                )
            logger.info(
                "[REQ] both `target` and `target_value` provided, value=%s",
                values.target,
            )
        return values


class Prediction(BaseModel):
    row: int
    col: int
    score: float  # 0~1 之間的置信度


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


@app.post("/predict", response_model=List[Prediction])
def predict(req: PredictRequest):
    board = np.array(req.board)
    rows, cols = board.shape
    n = rows * cols
    target = req.target
    if target is None:
        raise HTTPException(status_code=422, detail="`target` is required.")
    if not (1 <= target <= n):
        raise HTTPException(
            status_code=422, detail=f"target must be in [1, {n}], got {target}"
        )

    flat = board.flatten()
    mask_pos = np.where(flat == -1)[0]
    if mask_pos.size == 0:
        raise HTTPException(status_code=422, detail="no blank cells (-1) to predict")

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
        scores_all = probs[0, :, target]
        scores_np = scores_all.detach().cpu().numpy()
    else:
        inp = flat_input.reshape(1, -1)
        logits = model(inp)
        scores_np = logits[0, :, target]

    candidate_scores = scores_np[mask_pos]
    topk_local = np.argsort(candidate_scores)[-min(3, len(candidate_scores)) :][::-1]
    top_indices = mask_pos[topk_local]

    return [
        Prediction(
            row=int(idx // cols), col=int(idx % cols), score=float(scores_np[idx])
        )
        for idx in top_indices
    ]
