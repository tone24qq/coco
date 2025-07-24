import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from rf_infer.core import infer_top3_for_target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def initialize_resources() -> None:
    """Placeholder for startup logic."""


async def cleanup_resources() -> None:
    """Placeholder for shutdown logic."""


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_resources()
    yield
    await cleanup_resources()


class PredictRequest(BaseModel):
    board: List[List[int]]
    target: int
    kwargs: Optional[Dict[str, Any]] = None


class Prediction(BaseModel):
    row: int
    col: int
    score: float


def _predict_lgbm(
    board: list[list[int]], target: int, models_dir: str
) -> list[dict[str, float]]:
    """Return top-3 predictions using LightGBM models."""
    coords = infer_top3_for_target(
        np.array(board, dtype=int), target, models_dir=models_dir
    )
    return [{"row": r, "col": c, "score": 1.0} for r, c in coords]


app = FastAPI(title="Matrix Factorization Service", lifespan=lifespan)


@app.get("/")
async def root() -> Dict[str, str]:
    """Health check endpoint."""
    return {"message": "Hello World"}


@app.post("/predict", response_model=List[Prediction])
async def predict(req: PredictRequest) -> List[Prediction]:
    logger.info(
        "Received predict request: target=%s board=%dx%d",
        req.target,
        len(req.board),
        len(req.board[0]) if req.board else 0,
    )
    kwargs = req.kwargs or {}
    models_dir = kwargs.pop("models_dir", os.getenv("MODELS_DIR", "models"))
    predictions = _predict_lgbm(req.board, req.target, models_dir)
    logger.info("Returning %d predictions", len(predictions))
    return [Prediction(**p) for p in predictions]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
