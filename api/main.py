"""FastAPI service for the CSP solver."""

from __future__ import annotations

from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.csp_solver_agent import predict as solver_predict
from agents.hint_agent import predict as hint_predict

app = FastAPI(title="CSP Solver Service", version="0.2.0")


class PredictRequest(BaseModel):
    board: List[List[int]] = Field(..., min_items=1)
    target: int


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        board = np.asarray(req.board, dtype=int)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad board: {exc}") from exc
    if board.ndim != 2 or board.shape[0] != board.shape[1]:
        raise HTTPException(status_code=400, detail="board must be square")
    return {"predictions": solver_predict(board, req.target)}


@app.post("/hints")
def hints_endpoint(req: PredictRequest):
    try:
        board = np.asarray(req.board, dtype=int)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad board: {exc}") from exc
    if board.ndim != 2 or board.shape[0] != board.shape[1]:
        raise HTTPException(status_code=400, detail="board must be square")
    return {"hints": hint_predict(board, req.target)}
