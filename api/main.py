"""FastAPI service for the CSP solver."""

from __future__ import annotations

from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.csp_solver_agent import solve
from agents.hint_agent import predict as hint_predict
from agents.scratch_solver_agent import predict as scratch_predict

app = FastAPI(title="CSP Solver Service", version="0.2.0")


class PredictRequest(BaseModel):
    board: List[List[int]] = Field(..., min_items=1)
    target: int


class SolveRequest(BaseModel):
    board: List[List[int]] = Field(..., min_items=1)


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        board = np.asarray(req.board, dtype=int)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad board: {exc}") from exc

    if board.ndim != 2:
        raise HTTPException(status_code=400, detail="board must be 2D")

    rows, cols = board.shape
    numbers = board[board != -1]
    if numbers.size != len(np.unique(numbers)):
        raise HTTPException(status_code=400, detail="duplicate numbers found")
    if numbers.size and (numbers.min() < 1 or numbers.max() > rows * cols):
        raise HTTPException(status_code=400, detail="numbers out of range")
    if not (1 <= req.target <= rows * cols):
        raise HTTPException(status_code=400, detail="target out of range")

    preds = scratch_predict(board, req.target)
    top3 = sorted(preds, key=lambda x: x["score"], reverse=True)[:3]
    return {"predictions": top3}


@app.post("/hints")
def hints_endpoint(req: PredictRequest):
    try:
        board = np.asarray(req.board, dtype=int)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad board: {exc}") from exc
    if board.ndim != 2 or board.shape[0] != board.shape[1]:
        raise HTTPException(status_code=400, detail="board must be square")
    return {"hints": hint_predict(board, req.target)}


@app.post("/solve")
def solve_endpoint(req: SolveRequest):
    try:
        board = np.asarray(req.board, dtype=int)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad board: {exc}") from exc
    if board.ndim != 2 or board.shape[0] != board.shape[1]:
        raise HTTPException(status_code=400, detail="board must be square")
    solved = solve(board)
    if solved is None:
        raise HTTPException(status_code=400, detail="puzzle has no solution")
    return {"solution": solved.tolist()}
