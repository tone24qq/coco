"""FastAPI entry point for CSP solver agent."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from agents.csp_solver_agent import predict

app = FastAPI()


class PredictRequest(BaseModel):
    board: List[List[int]]
    target: int


@app.post("/predict")
def predict_endpoint(request: PredictRequest) -> List[Dict[str, Any]]:
    board = np.array(request.board, dtype=int)
    return predict(board, target=request.target)
