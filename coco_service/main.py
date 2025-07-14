from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from coco_agents.predict_agent import predict as agent_predict


class PredictRequest(BaseModel):
    board: List[List[int]]
    target: int
    kwargs: Optional[Dict[str, Any]] = None


class Prediction(BaseModel):
    row: int
    col: int
    score: float


app = FastAPI(title="Matrix Factorization Service")


@app.post("/predict", response_model=List[Prediction])
def predict(req: PredictRequest) -> List[Prediction]:
    board = np.array(req.board)
    kwargs = req.kwargs or {}
    predictions = agent_predict(board, req.target, **kwargs)
    return [Prediction(**p) for p in predictions]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
