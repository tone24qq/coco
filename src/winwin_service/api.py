from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from .fetcher import FetchError, fetch_latest_draws
from .scoring import PredictError, predict_top3
from .schemas import PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title="WinWin Bingo Predictor", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/predict", response_model=PredictionResponse)
def predict() -> PredictionResponse:
    try:
        draws, latest_period = fetch_latest_draws()
        result = predict_top3(draws, latest_period)
    except (FetchError, PredictError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return PredictionResponse(**result)
