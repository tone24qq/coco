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
logger = logging.getLogger(__name__)

app = FastAPI(title="WinWin Bingo Predictor", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/predict", response_model=PredictionResponse)
def predict() -> PredictionResponse:
    try:
        draws, latest_period = fetch_latest_draws()
    except FetchError as exc:
        logger.exception("fetch failed reason=%s", exc)
        raise HTTPException(
            status_code=502,
            detail={
                "error_code": "FETCH_FAILED",
                "detail": str(exc),
            },
        ) from exc

    try:
        result = predict_top3(draws, latest_period)
    except PredictError as exc:
        detail = str(exc)
        if "No combinations exceed min_score_threshold" in detail:
            logger.error("predict failed reason=%s", detail)
        elif "Valid number pool below 3" in detail:
            logger.error("predict failed reason=%s", detail)
        else:
            logger.exception("predict failed reason=%s", detail)
        raise HTTPException(
            status_code=502,
            detail={
                "error_code": "PREDICT_FAILED",
                "detail": detail,
            },
        ) from exc

    return PredictionResponse(**result)
