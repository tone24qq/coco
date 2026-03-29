from __future__ import annotations

import logging
import time
from copy import deepcopy

from fastapi import FastAPI, HTTPException

from .fetcher import FetchError, fetch_latest_draws
from .config import DEFAULT_CONFIG
from .scoring import PredictError, predict_top3
from .schemas import PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="WinWin Bingo Predictor", version="1.0.0")
_PREDICTION_CACHE: dict[str, object] = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/predict", response_model=PredictionResponse)
def predict(debug: bool = False) -> PredictionResponse:
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
        now = time.time()
        cached_period = _PREDICTION_CACHE.get("latest_period")
        cached_debug = _PREDICTION_CACHE.get("debug")
        cached_at = _PREDICTION_CACHE.get("created_at")
        ttl_seconds = DEFAULT_CONFIG.prediction_cache_ttl_seconds
        cache_result = _PREDICTION_CACHE.get("result")
        if (
            cached_period == latest_period
            and cached_debug == debug
            and isinstance(cached_at, float)
            and isinstance(cache_result, dict)
            and (now - cached_at) <= ttl_seconds
        ):
            cached = deepcopy(cache_result)
            cached_metadata = cached.setdefault("metadata", {})
            cache_age = max(0.0, now - cached_at)
            cached_metadata["cache_hit"] = True
            cached_metadata["cache_age_seconds"] = round(cache_age, 3)
            return PredictionResponse(**cached)

        result = predict_top3(
            draws,
            latest_period,
            include_regime_debug=debug,
        )
        result["metadata"]["cache_hit"] = False
        result["metadata"]["cache_age_seconds"] = 0.0
        _PREDICTION_CACHE["latest_period"] = latest_period
        _PREDICTION_CACHE["debug"] = debug
        _PREDICTION_CACHE["created_at"] = now
        _PREDICTION_CACHE["result"] = deepcopy(result)
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
