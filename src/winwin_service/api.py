from __future__ import annotations

import logging
import time
from copy import deepcopy
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .fetcher import FetchError, fetch_latest_draws
from .config import (
    AppConfig,
    DEFAULT_CONFIG,
    RECENT_WINDOW_MAX,
    RECENT_WINDOW_MIN,
    normalize_recent_window,
)
from .scoring import PredictError, predict_top3
from .schemas import PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="WinWin Bingo Predictor", version="1.0.0")
_PREDICTION_CACHE: dict[str, object] = {}
_BEST_CONFIG_PATH = Path("reports/final_holdout/best_config.json")
_LIVE_PARAM_KEYS = (
    "recent_draws_count",
    "max_recent_draws_count",
    "min_score_threshold",
    "skip_kill_threshold",
    "streak_kill_threshold",
    "candidate_trim_size",
    "warm_skip_min",
    "warm_skip_max",
    "streak_min",
    "streak_max",
)


def _load_validated_config() -> tuple[AppConfig, str, bool]:
    if not _BEST_CONFIG_PATH.exists():
        return DEFAULT_CONFIG, "default_config", False

    try:
        payload = json.loads(_BEST_CONFIG_PATH.read_text(encoding="utf-8"))
        params = payload.get("params", payload)
        kwargs = {
            key: int(params[key])
            for key in _LIVE_PARAM_KEYS
            if key in params
        }
        if not kwargs:
            return DEFAULT_CONFIG, "default_config_invalid_best_config", False
        raw_recent = int(
            kwargs.get(
                "recent_draws_count",
                DEFAULT_CONFIG.recent_draws_count,
            )
        )
        raw_max = int(
            kwargs.get(
                "max_recent_draws_count",
                DEFAULT_CONFIG.max_recent_draws_count
                or RECENT_WINDOW_MAX,
            )
        )
        normalized_recent, normalized_max = normalize_recent_window(
            raw_recent,
            raw_max,
        )
        if (
            raw_recent != normalized_recent
            or raw_max != normalized_max
            or raw_recent < RECENT_WINDOW_MIN
            or raw_recent > RECENT_WINDOW_MAX
            or raw_max < RECENT_WINDOW_MIN
            or raw_max > RECENT_WINDOW_MAX
        ):
            return (
                DEFAULT_CONFIG,
                "default_config_reject_long_window_best_config",
                False,
            )
        return AppConfig(**kwargs), "validated_best_config", True
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to load validated best config: %s", exc)
        return DEFAULT_CONFIG, "default_config_load_failed", False


(
    _ACTIVE_CONFIG,
    _ACTIVE_CONFIG_SOURCE,
    _ACTIVE_CONFIG_MATCH_VALIDATED,
) = _load_validated_config()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/predict", response_model=PredictionResponse)
def predict(debug: bool = False) -> PredictionResponse:
    now = time.time()
    cached_debug = _PREDICTION_CACHE.get("debug")
    cached_at = _PREDICTION_CACHE.get("created_at")
    ttl_seconds = _ACTIVE_CONFIG.prediction_cache_ttl_seconds
    cache_result = _PREDICTION_CACHE.get("result")
    if (
        cached_debug == debug
        and isinstance(cached_at, float)
        and isinstance(cache_result, dict)
        and (now - cached_at) <= ttl_seconds
    ):
        cached = deepcopy(cache_result)
        cached_metadata = cached.setdefault("metadata", {})
        cache_age = max(0.0, now - cached_at)
        cached_metadata["cache_hit"] = True
        cached_metadata["cache_age_seconds"] = round(cache_age, 3)
        cached_metadata["cache_strategy"] = "ttl_before_fetch"
        return PredictionResponse(**cached)

    try:
        draws, latest_period = fetch_latest_draws(config=_ACTIVE_CONFIG)
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
        result = predict_top3(
            draws,
            latest_period,
            config=_ACTIVE_CONFIG,
            include_regime_debug=debug,
        )
        result["metadata"]["active_config_source"] = _ACTIVE_CONFIG_SOURCE
        result["metadata"][
            "runtime_config_matches_validated_best"
        ] = _ACTIVE_CONFIG_MATCH_VALIDATED
        result["metadata"]["cache_hit"] = False
        result["metadata"]["cache_age_seconds"] = 0.0
        result["metadata"]["cache_strategy"] = "ttl_before_fetch"
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
