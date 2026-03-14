from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json
import logging
from typing import List

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.fetchers.auzo_bingo import (  # noqa: E402
    BingoDrawFetcher,
    FetchDrawsError,
    build_recent_draws,
)
from src.predict import Predictor  # noqa: E402
from src.utils import (  # noqa: E402
    CONFIG_DIR,
    MODELS_DIR,
    build_recent_report,
    load_yaml,
    min_required_history,
    normalize_feature_version,
)

app = FastAPI(title="BingoBingo Predictor", version="1.0.0")
LOGGER = logging.getLogger(__name__)
PREDICT_CFG = load_yaml(CONFIG_DIR / "predict.yaml")
MIN_RECENT_DRAWS = int(PREDICT_CFG.get("min_recent_draws", 1))
MAX_RECENT_DRAWS = int(PREDICT_CFG.get("max_recent_draws", 999))
FETCH_TIMEOUT = float(PREDICT_CFG.get("fetch_timeout_seconds", 8.0))
FETCH_RETRIES = int(PREDICT_CFG.get("fetch_retries", 2))
FETCH_SOURCES = list(PREDICT_CFG.get("auto_fetch_sources", []))
FETCH_BACKOFF_SECONDS = float(PREDICT_CFG.get("fetch_retry_backoff_seconds", 0.5))
METADATA = (
    json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    if (MODELS_DIR / "metadata.json").exists()
    else {}
)
MODEL_LOAD_ERROR: str | None = None
if (MODELS_DIR / "catboost_top20.cbm").exists():
    try:
        PREDICTOR = Predictor.load()
    except ValueError as exc:
        PREDICTOR = None
        MODEL_LOAD_ERROR = str(exc)
else:
    PREDICTOR = None


class PredictPayload(BaseModel):
    recent_draws: List[List[int]] | None = Field(
        default=None,
        description=(
            "optional; when provided, must contain min~max draws with exactly 20 "
            "unique numbers per draw "
            "between 1 and 80"
        ),
    )


def _required_history_for_predictor() -> int:
    if PREDICTOR is None:
        return int(PREDICT_CFG.get("feature_min_history", 22))
    predictor_feature_version = normalize_feature_version(
        getattr(
            PREDICTOR, "feature_version", METADATA.get("feature_version", "v3_core20")
        )
    )
    predictor_runtime_config = getattr(PREDICTOR, "runtime_config", {})
    return max(
        int(PREDICT_CFG.get("feature_min_history", 22)),
        min_required_history(predictor_feature_version, predictor_runtime_config),
    )


def _effective_min_history(recent_draws: List[List[int]]) -> int:
    required = _required_history_for_predictor()
    return max(0, min(required, len(recent_draws) - 1))


def _validate_recent_draws(recent_draws: List[List[int]]) -> None:
    if not (MIN_RECENT_DRAWS <= len(recent_draws) <= MAX_RECENT_DRAWS):
        raise HTTPException(
            status_code=400,
            detail=(
                f"recent_draws must contain {MIN_RECENT_DRAWS} to "
                f"{MAX_RECENT_DRAWS} draws"
            ),
        )

    for i, nums in enumerate(recent_draws):
        if len(nums) != 20:
            raise HTTPException(
                status_code=400,
                detail=f"draw index {i} must contain exactly 20 numbers",
            )
        if len(set(nums)) != 20:
            raise HTTPException(
                status_code=400,
                detail=f"draw index {i} contains duplicate numbers",
            )
        if any(n < 1 or n > 80 for n in nums):
            raise HTTPException(
                status_code=400,
                detail=f"draw index {i} contains out-of-range numbers",
            )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": PREDICTOR is not None,
        "model_load_error": MODEL_LOAD_ERROR,
    }


@app.get("/analysis")
def analysis() -> dict:
    required = int(PREDICT_CFG.get("feature_min_history", 22))
    return {
        "metadata": METADATA,
        "feature_min_history": required,
        "recent_draws_rules": {
            "min": MIN_RECENT_DRAWS,
            "max": MAX_RECENT_DRAWS,
            "numbers_per_draw": 20,
            "number_range": [1, 80],
            "required": False,
        },
    }


@app.post("/predict")
def predict(payload: PredictPayload) -> dict:
    if PREDICTOR is None:
        if MODEL_LOAD_ERROR:
            return {"error": f"model unavailable: {MODEL_LOAD_ERROR}"}
        return {"error": "model not found, please train first"}
    auto_fetched = payload.recent_draws is None
    data_source = "manual"
    records: list[dict] = []
    fetch_attempts: list[dict[str, str | bool | None]] = []

    if auto_fetched:
        fetcher = BingoDrawFetcher(
            sources=FETCH_SOURCES,
            timeout=FETCH_TIMEOUT,
            retries=FETCH_RETRIES,
            retry_backoff_seconds=FETCH_BACKOFF_SECONDS,
        )
        try:
            recent_draws, fetched_records, data_source, fetch_attempts = (
                build_recent_draws(
                    fetcher=fetcher,
                    min_draws=MIN_RECENT_DRAWS,
                    max_draws=MAX_RECENT_DRAWS,
                )
            )
        except FetchDrawsError as exc:
            raise HTTPException(
                status_code=502, detail=f"auto fetch failed: {exc}"
            ) from exc
        payload.recent_draws = recent_draws
        fetch_attempts = [
            {"source": a.source, "ok": a.ok, "error": a.error} for a in fetch_attempts
        ]
        records = [
            {
                "issue": record.issue,
                "draw_date": record.draw_time,
                "numbers": json.dumps(record.numbers, ensure_ascii=False),
                "size_label": getattr(record, "size_label", None)
                or getattr(record, "big_small", None),
                "odd_even_label": getattr(record, "odd_even_label", None)
                or getattr(record, "odd_even", None),
                "streak_count": getattr(record, "streak_count", None),
            }
            for record in fetched_records
        ]
    else:
        _validate_recent_draws(payload.recent_draws)

    assert payload.recent_draws is not None
    _validate_recent_draws(payload.recent_draws)
    effective_min_history = _effective_min_history(payload.recent_draws)

    if not records:
        records = [
            {
                "issue": i - len(payload.recent_draws) + 1,
                "draw_date": None,
                "numbers": json.dumps(sorted(nums), ensure_ascii=False),
                "size_label": None,
                "odd_even_label": None,
                "streak_count": None,
            }
            for i, nums in enumerate(payload.recent_draws)
        ]
    if len(records) != len(payload.recent_draws):
        raise HTTPException(
            status_code=400, detail="records and recent_draws length mismatch"
        )

    draws = records
    df = pd.DataFrame(draws)

    try:
        result = PREDICTOR.predict_from_draws(df, min_history=effective_min_history)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result["analysis_report"] = build_recent_report(payload.recent_draws)
    result["model_version"] = METADATA.get("model_type", "unknown")
    result["feature_version"] = "v3_core20"
    result["training_data_snapshot"] = {
        "train_issue_start": METADATA.get("train_issue_start"),
        "train_issue_end": METADATA.get("train_issue_end"),
        "feature_rows": METADATA.get("feature_rows"),
    }
    result["calibration_method"] = METADATA.get("calibration_method", "none")
    issues_used = [record["issue"] for record in records]
    result["data_source"] = data_source
    result["recent_draws_count"] = len(payload.recent_draws)
    result["first_issue_used"] = issues_used[0] if auto_fetched else None
    result["last_issue_used"] = issues_used[-1] if auto_fetched else None
    result["issues_used"] = (
        issues_used if auto_fetched else [None for _ in payload.recent_draws]
    )
    result["auto_fetched"] = auto_fetched
    result["fetch_attempts"] = fetch_attempts
    return result
