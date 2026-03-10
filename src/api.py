from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json
from typing import List

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.predict import Predictor  # noqa: E402
from src.utils import (  # noqa: E402
    CONFIG_DIR,
    MODELS_DIR,
    build_recent_report,
    load_yaml,
)

app = FastAPI(title="BingoBingo Predictor", version="1.0.0")
PREDICT_CFG = load_yaml(CONFIG_DIR / "predict.yaml")
MIN_RECENT_DRAWS = int(PREDICT_CFG.get("min_recent_draws", 22))
MAX_RECENT_DRAWS = int(PREDICT_CFG.get("max_recent_draws", 50))
METADATA = (
    json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    if (MODELS_DIR / "metadata.json").exists()
    else {}
)
PREDICTOR = Predictor.load() if (MODELS_DIR / "catboost_top20.cbm").exists() else None


class PredictPayload(BaseModel):
    recent_draws: List[List[int]] | None = Field(
        default=None,
        description=(
            "required: 22-50 draws, each contains exactly 20 unique numbers "
            "between 1 and 80"
        ),
    )


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
    return {"status": "ok", "model_loaded": PREDICTOR is not None}


@app.get("/analysis")
def analysis() -> dict:
    return {
        "metadata": METADATA,
        "feature_min_history": PREDICT_CFG["feature_min_history"],
        "recent_draws_rules": {
            "min": MIN_RECENT_DRAWS,
            "max": MAX_RECENT_DRAWS,
            "numbers_per_draw": 20,
            "number_range": [1, 80],
            "required": True,
        },
    }


@app.post("/predict")
def predict(payload: PredictPayload) -> dict:
    if PREDICTOR is None:
        return {"error": "model not found, please train first"}
    if payload.recent_draws is None:
        raise HTTPException(
            status_code=400,
            detail="請先提供最新 22–50 期資料（每期20顆），才可進行下一期預測。",
        )
    _validate_recent_draws(payload.recent_draws)

    draws = []
    start_issue = 900000000
    for i, nums in enumerate(payload.recent_draws):
        draws.append(
            {
                "issue": start_issue + i,
                "draw_date": f"2026-01-{(i % 28) + 1:02d}",
                "numbers": json.dumps(sorted(nums), ensure_ascii=False),
            }
        )
    df = pd.DataFrame(draws)

    try:
        result = PREDICTOR.predict_from_draws(
            df, min_history=int(PREDICT_CFG["feature_min_history"])
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result["analysis_report"] = build_recent_report(payload.recent_draws)
    result["model_version"] = METADATA.get("model_type", "unknown")
    result["feature_version"] = METADATA.get("feature_version", "unknown")
    result["training_data_snapshot"] = {
        "train_issue_start": METADATA.get("train_issue_start"),
        "train_issue_end": METADATA.get("train_issue_end"),
        "feature_rows": METADATA.get("feature_rows"),
    }
    result["calibration_method"] = METADATA.get("calibration_method", "none")
    return result
