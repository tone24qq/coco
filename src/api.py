from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json
from typing import List, Optional

import pandas as pd  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.predict import Predictor  # noqa: E402
from src.utils import (  # noqa: E402
    CONFIG_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    load_yaml,
)

app = FastAPI(title="BingoBingo Predictor", version="1.0.0")
PREDICT_CFG = load_yaml(CONFIG_DIR / "predict.yaml")
METADATA = (
    json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    if (MODELS_DIR / "metadata.json").exists()
    else {}
)
PREDICTOR = Predictor.load() if (MODELS_DIR / "lgbm_top20.txt").exists() else None


class PredictPayload(BaseModel):
    recent_draws: Optional[List[List[int]]] = Field(
        default=None, description="latest draws, each contains 20 numbers"
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": PREDICTOR is not None}


@app.get("/analysis")
def analysis() -> dict:
    return {
        "metadata": METADATA,
        "feature_min_history": PREDICT_CFG["feature_min_history"],
    }


@app.post("/predict")
def predict(payload: PredictPayload) -> dict:
    if PREDICTOR is None:
        return {"error": "model not found, please train first"}

    if payload.recent_draws:
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
    else:
        df = (
            pd.read_csv(DATA_PROCESSED_DIR / "bingo_draws.csv")
            .tail(int(PREDICT_CFG.get("recent_draws_limit", 3000)))
            .reset_index(drop=True)
        )
    return PREDICTOR.predict_from_draws(
        df, min_history=int(PREDICT_CFG["feature_min_history"])
    )
