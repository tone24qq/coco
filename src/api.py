from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.artifacts import ModelArtifacts, load_artifacts
from src.predict import run_prediction
from src.utils import DataContractError


class RecentDraw(BaseModel):
    issue: str
    draw_date: str
    numbers: list[int] = Field(..., min_length=20, max_length=20)
    day_issue_index: int = Field(..., ge=1)


class PredictPayload(BaseModel):
    recent_draws: list[RecentDraw] | None = None


class RankingScoreRow(BaseModel):
    number: int
    rank_final: int
    final_score: float
    ranker_score: float
    logistic_score: float
    retrieval_score: float
    history_prior_score: float
    analysis_rerank_score: float
    local_peak_score: float


class PredictResponse(BaseModel):
    issue: str
    source: str
    top20_numbers: list[int]
    top10_numbers: list[int]
    top3_numbers: list[int]
    top3_before_group_dedup: list[int]
    top3_after_group_dedup: list[int]
    ranking_score_table: list[RankingScoreRow]
    metadata: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    feature_count: int
    required_recent_draws_min: int
    source: str


app = FastAPI(title="BingoBingo Ranking API", version="1.0.1")


@lru_cache(maxsize=1)
def get_runtime() -> tuple[ModelArtifacts | None, dict[str, Any], str | None]:
    config = yaml.safe_load(Path("configs/predict.yaml").read_text(encoding="utf-8"))
    models_dir = Path(config.get("models", {}).get("dir", "models"))
    try:
        artifacts = load_artifacts(models_dir)
        return artifacts, config, None
    except Exception as exc:  # noqa: BLE001
        return None, config, str(exc)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    artifacts, config, err = get_runtime()
    return HealthResponse(
        status="ok" if artifacts else f"degraded: {err}",
        model_loaded=artifacts is not None,
        model_version=str((artifacts.metadata.get("created_at") if artifacts else "unavailable")),
        feature_count=(len(artifacts.feature_columns) if artifacts else 0),
        required_recent_draws_min=int(config.get("history", {}).get("lookback_n", 100)),
        source=str(config.get("auto_fetch", {}).get("sources", [config.get("auto_fetch", {}).get("source", "winwin")])[0]),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictPayload) -> PredictResponse:
    artifacts, config, err = get_runtime()
    if artifacts is None:
        raise HTTPException(status_code=503, detail=f"artifacts unavailable: {err}")
    try:
        recent = [r.model_dump() for r in payload.recent_draws] if payload.recent_draws else None
        result = run_prediction(artifacts, config, recent)
        return PredictResponse(**result)
    except DataContractError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
