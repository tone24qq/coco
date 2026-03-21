from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.analysis.snapshots import read_history_snapshot
from src.artifacts import ModelArtifacts, load_artifacts
from src.predict import _DEFAULT_PREDICT_CONFIG, _resolve_runtime_artifact_dir, normalize_predict_config_paths, run_prediction
from src.runtime_history import runtime_history_ready
from src.utils import DataContractError


class RecentDraw(BaseModel):
    issue: str
    draw_date: str
    numbers: list[int] = Field(..., min_length=20, max_length=20)
    day_issue_index: int = Field(..., ge=1)


class PredictPayload(BaseModel):
    recent_draws: list[RecentDraw] | None = None


class PredictResponse(BaseModel):
    latest_fetched_issue: str
    target_issue: str
    top20_numbers: list[int]
    big_count: int
    small_count: int
    odd_count: int
    even_count: int
    size_summary: str
    odd_even_summary: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    feature_count: int
    required_recent_draws_min: int
    source: str
    coverage_year_start: int | None = None
    coverage_year_end: int | None = None
    processed_history_exists: bool
    compact_history_ready: bool


app = FastAPI(title="BingoBingo Ranking API", version="1.2.0")


@lru_cache(maxsize=1)
def get_runtime() -> tuple[ModelArtifacts | None, dict[str, Any], str | None]:
    config_path = _DEFAULT_PREDICT_CONFIG
    config = normalize_predict_config_paths(yaml.safe_load(config_path.read_text(encoding="utf-8")), base_dir=config_path.parent.parent)
    models_dir = Path(config.get("models", {}).get("dir", "models"))
    try:
        artifacts = load_artifacts(models_dir)
        return artifacts, config, None
    except Exception as exc:  # noqa: BLE001
        return None, config, str(exc)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    artifacts, config, err = get_runtime()
    snap = read_history_snapshot(Path(config.get("snapshot", {}).get("path", "reports/history_snapshot.json")))
    processed_path = Path(config.get("history", {}).get("processed_path", "data/processed/history_processed.csv"))
    has_processed = processed_path.exists() or bool(sorted(processed_path.parent.glob(f"{processed_path.stem}.part*{processed_path.suffix}")))
    runtime_dir = _resolve_runtime_artifact_dir(config)
    return HealthResponse(
        status="ok" if artifacts else f"degraded: {err}",
        model_loaded=artifacts is not None,
        model_version=str((artifacts.metadata.get("model_version") if artifacts else "unavailable")),
        feature_count=(len(artifacts.feature_columns) if artifacts else 0),
        required_recent_draws_min=int(config.get("history", {}).get("min_dynamic_n", 20)),
        source=str(config.get("auto_fetch", {}).get("sources", [config.get("auto_fetch", {}).get("source", "winwin")])[0]),
        coverage_year_start=snap.get("coverage_year_start"),
        coverage_year_end=snap.get("coverage_year_end"),
        processed_history_exists=has_processed,
        compact_history_ready=runtime_history_ready(runtime_dir),
    )


def _minimal_response(result: dict[str, Any]) -> dict[str, Any]:
    top20 = [int(x) for x in result["top20_numbers"]]
    big = sum(1 for n in top20 if n >= 41)
    small = sum(1 for n in top20 if n <= 40)
    odd = sum(1 for n in top20 if n % 2 == 1)
    even = 20 - odd
    latest_issue = str((result.get("metadata") or {}).get("runtime_history_issue_range", [None, None])[-1] or "unknown")
    return {
        "latest_fetched_issue": latest_issue,
        "target_issue": str(result["issue"]),
        "top20_numbers": top20,
        "big_count": big,
        "small_count": small,
        "odd_count": odd,
        "even_count": even,
        "size_summary": f"大{big} / 小{small}",
        "odd_even_summary": f"單{odd} / 雙{even}",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictPayload) -> PredictResponse:
    artifacts, config, err = get_runtime()
    if artifacts is None:
        raise HTTPException(status_code=503, detail=f"artifacts unavailable: {err}")
    try:
        recent = [r.model_dump() for r in payload.recent_draws] if payload.recent_draws else None
        result = run_prediction(artifacts, config, recent)
        return PredictResponse(**_minimal_response(result))
    except DataContractError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
