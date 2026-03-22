from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
import threading
from typing import Any
import uuid

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.analysis.snapshots import read_history_snapshot
from src.artifacts import ModelArtifacts, load_artifacts
from src.predict import (
    PROJECT_ROOT,
    PredictionRuntimeState,
    _resolve_runtime_artifact_dir,
    build_prediction_runtime_state,
    normalize_predict_config_paths,
    run_prediction,
)
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
    fetched_same_day_issue_min: str | None = None
    fetched_same_day_issue_max: str | None = None
    fetched_same_day_issue_count: int | None = None
    dynamic_context_n: int | None = None
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
    runtime_history_ready: bool
    retrieval_index_ready: bool
    retrieval_index_version: str | None = None
    recent_cache_status: str | None = None
    recent_cache_updated_at: float | None = None
    fast_path_enabled: bool
    stale_allowed: bool


@lru_cache(maxsize=1)
def get_runtime() -> tuple[ModelArtifacts | None, dict[str, Any], str | None]:
    config = yaml.safe_load((PROJECT_ROOT / "configs/predict.yaml").read_text(encoding="utf-8"))
    config = normalize_predict_config_paths(config)
    models_dir = Path(config.get("models", {}).get("dir", "models"))
    try:
        artifacts = load_artifacts(models_dir)
        return artifacts, config, None
    except Exception as exc:  # noqa: BLE001
        return None, config, str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts, config, err = get_runtime()
    if artifacts is None:
        raise RuntimeError(f"startup fail-fast: artifacts unavailable: {err}")
    app.state.runtime_state = build_prediction_runtime_state(artifacts, config)
    app.state.config = config
    app.state.artifacts = artifacts
    yield


app = FastAPI(title="BingoBingo Ranking API", version="1.3.0", lifespan=lifespan)
_PREDICT_SINGLEFLIGHT_LOCK = threading.Lock()


def _minimal_response(result: dict[str, Any]) -> dict[str, Any]:
    top20 = [int(x) for x in result["top20_numbers"]]
    big = int(result.get("big_count", sum(1 for n in top20 if n >= 41)))
    small = int(result.get("small_count", sum(1 for n in top20 if n <= 40)))
    odd = int(result.get("odd_count", sum(1 for n in top20 if n % 2 == 1)))
    even = int(result.get("even_count", 20 - odd))
    meta = result.get("metadata") or {}
    latest_issue = str(
        meta.get("latest_fetched_issue")
        or meta.get("fetched_same_day_issue_max")
        or (meta.get("runtime_history_issue_range", [None, None])[-1])
        or "unknown"
    )
    return {
        "latest_fetched_issue": latest_issue,
        "fetched_same_day_issue_min": meta.get("fetched_same_day_issue_min"),
        "fetched_same_day_issue_max": meta.get("fetched_same_day_issue_max"),
        "fetched_same_day_issue_count": meta.get("fetched_same_day_issue_count"),
        "dynamic_context_n": meta.get("dynamic_context_n"),
        "target_issue": str(result["issue"]),
        "top20_numbers": top20,
        "big_count": big,
        "small_count": small,
        "odd_count": odd,
        "even_count": even,
        "size_summary": f"大{big} / 小{small}",
        "odd_even_summary": f"單{odd} / 雙{even}",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    artifacts, config, err = get_runtime()
    snap = read_history_snapshot(Path(config.get("snapshot", {}).get("path", "reports/history_snapshot.json")))
    processed_path = Path(config.get("history", {}).get("processed_path", "data/processed/history_processed.csv"))
    has_processed = processed_path.exists() or bool(sorted(processed_path.parent.glob(f"{processed_path.stem}.part*{processed_path.suffix}")))
    runtime_dir = _resolve_runtime_artifact_dir(config)
    runtime_state: PredictionRuntimeState | None = getattr(app.state, "runtime_state", None)
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
        runtime_history_ready=runtime_state is not None,
        retrieval_index_ready=runtime_state is not None,
        retrieval_index_version=(runtime_state.retrieval_index_version if runtime_state else None),
        recent_cache_status=(runtime_state.recent_cache.cache_status if runtime_state else None),
        recent_cache_updated_at=(runtime_state.recent_cache.updated_at_epoch if runtime_state else None),
        fast_path_enabled=bool(config.get("runtime", {}).get("fast_path_enabled", True)),
        stale_allowed=bool(config.get("recent_cache", {}).get("allow_stale", True)),
    )


@app.get("/debug/runtime")
def debug_runtime() -> dict[str, Any]:
    runtime_state: PredictionRuntimeState | None = getattr(app.state, "runtime_state", None)
    if runtime_state is None:
        raise HTTPException(status_code=503, detail="runtime not initialized")
    return {
        "model_loaded": True,
        "model_version": runtime_state.artifacts.metadata.get("model_version") or runtime_state.artifacts.metadata.get("created_at"),
        "runtime_history_ready": True,
        "runtime_history_version": runtime_state.runtime_history_version,
        "retrieval_index_ready": True,
        "retrieval_index_version": runtime_state.retrieval_index_version,
        "recent_cache_status": runtime_state.recent_cache.cache_status,
        "recent_cache_updated_at": runtime_state.recent_cache.updated_at_epoch,
        "recent_last_issue": runtime_state.recent_cache.recent_last_issue,
        "fast_path_enabled": bool(runtime_state.config.get("runtime", {}).get("fast_path_enabled", True)),
        "stale_allowed": bool(runtime_state.config.get("recent_cache", {}).get("allow_stale", True)),
        "latest_runtime_issue_range": [runtime_state.merged_history[0].issue, runtime_state.merged_history[-1].issue],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictPayload) -> PredictResponse:
    request_id = uuid.uuid4().hex[:8]
    artifacts, config, err = get_runtime()
    runtime_state: PredictionRuntimeState | None = getattr(app.state, "runtime_state", None)
    if artifacts is None or runtime_state is None:
        raise HTTPException(status_code=503, detail=f"runtime unavailable: {err}")
    acquired = _PREDICT_SINGLEFLIGHT_LOCK.acquire(blocking=False)
    if not acquired:
        print(f"[req={request_id}] /predict rejected: prediction already running", flush=True)
        raise HTTPException(status_code=429, detail="prediction already running")
    try:
        print(f"[req={request_id}] /predict start", flush=True)
        recent = [r.model_dump() for r in payload.recent_draws] if payload.recent_draws else None
        result = run_prediction(artifacts, config, recent, request_id=request_id, response_mode="minimal", runtime_state=runtime_state)
        response = PredictResponse(**_minimal_response(result))
        print(f"[req={request_id}] /predict done", flush=True)
        return response
    except DataContractError as exc:
        print(f"[req={request_id}] /predict error: {exc}", flush=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"[req={request_id}] /predict error: {exc}", flush=True)
        raise
    finally:
        if acquired:
            _PREDICT_SINGLEFLIGHT_LOCK.release()
