from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json
import logging
import time
from typing import List

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.analysis.engine import run_analysis_engine  # noqa: E402
from src.analysis.snapshots import (  # noqa: E402
    SNAPSHOT_CSV,
    SNAPSHOT_META,
    SNAPSHOT_PARQUET,
)
from src.fetchers.auzo_bingo import (  # noqa: E402
    BingoDrawFetcher,
    FetchDrawsError,
    build_recent_draws,
)
from src.fetchers.source_consensus import build_fetch_health_report  # noqa: E402
from src.integrations.auzo import AuzoConfig, fetch_auzo_external_analysis  # noqa: E402
from src.io.canonical_dataset import (  # noqa: E402
    CANONICAL_CSV,
    CANONICAL_PARQUET,
    load_canonical_with_diagnostics,
)
from src.io.raw_resolver import load_or_build_manifest  # noqa: E402
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
MODEL_LOAD_START = time.perf_counter()
LOGGER.info("model_load_start")
try:
    PREDICTOR = Predictor.load()
    LOGGER.info(
        "model_load_done elapsed_ms=%s",
        int((time.perf_counter() - MODEL_LOAD_START) * 1000),
    )
except Exception as exc:  # noqa: BLE001
    PREDICTOR = None
    MODEL_LOAD_ERROR = str(exc)
    LOGGER.exception(
        "model_load_failed elapsed_ms=%s",
        int((time.perf_counter() - MODEL_LOAD_START) * 1000),
    )


CANONICAL_CACHE: pd.DataFrame | None = None
CANONICAL_SOURCE: str | None = None


def _get_manifest_snapshot() -> dict:
    manifest = load_or_build_manifest()
    entries = manifest.get("entries", [])
    detected_files = [
        {
            "file": str(
                item.get("original_filename") or item.get("normalized_filename")
            ),
            "row_count": int(item.get("row_count", 0) or 0),
            "year_start": item.get("year_start"),
            "year_end": item.get("year_end"),
        }
        for item in entries
    ]
    return {
        "missing_years": manifest.get("missing_years", []),
        "file_count": int(manifest.get("file_count", 0) or 0),
        "total_rows": int(manifest.get("total_rows", 0) or 0),
        "coverage_year_start": manifest.get("coverage_year_start"),
        "coverage_year_end": manifest.get("coverage_year_end"),
        "detected_files": detected_files,
    }


def _get_canonical_cached(required: bool = True) -> pd.DataFrame:
    global CANONICAL_CACHE, CANONICAL_SOURCE
    if CANONICAL_CACHE is None:
        LOGGER.info("canonical_load_start")
        try:
            df, diag = load_canonical_with_diagnostics()
        except FileNotFoundError:
            if required:
                raise
            LOGGER.warning("canonical_missing_using_empty_cache")
            CANONICAL_CACHE = pd.DataFrame()
            return CANONICAL_CACHE.copy()
        CANONICAL_CACHE = df
        CANONICAL_SOURCE = str(diag["source"])
        LOGGER.info(
            "canonical_load_done rows=%s elapsed_ms=%s source=%s",
            diag["rows"],
            diag["elapsed_ms"],
            diag["source"],
        )
    return CANONICAL_CACHE.copy()


def _persist_canonical_cache(df: pd.DataFrame) -> None:
    global CANONICAL_CACHE
    CANONICAL_CACHE = df.copy()
    if CANONICAL_PARQUET.exists():
        df.to_parquet(CANONICAL_PARQUET, index=False)
    else:
        df.to_csv(CANONICAL_CSV, index=False)


def _issue_records(df: pd.DataFrame, limit: int) -> list[dict]:
    if "issue" not in df.columns:
        return []
    return [{"issue": int(x)} for x in df["issue"].tail(limit).tolist()]


@app.on_event("startup")
def preload_canonical() -> None:
    try:
        _get_canonical_cached(required=True)
    except FileNotFoundError:
        LOGGER.warning("canonical_preload_skipped_missing_dataset")


class PredictPayload(BaseModel):
    recent_draws: List[List[int]] | None = Field(
        default=None,
        description=(
            "optional; when provided, must contain min~max draws with exactly 20 "
            "unique numbers per draw "
            "between 1 and 80"
        ),
    )
    include_stage_details: bool | None = Field(
        default=None,
        description="when true and cascade pipeline is active, include stage debug payload",
    )
    include_external_analysis: bool = Field(
        default=False,
        description="when true, fetch auzo external analysis as auxiliary non-blocking payload",
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


@app.post("/fetch/history-backfill")
def fetch_history_backfill() -> dict:
    return {
        "status": "disabled",
        "message": (
            "history backfill is disabled in API runtime; "
            "run `python scripts/backfill_history.py` offline"
        ),
    }


@app.post("/fetch/latest")
def fetch_latest() -> dict:
    start = time.perf_counter()
    LOGGER.info("fetch_latest_start")
    manifest = _get_manifest_snapshot()
    canonical = _get_canonical_cached(required=True)
    latest_before = int(canonical["issue"].max()) if not canonical.empty else None

    fetcher = BingoDrawFetcher(
        sources=FETCH_SOURCES,
        timeout=FETCH_TIMEOUT,
        retries=FETCH_RETRIES,
        retry_backoff_seconds=FETCH_BACKOFF_SECONDS,
    )
    recent_draws, fetched_records, data_source, attempts = build_recent_draws(
        fetcher=fetcher,
        min_draws=MIN_RECENT_DRAWS,
        max_draws=MAX_RECENT_DRAWS,
    )
    _ = recent_draws

    incoming = pd.DataFrame(
        [
            {
                "issue": int(r.issue),
                "draw_date": str(r.draw_time or ""),
                "numbers": json.dumps(sorted(r.numbers), ensure_ascii=False),
                "numbers_draw_order": json.dumps(list(r.numbers), ensure_ascii=False),
                "draw_time": r.draw_time,
                "consecutive_count": r.streak_count,
                "size": r.size_label or r.big_small,
                "odd_even": r.odd_even_label or r.odd_even,
                "source": data_source,
                "source_priority": 4,
            }
            for r in fetched_records
        ]
    )
    if incoming.empty:
        LOGGER.info(
            "fetch_latest_done incremental_rows_added=0 elapsed_ms=%s canonical_rows=%s source_files=%s",
            int((time.perf_counter() - start) * 1000),
            int(len(canonical)),
            manifest.get("file_count", 0),
        )
        return {
            "status": "ok",
            "incremental_rows_added": 0,
            "latest_issue_before_fetch": latest_before,
        }

    if latest_before is not None:
        incoming = incoming[incoming["issue"] > latest_before].copy()
    merged = pd.concat([canonical, incoming], ignore_index=True)
    merged = merged.drop_duplicates(subset=["issue"], keep="first").sort_values("issue")
    _persist_canonical_cache(merged)

    health = build_fetch_health_report(
        {
            "canonical": _issue_records(canonical, 500),
            data_source: [{"issue": int(x)} for x in incoming["issue"].tolist()],
        }
    )
    LOGGER.info(
        "fetch_latest_done incremental_rows_added=%s elapsed_ms=%s canonical_rows=%s source_files=%s",
        int(len(incoming)),
        int((time.perf_counter() - start) * 1000),
        int(len(merged)),
        manifest.get("file_count", 0),
    )
    return {
        "status": "ok",
        "latest_issue_before_fetch": latest_before,
        "latest_issue_after_fetch": int(merged["issue"].max()),
        "incremental_rows_added": int(len(incoming)),
        "missing_years": manifest.get("missing_years", []),
        "fetch_attempts": [
            {"source": a.source, "ok": a.ok, "error": a.error} for a in attempts
        ],
        "source_consensus_status": health.get("source_consensus_status"),
    }


@app.post("/fetch/consensus-check")
def fetch_consensus_check() -> dict:
    start = time.perf_counter()
    canonical = _get_canonical_cached(required=False)
    report = build_fetch_health_report({"canonical": _issue_records(canonical, 2000)})
    LOGGER.info(
        "fetch_consensus_check_done elapsed_ms=%s canonical_rows=%s",
        int((time.perf_counter() - start) * 1000),
        int(len(canonical)),
    )
    return report


@app.post("/features/rebuild")
def features_rebuild() -> dict:
    from src.build_features import main as build_main

    build_main()
    return {"status": "ok"}


@app.post("/backtest/run")
def backtest_run() -> dict:
    from src.backtest import main as backtest_main

    backtest_main()
    return {"status": "ok"}


@app.get("/reports/source-consensus")
def report_source_consensus() -> dict:
    path = PROJECT_ROOT / "reports" / "source_consensus_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="source consensus report not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/reports/history-ablation")
def report_history_ablation() -> dict:
    path = PROJECT_ROOT / "reports" / "history_ablation_summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="history ablation report not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/analysis/full")
def analysis_full(payload: PredictPayload) -> dict:
    if payload.recent_draws is None:
        raise HTTPException(
            status_code=400, detail="recent_draws is required for analysis/full"
        )
    _validate_recent_draws(payload.recent_draws)
    return run_analysis_engine(payload.recent_draws)


@app.get("/analysis/external/auzo")
def analysis_external_auzo() -> dict:
    return fetch_auzo_external_analysis(AuzoConfig(timeout_seconds=2.0, ttl_seconds=60))


@app.post("/predict")
def predict(payload: PredictPayload) -> dict:
    start = time.perf_counter()
    LOGGER.info("predict_start")
    if PREDICTOR is None:
        if MODEL_LOAD_ERROR:
            return {"error": f"model unavailable: {MODEL_LOAD_ERROR}"}
        return {"error": "model not found, please train first"}
    auto_fetched = payload.recent_draws is None
    data_source = "manual"
    records: list[dict] = []
    response_records: list[dict] = []
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
        response_records = [
            {
                "issue": record.issue,
                "draw_date": record.draw_time,
                "numbers": list(record.numbers),
                "size_label": getattr(record, "size_label", None)
                or getattr(record, "big_small", None),
                "odd_even_label": getattr(record, "odd_even_label", None)
                or getattr(record, "odd_even", None),
                "streak_count": getattr(record, "streak_count", None),
            }
            for record in fetched_records
        ]
        records = [
            {
                "issue": item["issue"],
                "draw_date": item["draw_date"],
                "numbers": json.dumps(item["numbers"], ensure_ascii=False),
                "size_label": item["size_label"],
                "odd_even_label": item["odd_even_label"],
                "streak_count": item["streak_count"],
            }
            for item in response_records
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

    include_stage_details = payload.include_stage_details
    if include_stage_details is None:
        include_stage_details = bool(
            PREDICT_CFG.get("pipeline", {}).get("include_stage_details_default", False)
        )
    if not include_stage_details:
        result.pop("cascade_debug", None)

    result["analysis_report"] = build_recent_report(payload.recent_draws)
    result["analysis_engine"] = run_analysis_engine(payload.recent_draws)
    result["model_version"] = METADATA.get("model_type", "unknown")
    result["feature_version"] = "v3_core20"
    result["training_data_snapshot"] = {
        "train_issue_start": METADATA.get("train_issue_start"),
        "train_issue_end": METADATA.get("train_issue_end"),
        "feature_rows": METADATA.get("feature_rows"),
    }
    result["calibration_method"] = METADATA.get("calibration_method", "none")
    issues_used = [record["issue"] for record in records]
    canonical = _get_canonical_cached(required=False)
    manifest = _get_manifest_snapshot()
    latest_before = int(canonical["issue"].max()) if not canonical.empty else None
    latest_after = latest_before
    incremental_rows_added = 0
    source_consensus_status = "not_checked"

    if auto_fetched and response_records:
        incoming_df = pd.DataFrame(
            {
                "issue": [int(x["issue"]) for x in response_records],
                "draw_date": [str(x.get("draw_date") or "") for x in response_records],
                "numbers": [
                    json.dumps(sorted(x["numbers"]), ensure_ascii=False)
                    for x in response_records
                ],
                "numbers_draw_order": [
                    json.dumps(x["numbers"], ensure_ascii=False)
                    for x in response_records
                ],
                "draw_time": [x.get("draw_date") for x in response_records],
                "consecutive_count": [x.get("streak_count") for x in response_records],
                "size": [x.get("size_label") for x in response_records],
                "odd_even": [x.get("odd_even_label") for x in response_records],
                "source": [data_source for _ in response_records],
                "source_priority": [4 for _ in response_records],
            }
        )
        if latest_before is not None:
            incoming_df = incoming_df[incoming_df["issue"] > latest_before].copy()
        incremental_rows_added = int(len(incoming_df))
        if incremental_rows_added > 0:
            merged = pd.concat([canonical, incoming_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["issue"], keep="first").sort_values(
                "issue"
            )
            _persist_canonical_cache(merged)
            canonical = merged
            latest_after = int(merged["issue"].max())
            health = build_fetch_health_report(
                {
                    "canonical": _issue_records(canonical, 500),
                    data_source: [
                        {"issue": int(x)} for x in incoming_df["issue"].tolist()
                    ],
                }
            )
            source_consensus_status = str(
                health.get("source_consensus_status", "unknown")
            )

    canonical_issue_start = (
        int(canonical["issue"].min())
        if (not canonical.empty and "issue" in canonical)
        else "unavailable"
    )
    canonical_issue_end = (
        int(canonical["issue"].max())
        if (not canonical.empty and "issue" in canonical)
        else "unavailable"
    )

    result["data_source"] = data_source
    result["recent_draws_count"] = len(payload.recent_draws)
    result["first_issue_used"] = issues_used[0] if auto_fetched else None
    result["last_issue_used"] = issues_used[-1] if auto_fetched else None
    result["issues_used"] = (
        issues_used if auto_fetched else [None for _ in payload.recent_draws]
    )
    result["auto_fetched"] = auto_fetched
    result["fetch_attempts"] = fetch_attempts
    result["records"] = response_records if auto_fetched else []
    result["canonical_rows"] = int(len(canonical))
    result["canonical_issue_start"] = canonical_issue_start
    result["canonical_issue_end"] = canonical_issue_end
    result["raw_manifest_file_count"] = manifest.get("file_count", 0)
    result["raw_manifest_total_rows"] = manifest.get("total_rows", 0)
    result["coverage_year_start"] = manifest.get("coverage_year_start", "unavailable")
    result["coverage_year_end"] = manifest.get("coverage_year_end", "unavailable")
    result["detected_files"] = manifest.get("detected_files", [])
    result["training_window_used"] = {
        "max_draws_for_training": load_yaml(CONFIG_DIR / "train.yaml").get(
            "max_draws_for_training", "unavailable"
        ),
        "feature_rows": METADATA.get("feature_rows", "unavailable"),
        "train_issue_start": METADATA.get("train_issue_start", "unavailable"),
        "train_issue_end": METADATA.get("train_issue_end", "unavailable"),
    }

    snapshot_meta = (
        json.loads(SNAPSHOT_META.read_text(encoding="utf-8"))
        if SNAPSHOT_META.exists()
        else {}
    )
    snapshot_path = (
        snapshot_meta.get("paths", {}).get("history_snapshot")
        if snapshot_meta
        else None
    )
    snapshot_ready = bool(SNAPSHOT_PARQUET.exists() or SNAPSHOT_CSV.exists())
    result["history_snapshot"] = {
        "status": "ready" if snapshot_ready else "unavailable",
        "path": snapshot_path
        or (str(SNAPSHOT_PARQUET) if SNAPSHOT_PARQUET.exists() else str(SNAPSHOT_CSV)),
        "meta": snapshot_meta,
    }

    result["fetch_summary"] = {
        "local_history_used": True,
        "canonical_rows": int(len(canonical)),
        "latest_issue_before_fetch": latest_before,
        "latest_issue_after_fetch": latest_after,
        "incremental_rows_added": incremental_rows_added,
        "missing_years": manifest.get("missing_years", []),
        "missing_issues_detected": [],
        "source_consensus_status": source_consensus_status,
    }
    result["source_consensus"] = {
        "status": source_consensus_status,
        "report_path": "reports/source_consensus_report.json",
    }
    strategy_obj = getattr(PREDICTOR, "strategy", None)
    result["pipeline_version"] = getattr(strategy_obj, "pipeline_version", "unknown")
    result["strategy_version"] = getattr(strategy_obj, "version_id", "unknown")
    result["data_window_summary"] = {
        "min_recent_draws": MIN_RECENT_DRAWS,
        "max_recent_draws": MAX_RECENT_DRAWS,
        "used_recent_draws": len(payload.recent_draws),
    }

    if payload.include_external_analysis:
        external = fetch_auzo_external_analysis(
            AuzoConfig(timeout_seconds=2.0, ttl_seconds=60)
        )
    else:
        external = {
            "external_status": "not_requested",
            "provider": "auzo",
            "external_analysis": {},
            "cache_hit": False,
        }
    result.update(external)

    LOGGER.info(
        "predict_done elapsed_ms=%s canonical_rows=%s source_files=%s auto_fetched=%s",
        int((time.perf_counter() - start) * 1000),
        int(len(canonical)),
        manifest.get("file_count", 0),
        auto_fetched,
    )
    return result
