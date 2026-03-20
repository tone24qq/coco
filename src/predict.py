from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from src.analysis.explain import build_prediction_explain
from src.analysis.snapshots import build_history_snapshot, read_history_snapshot
from src.artifacts import ModelArtifacts, load_artifacts
from src.build_features import build_candidate_rows, resolve_dynamic_context
from src.fetch_winwin import AUZO_URL, WINWIN_URL, fetch_latest
from src.fetchers.source_consensus import run_source_consensus
from src.io.canonical_dataset import read_audit_summary
from src.runtime_history import (
    artifact_can_serve_missing_source,
    artifact_matches_source,
    build_runtime_history_artifact,
    load_runtime_history_store,
    resolve_processed_source_files,
    runtime_history_ready,
)
from src.runtime_scoring import DynamicWeightConfig, RuntimeWeights, score_candidates
from src.strategy import apply_top3_group_dedup
from src.utils import DataContractError, DrawRecord, enforce_dir_file_sizes, ensure_numbers, log_progress, parse_date, read_processed


def _next_issue(issue: str) -> str:
    if issue.isdigit():
        return str(int(issue) + 1)
    return f"{issue}_next"


def _issue_range(records: list[DrawRecord]) -> list[str]:
    if not records:
        return []
    issues = [r.issue for r in records]
    return [issues[0], issues[-1]]


def _records_from_payload(recent_draws: list[dict[str, Any]]) -> list[DrawRecord]:
    records = []
    for row in recent_draws:
        records.append(
            DrawRecord(
                issue=str(row["issue"]),
                draw_date=parse_date(str(row["draw_date"])),
                numbers=ensure_numbers(row["numbers"]),
                day_issue_index=int(row["day_issue_index"]),
            )
        )
    if not records:
        raise DataContractError("recent_draws is empty")
    issues = [r.issue for r in records]
    if len(set(issues)) != len(issues):
        raise DataContractError("recent_draws issue must be unique")
    if issues != sorted(issues):
        raise DataContractError("recent_draws issue must be sorted")
    return records


def _load_recent_draws(
    config: dict[str, Any], recent_draws: list[dict[str, Any]] | None
) -> tuple[list[DrawRecord], str, dict[str, Any]]:
    if recent_draws:
        rows = _records_from_payload(recent_draws)
        return rows, "manual", {"consensus_status": "manual", "fetch_attempts": 0, "actual_source_used": "manual"}

    if config.get("auto_fetch", {}).get("enabled", True):
        sources = list(config.get("auto_fetch", {}).get("sources") or [WINWIN_URL, AUZO_URL])
        if not sources:
            raise DataContractError("auto_fetch enabled but sources is empty")
        if len(sources) >= 2:
            consensus_cfg = config.get("auto_fetch", {}).get("consensus", {})
            mismatch_policy = str(consensus_cfg.get("on_mismatch", "fail_fast"))
            rows, report = run_source_consensus(
                sources,
                Path(config.get("provenance", {}).get("consensus_report_path", "reports/source_consensus_report.json")),
                mismatch_policy=mismatch_policy,
            )
            report.setdefault("failover_reason", None)
            report.setdefault("successful_sources", [])
            latest_day = max(r.draw_date for r in rows)
            today_rows = sorted([r for r in rows if r.draw_date == latest_day], key=lambda r: r.issue)
            if not today_rows:
                raise DataContractError("auto_fetch consensus failed: no same-day rows found")
            return today_rows, "winwin_auto_fetch", report
        fetched = fetch_latest(sources=sources)
        latest_day = max(r.draw_date for r in fetched.records)
        today_rows = sorted([r for r in fetched.records if r.draw_date == latest_day], key=lambda r: r.issue)
        if not today_rows:
            raise DataContractError("auto_fetch failed: no same-day rows found")
        return today_rows, "winwin_auto_fetch", {
            "consensus_status": "single_source",
            "fetch_attempts": fetched.attempts,
            "actual_source_used": fetched.source_url,
            "failover_reason": fetched.failover_reason,
            "source_consensus_report_path": None,
        }

    processed = Path(config.get("history", {}).get("processed_path", "data/processed/history_processed.csv"))
    records = read_processed(processed)
    latest_day = max(r.draw_date for r in records)
    today_rows = sorted([r for r in records if r.draw_date == latest_day], key=lambda r: r.issue)
    if not today_rows:
        raise DataContractError("processed_history failed: no latest-day rows found")
    return today_rows, "processed_history", {"consensus_status": "processed_history", "fetch_attempts": 0, "actual_source_used": str(processed)}


def _load_runtime_history(config: dict[str, Any]) -> Sequence[DrawRecord]:
    processed_path = Path(config.get("history", {}).get("processed_path", "data/processed/history_processed.csv"))
    runtime_dir = Path(config.get("history", {}).get("runtime_artifact_dir", "data/runtime_history"))
    try:
        store = _cached_runtime_history_store(str(processed_path), str(runtime_dir))
    except DataContractError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataContractError("processed history missing; build processed history before deploy") from exc

    if len(store) == 0:
        raise DataContractError("processed history missing; build processed history before deploy")
    return store


@lru_cache(maxsize=1)
def _cached_runtime_history_store(processed_path: str, runtime_dir: str):
    source = Path(processed_path)
    artifact_dir = Path(runtime_dir)
    artifact_ready = runtime_history_ready(artifact_dir)
    try:
        source_files = resolve_processed_source_files(source)
    except DataContractError:
        if artifact_ready and artifact_can_serve_missing_source(artifact_dir, source):
            return load_runtime_history_store(artifact_dir)
        raise
    if not artifact_ready or not artifact_matches_source(artifact_dir, source_files):
        build_runtime_history_artifact(source, artifact_dir)
    return load_runtime_history_store(artifact_dir)


def _clear_runtime_history_cache() -> None:
    _cached_runtime_history_store.cache_clear()


def _merge_history_with_context(
    processed_history: Sequence[DrawRecord], recent_context_rows: list[DrawRecord]
) -> Sequence[DrawRecord]:
    if not processed_history:
        raise DataContractError("processed history is empty")
    if not recent_context_rows:
        raise DataContractError("recent context is empty")

    recent_by_issue = {r.issue: r for r in recent_context_rows}
    matched_recent: set[str] = set()
    latest_processed_day = processed_history[-1].draw_date

    for base in processed_history:
        row = recent_by_issue.get(base.issue)
        if row is None:
            continue
        if base.draw_date != row.draw_date or base.numbers != row.numbers or base.day_issue_index != row.day_issue_index:
            raise DataContractError(f"history/recent merge mismatch on issue={row.issue}")
        matched_recent.add(base.issue)

    latest_recent_day = max(r.draw_date for r in recent_context_rows)
    if latest_recent_day < latest_processed_day:
        raise DataContractError("recent context date is older than processed history latest date")

    missing_rows = [r for r in recent_context_rows if r.issue not in matched_recent]
    if not missing_rows:
        return processed_history

    merged = list(processed_history) + sorted(missing_rows, key=lambda r: r.issue)
    if len({r.issue for r in merged}) != len(merged):
        raise DataContractError("merged history has duplicated issues")
    return merged


def _validate_feature_contract(feature_df: pd.DataFrame, artifacts: ModelArtifacts) -> None:
    expected = artifacts.feature_columns
    got = [c for c in feature_df.columns if c in expected]
    missing = [c for c in expected if c not in feature_df.columns]
    if missing:
        raise DataContractError(f"feature column mismatch, missing: {missing[:10]}")
    if got != expected:
        raise DataContractError("feature column order mismatch")
    non_numeric = [c for c in expected if not pd.api.types.is_numeric_dtype(feature_df[c])]
    if non_numeric:
        raise DataContractError(f"feature dtype mismatch, non-numeric: {non_numeric[:5]}")


def run_prediction(
    artifacts: ModelArtifacts,
    config: dict[str, Any],
    recent_draws: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    recent_context, source, fetch_meta = _load_recent_draws(config, recent_draws)
    log_progress(1, 5, "載入最近開獎上下文", f"來源={source}")
    processed_history = _load_runtime_history(config)
    history = _merge_history_with_context(processed_history, recent_context)
    log_progress(2, 5, "合併 processed + recent 歷史", f"rows={len(history)}")

    history_cfg = config.get("history", {})
    min_dynamic_n = int(history_cfg.get("min_dynamic_n", 20))
    max_dynamic_n = int(history_cfg.get("max_dynamic_n", 999))
    prefer_same_day_progress = bool(history_cfg.get("prefer_same_day_progress", True))

    context = resolve_dynamic_context(history, min_dynamic_n=min_dynamic_n, max_dynamic_n=max_dynamic_n)
    dynamic_n = len(context)

    target_issue = _next_issue(history[-1].issue)
    rows, matches = build_candidate_rows(
        history=history,
        issue=target_issue,
        draw_date=history[-1].draw_date.isoformat(),
        label_numbers=None,
        min_dynamic_n=min_dynamic_n,
        max_dynamic_n=max_dynamic_n,
        top_k=int(config.get("retrieval", {}).get("top_k", 50)),
        retrieval_weights=config.get("retrieval", {}).get("weights", {}),
        prefer_same_day_progress=prefer_same_day_progress,
    )
    feat_df = pd.DataFrame(rows)
    if len(feat_df) != 80:
        raise DataContractError("prediction contract violated: expected 80 candidates")

    _validate_feature_contract(feat_df, artifacts)

    x = feat_df[artifacts.feature_columns].fillna(0.0)
    ranker_score = artifacts.ranker.predict(x)
    lr_x = x.copy()
    lr_x["ranker_score"] = ranker_score
    logistic_score = artifacts.logistic.predict_proba(lr_x)[:, 1]

    weights = RuntimeWeights.from_mapping(config.get("runtime_scoring", {}).get("weights", {}))
    dynamic_cfg = DynamicWeightConfig.from_mapping(config.get("runtime_scoring", {}).get("dynamic"))
    scored, diagnostics = score_candidates(
        feat_df,
        ranker_score,
        logistic_score,
        weights,
        dynamic_cfg=dynamic_cfg,
        return_diagnostics=True,
    )
    log_progress(3, 5, "完成 ranking score chain", f"target_issue={target_issue}")

    top20 = scored.head(20)["candidate_number"].astype(int).tolist()
    top10 = top20[:10]
    top3_before = top10[:3]
    top3_after = apply_top3_group_dedup(top10)

    table = scored[
        [
            "candidate_number",
            "rank_final",
            "final_score",
            "ranker_score",
            "logistic_score",
            "retrieval_score",
            "history_prior_score",
            "analysis_rerank_score",
            "local_peak_score",
        ]
    ]

    retrieval_top_matches = [
        {
            "end_issue": m.end_issue,
            "similarity": m.similarity,
            "exact_draw_match_count": m.exact_draw_match_count,
            "same_day_progress": m.same_day_progress,
            "next_draw_numbers": list(m.next_draw_numbers),
        }
        for m in matches[: min(10, len(matches))]
    ]

    prov_cfg = config.get("provenance", {})
    audit_path = Path(prov_cfg.get("audit_path", "reports/local_data_audit.json"))
    audit = read_audit_summary(audit_path)
    snapshot_path = Path(config.get("snapshot", {}).get("path", "reports/history_snapshot.json"))
    snapshot = read_history_snapshot(snapshot_path)
    if not snapshot and history:
        snapshot = build_history_snapshot(history, output_path=snapshot_path)

    explain = build_prediction_explain(context, top20, top10, top3_after, matches)

    metadata = {
        "model_family": artifacts.metadata.get("model_family", "unknown"),
        "model_version": artifacts.metadata.get("created_at", "unknown"),
        "feature_count": len(artifacts.feature_columns),
        "score_type": "ranking_score",
        "auxiliary_score": "logistic_score",
        "data_source": source,
        "recent_draws_count": len(history),
        "runtime_history_rows": len(history),
        "runtime_recent_context_rows": len(recent_context),
        "dynamic_context_n": dynamic_n,
        "training_window_used": f"dynamic_n={dynamic_n}",
        "runtime_history_issue_range": _issue_range(history),
        "history_snapshot_summary": {
            "total_history_rows": snapshot.get("total_history_rows"),
            "issue_range": snapshot.get("issue_range"),
            "date_range": snapshot.get("date_range"),
        },
        "history_snapshot": snapshot,
        "coverage_year_start": snapshot.get("coverage_year_start", audit.get("coverage_year_start")),
        "coverage_year_end": snapshot.get("coverage_year_end", audit.get("coverage_year_end")),
        "detected_files": audit.get("detected_files", []),
        "canonical_rows": audit.get("canonical_rows", audit.get("total_rows")),
        "source_consensus_status": fetch_meta.get("consensus_status"),
        "source_consensus_report_path": config.get("provenance", {}).get("consensus_report_path", "reports/source_consensus_report.json"),
        "fetch_attempts": fetch_meta.get("fetch_attempts", 0),
        "actual_source_used": fetch_meta.get("actual_source_used", source),
        "failover_reason": fetch_meta.get("failover_reason"),
        "target_next_issue_contract": "passed",
        "predict_explain": explain,
        "effective_runtime_weights": (diagnostics.get("issues", {}).get(str(target_issue), {}) or {}).get(
            "effective_weights",
            {
                "ranker": weights.ranker,
                "logistic": weights.logistic,
                "retrieval": weights.retrieval,
                "history_prior": weights.history_prior,
                "analysis": weights.analysis,
                "local_peak": weights.local_peak,
            },
        ),
        "dynamic_weighting": {
            "enabled": dynamic_cfg.enabled,
            "mode": dynamic_cfg.mode,
            "gate_value": (diagnostics.get("issues", {}).get(str(target_issue), {}) or {}).get("gate_value", 0.0),
            "source": "retrieval_quality_gate",
        },
    }
    if dynamic_cfg.enabled and "effective_runtime_weights" not in metadata:
        raise DataContractError("dynamic enabled but effective_runtime_weights missing in metadata")
    log_progress(4, 5, "組裝預測輸出", f"top3={top3_after}")

    out = {
        "issue": target_issue,
        "source": source,
        "dynamic_context_n": dynamic_n,
        "top20_numbers": top20,
        "top10_numbers": top10,
        "top3_numbers": top3_after,
        "top3_before_group_dedup": top3_before,
        "top3_after_group_dedup": top3_after,
        "retrieval_top_matches": retrieval_top_matches,
        "ranking_score_table": table.to_dict(orient="records"),
        "metadata": metadata,
    }
    log_progress(5, 5, "預測主線完成", f"issue={target_issue}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/predict.yaml")
    parser.add_argument("--output", default="reports/latest_prediction.json")
    parser.add_argument("--recent-json", default="")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    artifacts = load_artifacts(Path(config.get("models", {}).get("dir", "models")))

    recent_draws = None
    if args.recent_json:
        recent_draws = json.loads(Path(args.recent_json).read_text(encoding="utf-8"))

    result = run_prediction(artifacts, config, recent_draws)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    enforce_dir_file_sizes([Path("reports"), Path("models"), Path("data/feature_store")])


if __name__ == "__main__":
    try:
        main()
    except DataContractError as exc:
        raise SystemExit(f"[fail-fast] {exc}")
