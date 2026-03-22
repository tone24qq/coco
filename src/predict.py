from __future__ import annotations

import argparse
import copy
import json
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import pandas as pd
import yaml

from src.analysis.explain import build_prediction_explain
from src.analysis.snapshots import build_history_snapshot, read_history_snapshot
from src.artifacts import ModelArtifacts, load_artifacts
from src.build_features import build_candidate_rows, build_history_runtime_cache, resolve_dynamic_context
from src.fetch_winwin import AUZO_URL, WINWIN_URL, fetch_authoritative_latest_issue, fetch_latest
from src.fetchers.source_consensus import run_source_consensus
from src.io.canonical_dataset import read_audit_summary
from src.runtime_history import (
    build_runtime_history_artifact,
    load_runtime_history_store,
    resolve_processed_source_files,
    runtime_history_ready,
)
from src.runtime_scoring import DynamicWeightConfig, RuntimeWeights, score_candidates
from src.strategy import apply_top3_group_dedup
from src.utils import DataContractError, DrawRecord, enforce_dir_file_sizes, ensure_numbers, log_progress, parse_date, read_processed


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _abs_path(value: str, base_dir: Path) -> str:
    p = Path(value)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    else:
        p = p.resolve()
    return str(p)


def normalize_predict_config_paths(config: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    root = (base_dir or PROJECT_ROOT).resolve()

    models = cfg.setdefault("models", {})
    if models.get("dir"):
        models["dir"] = _abs_path(str(models["dir"]), root)

    history = cfg.setdefault("history", {})
    history["processed_path"] = _abs_path(str(history.get("processed_path", "data/processed/history_processed.csv")), root)
    if history.get("runtime_artifact_dir"):
        history["runtime_artifact_dir"] = _abs_path(str(history["runtime_artifact_dir"]), root)

    provenance = cfg.setdefault("provenance", {})
    for key, default in [
        ("audit_path", "reports/local_data_audit.json"),
        ("manifest_path", "reports/raw_manifest.json"),
        ("consensus_report_path", "reports/source_consensus_report.json"),
    ]:
        provenance[key] = _abs_path(str(provenance.get(key, default)), root)
    raw_dirs = provenance.get("raw_dirs")
    if isinstance(raw_dirs, list):
        provenance["raw_dirs"] = [_abs_path(str(x), root) for x in raw_dirs]

    snapshot = cfg.setdefault("snapshot", {})
    snapshot["path"] = _abs_path(str(snapshot.get("path", "reports/history_snapshot.json")), root)
    return cfg


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
        timeout_s = float(config.get("auto_fetch", {}).get("fetch_timeout_seconds", 10.0))
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
                timeout_s=timeout_s,
            )
            report.setdefault("failover_reason", None)
            report.setdefault("successful_sources", [])
            latest_day = max(r.draw_date for r in rows)
            today_rows = sorted([r for r in rows if r.draw_date == latest_day], key=lambda r: r.issue)
            if not today_rows:
                raise DataContractError("auto_fetch consensus failed: no same-day rows found")
            _validate_auto_fetch_same_day(today_rows, source="consensus", source_max=report.get("source_same_day_max_issue", {}))
            return _verify_freshness_and_attach_meta(today_rows, "winwin_auto_fetch", report, timeout_s=timeout_s)
        fetched = fetch_latest(sources=sources, timeout_s=timeout_s)
        latest_day = max(r.draw_date for r in fetched.records)
        today_rows = sorted([r for r in fetched.records if r.draw_date == latest_day], key=lambda r: r.issue)
        if not today_rows:
            raise DataContractError("auto_fetch failed: no same-day rows found")
        _validate_auto_fetch_same_day(today_rows, source=fetched.source_url, source_max={fetched.source_url: today_rows[-1].issue})
        return _verify_freshness_and_attach_meta(today_rows, "winwin_auto_fetch", {
            "consensus_status": "single_source",
            "fetch_attempts": fetched.attempts,
            "actual_source_used": fetched.source_url,
            "failover_reason": fetched.failover_reason,
            "source_consensus_report_path": None,
            "source_same_day_max_issue": {fetched.source_url: today_rows[-1].issue},
        }, timeout_s=timeout_s)

    processed = Path(config.get("history", {}).get("processed_path", "data/processed/history_processed.csv"))
    records = read_processed(processed)
    latest_day = max(r.draw_date for r in records)
    today_rows = sorted([r for r in records if r.draw_date == latest_day], key=lambda r: r.issue)
    if not today_rows:
        raise DataContractError("processed_history failed: no latest-day rows found")
    return today_rows, "processed_history", {"consensus_status": "processed_history", "fetch_attempts": 0, "actual_source_used": str(processed)}


def _with_same_day_debug(meta: dict[str, Any], rows: list[DrawRecord]) -> dict[str, Any]:
    out = dict(meta)
    issues = [r.issue for r in rows]
    out["fetched_same_day_issue_min"] = issues[0]
    out["fetched_same_day_issue_max"] = issues[-1]
    out["fetched_same_day_issue_count"] = len(issues)
    out["fetched_same_day_issue_list_tail"] = issues[-10:]
    return out


def _verify_freshness_and_attach_meta(
    rows: list[DrawRecord],
    source: str,
    meta: dict[str, Any],
    *,
    timeout_s: float,
) -> tuple[list[DrawRecord], str, dict[str, Any]]:
    merged_same_day_issue_max = rows[-1].issue
    probe_start = perf_counter()
    authoritative_issue, authoritative_source = fetch_authoritative_latest_issue(timeout_s=timeout_s)
    freshness_probe_ms = (perf_counter() - probe_start) * 1000.0
    out = _with_same_day_debug(meta, rows)
    out["authoritative_latest_issue"] = authoritative_issue
    out["authoritative_source"] = authoritative_source
    out["merged_same_day_issue_max"] = merged_same_day_issue_max
    out["freshness_probe_elapsed_ms"] = freshness_probe_ms

    freshness_mismatch_reason = None
    if str(merged_same_day_issue_max) != str(authoritative_issue):
        freshness_mismatch_reason = (
            f"freshness mismatch(stale): merged max issue={merged_same_day_issue_max}, authoritative latest issue={authoritative_issue}"
        )
        out["freshness_check_passed"] = False
        out["freshness_mismatch_reason"] = freshness_mismatch_reason
        raise DataContractError(freshness_mismatch_reason)

    out["freshness_check_passed"] = True
    out["freshness_mismatch_reason"] = None
    out["verified_latest_fetched_issue"] = authoritative_issue
    return rows, source, out


def _validate_auto_fetch_same_day(rows: list[DrawRecord], source: str, source_max: dict[str, Any] | None = None) -> None:
    if not rows:
        raise DataContractError("auto_fetch same-day rows empty")
    ordered = sorted(rows, key=lambda r: int(r.issue) if r.issue.isdigit() else r.issue)
    suffixes: list[int] = []
    for row in ordered:
        if not row.issue.isdigit() or len(row.issue) < 3:
            raise DataContractError(f"auto_fetch same-day invalid issue format: {row.issue}")
        suffixes.append(int(row.issue[-3:]))
    expected = list(range(min(suffixes), max(suffixes) + 1))
    if suffixes != expected:
        raise DataContractError(
            f"auto_fetch same-day incomplete issue set from {source}: expected {expected[0]}..{expected[-1]}, got tail={suffixes[-10:]}"
        )
    expected_day_idx = list(range(1, len(ordered) + 1))
    got_day_idx = [r.day_issue_index for r in ordered]
    if got_day_idx != expected_day_idx:
        raise DataContractError("auto_fetch same-day day_issue_index must be 1..N contiguous")
    if source_max:
        max_issue = ordered[-1].issue
        has_stale_source = any(v is not None and str(v).isdigit() and int(str(v)) < int(max_issue) for v in source_max.values())
        if has_stale_source and len(source_max) == 1:
            raise DataContractError("auto_fetch single source appears stale for latest day; cannot safely continue")


def _load_runtime_history(config: dict[str, Any]) -> Sequence[DrawRecord]:
    processed_path = Path(config.get("history", {}).get("processed_path", "data/processed/history_processed.csv"))
    runtime_dir = _resolve_runtime_artifact_dir(config)
    try:
        store = _cached_runtime_history_store(str(processed_path), str(runtime_dir))
    except DataContractError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataContractError("processed history missing; build processed history before deploy") from exc

    if len(store) == 0:
        raise DataContractError("processed history missing; build processed history before deploy")
    return store


def _resolve_runtime_artifact_dir(config: dict[str, Any]) -> Path:
    history_cfg = config.get("history", {})
    explicit = history_cfg.get("runtime_artifact_dir")
    if explicit:
        return Path(explicit)
    processed_path = Path(history_cfg.get("processed_path", "data/processed/history_processed.csv"))
    if processed_path.parent.name == "processed":
        return processed_path.parent.parent / "runtime_history"
    return processed_path.parent / "runtime_history"


@lru_cache(maxsize=1)
def _cached_runtime_history_store(processed_path: str, runtime_dir: str):
    artifact_dir = Path(runtime_dir)
    if runtime_history_ready(artifact_dir):
        return load_runtime_history_store(artifact_dir)

    source = Path(processed_path)
    if not source.exists() and not sorted(source.parent.glob(f"{source.stem}.part*{source.suffix}")):
        raise DataContractError("runtime history artifact missing and processed history missing; cannot rebuild")
    resolve_processed_source_files(source)
    if not runtime_history_ready(artifact_dir):
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
    request_id: str | None = None,
    response_mode: str = "full",
) -> dict[str, Any]:
    if response_mode not in {"full", "minimal"}:
        raise DataContractError(f"unsupported response_mode: {response_mode}")

    t0 = perf_counter()
    tm: dict[str, float] = {}
    log_progress(0, 6, "收到預測請求，開始載入 recent_draws / auto_fetch", request_id=request_id)
    recent_context, source, fetch_meta = _load_recent_draws(config, recent_draws)
    tm["fetch"] = (perf_counter() - t0) * 1000.0
    tm["freshness_probe"] = float(fetch_meta.get("freshness_probe_elapsed_ms", 0.0))
    log_progress(1, 6, "載入最近開獎上下文", f"來源={source}", request_id=request_id)
    log_progress(2, 6, "開始載入 runtime history", request_id=request_id)
    t_merge = perf_counter()
    processed_history = _load_runtime_history(config)
    history = _merge_history_with_context(processed_history, recent_context)
    tm["merge"] = (perf_counter() - t_merge) * 1000.0
    log_progress(3, 6, "合併 processed + recent 歷史", f"rows={len(history)}", request_id=request_id)

    history_cfg = config.get("history", {})
    min_dynamic_n = int(history_cfg.get("min_dynamic_n", 20))
    max_dynamic_n = int(history_cfg.get("max_dynamic_n", 999))
    prefer_same_day_progress = bool(history_cfg.get("prefer_same_day_progress", True))

    context = resolve_dynamic_context(history, min_dynamic_n=min_dynamic_n, max_dynamic_n=max_dynamic_n)
    dynamic_n = len(context)

    verified_latest_fetched_issue = str(fetch_meta.get("verified_latest_fetched_issue") or recent_context[-1].issue)
    target_issue = _next_issue(verified_latest_fetched_issue)
    retrieval_start = perf_counter()
    runtime_cache = build_history_runtime_cache(list(history))
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
        progress_logging=True,
        runtime_cache=runtime_cache,
    )
    tm["retrieval_plus_feature_build"] = (perf_counter() - retrieval_start) * 1000.0
    feat_df = pd.DataFrame(rows)
    if len(feat_df) != 80:
        raise DataContractError("prediction contract violated: expected 80 candidates")

    _validate_feature_contract(feat_df, artifacts)
    log_progress(4, 6, "feature contract passed", f"columns={len(artifacts.feature_columns)}", request_id=request_id)

    x = feat_df[artifacts.feature_columns].fillna(0.0)
    t_score = perf_counter()
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
    tm["model_score"] = (perf_counter() - t_score) * 1000.0
    log_progress(4, 6, "完成 ranking score chain", f"target_issue={target_issue}", request_id=request_id)

    top20 = scored.head(20)["candidate_number"].astype(int).tolist()
    big = sum(1 for n in top20 if n >= 41)
    small = sum(1 for n in top20 if n <= 40)
    odd = sum(1 for n in top20 if n % 2 == 1)
    even = 20 - odd

    if response_mode == "minimal":
        log_progress(5, 6, "輸出 minimal 預測回應", f"top20={top20[:5]}...", request_id=request_id)
        log_progress(6, 6, "預測主線完成", f"issue={target_issue}", request_id=request_id)
        return {
            "issue": target_issue,
            "top20_numbers": top20,
            "big_count": big,
            "small_count": small,
            "odd_count": odd,
            "even_count": even,
            "metadata": {
                "latest_fetched_issue": verified_latest_fetched_issue,
                "fetched_same_day_issue_min": fetch_meta.get("fetched_same_day_issue_min"),
                "fetched_same_day_issue_max": fetch_meta.get("fetched_same_day_issue_max"),
                "fetched_same_day_issue_count": fetch_meta.get("fetched_same_day_issue_count"),
                "dynamic_context_n": dynamic_n,
            },
        }

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
        "fetched_same_day_issue_min": fetch_meta.get("fetched_same_day_issue_min"),
        "fetched_same_day_issue_max": fetch_meta.get("fetched_same_day_issue_max"),
        "fetched_same_day_issue_count": fetch_meta.get("fetched_same_day_issue_count"),
        "fetched_same_day_issue_list_tail": fetch_meta.get("fetched_same_day_issue_list_tail", []),
        "source_same_day_max_issue": fetch_meta.get("source_same_day_max_issue", {}),
        "authoritative_latest_issue": fetch_meta.get("authoritative_latest_issue"),
        "authoritative_source": fetch_meta.get("authoritative_source"),
        "verified_latest_fetched_issue": fetch_meta.get("verified_latest_fetched_issue"),
        "merged_same_day_issue_max": fetch_meta.get("merged_same_day_issue_max"),
        "freshness_check_passed": fetch_meta.get("freshness_check_passed"),
        "freshness_mismatch_reason": fetch_meta.get("freshness_mismatch_reason"),
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
        "latest_fetched_issue": verified_latest_fetched_issue,
        "elapsed_ms": {
            "fetch": round(tm.get("fetch", 0.0), 3),
            "freshness_probe": round(tm.get("freshness_probe", 0.0), 3),
            "merge": round(tm.get("merge", 0.0), 3),
            "retrieval_feature_build": round(tm.get("retrieval_plus_feature_build", 0.0), 3),
            "model_score": round(tm.get("model_score", 0.0), 3),
            "total": round((perf_counter() - t0) * 1000.0, 3),
        },
    }
    if dynamic_cfg.enabled and "effective_runtime_weights" not in metadata:
        raise DataContractError("dynamic enabled but effective_runtime_weights missing in metadata")
    log_progress(5, 6, "組裝預測輸出", f"top3={top3_after}", request_id=request_id)

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
    log_progress(6, 6, "預測主線完成", f"issue={target_issue}", request_id=request_id)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/predict.yaml"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reports/latest_prediction.json"))
    parser.add_argument("--recent-json", default="")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = normalize_predict_config_paths(config)
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
