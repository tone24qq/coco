from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from src.analysis.snapshots import build_history_snapshot
from src.io.canonical_dataset import build_canonical_audit, read_audit_summary
from src.modeling import (
    compute_metrics,
    fit_models,
    load_ranking_dataset,
    metadata_payload,
    resolve_feature_columns,
    run_cv,
    score_with_models,
)
from src.runtime_scoring import RuntimeWeights
from src.utils import DataContractError, read_processed


def _recompose_final(table: pd.DataFrame, weights: RuntimeWeights) -> pd.DataFrame:
    out = table.copy()
    out["final_score"] = (
        weights.ranker * out["ranker_score"]
        + weights.logistic * out["logistic_score"]
        + weights.retrieval * out["retrieval_score"]
        + weights.history_prior * out["history_prior_score"]
        + weights.analysis * out["analysis_rerank_score"]
        + weights.local_peak * out["local_peak_score"]
    )
    out = out.sort_values(["issue", "final_score"], ascending=[True, False]).reset_index(drop=True)
    out["rank_final"] = out.groupby("issue").cumcount() + 1
    return out


def _score_fixed_window_baseline(val_scored: pd.DataFrame) -> pd.DataFrame:
    out = val_scored.copy()
    out["final_score"] = 0.7 * out["cand_hits_last_100"].astype(float) + 0.3 * out["cand_hits_last_20"].astype(float)
    out = out.sort_values(["issue", "final_score"], ascending=[True, False]).reset_index(drop=True)
    out["rank_final"] = out.groupby("issue").cumcount() + 1
    return out


def _weights_for_experiment(name: str, base_weights: RuntimeWeights) -> RuntimeWeights:
    if name == "ablation_no_retrieval":
        return RuntimeWeights(
            ranker=base_weights.ranker,
            logistic=base_weights.logistic,
            retrieval=0.0,
            history_prior=base_weights.history_prior,
            analysis=base_weights.analysis,
            local_peak=base_weights.local_peak,
        )
    if name == "ablation_no_logistic":
        return RuntimeWeights(
            ranker=base_weights.ranker,
            logistic=0.0,
            retrieval=base_weights.retrieval,
            history_prior=base_weights.history_prior,
            analysis=base_weights.analysis,
            local_peak=base_weights.local_peak,
        )
    return base_weights


def _score_experiment(table: pd.DataFrame, name: str, base_weights: RuntimeWeights) -> pd.DataFrame:
    if name == "baseline_frequency":
        return _score_fixed_window_baseline(table)
    return _recompose_final(table, _weights_for_experiment(name, base_weights))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiments", default="configs/experiments.yaml")
    parser.add_argument("--input", default="data/feature_store/ranking_dataset.csv")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    exp_cfg = yaml.safe_load(Path(args.experiments).read_text(encoding="utf-8"))
    weights = RuntimeWeights.from_mapping(config.get("runtime_scoring", {}).get("weights", {}))

    provenance_cfg = config.get("provenance", {})
    raw_dirs = [Path(x) for x in provenance_cfg.get("raw_dirs", ["data/raw", "raw"])]
    audit_path = Path(provenance_cfg.get("audit_path", "reports/local_data_audit.json"))
    manifest_path = Path(provenance_cfg.get("manifest_path", "reports/raw_manifest.json"))
    audit, raw_records = build_canonical_audit(raw_dirs=raw_dirs, audit_output_path=audit_path, manifest_output_path=manifest_path)

    processed_path = Path(config.get("history", {}).get("processed_path", "data/processed/history_processed.csv"))
    if processed_path.exists():
        snapshot_records = read_processed(processed_path)
    else:
        snapshot_records = raw_records
    snapshot_path = Path(config.get("snapshot", {}).get("path", "reports/history_snapshot.json"))
    snapshot = build_history_snapshot(snapshot_records, output_path=snapshot_path) if snapshot_records else {}

    df = load_ranking_dataset(Path(args.input))
    feature_cols = resolve_feature_columns(df)
    n_splits = int(config.get("validation", {}).get("n_splits", 3))
    min_train_issues = int(config.get("validation", {}).get("min_train_issues", 30))

    folds = run_cv(df, feature_cols, weights, n_splits=n_splits, min_train_issues=min_train_issues)
    experiments = [str(x.get("name")) for x in exp_cfg.get("experiments", [])]
    if not experiments:
        experiments = ["baseline_frequency", "ranker_main_qsm", "ablation_no_retrieval", "ablation_no_logistic"]

    train_fold_rows: list[dict[str, Any]] = []
    backtest_fold_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    for exp in experiments:
        train_metrics_by_fold: list[float] = []
        backtest_metrics_by_fold: list[float] = []
        for fold in folds:
            train_scored = _score_experiment(fold.train_scored, exp, weights)
            val_scored = _score_experiment(fold.val_scored, exp, weights)
            train_metric = compute_metrics(train_scored)
            val_metric = compute_metrics(val_scored)
            train_fold_rows.append({"experiment": exp, "fold": fold.fold_id, **train_metric})
            backtest_fold_rows.append({"experiment": exp, "fold": fold.fold_id, **val_metric})
            train_metrics_by_fold.append(float(train_metric["top3_hit_rate"]))
            backtest_metrics_by_fold.append(float(val_metric["top3_hit_rate"]))
        registry_rows.append(
            {
                "experiment": exp,
                "status": "completed",
                "train_top3_hit_rate": float(sum(train_metrics_by_fold) / len(train_metrics_by_fold)),
                "backtest_top3_hit_rate": float(sum(backtest_metrics_by_fold) / len(backtest_metrics_by_fold)),
            }
        )

    fold_df = pd.DataFrame(train_fold_rows)
    Path("reports").mkdir(exist_ok=True)
    fold_df.to_csv("reports/train_experiment_per_fold_metrics.csv", index=False)
    pd.DataFrame(backtest_fold_rows).to_csv("reports/backtest_experiment_per_fold_metrics.csv", index=False)
    pd.DataFrame(registry_rows).to_csv("reports/train_experiment_registry.csv", index=False)

    train_main = [r["train_top3_hit_rate"] for r in registry_rows if r["experiment"] in {"ranker_main_qsm", "dynamic_n_fusion_main"}]
    backtest_main = [r["backtest_top3_hit_rate"] for r in registry_rows if r["experiment"] in {"ranker_main_qsm", "dynamic_n_fusion_main"}]
    if not train_main:
        train_main = [registry_rows[0]["train_top3_hit_rate"]]
    if not backtest_main:
        backtest_main = [registry_rows[0]["backtest_top3_hit_rate"]]
    backtest_summary = {
        "train_top3_hit_rate": float(sum(train_main) / len(train_main)),
        "mainline_top3_hit_rate": float(sum(backtest_main) / len(backtest_main)),
        "train_vs_backtest_gap_top3": float((sum(train_main) / len(train_main)) - (sum(backtest_main) / len(backtest_main))),
        "experiment_count": len(registry_rows),
    }
    Path("reports/backtest_experiment_summary.json").write_text(
        json.dumps(backtest_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ranker, logistic = fit_models(df, feature_cols)
    scored_all = score_with_models(df, feature_cols, ranker, logistic, weights)
    summary = compute_metrics(scored_all)

    Path("models").mkdir(exist_ok=True)
    ranker.booster_.save_model("models/lightgbm_ranker.txt")
    joblib.dump(logistic, "models/logistic_regression.pkl")
    Path("models/feature_columns.json").write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")

    issue_list = list(dict.fromkeys(df["issue"].tolist()))
    meta = metadata_payload(feature_cols, issue_list, {"train": config, "experiments": exp_cfg}, summary)
    meta["canonical_audit_summary"] = audit or read_audit_summary(audit_path)
    meta["history_snapshot_summary"] = {
        "total_history_rows": snapshot.get("total_history_rows"),
        "issue_range": snapshot.get("issue_range"),
        "date_range": snapshot.get("date_range"),
        "coverage_year_start": snapshot.get("coverage_year_start"),
        "coverage_year_end": snapshot.get("coverage_year_end"),
        "detected_files": (audit or {}).get("detected_files", []),
    }
    meta["backtest_summary"] = backtest_summary
    Path("models/metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except DataContractError as exc:
        raise SystemExit(f"[fail-fast] {exc}")
