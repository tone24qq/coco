from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    fold_df = pd.DataFrame([{"fold": f.fold_id, **compute_metrics(f.val_scored)} for f in folds])
    Path("reports").mkdir(exist_ok=True)
    fold_df.to_csv("reports/train_experiment_per_fold_metrics.csv", index=False)

    ranker, logistic = fit_models(df, feature_cols)
    scored_all = score_with_models(df, feature_cols, ranker, logistic, weights)
    summary = compute_metrics(scored_all)

    Path("models").mkdir(exist_ok=True)
    ranker.booster_.save_model("models/lightgbm_ranker.txt")
    joblib.dump(logistic, "models/logistic_regression.pkl")
    Path("models/feature_columns.json").write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")

    issue_list = list(dict.fromkeys(df["issue"].tolist()))
    backtest_summary_path = Path("reports/backtest_experiment_summary.json")
    backtest_summary = json.loads(backtest_summary_path.read_text(encoding="utf-8")) if backtest_summary_path.exists() else {}
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
    meta["backtest_summary"] = backtest_summary or meta.get("backtest_summary", {})
    Path("models/metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    experiments = [x.get("name") for x in exp_cfg.get("experiments", [])]
    reg = pd.DataFrame([{"experiment": name, "status": "configured"} for name in experiments])
    reg.to_csv("reports/train_experiment_registry.csv", index=False)


if __name__ == "__main__":
    try:
        main()
    except DataContractError as exc:
        raise SystemExit(f"[fail-fast] {exc}")
