from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

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
from src.utils import DataContractError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiments", default="configs/experiments.yaml")
    parser.add_argument("--input", default="data/feature_store/ranking_dataset.csv")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    exp_cfg = yaml.safe_load(Path(args.experiments).read_text(encoding="utf-8"))
    weights = RuntimeWeights.from_mapping(config.get("runtime_scoring", {}).get("weights", {}))

    df = load_ranking_dataset(Path(args.input))
    feature_cols = resolve_feature_columns(df)
    n_splits = int(config.get("validation", {}).get("n_splits", 3))
    min_train_issues = int(config.get("validation", {}).get("min_train_issues", 30))

    folds = run_cv(df, feature_cols, weights, n_splits=n_splits, min_train_issues=min_train_issues)
    fold_df = pd.DataFrame([{"fold": f.fold_id, **compute_metrics(f.val_scored)} for f in folds])
    Path("reports").mkdir(exist_ok=True)
    fold_df.to_csv("reports/experiment_per_fold_metrics.csv", index=False)

    ranker, logistic = fit_models(df, feature_cols)
    scored_all = score_with_models(df, feature_cols, ranker, logistic, weights)
    summary = compute_metrics(scored_all)

    Path("models").mkdir(exist_ok=True)
    ranker.booster_.save_model("models/lightgbm_ranker.txt")
    joblib.dump(logistic, "models/logistic_regression.pkl")
    Path("models/feature_columns.json").write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")

    issue_list = list(dict.fromkeys(df["issue"].tolist()))
    meta = metadata_payload(feature_cols, issue_list, {"train": config, "experiments": exp_cfg}, summary)
    Path("models/metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    experiments = [x.get("name") for x in exp_cfg.get("experiments", [])]
    reg = pd.DataFrame([{"experiment": name, "status": "configured"} for name in experiments])
    reg.to_csv("reports/experiment_registry.csv", index=False)


if __name__ == "__main__":
    try:
        main()
    except DataContractError as exc:
        raise SystemExit(f"[fail-fast] {exc}")
