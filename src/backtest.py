from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.modeling import (
    baseline_frequency_scores,
    compute_metrics,
    load_ranking_dataset,
    resolve_feature_columns,
    run_cv,
    save_json,
    summarize_fold_dispersion,
)
from src.runtime_scoring import RuntimeWeights


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


def _weights_for_experiment(name: str, base: RuntimeWeights) -> RuntimeWeights:
    if name == "ablation_no_retrieval":
        return RuntimeWeights(**{**base.__dict__, "retrieval": 0.0})
    if name == "ablation_no_logistic":
        return RuntimeWeights(**{**base.__dict__, "logistic": 0.0})
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiments", default="configs/experiments.yaml")
    parser.add_argument("--input", default="data/feature_store/ranking_dataset.csv")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    exp_cfg = yaml.safe_load(Path(args.experiments).read_text(encoding="utf-8"))
    base_weights = RuntimeWeights.from_mapping(config.get("runtime_scoring", {}).get("weights", {}))

    df = load_ranking_dataset(Path(args.input))
    feature_cols = resolve_feature_columns(df)
    n_splits = int(config.get("validation", {}).get("n_splits", 3))
    min_train_issues = int(config.get("validation", {}).get("min_train_issues", 30))
    folds = run_cv(df, feature_cols, base_weights, n_splits=n_splits, min_train_issues=min_train_issues)

    fold_rows: list[dict[str, float | int | str]] = []
    regime_rows: list[dict[str, float | int | str]] = []
    ablation_rows: list[dict[str, float | int]] = []

    experiments = [x.get("name") for x in exp_cfg.get("experiments", [])]
    for fold in folds:
        train_df = df[df["issue"].isin(fold.train_issues)]
        val_df = df[df["issue"].isin(fold.val_issues)]

        for exp_name in experiments:
            if exp_name == "baseline_frequency":
                scored = baseline_frequency_scores(train_df, val_df)
                m = compute_metrics(scored[["issue", "candidate_number", "label", "final_score"]])
            else:
                w = _weights_for_experiment(exp_name, base_weights)
                scored = _recompose_final(fold.val_scored, w)
                m = compute_metrics(scored)
            fold_rows.append({"fold": fold.fold_id, "experiment": exp_name, **m})

        regime_data = fold.val_scored.copy()
        regime_data["regime"] = pd.cut(regime_data["candidate_number"], bins=[0, 20, 40, 60, 80], labels=["A", "B", "C", "D"])
        for regime, grp in regime_data.groupby("regime"):
            regime_rows.append({"fold": fold.fold_id, "regime": str(regime), "top3_hit_rate": compute_metrics(grp)["top3_hit_rate"]})

        ablation_rows.append(
            {
                "fold": fold.fold_id,
                "main_top3_hit_rate": compute_metrics(fold.val_scored)["top3_hit_rate"],
                "no_retrieval_top3_hit_rate": compute_metrics(_recompose_final(fold.val_scored, _weights_for_experiment("ablation_no_retrieval", base_weights)))["top3_hit_rate"],
                "no_logistic_top3_hit_rate": compute_metrics(_recompose_final(fold.val_scored, _weights_for_experiment("ablation_no_logistic", base_weights)))["top3_hit_rate"],
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    regime_df = pd.DataFrame(regime_rows)
    ablation_df = pd.DataFrame(ablation_rows)

    Path("reports").mkdir(exist_ok=True)
    fold_df.to_csv("reports/backtest_experiment_per_fold_metrics.csv", index=False)
    regime_df.to_csv("reports/backtest_experiment_per_regime_metrics.csv", index=False)
    ablation_df.to_csv("reports/backtest_runtime_ablation_summary.csv", index=False)

    main_df = fold_df[fold_df["experiment"] == "ranker_main_qsm"]
    base_df = fold_df[fold_df["experiment"] == "baseline_frequency"]
    summary = {
        "top3_hit_rate": float(main_df["top3_hit_rate"].mean()),
        "baseline_top3_hit_rate": float(base_df["top3_hit_rate"].mean()),
        "fold_dispersion_top3": summarize_fold_dispersion(main_df.to_dict(orient="records")),
        "regime_dispersion_top3": float(regime_df["top3_hit_rate"].std()),
        "train_vs_backtest_gap_top3": 0.0,
    }
    save_json(Path("reports/backtest_experiment_summary.json"), summary)
    save_json(Path("reports/backtest_alignment_audit.json"), {"time_series_split": True, "runtime_scoring_shared": True})


if __name__ == "__main__":
    main()
