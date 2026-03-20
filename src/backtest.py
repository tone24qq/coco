from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.modeling import compute_metrics, load_ranking_dataset, resolve_feature_columns, run_cv, save_json
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


def _score_dynamic_retrieval_only(val_scored: pd.DataFrame) -> pd.DataFrame:
    out = val_scored.copy()
    out["final_score"] = out["retrieval_score"]
    out = out.sort_values(["issue", "final_score"], ascending=[True, False]).reset_index(drop=True)
    out["rank_final"] = out.groupby("issue").cumcount() + 1
    return out


def _score_fixed_window_baseline(val_scored: pd.DataFrame) -> pd.DataFrame:
    out = val_scored.copy()
    # fixed-window style baseline proxy: only fixed lookback prior features, no dynamic retrieval fusion
    out["final_score"] = 0.7 * out["cand_hits_last_100"].astype(float) + 0.3 * out["cand_hits_last_20"].astype(float)
    out = out.sort_values(["issue", "final_score"], ascending=[True, False]).reset_index(drop=True)
    out["rank_final"] = out.groupby("issue").cumcount() + 1
    return out


def _retrieval_hit_stats(scored: pd.DataFrame) -> dict[str, float]:
    topk = scored.sort_values(["issue", "final_score"], ascending=[True, False]).groupby("issue").head(20)
    return {
        "retrieval_topk_hit_rate": float(topk["retrieval_top3_hit_flag"].mean()),
        "exact_window_mean": float(topk["retrieval_exact_window_match_count"].mean()),
        "exact_draw_mean": float(topk["retrieval_exact_draw_match_count_mean"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--input", default="data/feature_store/ranking_dataset.csv")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_weights = RuntimeWeights.from_mapping(config.get("runtime_scoring", {}).get("weights", {}))

    df = load_ranking_dataset(Path(args.input))
    feature_cols = resolve_feature_columns(df)
    n_splits = int(config.get("validation", {}).get("n_splits", 3))
    min_train_issues = int(config.get("validation", {}).get("min_train_issues", 30))
    folds = run_cv(df, feature_cols, base_weights, n_splits=n_splits, min_train_issues=min_train_issues)

    rows: list[dict[str, float | int | str]] = []
    retrieval_rows: list[dict[str, float | int]] = []
    for fold in folds:
        dynamic_fusion = _recompose_final(fold.val_scored, base_weights)
        dynamic_retrieval_only = _score_dynamic_retrieval_only(fold.val_scored)
        fixed_baseline = _score_fixed_window_baseline(fold.val_scored)

        rows.append({"fold": fold.fold_id, "experiment": "fixed_window_baseline", **compute_metrics(fixed_baseline)})
        rows.append({"fold": fold.fold_id, "experiment": "dynamic_n_retrieval", **compute_metrics(dynamic_retrieval_only)})
        rows.append({"fold": fold.fold_id, "experiment": "dynamic_n_fusion_main", **compute_metrics(dynamic_fusion)})

        retrieval_rows.append({"fold": fold.fold_id, **_retrieval_hit_stats(dynamic_fusion)})

    out_df = pd.DataFrame(rows)
    retrieval_df = pd.DataFrame(retrieval_rows)
    Path("reports").mkdir(exist_ok=True)
    out_df.to_csv("reports/backtest_experiment_per_fold_metrics.csv", index=False)
    retrieval_df.to_csv("reports/backtest_retrieval_hit_stats.csv", index=False)

    main_df = out_df[out_df["experiment"] == "dynamic_n_fusion_main"]
    fixed_df = out_df[out_df["experiment"] == "fixed_window_baseline"]
    dyn_df = out_df[out_df["experiment"] == "dynamic_n_retrieval"]
    summary = {
        "dynamic_n_fusion_top3_hit_rate": float(main_df["top3_hit_rate"].mean()),
        "fixed_baseline_top3_hit_rate": float(fixed_df["top3_hit_rate"].mean()),
        "dynamic_retrieval_top3_hit_rate": float(dyn_df["top3_hit_rate"].mean()),
        "retrieval_topk_hit_rate": float(retrieval_df["retrieval_topk_hit_rate"].mean()),
        "exact_window_mean": float(retrieval_df["exact_window_mean"].mean()),
        "exact_draw_mean": float(retrieval_df["exact_draw_mean"].mean()),
    }
    save_json(Path("reports/backtest_experiment_summary.json"), summary)
    save_json(Path("reports/backtest_alignment_audit.json"), {"time_series_split": True, "runtime_scoring_shared": True})


if __name__ == "__main__":
    main()
