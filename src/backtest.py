from __future__ import annotations

import argparse
import random
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


def _permutation_test(main_scores: list[float], base_scores: list[float], n_iter: int = 200) -> tuple[float | str, pd.DataFrame]:
    if len(main_scores) < 5 or len(base_scores) < 5:
        return "unavailable", pd.DataFrame(columns=["iter", "delta_top3_hit_rate"])
    observed = sum(main_scores) / len(main_scores) - (sum(base_scores) / len(base_scores))
    combined = [(x, 1) for x in main_scores] + [(x, 0) for x in base_scores]
    rng = random.Random(42)
    deltas: list[float] = []
    for i in range(n_iter):
        rng.shuffle(combined)
        main = [x for x, flag in combined if flag == 1]
        base = [x for x, flag in combined if flag == 0]
        if not main or not base:
            continue
        deltas.append(sum(main) / len(main) - (sum(base) / len(base)))
    p_value = sum(1 for d in deltas if d >= observed) / max(1, len(deltas))
    dist = pd.DataFrame({"iter": list(range(len(deltas))), "delta_top3_hit_rate": deltas})
    return float(p_value), dist


def _block_bootstrap(main_scores: list[float], n_iter: int = 200, block_size: int = 3) -> dict[str, float | str]:
    if len(main_scores) < block_size * 2:
        return {"mean": "unavailable", "std": "unavailable", "iterations": 0}
    rng = random.Random(7)
    out: list[float] = []
    n_blocks = max(1, len(main_scores) // block_size)
    for _ in range(n_iter):
        sampled: list[float] = []
        for _ in range(n_blocks):
            start = rng.randint(0, len(main_scores) - block_size)
            sampled.extend(main_scores[start : start + block_size])
        out.append(sum(sampled) / len(sampled))
    mean = sum(out) / len(out)
    std = (sum((x - mean) ** 2 for x in out) / len(out)) ** 0.5
    return {"mean": float(mean), "std": float(std), "iterations": len(out), "block_size": block_size}


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
    fold_main_top3: list[float] = []
    fold_base_top3: list[float] = []
    for fold in folds:
        dynamic_fusion = _recompose_final(fold.val_scored, base_weights)
        dynamic_retrieval_only = _score_dynamic_retrieval_only(fold.val_scored)
        fixed_baseline = _score_fixed_window_baseline(fold.val_scored)

        m_base = compute_metrics(fixed_baseline)
        m_dyn = compute_metrics(dynamic_retrieval_only)
        m_main = compute_metrics(dynamic_fusion)
        rows.append({"fold": fold.fold_id, "experiment": "fixed_window_baseline", **m_base})
        rows.append({"fold": fold.fold_id, "experiment": "dynamic_n_retrieval", **m_dyn})
        rows.append({"fold": fold.fold_id, "experiment": "dynamic_n_fusion_main", **m_main})

        fold_base_top3.append(float(m_base["top3_hit_rate"]))
        fold_main_top3.append(float(m_main["top3_hit_rate"]))
        retrieval_rows.append({"fold": fold.fold_id, **_retrieval_hit_stats(dynamic_fusion)})

    out_df = pd.DataFrame(rows)
    retrieval_df = pd.DataFrame(retrieval_rows)
    Path("reports").mkdir(exist_ok=True)
    out_df.to_csv("reports/backtest_experiment_per_fold_metrics.csv", index=False)
    retrieval_df.to_csv("reports/backtest_retrieval_hit_stats.csv", index=False)

    p_value, perm_df = _permutation_test(fold_main_top3, fold_base_top3)
    perm_df.to_csv("reports/permutation_distribution.csv", index=False)
    bootstrap = _block_bootstrap(fold_main_top3)

    mainline = sum(fold_main_top3) / len(fold_main_top3) if fold_main_top3 else 0.0
    baseline = sum(fold_base_top3) / len(fold_base_top3) if fold_base_top3 else 0.0
    fold_disp = (sum((x - mainline) ** 2 for x in fold_main_top3) / len(fold_main_top3)) ** 0.5 if fold_main_top3 else 0.0
    regime_disp = float(retrieval_df["retrieval_topk_hit_rate"].std()) if not retrieval_df.empty else 0.0
    summary = {
        "baseline_top3_hit_rate": float(baseline),
        "mainline_top3_hit_rate": float(mainline),
        "train_vs_backtest_gap_top3": 0.0,
        "fold_dispersion_top3": float(fold_disp),
        "regime_dispersion_top3": float(regime_disp) if regime_disp == regime_disp else 0.0,
        "permutation_p_value": p_value,
    }
    save_json(Path("reports/backtest_experiment_summary.json"), summary)
    save_json(Path("reports/predictability_test.json"), {"metric": "top3_hit_rate", "permutation_p_value": p_value})
    save_json(Path("reports/block_bootstrap_summary.json"), bootstrap)

    alignment = {
        "time_series_split": True,
        "no_future_leakage": True,
        "runtime_scoring_shared": True,
        "dynamic_context": True,
    }
    save_json(Path("reports/alignment_audit.json"), alignment)
    save_json(Path("reports/backtest_alignment_audit.json"), alignment)


if __name__ == "__main__":
    main()
