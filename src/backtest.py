from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.modeling import compute_metrics, load_ranking_dataset, resolve_feature_columns, run_cv, save_json
from src.runtime_scoring import RuntimeWeights
from src.utils import enforce_dir_file_sizes, log_progress


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


def _weights_for_experiment(name: str, base_weights: RuntimeWeights) -> tuple[str, RuntimeWeights]:
    mapping = {
        "ranker_main_qsm": base_weights,
        "dynamic_n_fusion_main": base_weights,
        "ablation_no_retrieval": RuntimeWeights(
            ranker=base_weights.ranker,
            logistic=base_weights.logistic,
            retrieval=0.0,
            history_prior=base_weights.history_prior,
            analysis=base_weights.analysis,
            local_peak=base_weights.local_peak,
        ),
        "ablation_no_logistic": RuntimeWeights(
            ranker=base_weights.ranker,
            logistic=0.0,
            retrieval=base_weights.retrieval,
            history_prior=base_weights.history_prior,
            analysis=base_weights.analysis,
            local_peak=base_weights.local_peak,
        ),
    }
    return name, mapping.get(name, base_weights)


def _score_experiment(table: pd.DataFrame, name: str, base_weights: RuntimeWeights) -> pd.DataFrame:
    if name == "baseline_frequency":
        return _score_fixed_window_baseline(table)
    if name == "dynamic_n_retrieval":
        return _score_dynamic_retrieval_only(table)
    _, weights = _weights_for_experiment(name, base_weights)
    return _recompose_final(table, weights)


def _retrieval_hit_stats(scored: pd.DataFrame) -> dict[str, float]:
    topk = scored.sort_values(["issue", "final_score"], ascending=[True, False]).groupby("issue").head(20)
    return {
        "retrieval_topk_hit_rate": float(topk["retrieval_top3_hit_flag"].mean()),
        "exact_window_mean": float(topk["retrieval_exact_window_match_count"].mean()),
        "exact_draw_mean": float(topk["retrieval_exact_draw_match_count_mean"].mean()),
    }


def _permutation_test(deltas: list[float], n_iter: int = 500) -> tuple[float | str, pd.DataFrame]:
    if len(deltas) < 5:
        return "unavailable", pd.DataFrame(columns=["iter", "delta_top3_hit_rate"])
    observed = sum(deltas) / len(deltas)
    rng = random.Random(42)
    sampled_means: list[float] = []
    for _ in range(n_iter):
        sampled = [x if rng.random() >= 0.5 else -x for x in deltas]
        sampled_means.append(sum(sampled) / len(sampled))
    p_value = sum(1 for d in sampled_means if d >= observed) / max(1, len(sampled_means))
    dist = pd.DataFrame({"iter": list(range(len(sampled_means))), "delta_top3_hit_rate": sampled_means})
    return float(p_value), dist


def _block_bootstrap(deltas: list[float], n_iter: int = 300, block_size: int = 5) -> dict[str, float | str]:
    if len(deltas) < block_size * 2:
        return {"mean": "unavailable", "std": "unavailable", "iterations": 0}
    rng = random.Random(7)
    out: list[float] = []
    n_blocks = max(1, len(deltas) // block_size)
    for _ in range(n_iter):
        sampled: list[float] = []
        for _ in range(n_blocks):
            start = rng.randint(0, len(deltas) - block_size)
            sampled.extend(deltas[start : start + block_size])
        out.append(sum(sampled) / len(sampled))
    mean = sum(out) / len(out)
    std = (sum((x - mean) ** 2 for x in out) / len(out)) ** 0.5
    return {"mean": float(mean), "std": float(std), "iterations": len(out), "block_size": block_size}


def _issue_metric(scored: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for issue, grp in scored.groupby("issue"):
        g = grp.sort_values("final_score", ascending=False).head(3)
        out[str(issue)] = float(g["label"].sum() / 3.0)
    return out


def _alignment_audit(
    folds: list[Any], feature_cols: list[str], dataset: pd.DataFrame, weights: RuntimeWeights, models_dir: Path = Path("models")
) -> dict[str, Any]:
    monotonic = True
    overlap_issues = 0
    for fold in folds:
        if fold.train_issues and fold.val_issues and max(fold.train_issues) >= min(fold.val_issues):
            monotonic = False
        overlap_issues += len(set(fold.train_issues).intersection(set(fold.val_issues)))
    first = folds[0].val_scored if folds else pd.DataFrame()
    scoring_formula_match = False
    if not first.empty:
        expected = _recompose_final(first, weights)["final_score"].round(12)
        got = first.sort_values(["issue", "final_score"], ascending=[True, False])["final_score"].round(12)
        scoring_formula_match = bool((expected.reset_index(drop=True) == got.reset_index(drop=True)).all())
    artifact_contract_ok = (models_dir / "feature_columns.json").exists() and (models_dir / "metadata.json").exists()
    feature_contract_match = False
    if (models_dir / "feature_columns.json").exists():
        loaded = json.loads((models_dir / "feature_columns.json").read_text(encoding="utf-8"))
        feature_contract_match = loaded == feature_cols
    return {
        "time_series_split_forward_only": monotonic,
        "train_val_issue_overlap_count": overlap_issues,
        "runtime_backtest_scoring_formula_match": scoring_formula_match,
        "feature_contract_match": feature_contract_match,
        "artifact_contract_present": artifact_contract_ok,
        "dataset_issue_count": int(dataset["issue"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiments", default="configs/experiments.yaml")
    parser.add_argument("--input", default="data/feature_store/ranking_dataset.csv")
    args = parser.parse_args()

    log_progress(1, 6, "載入回測設定", f"config={args.config}")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    exp_cfg = yaml.safe_load(Path(args.experiments).read_text(encoding="utf-8"))
    base_weights = RuntimeWeights.from_mapping(config.get("runtime_scoring", {}).get("weights", {}))

    log_progress(2, 6, "讀取 ranking dataset", f"input={args.input}")
    df = load_ranking_dataset(Path(args.input))
    feature_cols = resolve_feature_columns(df)
    n_splits = int(config.get("validation", {}).get("n_splits", 3))
    min_train_issues = int(config.get("validation", {}).get("min_train_issues", 30))
    folds = run_cv(df, feature_cols, base_weights, n_splits=n_splits, min_train_issues=min_train_issues)
    log_progress(3, 6, "建立 walk-forward folds", f"folds={len(folds)}")

    experiment_names = [str(x.get("name")) for x in exp_cfg.get("experiments", [])] or [
        "baseline_frequency",
        "dynamic_n_retrieval",
        "dynamic_n_fusion_main",
    ]
    rows: list[dict[str, float | int | str]] = []
    retrieval_rows: list[dict[str, float | int]] = []
    fold_main_top3: list[float] = []
    fold_base_top3: list[float] = []
    per_issue_deltas: list[float] = []
    train_main_fold_top3: list[float] = []
    for fold in folds:
        val_scored_by_exp: dict[str, pd.DataFrame] = {}
        for name in experiment_names:
            scored = _score_experiment(fold.val_scored, name, base_weights)
            val_scored_by_exp[name] = scored
            rows.append({"fold": fold.fold_id, "experiment": name, **compute_metrics(scored)})
        main_exp_name = "dynamic_n_fusion_main" if "dynamic_n_fusion_main" in val_scored_by_exp else "ranker_main_qsm"
        if main_exp_name in val_scored_by_exp:
            m_main = compute_metrics(val_scored_by_exp[main_exp_name])
            fold_main_top3.append(float(m_main["top3_hit_rate"]))
            train_main = _score_experiment(fold.train_scored, main_exp_name, base_weights)
            train_main_fold_top3.append(float(compute_metrics(train_main)["top3_hit_rate"]))
            retrieval_rows.append({"fold": fold.fold_id, **_retrieval_hit_stats(val_scored_by_exp[main_exp_name])})
        if "baseline_frequency" in val_scored_by_exp:
            m_base = compute_metrics(val_scored_by_exp["baseline_frequency"])
            fold_base_top3.append(float(m_base["top3_hit_rate"]))
        if main_exp_name in val_scored_by_exp and "baseline_frequency" in val_scored_by_exp:
            base_issue = _issue_metric(val_scored_by_exp["baseline_frequency"])
            main_issue = _issue_metric(val_scored_by_exp[main_exp_name])
            shared_issues = sorted(set(base_issue).intersection(main_issue))
            per_issue_deltas.extend([main_issue[i] - base_issue[i] for i in shared_issues])
    log_progress(4, 6, "完成各實驗 fold scoring", f"experiments={len(experiment_names)}")

    out_df = pd.DataFrame(rows)
    retrieval_df = pd.DataFrame(retrieval_rows)
    Path("reports").mkdir(exist_ok=True)
    out_df.to_csv("reports/backtest_experiment_per_fold_metrics.csv", index=False)
    retrieval_df.to_csv("reports/backtest_retrieval_hit_stats.csv", index=False)

    p_value, perm_df = _permutation_test(per_issue_deltas)
    perm_df.to_csv("reports/permutation_distribution.csv", index=False)
    bootstrap = _block_bootstrap(per_issue_deltas)

    mainline = sum(fold_main_top3) / len(fold_main_top3) if fold_main_top3 else 0.0
    baseline = sum(fold_base_top3) / len(fold_base_top3) if fold_base_top3 else 0.0
    train_mainline = sum(train_main_fold_top3) / len(train_main_fold_top3) if train_main_fold_top3 else 0.0
    fold_disp = (sum((x - mainline) ** 2 for x in fold_main_top3) / len(fold_main_top3)) ** 0.5 if fold_main_top3 else 0.0
    regime_disp = float(retrieval_df["retrieval_topk_hit_rate"].std()) if not retrieval_df.empty else 0.0
    summary = {
        "baseline_top3_hit_rate": float(baseline),
        "mainline_top3_hit_rate": float(mainline),
        "train_top3_hit_rate": float(train_mainline),
        "train_vs_backtest_gap_top3": float(train_mainline - mainline),
        "fold_dispersion_top3": float(fold_disp),
        "regime_dispersion_top3": float(regime_disp) if regime_disp == regime_disp else 0.0,
        "permutation_p_value": p_value,
    }
    save_json(Path("reports/backtest_experiment_summary.json"), summary)
    save_json(
        Path("reports/predictability_test.json"),
        {"metric": "top3_hit_rate", "null_hypothesis": "mainline_minus_baseline_mean<=0", "permutation_p_value": p_value},
    )
    save_json(Path("reports/block_bootstrap_summary.json"), {"metric": "mainline_minus_baseline_top3", **bootstrap})

    alignment = _alignment_audit(folds, feature_cols, df, base_weights)
    save_json(Path("reports/alignment_audit.json"), alignment)
    save_json(Path("reports/backtest_alignment_audit.json"), alignment)
    log_progress(5, 6, "輸出回測報表完成", "reports/")
    enforce_dir_file_sizes([Path("models"), Path("reports"), Path("data/feature_store")])
    log_progress(6, 6, "回測主線完成", f"per_issue_deltas={len(per_issue_deltas)}")


if __name__ == "__main__":
    main()
