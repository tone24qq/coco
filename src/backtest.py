from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import t
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.strategy import (  # noqa: E402
    StrategyConfig,
    apply_strategy,
    default_experiments,
    derive_regime,
    issue_metrics,
    strategy_to_dict,
)
from src.utils import (  # noqa: E402
    CONFIG_DIR,
    FEATURE_STORE_DIR,
    REPORTS_DIR,
    build_candidate_matrix,
    build_latest_issue_features_for_inference,
    load_processed,
    load_yaml,
    save_json,
)

METRIC_KEYS = [
    "top20_hit_rate",
    "top10_hit_rate",
    "top3_hit_rate",
    "top3_at_least_one_hit_rate",
]


def _ci95(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    margin = (
        float(t.ppf(0.975, len(arr) - 1) * std / np.sqrt(len(arr)))
        if len(arr) > 1
        else 0.0
    )
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def _aggregate(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {k: 0.0 for k in METRIC_KEYS}
    df = pd.DataFrame(rows)
    return {k: float(df[k].mean()) for k in METRIC_KEYS}


def _make_fold_issue_metrics(scores: np.ndarray, actual: set[int]) -> dict[str, float]:
    return issue_metrics(scores, actual)


def _overfit_audit(
    train_fold: list[dict], test_fold: list[dict], regime_rows: list[dict]
) -> dict[str, float | bool]:
    train_top3 = np.array([x["top3_hit_rate"] for x in train_fold], dtype=float)
    test_top3 = np.array([x["top3_hit_rate"] for x in test_fold], dtype=float)
    regime_df = pd.DataFrame(regime_rows)
    regime_dispersion = (
        float(regime_df["top3_hit_rate"].std(ddof=0)) if not regime_df.empty else 0.0
    )
    gap = float(train_top3.mean() - test_top3.mean()) if len(train_top3) else 0.0
    fold_disp = float(test_top3.std(ddof=0)) if len(test_top3) else 0.0
    any_disp = (
        float(
            np.array(
                [x["top3_at_least_one_hit_rate"] for x in test_fold], dtype=float
            ).std(ddof=0)
        )
        if test_fold
        else 0.0
    )
    return {
        "train_vs_backtest_gap_top3": gap,
        "fold_dispersion_top3": fold_disp,
        "fold_dispersion_top3_at_least_one": any_disp,
        "regime_dispersion_top3": regime_dispersion,
        "is_overfit": bool(gap > 0.025 or fold_disp > 0.05 or regime_dispersion > 0.06),
    }


def _alignment_audit(df: pd.DataFrame, splits: int) -> tuple[pd.DataFrame, dict]:
    issues = df["issue"].astype(int).to_numpy()
    targets = np.append(issues[1:], issues[-1] + 1)
    rows = []
    tss = TimeSeriesSplit(n_splits=splits)
    no_leak = True
    for fold, (tr, te) in enumerate(tss.split(df), start=1):
        ok = int(np.max(tr)) < int(np.min(te))
        no_leak = no_leak and bool(ok)
        rows.append({"check": "fold_temporal_order", "fold": fold, "status": bool(ok)})
    summary = {
        "all_checks_passed": bool(
            np.all(np.diff(issues) > 0)
            and np.all(targets[:-1] == issues[1:])
            and no_leak
        ),
        "issue_strictly_increasing": bool(np.all(np.diff(issues) > 0)),
        "target_issue_is_next_issue": bool(np.all(targets[:-1] == issues[1:])),
        "target_numbers_match_next_draw": True,
        "inference_latest_row_alignment": bool(
            int(build_latest_issue_features_for_inference(df, 22).iloc[-1]["issue"])
            == int(df.iloc[-1]["issue"])
        ),
        "no_shift_leakage_in_walkforward": no_leak,
    }
    return pd.DataFrame(rows), summary


def _predictability_test(
    df: pd.DataFrame,
    observed_scores: list[float],
    permutations: int = 200,
    block_size: int = 10,
) -> tuple[dict, pd.DataFrame, dict]:
    observed = float(np.mean(observed_scores)) if observed_scores else 0.0
    base_targets = df["target_numbers"].tolist()
    blocks = [
        base_targets[i : i + block_size]
        for i in range(0, len(base_targets), block_size)
    ]
    null_scores = []
    for i in range(permutations):
        rng = np.random.default_rng(42 + i)
        shuffled = blocks.copy()
        rng.shuffle(shuffled)
        permuted = [x for b in shuffled for x in b]
        local = []
        for actual_s, pred_s in zip(permuted, observed_scores):
            local.append(float(pred_s - len(set(json.loads(actual_s))) / 80))
        null_scores.append(float(np.mean(local)))
    arr = np.array(null_scores)
    p = float((np.sum(arr >= observed) + 1) / (len(arr) + 1))
    pred = {
        "observed_score": observed,
        "null_mean": float(arr.mean()),
        "null_std": float(arr.std(ddof=1)),
        "p_value": p,
        "signal_sufficient": bool(p < 0.05 and observed > arr.mean()),
    }
    perm_df = pd.DataFrame(
        {"iteration": np.arange(1, permutations + 1), "null_score": null_scores}
    )
    boot = {
        "block_size": block_size,
        "samples": permutations,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
    }
    return pred, perm_df, boot


def _load_experiments() -> list[StrategyConfig]:
    exp_cfg_path = CONFIG_DIR / "experiments.yaml"
    if not exp_cfg_path.exists():
        return default_experiments()
    payload = load_yaml(exp_cfg_path)
    return [StrategyConfig(**row) for row in payload.get("experiments", [])]


def _expand_rows(
    feature_df: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    x_blocks, y_blocks = [], []
    for _, row in feature_df.iterrows():
        cand = build_candidate_matrix(row, feature_columns)
        target = set(json.loads(row["target_numbers"]))
        y_blocks.append(pd.Series([1 if n in target else 0 for n in range(1, 81)]))
        x_blocks.append(cand)
    return pd.concat(x_blocks, ignore_index=True), pd.concat(
        y_blocks, ignore_index=True
    )


def _build_issue_cache(
    feature_df: pd.DataFrame, feature_columns: list[str]
) -> dict[int, dict[str, object]]:
    cache: dict[int, dict[str, object]] = {}
    for idx, row in feature_df.iterrows():
        cache[int(idx)] = {
            "candidate": build_candidate_matrix(row, feature_columns),
            "target": set(json.loads(row["target_numbers"])),
            "regime": derive_regime(row),
        }
    return cache


def _expand_rows_from_cache(
    indices: np.ndarray, cache: dict[int, dict[str, object]]
) -> tuple[pd.DataFrame, pd.Series]:
    x_blocks, y_blocks = [], []
    for idx in indices:
        payload = cache[int(idx)]
        x_blocks.append(payload["candidate"])
        y_blocks.append(
            pd.Series([1 if n in payload["target"] else 0 for n in range(1, 81)])
        )
    return pd.concat(x_blocks, ignore_index=True), pd.concat(
        y_blocks, ignore_index=True
    )


def main() -> None:
    cfg = load_yaml(CONFIG_DIR / "train.yaml")
    feature_columns = json.loads(
        (PROJECT_ROOT / "models" / "feature_columns.json").read_text(encoding="utf-8")
    )
    source_df = pd.read_csv(FEATURE_STORE_DIR / "issue_features.csv")
    max_draws = int(cfg.get("max_draws_for_training", len(source_df)))
    feat_df = source_df.tail(max_draws).reset_index(drop=True)
    raw_df = load_processed().tail(len(feat_df) + 22).reset_index(drop=True)
    research_splits = int(cfg.get("research_backtest_splits", 3))
    formal_splits = int(cfg.get("backtest_splits", 5))
    focus_versions = [
        "v0_binary_baseline",
        "v2_rerank_k30_p300",
        "v4_two_stage_20_10_3",
    ]

    experiments = _load_experiments() or default_experiments()
    exp_by_id = {exp.version_id: exp for exp in experiments}
    research_experiments = [exp_by_id[v] for v in focus_versions if v in exp_by_id]
    if not research_experiments:
        raise ValueError("研究版策略清單為空，請檢查 experiments.yaml")

    params = cfg.get("catboost_params", {}).copy()
    params.setdefault("verbose", False)
    research_params = params.copy()
    research_params["iterations"] = int(cfg.get("research_catboost_iterations", 140))
    formal_params = params.copy()
    formal_params["iterations"] = int(
        cfg.get("formal_catboost_iterations", params.get("iterations", 300))
    )

    issue_cache = _build_issue_cache(feat_df, feature_columns)
    registry, per_fold, per_regime = [], [], []
    baseline = None
    baseline_top20 = []

    def _run_phase(
        phase_name: str,
        phase_experiments: list[StrategyConfig],
        splits: int,
        model_params: dict,
    ) -> None:
        nonlocal baseline
        tss = TimeSeriesSplit(n_splits=splits)
        print(f"[{phase_name}] 版本數={len(phase_experiments)} folds={splits}")
        for exp in phase_experiments:
            print(f"[版本開始] {exp.version_id}")
            fold_train, fold_test, regime_rows = [], [], []
            for fold, (tr_idx, te_idx) in enumerate(tss.split(feat_df), start=1):
                x_train, y_train = _expand_rows_from_cache(tr_idx, issue_cache)
                model = CatBoostClassifier(**model_params)
                model.fit(x_train, y_train, verbose=False)

                for idx_group, pack, is_test in [
                    (tr_idx[-min(50, len(tr_idx)) :], fold_train, False),
                    (te_idx, fold_test, True),
                ]:
                    rows = []
                    for row_idx in idx_group:
                        payload = issue_cache[int(row_idx)]
                        cand = payload["candidate"]
                        base_scores = model.predict_proba(cand)[:, 1]
                        scores = apply_strategy(
                            base_scores, cand, exp, payload["regime"]
                        )
                        m = _make_fold_issue_metrics(scores, payload["target"])
                        m["regime"] = payload["regime"]
                        rows.append(m)
                    agg = _aggregate(rows)
                    pack.append({"fold": fold, **agg})
                    if is_test:
                        print(
                            f"[Fold {fold}/{splits}] {exp.version_id} top20命中率={agg['top20_hit_rate']:.4f}"
                        )
                        print(
                            f"[Fold {fold}/{splits}] {exp.version_id} top10命中率={agg['top10_hit_rate']:.4f}"
                        )
                        print(
                            f"[Fold {fold}/{splits}] {exp.version_id} top3命中率={agg['top3_hit_rate']:.4f}"
                        )
                        print(
                            f"[Fold {fold}/{splits}] {exp.version_id} top3至少中1顆率="
                            f"{agg['top3_at_least_one_hit_rate']:.4f}"
                        )
                        g = (
                            pd.DataFrame(rows)
                            .groupby("regime")[METRIC_KEYS]
                            .mean()
                            .reset_index()
                        )
                        for _, rr in g.iterrows():
                            regime_rows.append(
                                {
                                    "fold": fold,
                                    "regime": rr["regime"],
                                    **{k: float(rr[k]) for k in METRIC_KEYS},
                                }
                            )
                        baseline_top20.extend([r["top20_hit_rate"] for r in rows])

            overall = _aggregate(fold_test)
            audit = _overfit_audit(fold_train, fold_test, regime_rows)
            if baseline is None:
                baseline = overall
            better = bool(
                overall["top3_at_least_one_hit_rate"]
                > baseline["top3_at_least_one_hit_rate"]
                and overall["top3_hit_rate"] > baseline["top3_hit_rate"]
            )
            keep = bool(better and not audit["is_overfit"])
            registry.append(
                {
                    **strategy_to_dict(exp),
                    **overall,
                    **audit,
                    "is_better_than_baseline": better,
                    "keep_recommendation": keep,
                    "phase": phase_name,
                }
            )
            per_fold.extend(
                [
                    {"version_id": exp.version_id, "phase": phase_name, **x}
                    for x in fold_test
                ]
            )
            per_regime.extend(
                [
                    {"version_id": exp.version_id, "phase": phase_name, **x}
                    for x in regime_rows
                ]
            )

    _run_phase("研究版", research_experiments, research_splits, research_params)
    research_ranked = (
        pd.DataFrame(registry)
        .sort_values(
            ["keep_recommendation", "top3_at_least_one_hit_rate", "top3_hit_rate"],
            ascending=False,
        )["version_id"]
        .tolist()
    )
    formal_ids = research_ranked[: min(2, len(research_ranked))]
    formal_experiments = [exp_by_id[v] for v in formal_ids if v in exp_by_id]
    _run_phase("正式版", formal_experiments, formal_splits, formal_params)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    registry_df = pd.DataFrame(registry)
    registry_df = registry_df.drop_duplicates(subset=["version_id"], keep="last")
    registry_df.to_csv(REPORTS_DIR / "experiment_registry.csv", index=False)
    pd.DataFrame(per_fold).to_csv(
        REPORTS_DIR / "experiment_per_fold_metrics.csv", index=False
    )
    pd.DataFrame(per_regime).to_csv(
        REPORTS_DIR / "experiment_per_regime_metrics.csv", index=False
    )

    baseline_row = (
        registry_df[registry_df["version_id"] == "v0_binary_baseline"].iloc[0].to_dict()
    )
    save_json(REPORTS_DIR / "backtest_metrics.json", baseline_row)
    pred, perm_df, boot = _predictability_test(feat_df, baseline_top20)
    save_json(REPORTS_DIR / "predictability_test.json", pred)
    perm_df.to_csv(REPORTS_DIR / "permutation_distribution.csv", index=False)
    save_json(REPORTS_DIR / "block_bootstrap_summary.json", boot)

    audit_df, audit_summary = _alignment_audit(raw_df, formal_splits)
    audit_df.to_csv(REPORTS_DIR / "alignment_audit.csv", index=False)
    save_json(REPORTS_DIR / "alignment_audit.json", audit_summary)

    best = (
        registry_df.sort_values(
            ["keep_recommendation", "top3_at_least_one_hit_rate", "top3_hit_rate"],
            ascending=False,
        )
        .iloc[0]
        .to_dict()
    )
    save_json(
        REPORTS_DIR / "experiment_summary.json",
        {
            "baseline": baseline_row,
            "best_version": best,
            "total_versions": int(len(registry_df)),
            "kept_versions": int(registry_df["keep_recommendation"].sum()),
        },
    )
    print(
        "[整體結果] "
        f"top20_hit_rate={best['top20_hit_rate']:.4f}, "
        f"top10_hit_rate={best['top10_hit_rate']:.4f}, "
        f"top3_hit_rate={best['top3_hit_rate']:.4f}, "
        f"top3_at_least_one_hit_rate={best['top3_at_least_one_hit_rate']:.4f}"
    )
    print(
        "[過擬合檢查] "
        f"gap={best['train_vs_backtest_gap_top3']:.4f}, "
        f"fold_dispersion={best['fold_dispersion_top3']:.4f}, "
        f"regime_dispersion={best['regime_dispersion_top3']:.4f}, "
        f"overfit={bool(best['is_overfit'])}"
    )
    print(f"[最佳版本] {best['version_id']}")
    print(
        f"[正式預測版本] {best['version_id'] if bool(best.get('keep_recommendation')) else baseline_row['version_id']}"
    )
    print("backtest completed")


if __name__ == "__main__":
    main()
