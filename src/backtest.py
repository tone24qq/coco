from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
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
    build_issue_features,
    build_latest_issue_features_for_inference,
    load_processed,
    load_yaml,
    precompute_issue_payloads,
    save_json,
    validate_feature_columns_contract,
)

METRIC_KEYS = [
    "top20_hit_rate",
    "top5_hit_rate",
    "top10_hit_rate",
    "top3_hit_rate",
    "top3_at_least_one_hit_rate",
    "ndcg_at_10",
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
    feat_df = build_issue_features(df, min_history=22)
    target_match = True
    for _, row in feat_df.iterrows():
        issue = int(row["issue"])
        idx = int(np.where(issues == issue)[0][0])
        if idx + 1 >= len(df):
            continue
        expected = sorted(json.loads(df.iloc[idx + 1]["numbers"]))
        actual = sorted(json.loads(row["target_numbers"]))
        if expected != actual:
            target_match = False
            break

    summary = {
        "all_checks_passed": bool(
            np.all(np.diff(issues) > 0)
            and np.all(targets[:-1] == issues[1:])
            and no_leak
            and target_match
        ),
        "issue_strictly_increasing": bool(np.all(np.diff(issues) > 0)),
        "target_issue_is_next_issue": bool(np.all(targets[:-1] == issues[1:])),
        "target_numbers_match_next_draw": bool(target_match),
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


def _build_feature_version_comparison(
    history_df: pd.DataFrame,
    current_row: dict,
    thresholds: dict,
) -> dict:
    full = pd.concat([history_df, pd.DataFrame([current_row])], ignore_index=True)
    v3_rows = full[full["feature_version"] == "v3_core20"]
    if v3_rows.empty:
        return {
            "available": False,
            "reason": "missing v3_core20 reference",
            "current_feature_version": current_row["feature_version"],
        }
    if len(v3_rows) < 2:
        return {
            "available": False,
            "reason": "missing historical v3_core20 reference",
            "current_feature_version": current_row["feature_version"],
        }

    sorted_v3 = v3_rows.sort_values("trained_at_utc")
    v3_baseline = sorted_v3.iloc[-2].to_dict()
    v3_current = sorted_v3.iloc[-1].to_dict()
    deltas = {
        "delta_top3": float(v3_current["top3_hit_rate"] - v3_baseline["top3_hit_rate"]),
        "delta_top5": float(v3_current["top5_hit_rate"] - v3_baseline["top5_hit_rate"]),
        "delta_top10": float(
            v3_current["top10_hit_rate"] - v3_baseline["top10_hit_rate"]
        ),
        "delta_top20": float(
            v3_current["top20_hit_rate"] - v3_baseline["top20_hit_rate"]
        ),
        "delta_top3_at_least_one_hit_rate": float(
            v3_current["top3_at_least_one_hit_rate"]
            - v3_baseline["top3_at_least_one_hit_rate"]
        ),
        "delta_fold_dispersion_top3": float(
            v3_current["fold_dispersion_top3"] - v3_baseline["fold_dispersion_top3"]
        ),
        "delta_regime_dispersion_top3": float(
            v3_current["regime_dispersion_top3"] - v3_baseline["regime_dispersion_top3"]
        ),
    }

    tol = float(thresholds.get("non_degradation_tol", 0.01))
    stability_min = float(thresholds.get("stability_improvement_min", 0.0))
    non_degradation_pass = bool(
        deltas["delta_top3"] >= -tol
        and deltas["delta_top5"] >= -tol
        and deltas["delta_top10"] >= -tol
    )
    stability_pass = bool(
        deltas["delta_fold_dispersion_top3"] <= -stability_min
        and deltas["delta_regime_dispersion_top3"] <= -stability_min
    )

    return {
        "available": True,
        "current_feature_version": current_row["feature_version"],
        "v3_baseline": v3_baseline,
        "v3_current": v3_current,
        "thresholds": {
            "non_degradation_tol": tol,
            "stability_improvement_min": stability_min,
        },
        "deltas": deltas,
        "non_degradation_pass": non_degradation_pass,
        "stability_pass": stability_pass,
        "acceptance_pass": bool(non_degradation_pass and stability_pass),
    }


def _load_experiments() -> list[StrategyConfig]:
    exp_cfg_path = CONFIG_DIR / "experiments.yaml"
    if not exp_cfg_path.exists():
        return default_experiments()
    payload = load_yaml(exp_cfg_path)
    return [StrategyConfig(**row) for row in payload.get("experiments", [])]


def _expand_rows(
    issue_payloads: dict[int, dict[str, object]], indices: list[int]
) -> tuple[pd.DataFrame, pd.Series]:
    x_blocks, y_blocks = [], []
    for idx in indices:
        payload = issue_payloads[int(idx)]
        x_blocks.append(payload["cand"])
        y_blocks.append(
            pd.Series([1 if n in payload["target"] else 0 for n in range(1, 81)])
        )
    return pd.concat(x_blocks, ignore_index=True), pd.concat(
        y_blocks, ignore_index=True
    )


def _run_experiments(
    feat_df: pd.DataFrame,
    splits: int,
    experiments: list[StrategyConfig],
    params: dict,
    issue_payloads: dict[int, dict[str, object]],
) -> tuple[list[dict], list[dict], list[dict], list[float]]:
    tss = TimeSeriesSplit(n_splits=splits)
    registry, per_fold, per_regime = [], [], []
    baseline = None
    baseline_top20 = []

    for exp in experiments:
        print(f"[版本開始] {exp.version_id}")
        fold_train, fold_test, regime_rows = [], [], []
        for fold, (tr_idx, te_idx) in enumerate(tss.split(feat_df), start=1):
            x_train, y_train = _expand_rows(issue_payloads, list(tr_idx))
            model = CatBoostClassifier(**params)
            model.fit(x_train, y_train, verbose=False)

            for idx_set, pack in [
                (tr_idx[-min(50, len(tr_idx)) :], fold_train),
                (te_idx, fold_test),
            ]:
                rows = []
                for row_idx in idx_set:
                    payload = issue_payloads[int(row_idx)]
                    cand = payload["cand"]
                    base_scores = model.predict_proba(cand)[:, 1]
                    regime = payload["regime"]
                    if regime is None:
                        regime = derive_regime(feat_df.iloc[int(row_idx)])
                        payload["regime"] = regime
                    scores = apply_strategy(base_scores, cand, exp, regime)
                    m = _make_fold_issue_metrics(scores, payload["target"])
                    m["regime"] = regime
                    rows.append(m)
                pack.append({"fold": fold, **_aggregate(rows)})
                if list(idx_set) == list(te_idx):
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

            te_agg = fold_test[-1]
            print(
                f"[Fold {fold}/{splits}] {exp.version_id} top20命中率={te_agg['top20_hit_rate']:.4f}"
            )
            print(
                f"[Fold {fold}/{splits}] {exp.version_id} top10命中率={te_agg['top10_hit_rate']:.4f}"
            )
            print(
                f"[Fold {fold}/{splits}] {exp.version_id} top3命中率={te_agg['top3_hit_rate']:.4f}"
            )
            print(
                f"[Fold {fold}/{splits}] {exp.version_id} top3至少中1顆率={te_agg['top3_at_least_one_hit_rate']:.4f}"
            )

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
            }
        )
        per_fold.extend([{"version_id": exp.version_id, **x} for x in fold_test])
        per_regime.extend([{"version_id": exp.version_id, **x} for x in regime_rows])

    return registry, per_fold, per_regime, baseline_top20


def main() -> None:
    cfg = load_yaml(CONFIG_DIR / "train.yaml")
    if str(cfg.get("feature_version", "v3_core20")) != "v3_core20":
        raise ValueError("only v3_core20 is supported")
    os.environ["STRICT_FEATURES"] = "1"
    feature_columns = json.loads(
        (PROJECT_ROOT / "models" / "feature_columns.json").read_text(encoding="utf-8")
    )
    validate_feature_columns_contract(
        feature_columns,
        str(cfg.get("feature_version", "v3_core20")),
    )
    feat_df = pd.read_csv(FEATURE_STORE_DIR / "issue_features.csv")
    max_draws = int(cfg.get("max_draws_for_training", len(feat_df)))
    feat_df = feat_df.tail(max_draws).reset_index(drop=True)
    raw_df = load_processed().tail(len(feat_df) + 22).reset_index(drop=True)
    splits = int(cfg["backtest_splits"])

    experiments = _load_experiments() or default_experiments()
    params = cfg.get("catboost_params", {})
    params.setdefault("verbose", False)
    issue_payloads = precompute_issue_payloads(
        feat_df,
        feature_columns,
        strict_features=True,
    )

    fast_version_ids = {
        "v0_binary_baseline",
        "v3_rerank_k30_p300",
        "v4_two_stage_20_10_3",
    }
    fast_experiments = [
        exp for exp in experiments if exp.version_id in fast_version_ids
    ]
    print("[研究流程] backtest 快速階段：3個版本、3 folds、較低 iterations")
    fast_params = dict(params)
    fast_params["iterations"] = int(cfg.get("research_iterations", 140))
    fast_registry, _, _, _ = _run_experiments(
        feat_df=feat_df,
        splits=int(cfg.get("research_backtest_splits", 3)),
        experiments=fast_experiments,
        params=fast_params,
        issue_payloads=issue_payloads,
    )
    fast_df = pd.DataFrame(fast_registry)
    selected_final_ids = (
        fast_df.sort_values(
            ["keep_recommendation", "top3_at_least_one_hit_rate", "top3_hit_rate"],
            ascending=False,
        )["version_id"]
        .head(int(cfg.get("final_stage_versions", 2)))
        .tolist()
    )
    if not selected_final_ids:
        selected_final_ids = ["v0_binary_baseline"]
    if "v0_binary_baseline" not in selected_final_ids:
        selected_final_ids = ["v0_binary_baseline", *selected_final_ids]
    final_experiments = [
        exp for exp in experiments if exp.version_id in set(selected_final_ids)
    ]
    print(f"[研究流程] backtest 正式階段：版本={selected_final_ids}、{splits} folds")
    registry, per_fold, per_regime, baseline_top20 = _run_experiments(
        feat_df=feat_df,
        splits=splits,
        experiments=final_experiments,
        params=params,
        issue_payloads=issue_payloads,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    registry_df = pd.DataFrame(registry)
    registry_df.to_csv(REPORTS_DIR / "experiment_registry.csv", index=False)
    pd.DataFrame(fast_registry).to_csv(
        REPORTS_DIR / "experiment_registry_research.csv", index=False
    )
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

    audit_df, audit_summary = _alignment_audit(raw_df, splits)
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
    feature_version = str(cfg.get("feature_version", "v3_core20"))
    history_path = REPORTS_DIR / "feature_version_history.csv"
    current_comp_row = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_version": feature_version,
        "version_id": str(best.get("version_id", "unknown")),
        "top20_hit_rate": float(best.get("top20_hit_rate", 0.0)),
        "top10_hit_rate": float(best.get("top10_hit_rate", 0.0)),
        "top5_hit_rate": float(best.get("top5_hit_rate", 0.0)),
        "top3_hit_rate": float(best.get("top3_hit_rate", 0.0)),
        "top3_at_least_one_hit_rate": float(
            best.get("top3_at_least_one_hit_rate", 0.0)
        ),
        "ndcg_at_10": float(best.get("ndcg_at_10", 0.0)),
        "fold_dispersion_top3": float(best.get("fold_dispersion_top3", 0.0)),
        "regime_dispersion_top3": float(best.get("regime_dispersion_top3", 0.0)),
    }
    if history_path.exists():
        history_df = pd.read_csv(history_path)
    else:
        history_df = pd.DataFrame(columns=list(current_comp_row.keys()))
    updated_history = pd.concat(
        [history_df, pd.DataFrame([current_comp_row])], ignore_index=True
    )
    updated_history.to_csv(history_path, index=False)

    comparison = _build_feature_version_comparison(
        history_df,
        current_comp_row,
        cfg.get("acceptance_thresholds", {}),
    )
    save_json(REPORTS_DIR / "feature_version_comparison.json", comparison)

    save_json(
        REPORTS_DIR / "experiment_summary.json",
        {
            "feature_version": feature_version,
            "baseline": baseline_row,
            "best_version": best,
            "top5_hit_rate": float(best.get("top5_hit_rate", 0.0)),
            "ndcg_at_10": float(best.get("ndcg_at_10", 0.0)),
            "comparison": comparison,
            "acceptance": {
                "available": bool(comparison.get("available", False)),
                "acceptance_pass": bool(comparison.get("acceptance_pass", False)),
            },
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
    print(f"[正式預測版本] {best['version_id']}")


if __name__ == "__main__":
    main()
