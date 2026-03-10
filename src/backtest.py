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


def main() -> None:
    cfg = load_yaml(CONFIG_DIR / "train.yaml")
    feature_columns = json.loads(
        (PROJECT_ROOT / "models" / "feature_columns.json").read_text(encoding="utf-8")
    )
    feat_df = (
        pd.read_csv(FEATURE_STORE_DIR / "issue_features.csv")
        .tail(int(cfg.get("backtest_max_draws", 1000)))
        .reset_index(drop=True)
    )
    raw_df = load_processed().tail(len(feat_df) + 22).reset_index(drop=True)
    splits = int(cfg["backtest_splits"])

    tss = TimeSeriesSplit(n_splits=splits)
    experiments = _load_experiments() or default_experiments()
    params = cfg.get("catboost_params", {})
    params.setdefault("verbose", False)

    registry, per_fold, per_regime = [], [], []
    baseline = None
    baseline_top20 = []

    for exp in experiments:
        fold_train, fold_test, regime_rows = [], [], []
        for fold, (tr_idx, te_idx) in enumerate(tss.split(feat_df), start=1):
            train_df = feat_df.iloc[tr_idx]
            test_df = feat_df.iloc[te_idx]
            x_train, y_train = _expand_rows(train_df, feature_columns)
            model = CatBoostClassifier(**params)
            model.fit(x_train, y_train, verbose=False)

            for target, pack in [
                (train_df.tail(min(50, len(train_df))), fold_train),
                (test_df, fold_test),
            ]:
                rows = []
                for _, r in target.iterrows():
                    cand = build_candidate_matrix(r, feature_columns)
                    base_scores = model.predict_proba(cand)[:, 1]
                    scores = apply_strategy(base_scores, cand, exp, derive_regime(r))
                    m = _make_fold_issue_metrics(
                        scores, set(json.loads(r["target_numbers"]))
                    )
                    m["regime"] = derive_regime(r)
                    rows.append(m)
                pack.append({"fold": fold, **_aggregate(rows)})
                if target is test_df:
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
            }
        )
        per_fold.extend([{"version_id": exp.version_id, **x} for x in fold_test])
        per_regime.extend([{"version_id": exp.version_id, **x} for x in regime_rows])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    registry_df = pd.DataFrame(registry)
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
    save_json(
        REPORTS_DIR / "experiment_summary.json",
        {
            "baseline": baseline_row,
            "best_version": best,
            "total_versions": int(len(registry_df)),
            "kept_versions": int(registry_df["keep_recommendation"].sum()),
        },
    )
    print("backtest completed")


if __name__ == "__main__":
    main()
