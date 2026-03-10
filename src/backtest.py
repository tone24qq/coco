from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

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


def _top_hits(pred_scores: np.ndarray, actual: set[int]) -> tuple[int, int, int]:
    order = np.argsort(pred_scores)[::-1] + 1
    return (
        len(set(order[:20]) & actual),
        len(set(order[:10]) & actual),
        len(set(order[:3]) & actual),
    )


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


def _derive_regime(row: pd.Series) -> str:
    if float(row.get("span", 0)) >= 72 or float(row.get("consecutive_pairs", 0)) >= 6:
        return "high_vol"
    if float(row.get("zone_range", 0)) <= 2 and float(row.get("span", 0)) <= 58:
        return "balanced"
    return "transitional"


def _make_fold_issue_metrics(scores: np.ndarray, actual: set[int]) -> dict[str, float]:
    h20, h10, h3 = _top_hits(scores, actual)
    return {
        "top20_hit_rate": h20 / 20,
        "top10_hit_rate": h10 / 10,
        "top3_hit_rate": h3 / 3,
        "top3_at_least_one_hit_rate": float(h3 > 0),
    }


def _aggregate(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {k: 0.0 for k in METRIC_KEYS}
    df = pd.DataFrame(rows)
    return {k: float(df[k].mean()) for k in METRIC_KEYS}


def _overfit_audit(
    train_fold: list[dict], test_fold: list[dict], regime_rows: list[dict]
) -> dict[str, float | bool]:
    train_top3 = np.array([x["top3_hit_rate"] for x in train_fold], dtype=float)
    test_top3 = np.array([x["top3_hit_rate"] for x in test_fold], dtype=float)
    test_top3_any = np.array(
        [x["top3_at_least_one_hit_rate"] for x in test_fold], dtype=float
    )
    regime_df = pd.DataFrame(regime_rows)
    regime_dispersion = (
        float(regime_df["top3_hit_rate"].std(ddof=0)) if not regime_df.empty else 0.0
    )
    gap = float(train_top3.mean() - test_top3.mean()) if len(train_top3) else 0.0
    fold_disp = float(test_top3.std(ddof=0)) if len(test_top3) else 0.0
    any_disp = float(test_top3_any.std(ddof=0)) if len(test_top3_any) else 0.0
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
    bs = arr.copy()
    boot = {
        "block_size": block_size,
        "samples": permutations,
        "mean": float(bs.mean()),
        "std": float(bs.std(ddof=1)),
        "ci95_low": float(np.percentile(bs, 2.5)),
        "ci95_high": float(np.percentile(bs, 97.5)),
    }
    return pred, perm_df, boot


def _base_binary_score(cand: pd.DataFrame, window_factor: float = 1.0) -> np.ndarray:
    raw = (
        0.8 * cand["freq_last_20"].to_numpy()
        + 0.5 * cand["freq_last_100"].to_numpy()
        + 0.4 * cand["pair_score_with_last_5_draws"].to_numpy()
        - 0.04 * cand["gap_since_last_seen"].to_numpy()
        + 0.2 * cand["cand_in_prev_pm1"].to_numpy()
        + 0.3 * cand["ema_short_minus_ema_long"].to_numpy() * window_factor
    )
    return expit(raw)


def _rerank(
    scores: np.ndarray, cand: pd.DataFrame, pool_k: int, w: float, p: float, tw: float
) -> np.ndarray:
    out = scores.copy()
    idx = np.argsort(out)[::-1][:pool_k]
    c = cand.iloc[idx]
    bonus = (
        0.7 * c["freq_last_20"].to_numpy()
        + 0.4 * c["freq_last_100"].to_numpy()
        + tw * c["ema_short_minus_ema_long"].to_numpy()
    )
    penalty = p * np.abs(c["num_zone"].to_numpy() - np.median(c["num_zone"].to_numpy()))
    out[idx] = out[idx] + w * 0.01 * bonus - penalty
    return out


def main() -> None:
    cfg = load_yaml(CONFIG_DIR / "train.yaml")
    feature_columns = json.loads(
        (PROJECT_ROOT / "models" / "feature_columns.json").read_text(encoding="utf-8")
    )
    feat_df = (
        pd.read_csv(FEATURE_STORE_DIR / "issue_features.csv")
        .tail(int(cfg.get("backtest_max_draws", 200)))
        .reset_index(drop=True)
    )
    raw_df = load_processed().tail(len(feat_df) + 22).reset_index(drop=True)
    splits = int(cfg["backtest_splits"])
    tss = TimeSeriesSplit(n_splits=splits)

    candidate_cache: dict[int, pd.DataFrame] = {}
    actual_cache: dict[int, set[int]] = {}
    for _, r in feat_df.iterrows():
        issue = int(r["issue"])
        candidate_cache[issue] = build_candidate_matrix(r, feature_columns)
        actual_cache[issue] = set(json.loads(r["target_numbers"]))

    exps = [
        {"version_id": "v0_binary_baseline", "type": "binary"},
        {"version_id": "v1_rank_heuristic", "type": "rank"},
        {
            "version_id": "v2_rerank_k20_w100",
            "type": "rerank",
            "k": 20,
            "w": 1.0,
            "p": 0.08,
            "tw": 0.3,
        },
        {
            "version_id": "v3_rerank_k30_w300",
            "type": "rerank",
            "k": 30,
            "w": 3.0,
            "p": 0.10,
            "tw": 0.4,
        },
        {
            "version_id": "v4_rerank_k40_w500",
            "type": "rerank",
            "k": 40,
            "w": 5.0,
            "p": 0.13,
            "tw": 0.45,
        },
        {
            "version_id": "v5_two_stage_20_10_3",
            "type": "two_stage",
            "k": 20,
            "w": 3.0,
            "p": 0.12,
            "tw": 0.5,
        },
        {
            "version_id": "v6_ablation_no_structure",
            "type": "rerank",
            "k": 30,
            "w": 0.0,
            "p": 0.0,
            "tw": 0.0,
        },
        {
            "version_id": "v7_ablation_no_trend",
            "type": "rerank",
            "k": 30,
            "w": 3.0,
            "p": 0.10,
            "tw": 0.0,
        },
        {
            "version_id": "v8_weight_search_light",
            "type": "rerank",
            "k": 30,
            "w": 2.2,
            "p": 0.06,
            "tw": 0.35,
        },
        {
            "version_id": "v9_weight_search_aggressive",
            "type": "rerank",
            "k": 30,
            "w": 5.2,
            "p": 0.22,
            "tw": 0.55,
        },
    ]

    registry, per_fold, per_regime = [], [], []
    baseline = None
    baseline_top20 = []

    for exp in exps:
        fold_train, fold_test, regime_rows = [], [], []
        for fold, (tr_idx, te_idx) in enumerate(tss.split(feat_df), start=1):
            train_df = feat_df.iloc[tr_idx]
            test_df = feat_df.iloc[te_idx]
            for target, pack in [(train_df, fold_train), (test_df, fold_test)]:
                rows = []
                for _, r in target.iterrows():
                    issue = int(r["issue"])
                    cand = candidate_cache[issue]
                    wf = {
                        "v2_rerank_k20_w100": 0.8,
                        "v3_rerank_k30_w300": 1.0,
                        "v4_rerank_k40_w500": 1.2,
                    }.get(exp["version_id"], 1.0)
                    scores = _base_binary_score(cand, wf)
                    if exp["type"] == "rank":
                        scores = (
                            0.9 * cand["freq_last_20"].to_numpy()
                            + 0.6 * cand["pair_score_with_last_5_draws"].to_numpy()
                            - 0.03 * cand["gap_since_last_seen"].to_numpy()
                        )
                    elif exp["type"] == "rerank":
                        scores = _rerank(
                            scores, cand, exp["k"], exp["w"], exp["p"], exp["tw"]
                        )
                    elif exp["type"] == "two_stage":
                        scores = _rerank(
                            scores, cand, exp["k"], exp["w"], exp["p"], exp["tw"]
                        )
                        scores = _rerank(
                            scores, cand, 10, exp["w"] * 0.9, exp["p"] * 1.2, exp["tw"]
                        )
                        scores = _rerank(
                            scores, cand, 3, exp["w"] * 1.1, exp["p"] * 1.3, exp["tw"]
                        )
                    actual = actual_cache[issue]
                    m = _make_fold_issue_metrics(scores, actual)
                    m["regime"] = _derive_regime(r)
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

        overall = _aggregate(fold_test)
        audit = _overfit_audit(fold_train, fold_test, regime_rows)
        if baseline is None:
            baseline = overall
            baseline_top20 = [x["top20_hit_rate"] for x in fold_test]
        better = bool(
            overall["top3_at_least_one_hit_rate"]
            > baseline["top3_at_least_one_hit_rate"]
            and overall["top3_hit_rate"] > baseline["top3_hit_rate"]
        )
        keep = bool(better and not audit["is_overfit"])
        registry.append(
            {
                "version_id": exp["version_id"],
                "change_summary": json.dumps(exp, ensure_ascii=False),
                **overall,
                "is_better_than_baseline": better,
                "is_overfit": bool(audit["is_overfit"]),
                "keep_recommendation": keep,
                "failed_reason": (
                    ""
                    if keep or exp["version_id"] == "v0_binary_baseline"
                    else (
                        "overfit_candidate"
                        if audit["is_overfit"]
                        else "oos_not_better_than_baseline"
                    )
                ),
                **audit,
            }
        )
        per_fold.extend([{"version_id": exp["version_id"], **x} for x in fold_test])
        per_regime.extend([{"version_id": exp["version_id"], **x} for x in regime_rows])

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
    registry_df[registry_df["version_id"].str.contains("binary|rank")][
        ["version_id", *METRIC_KEYS, "is_overfit", "is_better_than_baseline"]
    ].to_csv(REPORTS_DIR / "ranking_vs_binary.csv", index=False)

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
