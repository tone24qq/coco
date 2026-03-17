from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from scipy.stats import t
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analysis.snapshots import load_history_snapshot_payload  # noqa: E402
from src.pipeline import CascadePipeline  # noqa: E402
from src.ranking_dataset import (  # noqa: E402
    build_ranker_training_rows,
    split_ranker_training_frame,
)
from src.runtime_scoring import (  # noqa: E402
    RUNTIME_SCORE_REQUIRED_COLUMNS,
    score_candidates_runtime,
)
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
    apply_local_peak_correction,
    apply_topk_group_dedup,
    build_issue_features,
    build_latest_issue_features_for_inference,
    load_processed,
    load_yaml,
    normalize_pipeline_version,
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
        experiments = default_experiments()
    else:
        payload = load_yaml(exp_cfg_path)
        experiments = [StrategyConfig(**row) for row in payload.get("experiments", [])]
    if not any(exp.version_id == "ranker_main_qsm" for exp in experiments):
        experiments.append(
            StrategyConfig(
                version_id="ranker_main_qsm",
                stage_type="ranker_main",
                pipeline_version="baseline_flat_score",
                stage1_keep=30,
                stage2_keep=10,
                candidate_pool=20,
                prior_window=100,
                rerank_weight=0.0,
                penalty_weight=0.0,
                trend_weight=0.0,
                regime_aware=False,
            )
        )
    return experiments


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


def _fit_cascade_pipeline(
    feat_df: pd.DataFrame,
    indices: list[int],
    params: dict,
    stage1_keep: int,
    stage2_keep: int,
) -> CascadePipeline:
    local_df = feat_df.iloc[indices].reset_index(drop=True)
    pipeline, _ = CascadePipeline.train(
        local_df,
        stage1_keep=stage1_keep,
        stage2_keep=stage2_keep,
        catboost_params=params,
    )
    return pipeline


def _load_runtime_scoring_bundle() -> dict:
    predict_cfg = load_yaml(CONFIG_DIR / "predict.yaml")
    metadata_path = PROJECT_ROOT / "models" / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    runtime_cfg = dict(metadata.get("runtime_config", {}))
    for key in [
        "history_prior",
        "analysis_rerank",
        "long_feature_injection",
        "neighbor_peak_correction",
        "topk_group_dedup",
        "soft_label_training",
        "proximity_model",
    ]:
        if key in predict_cfg:
            runtime_cfg[key] = predict_cfg.get(key, {})

    soft_model = None
    pm1_model = None
    soft_path = PROJECT_ROOT / "models" / "catboost_soft_ce.cbm"
    pm1_path = PROJECT_ROOT / "models" / "catboost_pm1_proximity.cbm"
    if soft_path.exists():
        try:
            soft_model = CatBoostClassifier()
            soft_model.load_model(str(soft_path))
        except Exception:
            soft_model = None
    if pm1_path.exists():
        try:
            pm1_model = CatBoostClassifier()
            pm1_model.load_model(str(pm1_path))
        except Exception:
            pm1_model = None

    snapshot_payload = load_history_snapshot_payload()
    board_priors = snapshot_payload.get("meta", {}).get("board_priors", {})
    return {
        "runtime_config": runtime_cfg,
        "snapshot_payload": snapshot_payload,
        "board_priors": board_priors,
        "soft_model": soft_model,
        "pm1_model": pm1_model,
    }


def _score_issue_with_runtime_pipeline(
    payload: dict[str, object],
    scores: np.ndarray,
    runtime_bundle: dict,
):
    cand_for_runtime = payload["cand"].copy().reset_index(drop=True)
    if "number" not in cand_for_runtime.columns:
        cand_for_runtime.insert(0, "number", np.arange(1, 81, dtype=int))
    issue_row = payload["issue_row"]
    recent_draws = [
        sorted(x) for x in json.loads(str(issue_row.get("history_numbers", "[]")))
    ]
    soft_raw = (
        runtime_bundle["soft_model"].predict_proba(cand_for_runtime)[:, 1]
        if runtime_bundle.get("soft_model") is not None
        else None
    )
    pm1_raw = (
        runtime_bundle["pm1_model"].predict_proba(cand_for_runtime)[:, 1]
        if runtime_bundle.get("pm1_model") is not None
        else None
    )
    outputs = score_candidates_runtime(
        base_scores=np.array(scores, dtype=float),
        candidate_df=cand_for_runtime,
        recent_draws=recent_draws,
        runtime_config=runtime_bundle["runtime_config"],
        snapshot_payload=runtime_bundle["snapshot_payload"],
        board_priors=runtime_bundle["board_priors"],
        soft_label_raw=soft_raw,
        pm1_proximity_raw=pm1_raw,
    )
    for col in RUNTIME_SCORE_REQUIRED_COLUMNS:
        if col not in outputs.score_table.columns:
            outputs.score_table[col] = 0.0
    return outputs


def _run_experiments(
    feat_df: pd.DataFrame,
    splits: int,
    experiments: list[StrategyConfig],
    params: dict,
    issue_payloads: dict[int, dict[str, object]],
    runtime_bundle: dict,
) -> tuple[list[dict], list[dict], list[dict], list[float], list[dict]]:
    tss = TimeSeriesSplit(n_splits=splits)
    registry, per_fold, per_regime, per_issue = [], [], [], []
    baseline = None
    baseline_top20 = []

    for exp in experiments:
        print(f"[版本開始] {exp.version_id}")
        fold_train, fold_test, regime_rows = [], [], []
        for fold, (tr_idx, te_idx) in enumerate(tss.split(feat_df), start=1):
            model = None
            ranker_model = None
            if exp.stage_type == "ranker_main":
                rank_rows = build_ranker_training_rows(
                    issue_payloads,
                    list(tr_idx),
                    [
                        c
                        for c in issue_payloads[int(tr_idx[0])]["cand"].columns
                        if c != "number"
                    ],
                )
                rank_feature_cols = [
                    c
                    for c in rank_rows.columns
                    if c not in {"issue", "number", "label", "group_id"}
                ]
                rank_x, rank_y, rank_gid = split_ranker_training_frame(
                    rank_rows, rank_feature_cols
                )
                ranker_params = dict(params)
                ranker_params["loss_function"] = "QuerySoftMax"
                ranker_params["eval_metric"] = "NDCG:top=10"
                ranker_params["custom_metric"] = [
                    "NDCG:top=3",
                    "NDCG:top=10",
                    "PrecisionAt:top=3",
                    "PrecisionAt:top=10",
                    "RecallAt:top=3",
                    "RecallAt:top=10",
                ]
                ranker_model = CatBoostRanker(**ranker_params)
                ranker_model.fit(
                    Pool(rank_x, label=rank_y, group_id=rank_gid), verbose=False
                )
            elif exp.stage_type != "cascade":
                x_train, y_train = _expand_rows(issue_payloads, list(tr_idx))
                model = CatBoostClassifier(**params)
                model.fit(x_train, y_train, verbose=False)
            cascade_pipeline = None
            if exp.stage_type == "cascade":
                cascade_pipeline = _fit_cascade_pipeline(
                    feat_df,
                    list(tr_idx),
                    params=params,
                    stage1_keep=exp.stage1_keep,
                    stage2_keep=exp.stage2_keep,
                )

            for idx_set, pack in [
                (tr_idx[-min(50, len(tr_idx)) :], fold_train),
                (te_idx, fold_test),
            ]:
                rows = []
                for row_idx in idx_set:
                    payload = issue_payloads[int(row_idx)]
                    cand = payload["cand"]
                    regime = payload["regime"]
                    if regime is None:
                        regime = derive_regime(feat_df.iloc[int(row_idx)])
                        payload["regime"] = regime
                    stage_meta: dict[str, object] = {}
                    if exp.stage_type == "cascade":
                        if cascade_pipeline is None:
                            raise ValueError("cascade pipeline not available")
                        cascade = cascade_pipeline.predict_issue(payload["issue_row"])
                        scores = cascade["final_scores"]
                        target = set(payload["target"])
                        stage1_df = cascade["stage1"]
                        stage2_df = cascade["stage2"]
                        stage1_pool = set(
                            int(x)
                            for x in stage1_df[stage1_df["stage1_keep_flag"] == 1][
                                "number"
                            ].tolist()
                        )
                        stage2_pool = set(
                            int(x)
                            for x in stage2_df[stage2_df["stage2_keep_flag"] == 1][
                                "number"
                            ].tolist()
                        )
                        no_selector_top3 = [int(x) for x in cascade["no_selector_top3"]]
                        sel_top3 = [int(x) for x in cascade["final_top3"]]

                        def _top3_diag(top3_vals: list[int]) -> dict[str, float]:
                            min_d = [
                                min(abs(n - a) for a in target) if target else 80.0
                                for n in top3_vals
                            ]
                            return {
                                "exact": float(
                                    sum(1 for n in top3_vals if n in target) / 3.0
                                ),
                                "at_least_one": float(
                                    any(n in target for n in top3_vals)
                                ),
                                "adj": float(
                                    sum(
                                        1
                                        for n in top3_vals
                                        if any(abs(n - a) <= 1 for a in target)
                                    )
                                    / 3.0
                                ),
                                "strict_adj_only": float(
                                    sum(
                                        1
                                        for n in top3_vals
                                        if n not in target
                                        and any(abs(n - a) == 1 for a in target)
                                    )
                                    / 3.0
                                ),
                                "mean_min_distance": float(np.mean(min_d)),
                            }

                        sel_diag = _top3_diag(sel_top3)
                        no_sel_diag = _top3_diag(no_selector_top3)
                        rel10 = [
                            1.0 if int(n) in target else 0.0
                            for n in stage2_df.sort_values(
                                "stage2_score", ascending=False
                            )["number"]
                            .head(10)
                            .tolist()
                        ]
                        discounts = 1.0 / np.log2(np.arange(2, 12, dtype=float))
                        dcg = float(np.sum(np.array(rel10, dtype=float) * discounts))
                        ideal_rel = np.array(
                            [1.0] * min(len(target), 10)
                            + [0.0] * max(0, 10 - len(target))
                        )
                        idcg = float(np.sum(ideal_rel * discounts))
                        stage_meta = {
                            "stage1_recall_at_30": float(
                                len(stage1_pool & target) / 20.0
                            ),
                            "stage1_retained_actual_count": int(
                                len(stage1_pool & target)
                            ),
                            "stage2_top10_hit_rate": float(
                                len(stage2_pool & target) / 10.0
                            ),
                            "stage2_ndcg_at_10": float(dcg / idcg) if idcg > 0 else 0.0,
                            "stage3_selector_exact_hit_at_3": float(sel_diag["exact"]),
                            "stage3_no_selector_exact_hit_at_3": float(
                                no_sel_diag["exact"]
                            ),
                            "stage3_selector_top3_at_least_one": float(
                                sel_diag["at_least_one"]
                            ),
                            "stage3_no_selector_top3_at_least_one": float(
                                no_sel_diag["at_least_one"]
                            ),
                            "stage3_selector_adj_hit_pm1_at_3": float(sel_diag["adj"]),
                            "stage3_no_selector_adj_hit_pm1_at_3": float(
                                no_sel_diag["adj"]
                            ),
                            "stage3_selector_strict_adj_only_pm1_at_3": float(
                                sel_diag["strict_adj_only"]
                            ),
                            "stage3_no_selector_strict_adj_only_pm1_at_3": float(
                                no_sel_diag["strict_adj_only"]
                            ),
                            "stage3_selector_mean_min_distance_at_3": float(
                                sel_diag["mean_min_distance"]
                            ),
                            "stage3_no_selector_mean_min_distance_at_3": float(
                                no_sel_diag["mean_min_distance"]
                            ),
                            "selector_uplift_exact_hit_at_3": float(
                                sel_diag["exact"] - no_sel_diag["exact"]
                            ),
                            "selector_uplift_adj_hit_pm1_at_3": float(
                                sel_diag["adj"] - no_sel_diag["adj"]
                            ),
                            "selector_uplift_strict_adj_only_pm1_at_3": float(
                                sel_diag["strict_adj_only"]
                                - no_sel_diag["strict_adj_only"]
                            ),
                            "selector_uplift_mean_min_distance_at_3": float(
                                sel_diag["mean_min_distance"]
                                - no_sel_diag["mean_min_distance"]
                            ),
                        }
                    else:
                        if exp.stage_type == "ranker_main":
                            if ranker_model is None:
                                raise ValueError("ranker model not available")
                            scores = ranker_model.predict(cand)
                        else:
                            if model is None:
                                raise ValueError("legacy model not available")
                            base_scores = model.predict_proba(cand)[:, 1]
                            scores = apply_strategy(base_scores, cand, exp, regime)
                    runtime_outputs = _score_issue_with_runtime_pipeline(
                        payload=payload,
                        scores=np.array(scores, dtype=float),
                        runtime_bundle=runtime_bundle,
                    )
                    runtime_table = runtime_outputs.score_table.copy()
                    for col in RUNTIME_SCORE_REQUIRED_COLUMNS:
                        if col not in runtime_table.columns:
                            runtime_table[col] = 0.0
                    final_scores_arr = np.zeros(80, dtype=float)
                    for rec in runtime_table[["number", "final_score"]].to_dict(
                        orient="records"
                    ):
                        final_scores_arr[int(rec["number"]) - 1] = float(
                            rec["final_score"]
                        )

                    m = _make_fold_issue_metrics(final_scores_arr, payload["target"])
                    m["regime"] = regime
                    m.update(stage_meta)
                    rows.append(m)

                    if list(idx_set) == list(te_idx):
                        pred_top10 = (
                            runtime_table["number"].head(10).astype(int).tolist()
                        )
                        pred_top3 = runtime_outputs.dedup_summary.get(
                            "top3_after_group_dedup", pred_top10[:3]
                        )
                        if exp.stage_type == "cascade":
                            pred_top3 = [int(x) for x in cascade["final_top3"]]
                        feat_row = feat_df.iloc[int(row_idx)]
                        actual = sorted(int(x) for x in payload["target"])
                        prev_numbers = sorted(
                            int(x)
                            for x in json.loads(str(feat_row.get("prev_numbers", "[]")))
                        )
                        per_issue.append(
                            {
                                "version_id": exp.version_id,
                                "fold": fold,
                                "regime": regime,
                                "issue": int(feat_row["issue"]),
                                "history_length": int(
                                    len(
                                        json.loads(
                                            str(feat_row.get("history_numbers", "[]"))
                                        )
                                    )
                                ),
                                "pred_top3": pred_top3,
                                "pred_top3_no_selector": (
                                    [int(x) for x in cascade["no_selector_top3"]]
                                    if exp.stage_type == "cascade"
                                    else pred_top3
                                ),
                                "pred_top10": pred_top10,
                                "actual": actual,
                                "prev_numbers": prev_numbers,
                                "score_table": runtime_table.to_dict(orient="records"),
                                "runtime_dedup_summary": runtime_outputs.dedup_summary,
                                **stage_meta,
                            }
                        )

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
                    if exp.version_id == "v0_binary_baseline":
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

    return registry, per_fold, per_regime, baseline_top20, per_issue


def _bucket_label(history_len: int) -> str:
    if history_len <= 20:
        return "1-20"
    if history_len <= 50:
        return "21-50"
    if history_len <= 100:
        return "51-100"
    if history_len <= 200:
        return "101-200"
    return "201+"


def _build_history_bucket_report(issue_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in issue_rows.iterrows():
        pred_top10 = [int(x) for x in row.get("pred_top10", [])]
        pred_top3 = [int(x) for x in row.get("pred_top3", [])]
        actual = set(int(x) for x in row.get("actual", []))
        prev = set(int(x) for x in row.get("prev_numbers", []))
        if not pred_top3:
            continue
        min_dist_to_actual = [
            min(abs(n - a) for a in actual) if actual else 80.0 for n in pred_top3
        ]
        min_dist_to_prev = [
            min(abs(n - p) for p in prev) if prev else 80.0 for n in pred_top3
        ]
        rows.append(
            {
                "history_bucket": _bucket_label(int(row["history_length"])),
                "exact_hit@3": float(sum(1 for n in pred_top3 if n in actual) / 3.0),
                "exact_hit@10": float(
                    sum(1 for n in pred_top10 if n in actual)
                    / max(1.0, float(len(pred_top10)))
                ),
                "top3_at_least_one_exact": float(any(n in actual for n in pred_top3)),
                "adj_hit_pm1@3": float(
                    sum(1 for n in pred_top3 if any(abs(n - a) <= 1 for a in actual))
                    / 3.0
                ),
                "adj_hit_pm1@10": float(
                    sum(1 for n in pred_top10 if any(abs(n - a) <= 1 for a in actual))
                    / max(1.0, float(len(pred_top10)))
                ),
                "adj_hit_pm2@3": float(
                    sum(1 for n in pred_top3 if any(abs(n - a) <= 2 for a in actual))
                    / 3.0
                ),
                "strict_adj_only_pm1@3": float(
                    sum(
                        1
                        for n in pred_top3
                        if n not in actual and any(abs(n - a) == 1 for a in actual)
                    )
                    / 3.0
                ),
                "strict_pm1_error_rate_at_3": float(
                    sum(1 for n in pred_top3 if any(abs(n - a) == 1 for a in actual))
                    / 3.0
                ),
                "strict_pm2_error_rate_at_3": float(
                    sum(1 for n in pred_top3 if any(abs(n - a) == 2 for a in actual))
                    / 3.0
                ),
                "exact_or_pm1_rate_at_3": float(
                    sum(1 for n in pred_top3 if any(abs(n - a) <= 1 for a in actual))
                    / 3.0
                ),
                "mean_min_distance_at_3": float(np.mean(min_dist_to_actual)),
                "over_shoot_rate_at_3": float(
                    np.mean(
                        [
                            (
                                1.0
                                if (
                                    actual and min(actual, key=lambda a: abs(a - n)) < n
                                )
                                else 0.0
                            )
                            for n in pred_top3
                        ]
                    )
                ),
                "under_shoot_rate_at_3": float(
                    np.mean(
                        [
                            (
                                1.0
                                if (
                                    actual and min(actual, key=lambda a: abs(a - n)) > n
                                )
                                else 0.0
                            )
                            for n in pred_top3
                        ]
                    )
                ),
                "top3_prev_draw_mean_min_distance": float(np.mean(min_dist_to_prev)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    metric_cols = [c for c in out.columns if c != "history_bucket"]
    return out.groupby("history_bucket")[metric_cols].mean().reset_index()


def _build_error_shift_report(issue_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if issue_rows.empty:
        return pd.DataFrame(), {"rows": []}
    plus1: dict[str, int] = {}
    minus1: dict[str, int] = {}
    zone_rows: list[dict] = []
    feature_hits = {
        "history_length_gt_100_ratio": 0,
        "prev_draw_overlap_ge_1_ratio": 0,
    }
    total_cases = 0

    for _, row in issue_rows.iterrows():
        pred_top3 = [int(x) for x in row.get("pred_top3", [])]
        actual = [int(x) for x in row.get("actual", [])]
        if not pred_top3 or not actual:
            continue
        total_cases += 1
        prev = set(int(x) for x in row.get("prev_numbers", []))

        if int(row.get("history_length", 0)) > 100:
            feature_hits["history_length_gt_100_ratio"] += 1
        if any(n in prev for n in pred_top3):
            feature_hits["prev_draw_overlap_ge_1_ratio"] += 1

        for n in pred_top3:
            nearest = min(actual, key=lambda a: abs(a - n))
            diff = int(n - nearest)
            if diff == 1:
                plus1[str(nearest)] = plus1.get(str(nearest), 0) + 1
            if diff == -1:
                minus1[str(nearest)] = minus1.get(str(nearest), 0) + 1
            zone_rows.append(
                {
                    "zone": (
                        "A" if n <= 20 else "B" if n <= 40 else "C" if n <= 60 else "D"
                    ),
                    "pm1_proximity": float(abs(diff) <= 1),
                    "strict_pm1_error": float(abs(diff) == 1),
                }
            )

    zone_df = pd.DataFrame(zone_rows)
    zone_pm1_prox = []
    zone_strict_pm1 = []
    if not zone_df.empty:
        zone_pm1_prox = (
            zone_df.groupby("zone")["pm1_proximity"]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        )
        zone_strict_pm1 = (
            zone_df.groupby("zone")["strict_pm1_error"]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        )

    summary = {
        "most_predicted_as_n_plus_1": sorted(
            plus1.items(), key=lambda x: x[1], reverse=True
        )[:20],
        "most_predicted_as_n_minus_1": sorted(
            minus1.items(), key=lambda x: x[1], reverse=True
        )[:20],
        "zone_pm1_proximity_rate": zone_pm1_prox,
        "zone_strict_pm1_error_rate": zone_strict_pm1,
        "history_length_gt_100_ratio": float(
            feature_hits["history_length_gt_100_ratio"] / max(1, total_cases)
        ),
        "prev_draw_overlap_ge_1_ratio": float(
            feature_hits["prev_draw_overlap_ge_1_ratio"] / max(1, total_cases)
        ),
    }

    rows: list[dict] = []
    for num, cnt in summary["most_predicted_as_n_plus_1"]:
        rows.append(
            {"pattern": "n_plus_1", "actual_number": int(num), "count": int(cnt)}
        )
    for num, cnt in summary["most_predicted_as_n_minus_1"]:
        rows.append(
            {"pattern": "n_minus_1", "actual_number": int(num), "count": int(cnt)}
        )
    for rec in zone_pm1_prox:
        rows.append(
            {
                "pattern": "zone_pm1_proximity_rate",
                "zone": rec["zone"],
                "rate": float(rec["pm1_proximity"]),
            }
        )
    for rec in zone_strict_pm1:
        rows.append(
            {
                "pattern": "zone_strict_pm1_error_rate",
                "zone": rec["zone"],
                "rate": float(rec["strict_pm1_error"]),
            }
        )

    return pd.DataFrame(rows), summary


def _ablation_report_from_issue_rows(issue_rows: pd.DataFrame) -> dict:
    if issue_rows.empty:
        return {"experiments": [], "comparisons": []}

    predict_cfg = load_yaml(CONFIG_DIR / "predict.yaml")
    local_cfg = dict(predict_cfg.get("neighbor_peak_correction", {}))
    dedup_cfg = dict(predict_cfg.get("topk_group_dedup", {}))
    soft_cfg = dict(predict_cfg.get("soft_label_training", {}))
    pm1_cfg = dict(predict_cfg.get("proximity_model", {}))

    def _metrics(top3: list[int], top10: list[int], actual: list[int]) -> dict:
        a = set(int(x) for x in actual)
        min_dist = [min(abs(n - x) for x in a) if a else 80.0 for n in top3]
        return {
            "exact_hit@3": float(sum(1 for n in top3 if n in a) / 3.0),
            "exact_hit@10": float(
                sum(1 for n in top10 if n in a) / max(1.0, float(len(top10)))
            ),
            "top3_at_least_one_exact": float(any(n in a for n in top3)),
            "adj_hit_pm1@3": float(
                sum(1 for n in top3 if any(abs(n - x) <= 1 for x in a)) / 3.0
            ),
            "adj_hit_pm1@10": float(
                sum(1 for n in top10 if any(abs(n - x) <= 1 for x in a))
                / max(1.0, float(len(top10)))
            ),
            "adj_hit_pm2@3": float(
                sum(1 for n in top3 if any(abs(n - x) <= 2 for x in a)) / 3.0
            ),
            "mean_min_distance_at_3": float(np.mean(min_dist)),
            "strict_pm1_error_rate_at_3": float(
                sum(1 for n in top3 if any(abs(n - x) == 1 for x in a)) / 3.0
            ),
        }

    experiments: dict[str, list[dict]] = {
        "exact_only": [],
        "exact_plus_local_peak": [],
        "exact_plus_group_dedup": [],
        "exact_plus_local_peak_plus_group_dedup": [],
        "exact_plus_soft_label": [],
        "exact_plus_pm1_proximity": [],
        "exact_plus_soft_label_plus_pm1_proximity": [],
        "full_runtime_chain": [],
    }
    skipped_reasons: dict[str, str] = {}

    for _, row in issue_rows.iterrows():
        raw_table = row.get("score_table", [])
        actual = [int(x) for x in row.get("actual", [])]
        if not raw_table or not actual:
            continue
        df = pd.DataFrame(raw_table)
        if df.empty or "number" not in df.columns:
            continue

        base = df.copy()
        if "model_score" not in base.columns:
            base["model_score"] = base.get("final_score", 0.0).astype(float)
        if "soft_label_score" not in base.columns:
            base["soft_label_score"] = 0.0
        if "pm1_proximity_score" not in base.columns:
            base["pm1_proximity_score"] = 0.0
        base["final_score"] = base["model_score"].astype(float)
        base_rank = base.sort_values("final_score", ascending=False).reset_index(
            drop=True
        )
        exact_top10 = base_rank["number"].head(10).astype(int).tolist()
        exact_top3 = base_rank["number"].head(3).astype(int).tolist()
        experiments["exact_only"].append(_metrics(exact_top3, exact_top10, actual))

        lp_df, _ = apply_local_peak_correction(
            base_rank,
            cfg={**local_cfg, "enabled": True},
            input_score_column="final_score",
            output_score_column="score_after_local_peak",
        )
        lp_df["final_score"] = lp_df["score_after_local_peak"].astype(float)
        lp_rank = lp_df.sort_values("final_score", ascending=False).reset_index(
            drop=True
        )
        lp_top10 = lp_rank["number"].head(10).astype(int).tolist()
        lp_top3 = lp_rank["number"].head(3).astype(int).tolist()
        experiments["exact_plus_local_peak"].append(_metrics(lp_top3, lp_top10, actual))

        gd_rank, gd_summary = apply_topk_group_dedup(
            base_rank,
            cfg={**dedup_cfg, "enabled": True, "apply_to_top3_only": False},
            top_k=3,
        )
        gd_top10 = gd_rank["number"].head(10).astype(int).tolist()
        gd_top3 = gd_summary["top3_after_group_dedup"]
        experiments["exact_plus_group_dedup"].append(
            _metrics(gd_top3, gd_top10, actual)
        )

        lpgd_rank, lpgd_summary = apply_topk_group_dedup(
            lp_rank,
            cfg={**dedup_cfg, "enabled": True, "apply_to_top3_only": False},
            top_k=3,
        )
        lpgd_top10 = lpgd_rank["number"].head(10).astype(int).tolist()
        lpgd_top3 = lpgd_summary["top3_after_group_dedup"]
        experiments["exact_plus_local_peak_plus_group_dedup"].append(
            _metrics(lpgd_top3, lpgd_top10, actual)
        )

        soft_weight = float(soft_cfg.get("blend_weight", 0.15))
        soft_enabled = (
            bool(soft_cfg.get("enabled", False))
            and float(df["soft_label_score"].abs().sum()) > 0.0
        )
        if soft_enabled:
            soft_df = df.copy()
            soft_df["final_score"] = soft_df["model_score"].astype(
                float
            ) + soft_weight * soft_df["soft_label_score"].astype(float)
            soft_rank = soft_df.sort_values("final_score", ascending=False).reset_index(
                drop=True
            )
            experiments["exact_plus_soft_label"].append(
                _metrics(
                    soft_rank["number"].head(3).astype(int).tolist(),
                    soft_rank["number"].head(10).astype(int).tolist(),
                    actual,
                )
            )
        else:
            skipped_reasons.setdefault(
                "exact_plus_soft_label", "soft_label disabled or artifact missing"
            )

        pm1_weight = float(pm1_cfg.get("pm1_weight", 0.12))
        pm1_enabled = (
            bool(pm1_cfg.get("enabled", False))
            and float(df["pm1_proximity_score"].abs().sum()) > 0.0
        )
        if pm1_enabled:
            pm1_df = df.copy()
            pm1_df["final_score"] = pm1_df["model_score"].astype(
                float
            ) + pm1_weight * pm1_df["pm1_proximity_score"].astype(float)
            pm1_rank = pm1_df.sort_values("final_score", ascending=False).reset_index(
                drop=True
            )
            experiments["exact_plus_pm1_proximity"].append(
                _metrics(
                    pm1_rank["number"].head(3).astype(int).tolist(),
                    pm1_rank["number"].head(10).astype(int).tolist(),
                    actual,
                )
            )
        else:
            skipped_reasons.setdefault(
                "exact_plus_pm1_proximity", "pm1 proximity disabled or artifact missing"
            )

        if soft_enabled and pm1_enabled:
            sp_df = df.copy()
            sp_df["final_score"] = (
                sp_df["model_score"].astype(float)
                + soft_weight * sp_df["soft_label_score"].astype(float)
                + pm1_weight * sp_df["pm1_proximity_score"].astype(float)
            )
            sp_rank = sp_df.sort_values("final_score", ascending=False).reset_index(
                drop=True
            )
            experiments["exact_plus_soft_label_plus_pm1_proximity"].append(
                _metrics(
                    sp_rank["number"].head(3).astype(int).tolist(),
                    sp_rank["number"].head(10).astype(int).tolist(),
                    actual,
                )
            )
        else:
            skipped_reasons.setdefault(
                "exact_plus_soft_label_plus_pm1_proximity",
                "soft_label or pm1 proximity disabled/missing",
            )

        full_rank = df.sort_values("final_score", ascending=False).reset_index(
            drop=True
        )
        full_top3 = (
            row.get("pred_top3") or full_rank["number"].head(3).astype(int).tolist()
        )
        full_top10 = (
            row.get("pred_top10") or full_rank["number"].head(10).astype(int).tolist()
        )
        experiments["full_runtime_chain"].append(
            _metrics(
                [int(x) for x in full_top3], [int(x) for x in full_top10][:10], actual
            )
        )

    rows = []
    for name, vals in experiments.items():
        if vals:
            rows.append(
                {
                    "name": name,
                    "skipped_reason": "",
                    **pd.DataFrame(vals).mean().to_dict(),
                }
            )
        else:
            rows.append(
                {"name": name, "skipped_reason": skipped_reasons.get(name, "no_data")}
            )

    by_name = {r["name"]: r for r in rows}
    baseline = by_name.get("exact_only", {})
    comparisons = []
    for name in [
        "exact_plus_local_peak",
        "exact_plus_group_dedup",
        "exact_plus_soft_label",
        "exact_plus_pm1_proximity",
        "full_runtime_chain",
    ]:
        cur = by_name.get(name, {})
        if baseline.get("exact_hit@3") is None or cur.get("exact_hit@3") is None:
            continue
        comparisons.append(
            {
                "compare": f"exact_only vs {name}",
                "delta_exact_hit@3": float(
                    cur.get("exact_hit@3", 0.0) - baseline.get("exact_hit@3", 0.0)
                ),
                "delta_mean_min_distance_at_3": float(
                    cur.get("mean_min_distance_at_3", 0.0)
                    - baseline.get("mean_min_distance_at_3", 0.0)
                ),
            }
        )

    return {"experiments": rows, "comparisons": comparisons}


def _build_stagewise_uplift_report(per_issue_df: pd.DataFrame) -> dict:
    if per_issue_df.empty:
        return {}
    out: dict[str, object] = {}

    def _window(df: pd.DataFrame, name: str) -> None:
        if df.empty:
            return
        cols = [
            "stage1_recall_at_30",
            "stage1_retained_actual_count",
            "stage2_top10_hit_rate",
            "stage3_selector_exact_hit_at_3",
            "stage3_no_selector_exact_hit_at_3",
            "stage3_selector_adj_hit_pm1_at_3",
            "stage3_no_selector_adj_hit_pm1_at_3",
            "stage3_selector_strict_adj_only_pm1_at_3",
            "stage3_no_selector_strict_adj_only_pm1_at_3",
            "stage3_selector_mean_min_distance_at_3",
            "stage3_no_selector_mean_min_distance_at_3",
            "selector_uplift_exact_hit_at_3",
            "selector_uplift_adj_hit_pm1_at_3",
            "selector_uplift_strict_adj_only_pm1_at_3",
            "selector_uplift_mean_min_distance_at_3",
            "stage3_selector_top3_at_least_one",
            "stage3_no_selector_top3_at_least_one",
        ]
        avail = [c for c in cols if c in df.columns]
        out[name] = {c: float(df[c].mean()) for c in avail}

    cascade_rows = per_issue_df[
        per_issue_df["version_id"].astype(str).str.contains("cascade")
    ]
    _window(cascade_rows, "full_window")
    _window(cascade_rows.tail(100), "recent_100")
    _window(cascade_rows.tail(300), "recent_300")

    if "regime" in cascade_rows.columns and not cascade_rows.empty:
        regime_cols = [
            c
            for c in [
                "stage1_recall_at_30",
                "stage2_top10_hit_rate",
                "stage3_selector_exact_hit_at_3",
                "selector_uplift_exact_hit_at_3",
            ]
            if c in cascade_rows.columns
        ]
        out["regime_bucket"] = (
            cascade_rows.groupby("regime")[regime_cols]
            .mean()
            .reset_index()
            .to_dict(orient="records")
            if regime_cols
            else []
        )

    if "version_id" in per_issue_df.columns and "actual" in per_issue_df.columns:
        version_rows = []
        for ver, g in per_issue_df.groupby("version_id"):
            exact3 = []
            one3 = []
            for _, row in g.iterrows():
                actual = set(int(x) for x in row.get("actual", []))
                top3 = [int(x) for x in row.get("pred_top3", [])]
                if not top3:
                    continue
                exact3.append(sum(1 for n in top3 if n in actual) / 3.0)
                one3.append(float(any(n in actual for n in top3)))
            if exact3:
                version_rows.append(
                    {
                        "version_id": str(ver),
                        "exact_hit@3": float(np.mean(exact3)),
                        "top3_at_least_one_exact": float(np.mean(one3)),
                    }
                )
        out["version_compare"] = version_rows

    if not cascade_rows.empty and "pred_top3_no_selector" in cascade_rows.columns:
        exact_sel = []
        exact_no_sel = []
        adj_sel = []
        adj_no_sel = []
        strict_sel = []
        strict_no_sel = []
        dist_sel = []
        dist_no_sel = []
        for _, row in cascade_rows.iterrows():
            actual = set(int(x) for x in row.get("actual", []))
            top3_sel = [int(x) for x in row.get("pred_top3", [])]
            top3_no_sel = [int(x) for x in row.get("pred_top3_no_selector", [])]
            if len(top3_sel) == 3 and len(top3_no_sel) == 3:
                exact_sel.append(sum(1 for n in top3_sel if n in actual) / 3.0)
                exact_no_sel.append(sum(1 for n in top3_no_sel if n in actual) / 3.0)
                adj_sel.append(
                    sum(1 for n in top3_sel if any(abs(n - a) <= 1 for a in actual))
                    / 3.0
                )
                adj_no_sel.append(
                    sum(1 for n in top3_no_sel if any(abs(n - a) <= 1 for a in actual))
                    / 3.0
                )
                strict_sel.append(
                    sum(
                        1
                        for n in top3_sel
                        if n not in actual and any(abs(n - a) == 1 for a in actual)
                    )
                    / 3.0
                )
                strict_no_sel.append(
                    sum(
                        1
                        for n in top3_no_sel
                        if n not in actual and any(abs(n - a) == 1 for a in actual)
                    )
                    / 3.0
                )
                dist_sel.append(
                    float(np.mean([min(abs(n - a) for a in actual) for n in top3_sel]))
                    if actual
                    else 80.0
                )
                dist_no_sel.append(
                    float(
                        np.mean([min(abs(n - a) for a in actual) for n in top3_no_sel])
                    )
                    if actual
                    else 80.0
                )
        if exact_sel and exact_no_sel:
            out["cascade_selector_vs_no_selector"] = {
                "cascade_v1_without_selector_exact_hit@3": float(np.mean(exact_no_sel)),
                "cascade_v1_with_selector_exact_hit@3": float(np.mean(exact_sel)),
                "uplift_exact_hit@3": float(np.mean(exact_sel) - np.mean(exact_no_sel)),
                "cascade_v1_without_selector_adj_hit_pm1@3": float(np.mean(adj_no_sel)),
                "cascade_v1_with_selector_adj_hit_pm1@3": float(np.mean(adj_sel)),
                "uplift_adj_hit_pm1@3": float(np.mean(adj_sel) - np.mean(adj_no_sel)),
                "cascade_v1_without_selector_strict_adj_only_pm1@3": float(
                    np.mean(strict_no_sel)
                ),
                "cascade_v1_with_selector_strict_adj_only_pm1@3": float(
                    np.mean(strict_sel)
                ),
                "uplift_strict_adj_only_pm1@3": float(
                    np.mean(strict_sel) - np.mean(strict_no_sel)
                ),
                "cascade_v1_without_selector_mean_min_distance_at_3": float(
                    np.mean(dist_no_sel)
                ),
                "cascade_v1_with_selector_mean_min_distance_at_3": float(
                    np.mean(dist_sel)
                ),
                "uplift_mean_min_distance_at_3": float(
                    np.mean(dist_sel) - np.mean(dist_no_sel)
                ),
            }

    if "history_length" in cascade_rows.columns:
        h = cascade_rows.copy()
        h["history_bucket"] = h["history_length"].astype(int).map(_bucket_label)
        cols = [
            col
            for col in [
                "selector_uplift_exact_hit_at_3",
                "selector_uplift_adj_hit_pm1_at_3",
                "selector_uplift_strict_adj_only_pm1_at_3",
                "selector_uplift_mean_min_distance_at_3",
            ]
            if col in h.columns
        ]
        if cols:
            out["history_bucket_selector_uplift"] = (
                h.groupby("history_bucket")[cols]
                .mean()
                .reset_index()
                .to_dict(orient="records")
            )

    return out


def main() -> None:
    cfg = load_yaml(CONFIG_DIR / "train.yaml")
    pipeline_cfg = cfg.get("pipeline", {})
    normalize_pipeline_version(pipeline_cfg.get("version", "baseline_flat_score"))
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
        "cascade_v1_flow",
        "ranker_main_qsm",
    }
    fast_experiments = [
        exp for exp in experiments if exp.version_id in fast_version_ids
    ]
    print("[研究流程] backtest 快速階段：3個版本、3 folds、較低 iterations")
    fast_params = dict(params)
    fast_params["iterations"] = int(cfg.get("research_iterations", 140))
    runtime_bundle = _load_runtime_scoring_bundle()
    fast_registry, _, _, _, _ = _run_experiments(
        feat_df=feat_df,
        splits=int(cfg.get("research_backtest_splits", 3)),
        experiments=fast_experiments,
        params=fast_params,
        issue_payloads=issue_payloads,
        runtime_bundle=runtime_bundle,
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
    if (
        "ranker_main_qsm" in fast_df["version_id"].tolist()
        and "ranker_main_qsm" not in selected_final_ids
    ):
        selected_final_ids.append("ranker_main_qsm")
    final_experiments = [
        exp for exp in experiments if exp.version_id in set(selected_final_ids)
    ]
    print(f"[研究流程] backtest 正式階段：版本={selected_final_ids}、{splits} folds")
    registry, per_fold, per_regime, baseline_top20, per_issue_rows = _run_experiments(
        feat_df=feat_df,
        splits=splits,
        experiments=final_experiments,
        params=params,
        issue_payloads=issue_payloads,
        runtime_bundle=runtime_bundle,
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
    per_issue_df = pd.DataFrame(per_issue_rows)
    selected_issue_df = (
        per_issue_df[per_issue_df["version_id"] == best["version_id"]]
        if "version_id" in per_issue_df.columns
        else pd.DataFrame()
    )
    history_bucket_df = _build_history_bucket_report(selected_issue_df)
    history_bucket_df.to_csv(REPORTS_DIR / "history_bucket_report.csv", index=False)
    save_json(
        REPORTS_DIR / "history_bucket_report.json",
        {"rows": history_bucket_df.to_dict(orient="records")},
    )
    stagewise_uplift = _build_stagewise_uplift_report(per_issue_df)
    save_json(REPORTS_DIR / "cascade_stagewise_report.json", stagewise_uplift)
    error_shift_csv, error_shift_json = _build_error_shift_report(selected_issue_df)
    error_shift_csv.to_csv(REPORTS_DIR / "error_shift_analysis.csv", index=False)
    save_json(REPORTS_DIR / "error_shift_analysis.json", error_shift_json)
    ablation_summary = _ablation_report_from_issue_rows(selected_issue_df)
    save_json(REPORTS_DIR / "ablation_shift_analysis.json", ablation_summary)
    save_json(REPORTS_DIR / "runtime_ablation_summary.json", ablation_summary)
    pd.DataFrame(ablation_summary.get("experiments", [])).to_csv(
        REPORTS_DIR / "runtime_ablation_summary.csv", index=False
    )

    classifier_rows = registry_df[registry_df["version_id"] == "v0_binary_baseline"]
    classifier_baseline_summary = (
        classifier_rows.iloc[0].to_dict() if not classifier_rows.empty else baseline_row
    )
    ranker_rows = registry_df[registry_df["stage_type"] == "ranker_main"]
    ranker_main_summary = (
        ranker_rows.sort_values(
            ["top3_at_least_one_hit_rate", "top3_hit_rate"], ascending=False
        )
        .iloc[0]
        .to_dict()
        if not ranker_rows.empty
        else {}
    )
    save_json(
        REPORTS_DIR / "experiment_summary.json",
        {
            "feature_version": feature_version,
            "baseline": baseline_row,
            "best_version": best,
            "model_family_best": str(best.get("stage_type", "baseline")),
            "classifier_baseline_summary": classifier_baseline_summary,
            "ranker_main_summary": ranker_main_summary,
            "selected_formal_strategy": str(
                best.get("version_id", "v0_binary_baseline")
            ),
            "selected_formal_model_family": str(best.get("stage_type", "baseline")),
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
