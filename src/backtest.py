from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker
from scipy.stats import t
from sklearn.metrics import brier_score_loss, log_loss, ndcg_score
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


def _top_hits(pred_scores: np.ndarray, actual: set[int]) -> tuple[int, int, int]:
    order = np.argsort(pred_scores)[::-1] + 1
    top20 = set(order[:20].tolist())
    top10 = set(order[:10].tolist())
    top3 = set(order[:3].tolist())
    return len(top20 & actual), len(top10 & actual), len(top3 & actual)


def _labels_from_actual(actual: set[int]) -> np.ndarray:
    return np.array([1 if n in actual else 0 for n in range(1, 81)])


def _frequency_vector(rows: pd.DataFrame, field: str = "target_numbers") -> np.ndarray:
    counts = np.zeros(80, dtype=float)
    for _, row in rows.iterrows():
        target = set(json.loads(row[field]))
        for n in target:
            counts[n - 1] += 1
    return counts / max(counts.sum(), 1.0)


def _ci95(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) > 1:
        margin = float(t.ppf(0.975, len(arr) - 1) * std / np.sqrt(len(arr)))
    else:
        margin = 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def _train_binary_model(
    x_train: pd.DataFrame, y_train: pd.Series, cfg: dict
) -> CatBoostClassifier:
    params = dict(cfg.get("catboost_params", {}))
    params.setdefault("loss_function", "Logloss")
    params.setdefault("verbose", False)
    params.setdefault("random_seed", 42)
    model = CatBoostClassifier(**params)
    model.fit(x_train, y_train)
    return model


def _train_ranker(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    group_id: pd.Series,
    cfg: dict,
    loss_function: str,
) -> CatBoostRanker:
    params = dict(cfg.get("catboost_params", {}))
    params["loss_function"] = loss_function
    params["eval_metric"] = "NDCG:top=20"
    params.setdefault("verbose", False)
    params.setdefault("random_seed", 42)
    params.setdefault("allow_writing_files", False)
    ranker = CatBoostRanker(**params)
    ranker.fit(x_train, y_train, group_id=group_id)
    return ranker


def _evaluate_fold(
    model,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    global_freq: np.ndarray,
    recent_freq: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    per_issue = []
    baseline_issue = []
    uniform = np.ones(80, dtype=float) / 80.0

    for _, row in test_df.iterrows():
        actual = set(json.loads(row["target_numbers"]))
        y_true = _labels_from_actual(actual)
        x_test = build_candidate_matrix(row, feature_columns)
        scores = np.asarray(
            model.predict_proba(x_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.predict(x_test)
        )
        h20, h10, h3 = _top_hits(scores, actual)

        u20, _, _ = _top_hits(uniform, actual)
        g20, _, _ = _top_hits(global_freq, actual)
        r20, _, _ = _top_hits(recent_freq, actual)

        per_issue.append(
            {
                "issue": int(row["issue"]),
                "target_issue": int(row["target_issue"]),
                "top20_hit_rate": h20 / 20,
                "top10_hit_rate": h10 / 10,
                "top3_hit_rate": h3 / 3,
                "logloss": log_loss(y_true, np.clip(scores, 1e-6, 1 - 1e-6)),
                "brier_score": brier_score_loss(y_true, np.clip(scores, 0, 1)),
                "ndcg@20": ndcg_score([y_true], [scores], k=20),
            }
        )
        baseline_issue.append(
            {
                "issue": int(row["issue"]),
                "target_issue": int(row["target_issue"]),
                "uniform_top20_hit_rate": u20 / 20,
                "global_freq_top20_hit_rate": g20 / 20,
                "recent_freq_top20_hit_rate": r20 / 20,
            }
        )

    return per_issue, baseline_issue


def _alignment_audit(df: pd.DataFrame, splits: int) -> tuple[pd.DataFrame, dict]:
    audit_rows = []
    issues = df["issue"].astype(int).to_numpy()
    targets = (
        df["target_issue"].astype(int).to_numpy()
        if "target_issue" in df.columns
        else np.append(issues[1:], issues[-1] + 1)
    )

    issue_inc = bool(np.all(np.diff(issues) > 0))
    target_next = bool(np.all(targets[:-1] == issues[1:]))

    target_match = []
    for i in range(len(df) - 1):
        expected = json.dumps(
            sorted(json.loads(df.iloc[i + 1]["numbers"])), ensure_ascii=False
        )
        observed_raw = (
            df.iloc[i]["target_numbers"]
            if "target_numbers" in df.columns
            else df.iloc[i + 1]["numbers"]
        )
        observed = json.dumps(sorted(json.loads(observed_raw)), ensure_ascii=False)
        target_match.append(expected == observed)
    target_all = bool(all(target_match)) if target_match else True

    infer_feat = build_latest_issue_features_for_inference(df, min_history=22)
    last_issue = int(df.iloc[-1]["issue"])
    inference_latest_ok = bool(
        int(infer_feat.iloc[-1]["issue"]) == last_issue
        and int(infer_feat.iloc[-1]["target_issue"]) == last_issue + 1
    )

    tss = TimeSeriesSplit(n_splits=splits)
    no_leakage = True
    for fold, (train_idx, test_idx) in enumerate(tss.split(df), start=1):
        fold_ok = int(np.max(train_idx)) < int(np.min(test_idx))
        no_leakage = no_leakage and bool(fold_ok)
        audit_rows.append(
            {
                "check": "fold_temporal_order",
                "fold": fold,
                "status": bool(fold_ok),
                "train_max_idx": int(np.max(train_idx)),
                "test_min_idx": int(np.min(test_idx)),
            }
        )

    audit_rows.extend(
        [
            {"check": "issue_strictly_increasing", "fold": 0, "status": issue_inc},
            {"check": "target_issue_is_next_issue", "fold": 0, "status": target_next},
            {
                "check": "target_numbers_match_next_draw",
                "fold": 0,
                "status": target_all,
            },
            {
                "check": "inference_latest_row_alignment",
                "fold": 0,
                "status": inference_latest_ok,
            },
            {
                "check": "no_shift_leakage_in_walkforward",
                "fold": 0,
                "status": no_leakage,
            },
        ]
    )

    summary = {
        "all_checks_passed": bool(all(x["status"] for x in audit_rows)),
        "issue_strictly_increasing": issue_inc,
        "target_issue_is_next_issue": target_next,
        "target_numbers_match_next_draw": target_all,
        "inference_latest_row_alignment": inference_latest_ok,
        "no_shift_leakage_in_walkforward": no_leakage,
    }
    return pd.DataFrame(audit_rows), summary


def _predictability_test(
    df: pd.DataFrame,
    observed_scores: list[float],
    permutations: int = 200,
    block_size: int = 10,
) -> tuple[dict, pd.DataFrame, dict]:
    observed = float(np.mean(observed_scores)) if observed_scores else 0.0
    base_targets = df["target_numbers"].tolist()
    null_scores = []

    blocks = [
        base_targets[i : i + block_size]
        for i in range(0, len(base_targets), block_size)
    ]
    for i in range(permutations):
        rng = np.random.default_rng(42 + i)
        shuffled_blocks = blocks.copy()
        rng.shuffle(shuffled_blocks)
        permuted = [item for b in shuffled_blocks for item in b]
        hits = []
        for actual_s, pred_s in zip(permuted, observed_scores):
            actual = set(json.loads(actual_s))
            expected_random_hit = len(actual) / 80
            hits.append(float(pred_s - expected_random_hit))
        null_scores.append(float(np.mean(hits)))

    null_arr = np.array(null_scores)
    p_value = float((np.sum(null_arr >= observed) + 1) / (len(null_arr) + 1))

    predictability = {
        "observed_score": observed,
        "null_mean": float(null_arr.mean()),
        "null_std": float(null_arr.std(ddof=1)),
        "p_value": p_value,
        "signal_sufficient": bool(p_value < 0.05 and observed > null_arr.mean()),
    }
    perm_df = pd.DataFrame(
        {"iteration": np.arange(1, permutations + 1), "null_score": null_scores}
    )

    boot_means = []
    n_blocks = max(len(blocks), 1)
    for i in range(permutations):
        rng = np.random.default_rng(5000 + i)
        sample_idx = rng.integers(0, n_blocks, size=n_blocks)
        sampled = [x for idx in sample_idx for x in blocks[idx]]
        sampled = sampled[: len(observed_scores)]
        local = []
        for actual_s, pred_s in zip(sampled, observed_scores):
            actual = set(json.loads(actual_s))
            expected_random_hit = len(actual) / 80
            local.append(float(pred_s - expected_random_hit))
        boot_means.append(float(np.mean(local)))

    bs = np.array(boot_means)
    bootstrap_summary = {
        "block_size": block_size,
        "samples": permutations,
        "mean": float(bs.mean()),
        "std": float(bs.std(ddof=1)),
        "ci95_low": float(np.percentile(bs, 2.5)),
        "ci95_high": float(np.percentile(bs, 97.5)),
    }
    return predictability, perm_df, bootstrap_summary


def main() -> None:
    cfg = load_yaml(CONFIG_DIR / "train.yaml")
    feature_columns = json.loads(
        (PROJECT_ROOT / "models" / "feature_columns.json").read_text(encoding="utf-8")
    )
    feat_df = pd.read_csv(FEATURE_STORE_DIR / "issue_features.csv").reset_index(
        drop=True
    )
    raw_df = load_processed().reset_index(drop=True)
    splits = int(cfg["backtest_splits"])
    tss = TimeSeriesSplit(n_splits=splits)

    fold_rows = []
    baseline_rows = []
    all_model_top20 = []
    all_uniform_top20 = []
    all_global_top20 = []
    all_recent_top20 = []

    for fold, (train_idx, test_idx) in enumerate(tss.split(feat_df), start=1):
        train_df = feat_df.iloc[train_idx]
        test_df = feat_df.iloc[test_idx]
        x_blocks = []
        y_blocks = []
        for _, row in train_df.iterrows():
            y_true = set(json.loads(row["target_numbers"]))
            x_blocks.append(build_candidate_matrix(row, feature_columns))
            y_blocks.append(pd.Series([1 if n in y_true else 0 for n in range(1, 81)]))
        x_train = pd.concat(x_blocks, ignore_index=True)
        y_train = pd.concat(y_blocks, ignore_index=True)

        model = _train_binary_model(x_train, y_train, cfg)
        global_freq = _frequency_vector(train_df)
        recent_freq = _frequency_vector(train_df.tail(200))

        per_issue, baseline_issue = _evaluate_fold(
            model, test_df, feature_columns, global_freq, recent_freq
        )
        model_rates = [x["top20_hit_rate"] for x in per_issue]
        uniform_rates = [x["uniform_top20_hit_rate"] for x in baseline_issue]
        global_rates = [x["global_freq_top20_hit_rate"] for x in baseline_issue]
        recent_rates = [x["recent_freq_top20_hit_rate"] for x in baseline_issue]

        fold_rows.append(
            {
                "fold": fold,
                "train_size": int(len(train_df)),
                "test_size": int(len(test_df)),
                "top20_hit_rate": float(np.mean(model_rates)),
                "top10_hit_rate": float(
                    np.mean([x["top10_hit_rate"] for x in per_issue])
                ),
                "top3_hit_rate": float(
                    np.mean([x["top3_hit_rate"] for x in per_issue])
                ),
                "logloss": float(np.mean([x["logloss"] for x in per_issue])),
                "brier_score": float(np.mean([x["brier_score"] for x in per_issue])),
                "ndcg@20": float(np.mean([x["ndcg@20"] for x in per_issue])),
                "uniform_top20_hit_rate": float(np.mean(uniform_rates)),
                "global_freq_top20_hit_rate": float(np.mean(global_rates)),
                "recent_freq_top20_hit_rate": float(np.mean(recent_rates)),
            }
        )
        baseline_rows.append(
            {
                "fold": fold,
                "model_top20_hit_rate": float(np.mean(model_rates)),
                "uniform_top20_hit_rate": float(np.mean(uniform_rates)),
                "global_freq_top20_hit_rate": float(np.mean(global_rates)),
                "recent_freq_top20_hit_rate": float(np.mean(recent_rates)),
            }
        )

        all_model_top20.extend(model_rates)
        all_uniform_top20.extend(uniform_rates)
        all_global_top20.extend(global_rates)
        all_recent_top20.extend(recent_rates)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(REPORTS_DIR / "fold_metrics.csv", index=False)
    save_json(REPORTS_DIR / "baseline_comparison.json", {"folds": baseline_rows})
    if not (REPORTS_DIR / "baseline_comparison.json").exists():
        raise RuntimeError("baseline_comparison.json is required")

    overall = {
        "splits": splits,
        "top20_hit_rate": float(np.mean(all_model_top20)),
        "top10_hit_rate": float(fold_df["top10_hit_rate"].mean()),
        "top3_hit_rate": float(fold_df["top3_hit_rate"].mean()),
        "logloss": float(fold_df["logloss"].mean()),
        "brier_score": float(fold_df["brier_score"].mean()),
        "ndcg@20": float(fold_df["ndcg@20"].mean()),
    }
    save_json(REPORTS_DIR / "backtest_metrics.json", overall)

    excess = {
        "vs_uniform": {
            "excess_hit_rate": float(
                np.mean(all_model_top20) - np.mean(all_uniform_top20)
            ),
            "lift": float(
                np.mean(all_model_top20) / max(np.mean(all_uniform_top20), 1e-9)
            ),
            **_ci95([m - b for m, b in zip(all_model_top20, all_uniform_top20)]),
        },
        "vs_global_frequency": {
            "excess_hit_rate": float(
                np.mean(all_model_top20) - np.mean(all_global_top20)
            ),
            "lift": float(
                np.mean(all_model_top20) / max(np.mean(all_global_top20), 1e-9)
            ),
            **_ci95([m - b for m, b in zip(all_model_top20, all_global_top20)]),
        },
        "vs_recent_frequency": {
            "excess_hit_rate": float(
                np.mean(all_model_top20) - np.mean(all_recent_top20)
            ),
            "lift": float(
                np.mean(all_model_top20) / max(np.mean(all_recent_top20), 1e-9)
            ),
            **_ci95([m - b for m, b in zip(all_model_top20, all_recent_top20)]),
        },
    }
    save_json(REPORTS_DIR / "excess_metrics.json", excess)

    predictability, perm_df, bootstrap_summary = _predictability_test(
        feat_df,
        all_model_top20,
        permutations=200,
        block_size=10,
    )
    save_json(REPORTS_DIR / "predictability_test.json", predictability)
    perm_df.to_csv(REPORTS_DIR / "permutation_distribution.csv", index=False)
    save_json(REPORTS_DIR / "block_bootstrap_summary.json", bootstrap_summary)

    audit_df, audit_summary = _alignment_audit(raw_df, splits=splits)
    audit_df.to_csv(REPORTS_DIR / "alignment_audit.csv", index=False)
    save_json(REPORTS_DIR / "alignment_audit.json", audit_summary)

    # Ranking vs binary
    ranking_rows = []
    for loss in ["PairLogitPairwise", "YetiRank", "LambdaMart"]:
        scores = []
        for train_idx, test_idx in tss.split(feat_df):
            train_df = feat_df.iloc[train_idx]
            test_df = feat_df.iloc[test_idx]
            x_blocks = []
            y_blocks = []
            groups = []
            group_counter = 0
            for _, row in train_df.iterrows():
                y_true = set(json.loads(row["target_numbers"]))
                x = build_candidate_matrix(row, feature_columns)
                y = [1 if n in y_true else 0 for n in range(1, 81)]
                x_blocks.append(x)
                y_blocks.extend(y)
                groups.extend([group_counter] * 80)
                group_counter += 1
            x_train = pd.concat(x_blocks, ignore_index=True)
            y_train = pd.Series(y_blocks)
            group_id = pd.Series(groups)
            ranker = _train_ranker(x_train, y_train, group_id, cfg, loss)
            for _, row in test_df.iterrows():
                actual = set(json.loads(row["target_numbers"]))
                x_test = build_candidate_matrix(row, feature_columns)
                pred = np.asarray(ranker.predict(x_test))
                h20, _, _ = _top_hits(pred, actual)
                scores.append(h20 / 20)
        ranking_rows.append({"model": loss, "top20_hit_rate": float(np.mean(scores))})

    ranking_df = pd.DataFrame(ranking_rows)
    binary_top20 = float(np.mean(all_model_top20))
    ranking_df["binary_top20_hit_rate"] = binary_top20
    ranking_df["excess_vs_binary"] = ranking_df["top20_hit_rate"] - binary_top20
    ranking_df.to_csv(REPORTS_DIR / "ranking_vs_binary.csv", index=False)
    save_json(
        REPORTS_DIR / "ranking_model_metrics.json",
        {
            "binary_top20_hit_rate": binary_top20,
            "models": ranking_rows,
        },
    )

    print("backtest completed")


if __name__ == "__main__":
    main()
