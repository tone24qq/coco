from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, ndcg_score
from sklearn.model_selection import TimeSeriesSplit

from src.utils import (  # noqa: E402
    CONFIG_DIR,
    FEATURE_STORE_DIR,
    REPORTS_DIR,
    build_candidate_matrix,
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


def _recent_frequency_baseline(train_rows: pd.DataFrame) -> np.ndarray:
    counts = np.zeros(80, dtype=float)
    for _, row in train_rows.iterrows():
        target = set(json.loads(row["target_numbers"]))
        for n in target:
            counts[n - 1] += 1
    return counts / max(counts.sum(), 1.0)


def main() -> None:
    cfg = load_yaml(CONFIG_DIR / "train.yaml")
    feature_columns = json.loads(
        open("models/feature_columns.json", encoding="utf-8").read()
    )
    df = pd.read_csv(FEATURE_STORE_DIR / "issue_features.csv").reset_index(drop=True)
    tss = TimeSeriesSplit(n_splits=int(cfg["backtest_splits"]))

    fold_rows = []
    total_top20 = []
    total_top10 = []
    total_top3 = []
    total_logloss = []
    total_brier = []
    total_ndcg20 = []
    baseline_rows = []

    for fold, (train_idx, test_idx) in enumerate(tss.split(df), start=1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        x_blocks = []
        y_blocks = []
        for _, row in train_df.iterrows():
            y_true = set(json.loads(row["target_numbers"]))
            x_blocks.append(build_candidate_matrix(row, feature_columns))
            y_blocks.append(pd.Series([1 if n in y_true else 0 for n in range(1, 81)]))
        x_train = pd.concat(x_blocks, ignore_index=True)
        y_train = pd.concat(y_blocks, ignore_index=True)

        params = cfg.get("catboost_params", {})
        params.setdefault("loss_function", "Logloss")
        params.setdefault("verbose", False)
        params.setdefault("random_seed", 42)
        model = CatBoostClassifier(**params)
        model.fit(x_train, y_train)
        global_freq = _recent_frequency_baseline(train_df)
        uniform = np.ones(80, dtype=float) / 80.0

        fold_top20 = []
        fold_top10 = []
        fold_top3 = []
        fold_logloss = []
        fold_brier = []
        fold_ndcg20 = []
        fold_uniform_top20 = []
        fold_global_freq_top20 = []
        for _, row in test_df.iterrows():
            actual = set(json.loads(row["target_numbers"]))
            y_true = _labels_from_actual(actual)
            x_test = build_candidate_matrix(row, feature_columns)
            scores = model.predict_proba(x_test)[:, 1]
            h20, h10, h3 = _top_hits(scores, actual)
            fold_top20.append(h20 / 20)
            fold_top10.append(h10 / 10)
            fold_top3.append(h3 / 3)
            fold_logloss.append(log_loss(y_true, np.clip(scores, 1e-6, 1 - 1e-6)))
            fold_brier.append(brier_score_loss(y_true, scores))
            fold_ndcg20.append(ndcg_score([y_true], [scores], k=20))

            u20, _, _ = _top_hits(uniform, actual)
            g20, _, _ = _top_hits(global_freq, actual)
            fold_uniform_top20.append(u20 / 20)
            fold_global_freq_top20.append(g20 / 20)

        fold_rows.append(
            {
                "fold": fold,
                "train_size": int(len(train_df)),
                "test_size": int(len(test_df)),
                "top20_hit_rate": float(np.mean(fold_top20)),
                "top10_hit_rate": float(np.mean(fold_top10)),
                "top3_hit_rate": float(np.mean(fold_top3)),
                "logloss": float(np.mean(fold_logloss)),
                "brier_score": float(np.mean(fold_brier)),
                "ndcg@20": float(np.mean(fold_ndcg20)),
            }
        )
        total_top20.extend(fold_top20)
        total_top10.extend(fold_top10)
        total_top3.extend(fold_top3)
        total_logloss.extend(fold_logloss)
        total_brier.extend(fold_brier)
        total_ndcg20.extend(fold_ndcg20)
        baseline_rows.append(
            {
                "fold": fold,
                "model_top20_hit_rate": float(np.mean(fold_top20)),
                "uniform_top20_hit_rate": float(np.mean(fold_uniform_top20)),
                "global_freq_top20_hit_rate": float(np.mean(fold_global_freq_top20)),
            }
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_df = pd.DataFrame(fold_rows)
    report_df.to_csv(REPORTS_DIR / "walkforward_report.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(REPORTS_DIR / "fold_metrics.csv", index=False)
    save_json(REPORTS_DIR / "baseline_comparison.json", {"folds": baseline_rows})
    save_json(
        REPORTS_DIR / "backtest_metrics.json",
        {
            "splits": int(cfg["backtest_splits"]),
            "top20_hit_rate": float(np.mean(total_top20)),
            "top10_hit_rate": float(np.mean(total_top10)),
            "top3_hit_rate": float(np.mean(total_top3)),
            "logloss": float(np.mean(total_logloss)),
            "brier_score": float(np.mean(total_brier)),
            "ndcg@20": float(np.mean(total_ndcg20)),
        },
    )
    print("backtest completed")


if __name__ == "__main__":
    main()
