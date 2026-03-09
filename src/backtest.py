from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
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


def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_true - y_prob) ** 2))


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    order = np.argsort(y_score)[::-1][:k]
    gains = y_true[order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(y_true)[::-1][:k]
    idcg = float(np.sum(ideal * discounts))
    return 0.0 if idcg == 0 else dcg / idcg


def _labels(actual: set[int]) -> np.ndarray:
    return np.array([1 if n in actual else 0 for n in range(1, 81)], dtype=float)


def _normalize(arr: np.ndarray) -> np.ndarray:
    a = np.array(arr, dtype=float)
    if np.allclose(a.sum(), 0):
        return np.full_like(a, 1 / len(a))
    return np.clip(a / a.sum(), 1e-6, 1 - 1e-6)


def _build_freq_baseline(
    history_numbers: list[list[int]], window: int | None = None
) -> np.ndarray:
    if not history_numbers:
        return np.full(80, 1 / 80)
    draws = history_numbers if window is None else history_numbers[-window:]
    freq = np.zeros(80, dtype=float)
    for draw in draws:
        for n in draw:
            freq[n - 1] += 1
    return _normalize(freq)


def main() -> None:
    cfg = load_yaml(CONFIG_DIR / "train.yaml")
    feature_columns = json.loads(
        open("models/feature_columns.json", encoding="utf-8").read()
    )
    df = pd.read_csv(FEATURE_STORE_DIR / "issue_features.csv").reset_index(drop=True)
    tss = TimeSeriesSplit(n_splits=int(cfg["backtest_splits"]))

    fold_rows = []
    global_metrics: dict[str, list[float]] = {
        "model_top20": [],
        "model_top10": [],
        "model_top3": [],
        "model_ndcg20": [],
        "model_logloss": [],
        "model_brier": [],
        "uniform_top20": [],
        "global_freq_top20": [],
        "recent_freq_top20": [],
    }

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

        model = lgb.LGBMClassifier(**cfg["lgbm_params"])
        model.fit(x_train, y_train)

        fold_model_top20 = []
        fold_model_top10 = []
        fold_model_top3 = []
        fold_model_ndcg20 = []
        fold_model_logloss = []
        fold_model_brier = []
        fold_uniform_top20 = []
        fold_global_freq_top20 = []
        fold_recent_freq_top20 = []

        for _, row in test_df.iterrows():
            actual = set(json.loads(row["target_numbers"]))
            y_true = _labels(actual)
            x_test = build_candidate_matrix(row, feature_columns)
            scores = np.clip(model.predict_proba(x_test)[:, 1], 1e-6, 1 - 1e-6)
            h20, h10, h3 = _top_hits(scores, actual)
            fold_model_top20.append(h20 / 20)
            fold_model_top10.append(h10 / 10)
            fold_model_top3.append(h3 / 3)
            fold_model_ndcg20.append(_ndcg_at_k(y_true, scores, 20))
            fold_model_logloss.append(float(log_loss(y_true, scores, labels=[0, 1])))
            fold_model_brier.append(_brier_score(y_true, scores))

            history_numbers = json.loads(row.get("history_numbers", "[]"))
            uniform = np.full(80, 1 / 80)
            global_freq = _build_freq_baseline(history_numbers)
            recent_freq = _build_freq_baseline(history_numbers, window=20)

            fold_uniform_top20.append(_top_hits(uniform, actual)[0] / 20)
            fold_global_freq_top20.append(_top_hits(global_freq, actual)[0] / 20)
            fold_recent_freq_top20.append(_top_hits(recent_freq, actual)[0] / 20)

        fold_rows.append(
            {
                "fold": fold,
                "train_size": int(len(train_df)),
                "test_size": int(len(test_df)),
                "model_top20_hit_rate": float(np.mean(fold_model_top20)),
                "model_top10_hit_rate": float(np.mean(fold_model_top10)),
                "model_top3_hit_rate": float(np.mean(fold_model_top3)),
                "model_ndcg20": float(np.mean(fold_model_ndcg20)),
                "model_logloss": float(np.mean(fold_model_logloss)),
                "model_brier": float(np.mean(fold_model_brier)),
                "uniform_top20_hit_rate": float(np.mean(fold_uniform_top20)),
                "global_freq_top20_hit_rate": float(np.mean(fold_global_freq_top20)),
                "recent_freq_top20_hit_rate": float(np.mean(fold_recent_freq_top20)),
            }
        )

        global_metrics["model_top20"].extend(fold_model_top20)
        global_metrics["model_top10"].extend(fold_model_top10)
        global_metrics["model_top3"].extend(fold_model_top3)
        global_metrics["model_ndcg20"].extend(fold_model_ndcg20)
        global_metrics["model_logloss"].extend(fold_model_logloss)
        global_metrics["model_brier"].extend(fold_model_brier)
        global_metrics["uniform_top20"].extend(fold_uniform_top20)
        global_metrics["global_freq_top20"].extend(fold_global_freq_top20)
        global_metrics["recent_freq_top20"].extend(fold_recent_freq_top20)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_df = pd.DataFrame(fold_rows)
    report_df.to_csv(REPORTS_DIR / "walkforward_report.csv", index=False)
    report_df.to_csv(REPORTS_DIR / "fold_metrics.csv", index=False)

    summary = {
        "splits": int(cfg["backtest_splits"]),
        "top20_hit_rate": float(np.mean(global_metrics["model_top20"])),
        "top10_hit_rate": float(np.mean(global_metrics["model_top10"])),
        "top3_hit_rate": float(np.mean(global_metrics["model_top3"])),
        "ndcg@20": float(np.mean(global_metrics["model_ndcg20"])),
        "logloss": float(np.mean(global_metrics["model_logloss"])),
        "brier": float(np.mean(global_metrics["model_brier"])),
    }
    save_json(REPORTS_DIR / "backtest_metrics.json", summary)

    baseline_comparison = {
        "model_top20_hit_rate": summary["top20_hit_rate"],
        "uniform_top20_hit_rate": float(np.mean(global_metrics["uniform_top20"])),
        "global_frequency_top20_hit_rate": float(
            np.mean(global_metrics["global_freq_top20"])
        ),
        "recent_frequency_top20_hit_rate": float(
            np.mean(global_metrics["recent_freq_top20"])
        ),
    }
    save_json(REPORTS_DIR / "baseline_comparison.json", baseline_comparison)
    save_json(
        REPORTS_DIR / "calibration_metrics.json",
        {
            "method": "none",
            "ece": None,
            "note": "calibration pipeline not enabled yet",
        },
    )
    print("backtest completed")


if __name__ == "__main__":
    main()
