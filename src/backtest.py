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

        fold_top20 = []
        fold_top10 = []
        fold_top3 = []
        for _, row in test_df.iterrows():
            actual = set(json.loads(row["target_numbers"]))
            x_test = build_candidate_matrix(row, feature_columns)
            scores = model.predict_proba(x_test)[:, 1]
            h20, h10, h3 = _top_hits(scores, actual)
            fold_top20.append(h20 / 20)
            fold_top10.append(h10 / 10)
            fold_top3.append(h3 / 3)

        fold_rows.append(
            {
                "fold": fold,
                "train_size": int(len(train_df)),
                "test_size": int(len(test_df)),
                "top20_hit_rate": float(np.mean(fold_top20)),
                "top10_hit_rate": float(np.mean(fold_top10)),
                "top3_hit_rate": float(np.mean(fold_top3)),
            }
        )
        total_top20.extend(fold_top20)
        total_top10.extend(fold_top10)
        total_top3.extend(fold_top3)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_df = pd.DataFrame(fold_rows)
    report_df.to_csv(REPORTS_DIR / "walkforward_report.csv", index=False)
    save_json(
        REPORTS_DIR / "backtest_metrics.json",
        {
            "splits": int(cfg["backtest_splits"]),
            "top20_hit_rate": float(np.mean(total_top20)),
            "top10_hit_rate": float(np.mean(total_top10)),
            "top3_hit_rate": float(np.mean(total_top3)),
        },
    )
    print("backtest completed")


if __name__ == "__main__":
    main()
