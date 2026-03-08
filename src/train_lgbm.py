from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json
from datetime import datetime, timezone

import lightgbm as lgb  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils import (  # noqa: E402
    CONFIG_DIR,
    FEATURE_STORE_DIR,
    MODELS_DIR,
    build_candidate_matrix,
    load_yaml,
    save_json,
)


def _expand_training_rows(
    feature_df: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    x_blocks = []
    y_blocks = []
    for _, row in feature_df.iterrows():
        target = set(json.loads(row["target_numbers"]))
        candidates = build_candidate_matrix(row, feature_columns)
        labels = pd.Series([1 if n in target else 0 for n in range(1, 81)])
        x_blocks.append(candidates)
        y_blocks.append(labels)
    return pd.concat(x_blocks, ignore_index=True), pd.concat(
        y_blocks, ignore_index=True
    )


def main() -> None:
    cfg = load_yaml(CONFIG_DIR / "train.yaml")
    feature_df = pd.read_csv(FEATURE_STORE_DIR / "issue_features.csv")
    feature_columns = json.loads(
        (MODELS_DIR / "feature_columns.json").read_text(encoding="utf-8")
    )

    x_train, y_train = _expand_training_rows(feature_df, feature_columns)
    params = cfg["lgbm_params"]
    model = lgb.LGBMClassifier(**params)
    model.fit(x_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODELS_DIR / "lgbm_top20.txt"))

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)

    metadata = {
        "model_type": "lightgbm_binary_v1",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_rows": int(len(feature_df)),
        "sample_rows": int(len(x_train)),
        "train_issue_start": int(feature_df["issue"].min()),
        "train_issue_end": int(feature_df["target_issue"].max()),
        "feature_columns_path": "models/feature_columns.json",
        "model_path": "models/lgbm_top20.txt",
        "best_params": params,
        "feature_version": "v1",
    }

    save_json(MODELS_DIR / "metadata.json", metadata)
    importance.to_csv("reports/feature_importance.csv", index=False)
    print("saved model and metadata")


if __name__ == "__main__":
    main()
