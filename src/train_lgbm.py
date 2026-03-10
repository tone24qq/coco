from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json
from datetime import datetime, timezone

import pandas as pd  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

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
    (MODELS_DIR / "feature_columns.json").write_text(
        json.dumps(feature_columns, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    x_train, y_train = _expand_training_rows(feature_df, feature_columns)
    split_at = int(len(x_train) * 0.8)
    x_fit, x_valid = x_train.iloc[:split_at], x_train.iloc[split_at:]
    y_fit, y_valid = y_train.iloc[:split_at], y_train.iloc[split_at:]

    params = cfg.get("catboost_params", {})
    params.setdefault("loss_function", "Logloss")
    params.setdefault("verbose", False)
    params.setdefault("random_seed", 42)
    model = CatBoostClassifier(**params)
    model.fit(x_fit, y_fit, eval_set=(x_valid, y_valid), use_best_model=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODELS_DIR / "catboost_top20.cbm"))

    valid_proba = model.predict_proba(x_valid)[:, 1]
    valid_logloss = log_loss(y_valid, valid_proba)

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.get_feature_importance(type="PredictionValuesChange"),
        }
    ).sort_values("importance", ascending=False)

    metadata = {
        "model_type": "catboost",
        "loss_function": params["loss_function"],
        "feature_count": len(feature_columns),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_rows": int(len(feature_df)),
        "training_rows": int(len(x_train)),
        "train_issue_start": int(feature_df["issue"].min()),
        "train_issue_end": int(feature_df["target_issue"].max()),
        "feature_columns_path": "models/feature_columns.json",
        "model_path": "models/catboost_top20.cbm",
        "params": params,
        "validation_metric": {"logloss": float(valid_logloss)},
        "feature_version": "v1",
        "calibration_method": "none",
    }

    save_json(MODELS_DIR / "metadata.json", metadata)
    importance.to_csv("reports/feature_importance.csv", index=False)
    save_json(
        Path("reports") / "feature_importance.json",
        {"type": "PredictionValuesChange", "features": importance.to_dict("records")},
    )
    print("saved model and metadata")


if __name__ == "__main__":
    main()
