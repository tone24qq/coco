from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json

from src.utils import (  # noqa: E402
    CONFIG_DIR,
    FEATURE_STORE_DIR,
    MODELS_DIR,
    build_issue_features,
    issue_feature_columns,
    load_processed,
    load_yaml,
)


def main() -> None:
    cfg = load_yaml(CONFIG_DIR / "train.yaml")
    df = load_processed()
    max_draws = int(cfg.get("max_draws_for_training", len(df)))
    df = df.tail(max_draws).reset_index(drop=True)
    feat_df = build_issue_features(df, min_history=int(cfg["feature_min_history"]))
    FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(FEATURE_STORE_DIR / "issue_features.csv", index=False)

    cols = issue_feature_columns(feat_df)
    model_cols = cols + ["num", "num_norm", "num_zone", "num_is_odd", "num_is_big"]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "feature_columns.json").write_text(
        json.dumps(model_cols, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved features: {len(feat_df)} rows")


if __name__ == "__main__":
    main()
