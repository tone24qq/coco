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
    V3_CORE20_COLUMNS,
    build_issue_features,
    load_processed,
    load_yaml,
    validate_feature_columns_contract,
)


def main() -> None:
    cfg = load_yaml(CONFIG_DIR / "train.yaml")
    if str(cfg.get("feature_version", "v3_core20")) != "v3_core20":
        raise ValueError("only v3_core20 is supported")
    df = load_processed()
    max_draws = int(cfg.get("max_draws_for_training", len(df)))
    if max_draws < 3000:
        raise ValueError("max_draws_for_training 不可小於 3000。")
    df = df.tail(max_draws).reset_index(drop=True)
    feat_df = build_issue_features(df, min_history=int(cfg["feature_min_history"]))
    FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(FEATURE_STORE_DIR / "issue_features.csv", index=False)

    model_cols = V3_CORE20_COLUMNS
    validate_feature_columns_contract(
        model_cols,
        str(cfg.get("feature_version", "v3_core20")),
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "feature_columns.json").write_text(
        json.dumps(model_cols, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved features: {len(feat_df)} rows")


if __name__ == "__main__":
    main()
