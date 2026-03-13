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
    issue_feature_columns,
    load_processed,
    load_yaml,
    validate_feature_columns_contract,
)


def main() -> None:
    cfg = load_yaml(CONFIG_DIR / "train.yaml")
    df = load_processed()
    max_draws = int(cfg.get("max_draws_for_training", len(df)))
    if max_draws < 3000:
        raise ValueError("max_draws_for_training 不可小於 3000。")
    df = df.tail(max_draws).reset_index(drop=True)
    feat_df = build_issue_features(df, min_history=int(cfg["feature_min_history"]))
    FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(FEATURE_STORE_DIR / "issue_features.csv", index=False)

    if cfg.get("feature_version", "v2_legacy") == "v3_core20":
        model_cols = V3_CORE20_COLUMNS
    else:
        cols = issue_feature_columns(feat_df)
        model_cols = cols + [
            "num",
            "num_norm",
            "num_zone",
            "num_is_odd",
            "num_is_big",
            "cand_in_prev_plus1",
            "cand_in_prev_plus2",
            "cand_in_prev_minus1",
            "cand_in_prev_pm1",
            "freq_last_10",
            "freq_last_20",
            "freq_last_50",
            "freq_last_100",
            "freq_last_200",
            "freq_last_300",
            "freq_last_500",
            "freq_last_1000",
            "ema_freq_alpha_0_05",
            "ema_freq_alpha_0_1",
            "ema_freq_alpha_0_2",
            "gap_since_last_seen",
            "avg_gap_last_3",
            "avg_gap_last_5",
            "std_gap_last_5",
            "min_gap_last_5",
            "max_gap_last_5",
            "freq_10_minus_50",
            "freq_20_minus_100",
            "recent_trend_up_down",
            "ema_short_minus_ema_long",
            "cooccur_with_last_draw_sum",
            "cooccur_with_last_draw_mean",
            "cooccur_with_last_draw_max",
            "distance_to_last_draw_min",
            "distance_to_last_draw_mean",
            "count_close_to_last_draw_within_1",
            "count_close_to_last_draw_within_2",
            "count_close_to_last_draw_within_3",
            "is_adjacent_to_last_draw",
            "adjacent_count_vs_last_draw",
            "pair_score_with_last_5_draws",
            "rank_by_recent_freq",
            "rank_by_gap_inverse",
            "rank_by_cooccur_score",
        ]
    validate_feature_columns_contract(
        model_cols,
        str(cfg.get("feature_version", "v2_legacy")),
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "feature_columns.json").write_text(
        json.dumps(model_cols, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved features: {len(feat_df)} rows")


if __name__ == "__main__":
    main()
