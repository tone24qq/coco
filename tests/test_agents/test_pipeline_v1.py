import json

import pandas as pd

from src.utils import (
    build_candidate_matrix,
    build_issue_features,
    issue_feature_columns,
)


def test_build_issue_features_has_required_columns() -> None:
    rows = []
    for i in range(35):
        nums = [((i + k) % 80) + 1 for k in range(20)]
        rows.append(
            {
                "issue": 1000 + i,
                "draw_date": "2026-01-01",
                "numbers": json.dumps(sorted(nums)),
            }
        )
    df = pd.DataFrame(rows)
    feat_df = build_issue_features(df, min_history=20)

    required = {
        "sum_all",
        "zone_A_cnt",
        "small_cnt",
        "consecutive_pairs",
        "tail_0_cnt",
        "recent_freq_20",
        "delta_sum_1",
        "roll5_sum_mean",
        "sim_top1_score",
    }
    assert required.issubset(set(feat_df.columns))


def test_candidate_matrix_matches_feature_columns() -> None:
    rows = []
    for i in range(30):
        nums = [((i + k) % 80) + 1 for k in range(20)]
        rows.append(
            {
                "issue": 2000 + i,
                "draw_date": "2026-01-01",
                "numbers": json.dumps(sorted(nums)),
            }
        )
    feat_df = build_issue_features(pd.DataFrame(rows), min_history=20)
    cols = issue_feature_columns(feat_df) + [
        "num",
        "num_norm",
        "num_zone",
        "num_is_odd",
        "num_is_big",
    ]
    x = build_candidate_matrix(feat_df.iloc[-1], cols)
    assert list(x.columns) == cols
    assert len(x) == 80
