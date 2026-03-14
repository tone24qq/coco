import json

import pandas as pd
import pytest

from src.utils import V3_CORE20_COLUMNS, build_candidate_matrix, build_issue_features


def _make_draws(n_rows: int = 80) -> pd.DataFrame:
    rows = []
    for i in range(n_rows):
        nums = [((i + k) % 80) + 1 for k in range(20)]
        rows.append(
            {
                "issue": 5000 + i,
                "draw_date": "2026-01-01",
                "numbers": json.dumps(sorted(nums)),
            }
        )
    return pd.DataFrame(rows)


def test_v3_core20_columns_complete() -> None:
    feat_df = build_issue_features(_make_draws(), min_history=22)
    x = build_candidate_matrix(feat_df.iloc[-1], V3_CORE20_COLUMNS)
    assert list(x.columns) == V3_CORE20_COLUMNS
    assert x.shape == (80, len(V3_CORE20_COLUMNS))


def test_v3_boundary_features_for_num_1_and_80() -> None:
    feat_df = build_issue_features(_make_draws(), min_history=22)
    x = build_candidate_matrix(feat_df.iloc[-1], V3_CORE20_COLUMNS)

    row_1 = x.iloc[0]
    row_80 = x.iloc[-1]
    assert row_1["num_zone"] == 0.0
    assert row_80["num_zone"] == 3.0
    assert row_1["cand_neighbor_pm1_decay_hl10"] >= 0.0
    assert row_80["cand_neighbor_pm2_decay_hl10"] >= 0.0
    assert row_1["cand_pmi_last_draw_max_200"] >= 0.0
    assert row_80["cand_handoff_pm1_lift_200"] == row_80["cand_handoff_pm1_lift_200"]


def test_cross_issue_distance_features_cover_prev_draw_example() -> None:
    rows = [
        {
            "issue": 1,
            "draw_date": "2026-01-01",
            "numbers": json.dumps(list(range(1, 21))),
        },
        {
            "issue": 2,
            "draw_date": "2026-01-01",
            "numbers": json.dumps(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 70, 80]
            ),
        },
        {
            "issue": 3,
            "draw_date": "2026-01-01",
            "numbers": json.dumps(
                [
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    60,
                    79,
                ]
            ),
        },
    ]
    feat_df = build_issue_features(pd.DataFrame(rows), min_history=1)
    x = build_candidate_matrix(feat_df.iloc[-1], V3_CORE20_COLUMNS)
    row_60 = x.iloc[59]
    row_79 = x.iloc[78]
    assert row_60["cand_min_abs_distance_to_prev_draw"] == 10
    assert row_79["cand_min_abs_distance_to_prev_draw"] == 1
    assert row_79["cand_has_prev_pm1"] == 1.0


def test_strict_features_raise_on_missing_columns() -> None:
    feat_df = build_issue_features(_make_draws(), min_history=22)
    with pytest.raises(ValueError):
        build_candidate_matrix(
            feat_df.iloc[-1],
            V3_CORE20_COLUMNS + ["missing_col"],
            strict_features=True,
        )
