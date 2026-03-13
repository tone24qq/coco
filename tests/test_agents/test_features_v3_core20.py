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
    assert x.shape == (80, 20)


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


def test_strict_features_raise_on_missing_columns() -> None:
    feat_df = build_issue_features(_make_draws(), min_history=22)
    with pytest.raises(ValueError):
        build_candidate_matrix(
            feat_df.iloc[-1],
            V3_CORE20_COLUMNS + ["missing_col"],
            strict_features=True,
        )
