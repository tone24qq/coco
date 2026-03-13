import json

import pandas as pd
from pandas.testing import assert_frame_equal

from src.utils import V3_CORE20_COLUMNS, build_candidate_matrix, build_issue_features


def _draws(n: int = 45) -> pd.DataFrame:
    rows = []
    for i in range(n):
        nums = sorted([((i * 3 + k) % 80) + 1 for k in range(20)])
        rows.append(
            {
                "issue": 100000 + i,
                "draw_date": "2026-01-01",
                "numbers": json.dumps(nums, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def test_v3_candidate_matrix_not_affected_by_target_numbers(monkeypatch) -> None:
    monkeypatch.setenv("FEATURE_VERSION_OVERRIDE", "v3_core20")
    feat_df = build_issue_features(_draws(), min_history=22)
    row = feat_df.iloc[-1].copy()

    x1 = build_candidate_matrix(row, V3_CORE20_COLUMNS)
    row["target_numbers"] = json.dumps(list(range(61, 81)), ensure_ascii=False)
    x2 = build_candidate_matrix(row, V3_CORE20_COLUMNS)

    assert_frame_equal(x1, x2)
