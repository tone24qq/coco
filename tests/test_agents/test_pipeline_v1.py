import json

import pandas as pd

from src.utils import (
    V3_CORE20_COLUMNS,
    build_candidate_matrix,
    build_issue_features,
    build_latest_issue_features_for_inference,
    build_recent_report,
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
        "issue_zone_entropy",
        "issue_span_z50",
        "issue_sum_z50",
        "issue_consecutive_z50",
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
    cols = V3_CORE20_COLUMNS
    x = build_candidate_matrix(feat_df.iloc[-1], cols)
    assert list(x.columns) == cols
    assert len(x) == 80
    assert x["cand_freq_smooth_20"].sum() >= 0


def test_latest_inference_row_aligns_to_next_issue() -> None:
    rows = []
    for i in range(25):
        nums = [((i + k) % 80) + 1 for k in range(20)]
        rows.append(
            {
                "issue": 3000 + i,
                "draw_date": "2026-01-01",
                "numbers": json.dumps(sorted(nums)),
            }
        )
    df = pd.DataFrame(rows)
    train_feat = build_issue_features(df, min_history=20)
    infer_feat = build_latest_issue_features_for_inference(df, min_history=20)

    assert int(train_feat.iloc[-1]["issue"]) == 3000 + 23
    assert int(infer_feat.iloc[-1]["issue"]) == 3000 + 24
    assert int(infer_feat.iloc[-1]["target_issue"]) == 3000 + 25


def test_build_recent_report_contains_expected_sections() -> None:
    recent_draws = []
    for i in range(25):
        recent_draws.append([((i + k) % 80) + 1 for k in range(20)])

    report = build_recent_report(recent_draws)

    assert set(report) == {"odd_even", "big_small", "zone", "recent_frequency"}
    assert "odd_cnt" in report["odd_even"]
    assert "big_cnt" in report["big_small"]
    assert "board_regime" in report["zone"]
    assert "recent_freq_20" in report["recent_frequency"]
