import json

import pandas as pd

from src.utils import (
    CANDIDATE_FEATURE_COLUMNS,
    build_candidate_matrix,
    build_issue_features,
    build_latest_issue_features_for_inference,
    build_recent_report,
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
        "shift_p1_hit_rate",
        "shift_pm1_hit_rate",
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
    cols = issue_feature_columns(feat_df) + CANDIDATE_FEATURE_COLUMNS
    x = build_candidate_matrix(feat_df.iloc[-1], cols)
    assert list(x.columns) == cols
    assert len(x) == 80
    assert "freq_last_10" in x.columns
    assert "cooccur_mean_last_200" in x.columns


def test_inference_feature_uses_latest_issue_alignment() -> None:
    rows = []
    for i in range(30):
        nums = [((i + k) % 80) + 1 for k in range(20)]
        rows.append(
            {
                "issue": 3000 + i,
                "draw_date": "2026-01-01",
                "numbers": json.dumps(sorted(nums)),
            }
        )
    df = pd.DataFrame(rows)
    latest_row = build_latest_issue_features_for_inference(df, min_history=20)
    assert int(latest_row["issue"]) == 3029
    assert int(latest_row["target_issue"]) == 3030


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
