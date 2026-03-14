import pandas as pd

from src.backtest import _build_history_bucket_report


def test_history_bucket_report_uses_prediction_rows_not_prev_copy() -> None:
    issue_rows = pd.DataFrame(
        [
            {
                "version_id": "v3_rerank_k30_p300",
                "history_length": 30,
                "pred_top3": [10, 20, 30],
                "pred_top10": [10, 20, 30, 40, 50, 60, 70, 71, 72, 73],
                "actual": [11, 21, 31],
                "prev_numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 80],
            }
        ]
    )

    report = _build_history_bucket_report(issue_rows)

    row = report.iloc[0].to_dict()
    assert row["mean_min_distance_at_3"] == 1.0
    assert row["top3_prev_draw_mean_min_distance"] > 0.0


def test_history_bucket_distance_metrics_change_with_predictions() -> None:
    issue_rows = pd.DataFrame(
        [
            {
                "version_id": "v3_rerank_k30_p300",
                "history_length": 15,
                "pred_top3": [10, 20, 30],
                "pred_top10": [10, 20, 30, 40, 50, 60, 70, 71, 72, 73],
                "actual": [11, 21, 31],
                "prev_numbers": [9, 19, 29, 39, 49, 59, 69, 79, 1, 2],
            },
            {
                "version_id": "v3_rerank_k30_p300",
                "history_length": 15,
                "pred_top3": [11, 21, 31],
                "pred_top10": [11, 21, 31, 41, 51, 61, 71, 72, 73, 74],
                "actual": [11, 21, 31],
                "prev_numbers": [9, 19, 29, 39, 49, 59, 69, 79, 1, 2],
            },
        ]
    )

    report = _build_history_bucket_report(issue_rows)

    row = report.iloc[0].to_dict()
    assert row["mean_min_distance_at_3"] == 0.5
    assert row["top3_prev_draw_mean_min_distance"] == 1.5


def test_history_bucket_report_would_expose_old_prev_self_compare_bug() -> None:
    issue_rows = pd.DataFrame(
        [
            {
                "version_id": "v3_rerank_k30_p300",
                "history_length": 40,
                "pred_top3": [30, 40, 50],
                "pred_top10": [30, 40, 50, 60, 61, 62, 63, 64, 65, 66],
                "actual": [31, 41, 51],
                "prev_numbers": [30, 40, 50, 1, 2, 3, 4, 5, 6, 7],
            }
        ]
    )

    report = _build_history_bucket_report(issue_rows)

    row = report.iloc[0].to_dict()
    assert row["mean_min_distance_at_3"] == 1.0
    assert row["top3_prev_draw_mean_min_distance"] == 0.0
