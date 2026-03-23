import pandas as pd

from src.backtest import (
    _build_match_pairs,
    _build_near_miss_report,
    _near_miss_issue_metrics,
)


def test_exact_and_pm1_metrics_with_manual_fixture() -> None:
    actual = [10, 20, 30]
    pred_top3 = [9, 20, 31]

    out = _near_miss_issue_metrics(pred_top3, actual, top_k=3)

    assert out["exact_hit@3"] == 1 / 3
    assert out["adj_hit_pm1@3"] == 1.0
    assert out["strict_adj_only_pm1@3"] == 2 / 3
    assert out["top3_at_least_one_exact"] == 1.0
    assert out["top3_at_least_one_adj_pm1"] == 1.0
    assert out["top3_at_least_one_strict_adj_only_pm1"] == 1.0


def test_one_to_one_matching_prevents_double_counting() -> None:
    pred_top3 = [9, 10, 11]
    actual = [10]

    adj_pairs = _build_match_pairs(pred_top3, actual, max_dist=1, include_exact=True)
    strict_pairs = _build_match_pairs(
        pred_top3, actual, max_dist=1, include_exact=False
    )

    assert len(adj_pairs) == 1
    assert len(strict_pairs) == 1


def test_strict_pm1_and_boundaries_for_1_and_80() -> None:
    actual = [1, 80, 50]
    pred_top3 = [2, 79, 50]

    out = _near_miss_issue_metrics(pred_top3, actual, top_k=3)

    assert out["strict_adj_only_pm1@3"] == 2 / 3
    assert out["near_miss_plus1_count"] == 1
    assert out["near_miss_minus1_count"] == 1


def test_near_miss_report_contains_baselines_and_required_columns() -> None:
    feat_df = pd.DataFrame(
        [
            {"issue": 100, "history_numbers": "[1, 2, 3, 4, 5, 5, 6]"},
            {"issue": 101, "history_numbers": "[10, 11, 12, 13, 14, 14, 14]"},
        ]
    )
    per_issue_df = pd.DataFrame(
        [
            {
                "version_id": "v9_model",
                "fold": 1,
                "issue": 100,
                "pred_top3": [9, 20, 31],
                "pred_top10": [9, 20, 31, 40, 41, 42, 43, 44, 45, 46],
                "actual": [10, 20, 30],
            },
            {
                "version_id": "v0_binary_baseline",
                "fold": 1,
                "issue": 100,
                "pred_top3": [8, 22, 33],
                "pred_top10": [8, 22, 33, 40, 41, 42, 43, 44, 45, 46],
                "actual": [10, 20, 30],
            },
        ]
    )

    per_fold, overall, baseline_comp, ci_summary = _build_near_miss_report(
        feat_df=feat_df,
        per_issue_df=per_issue_df,
        best_version="v9_model",
        splits=1,
    )

    assert not per_fold.empty
    assert not overall.empty
    assert {"random_baseline", "frequency_baseline", "v9_model"}.issubset(
        set(overall["baseline"].tolist())
    )
    assert "delta_adj_hit_pm1@3" in baseline_comp.columns
    assert "model" in ci_summary
    assert "exact_hit@3" in ci_summary["model"]
