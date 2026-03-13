import pandas as pd

from src.backtest import _build_feature_version_comparison


def test_backtest_version_comparison_contains_v2_v3_baselines() -> None:
    history = pd.DataFrame(
        [
            {
                "trained_at_utc": "2026-01-01T00:00:00+00:00",
                "feature_version": "v2_legacy",
                "top20_hit_rate": 0.20,
                "top10_hit_rate": 0.10,
                "top5_hit_rate": 0.05,
                "top3_hit_rate": 0.03,
                "top3_at_least_one_hit_rate": 0.10,
                "fold_dispersion_top3": 0.09,
                "regime_dispersion_top3": 0.07,
            }
        ]
    )
    current = {
        "trained_at_utc": "2026-01-02T00:00:00+00:00",
        "feature_version": "v3_core20",
        "top20_hit_rate": 0.21,
        "top10_hit_rate": 0.12,
        "top5_hit_rate": 0.07,
        "top3_hit_rate": 0.05,
        "top3_at_least_one_hit_rate": 0.11,
        "fold_dispersion_top3": 0.08,
        "regime_dispersion_top3": 0.06,
    }
    out = _build_feature_version_comparison(history, current, {})
    assert out["available"] is True
    assert "v2_baseline" in out and "v3_baseline" in out
    assert "delta_top3" in out["deltas"]


def test_backtest_version_comparison_missing_v2_message() -> None:
    history = pd.DataFrame(
        [
            {
                "trained_at_utc": "2026-01-01T00:00:00+00:00",
                "feature_version": "v3_core20",
                "top20_hit_rate": 0.21,
                "top10_hit_rate": 0.12,
                "top5_hit_rate": 0.07,
                "top3_hit_rate": 0.05,
                "top3_at_least_one_hit_rate": 0.11,
                "fold_dispersion_top3": 0.08,
                "regime_dispersion_top3": 0.06,
            }
        ]
    )
    current = history.iloc[0].to_dict()
    out = _build_feature_version_comparison(history, current, {})
    assert out["available"] is False
    assert out["reason"] == "missing v2_legacy reference"
