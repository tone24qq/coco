import json

import numpy as np
import pandas as pd

from src.backtest import (
    _alignment_audit,
    _build_feature_version_comparison,
    _ci95,
    _make_fold_issue_metrics,
    _overfit_audit,
    _predictability_test,
)


def test_ci95_returns_expected_keys() -> None:
    out = _ci95([0.1, 0.2, 0.3, 0.4])
    assert set(out) == {"mean", "std", "ci95_low", "ci95_high"}
    assert out["ci95_low"] <= out["mean"] <= out["ci95_high"]


def test_predictability_test_outputs_distribution() -> None:
    df = pd.DataFrame(
        {
            "target_numbers": [json.dumps(list(range(1, 21))) for _ in range(50)],
        }
    )
    observed = [0.28 + 0.01 * (i % 3) for i in range(50)]

    summary, perm_df, bootstrap = _predictability_test(
        df=df,
        observed_scores=observed,
        permutations=40,
        block_size=5,
    )

    assert set(summary) >= {
        "observed_score",
        "null_mean",
        "null_std",
        "p_value",
        "signal_sufficient",
    }
    assert len(perm_df) == 40
    assert np.isfinite(bootstrap["mean"])


def test_alignment_audit_happy_path() -> None:
    rows = []
    for i in range(35):
        rows.append(
            {
                "issue": 1000 + i,
                "draw_date": "2026-01-01",
                "numbers": json.dumps(sorted([((i + k) % 80) + 1 for k in range(20)])),
            }
        )
    df = pd.DataFrame(rows)

    audit_df, summary = _alignment_audit(df, splits=3)

    assert not audit_df.empty
    assert bool(summary["issue_strictly_increasing"])
    assert bool(summary["target_issue_is_next_issue"])


def test_top3_at_least_one_metric() -> None:
    scores = np.linspace(0.0, 1.0, 80)
    actual = {78, 79, 80}
    metrics = _make_fold_issue_metrics(scores, actual)
    assert metrics["top3_at_least_one_hit_rate"] == 1.0
    assert "top5_hit_rate" in metrics
    assert "ndcg_at_10" in metrics


def test_overfit_audit_flags_large_gap() -> None:
    train_fold = [
        {"top3_hit_rate": 0.5, "top3_at_least_one_hit_rate": 0.9},
        {"top3_hit_rate": 0.48, "top3_at_least_one_hit_rate": 0.88},
    ]
    test_fold = [
        {"top3_hit_rate": 0.2, "top3_at_least_one_hit_rate": 0.5},
        {"top3_hit_rate": 0.18, "top3_at_least_one_hit_rate": 0.48},
    ]
    regime_rows = [
        {"regime": "balanced", "top3_hit_rate": 0.1},
        {"regime": "high_vol", "top3_hit_rate": 0.4},
    ]
    out = _overfit_audit(train_fold, test_fold, regime_rows)
    assert out["is_overfit"] is True


def test_feature_version_comparison_no_reference() -> None:
    history = pd.DataFrame(
        [
            {
                "trained_at_utc": "2026-01-01T00:00:00+00:00",
                "feature_version": "v3_core20",
                "top20_hit_rate": 0.2,
                "top10_hit_rate": 0.2,
                "top5_hit_rate": 0.2,
                "top3_hit_rate": 0.2,
                "top3_at_least_one_hit_rate": 0.2,
                "fold_dispersion_top3": 0.1,
                "regime_dispersion_top3": 0.1,
            }
        ]
    )
    current = history.iloc[0].to_dict()
    out = _build_feature_version_comparison(history, current, {})
    assert out["available"] is False


def test_feature_version_comparison_acceptance() -> None:
    history = pd.DataFrame(
        [
            {
                "trained_at_utc": "2026-01-01T00:00:00+00:00",
                "feature_version": "v2_legacy",
                "top20_hit_rate": 0.20,
                "top10_hit_rate": 0.18,
                "top5_hit_rate": 0.16,
                "top3_hit_rate": 0.14,
                "top3_at_least_one_hit_rate": 0.35,
                "fold_dispersion_top3": 0.08,
                "regime_dispersion_top3": 0.09,
            },
            {
                "trained_at_utc": "2026-01-02T00:00:00+00:00",
                "feature_version": "v3_core20",
                "top20_hit_rate": 0.21,
                "top10_hit_rate": 0.19,
                "top5_hit_rate": 0.17,
                "top3_hit_rate": 0.145,
                "top3_at_least_one_hit_rate": 0.36,
                "fold_dispersion_top3": 0.07,
                "regime_dispersion_top3": 0.08,
            },
        ]
    )
    current = history.iloc[-1].to_dict()
    out = _build_feature_version_comparison(
        history,
        current,
        {"non_degradation_tol": 0.01, "stability_improvement_min": 0.0},
    )
    assert out["available"] is True
    assert out["acceptance_pass"] is True
