from __future__ import annotations

import random

from winwin_service.config import AppConfig, RegimeConfig
from winwin_service.scoring import (
    _apply_regime_adjustment,
    detect_regime,
    predict_top3,
)


def _baseline_draws() -> list[list[int]]:
    rng = random.Random(42)
    draws: list[list[int]] = []
    for _ in range(50):
        draws.append(sorted(rng.sample(range(1, 81), 20)))
    return draws


def _concentrated_draw(base: int = 1) -> list[int]:
    return list(range(base, base + 20))


def test_predict_contract_top3_schema_unchanged() -> None:
    result = predict_top3(
        _baseline_draws(),
        latest_period=4000,
        config=AppConfig(min_score_threshold=10),
    )

    assert set(result.keys()) == {
        "target_period",
        "latest_period",
        "top3",
        "kill_zone",
        "metadata",
    }
    assert isinstance(result["top3"], list)
    assert all(
        isinstance(item, list) and len(item) == 3
        for item in result["top3"]
    )


def test_top3_length_is_always_three() -> None:
    result = predict_top3(
        _baseline_draws(),
        latest_period=1000,
        config=AppConfig(min_score_threshold=10),
    )
    assert len(result["top3"]) == 3


def test_diversified_strict_overlap() -> None:
    result = predict_top3(
        _baseline_draws(),
        latest_period=2000,
        config=AppConfig(min_score_threshold=10),
    )
    top3 = result["top3"]
    assert len(set(top3[0]) & set(top3[1])) <= 1
    assert len(set(top3[0]) & set(top3[2])) <= 1


def test_fallback_path_can_fill_top3() -> None:
    draws = _baseline_draws()
    # shrink effective pool to force fallback path more often
    limited_draws = [draw[:20] for draw in draws]
    result = predict_top3(
        limited_draws,
        latest_period=3000,
        config=AppConfig(
            min_score_threshold=-100,
            skip_kill_threshold=1,
            streak_kill_threshold=100,
        ),
    )
    assert len(result["top3"]) == 3
    assert "fallback_used" in result["metadata"]


def test_detector_stable_data_returns_normal() -> None:
    regime = detect_regime(_baseline_draws(), config=AppConfig())
    assert regime["regime"] == "normal"
    assert regime["fallback_to_normal"] is True


def test_adjustment_strength_capped() -> None:
    draws = _baseline_draws()
    dense = _concentrated_draw(1)
    draws[-2] = dense
    draws[-1] = dense
    regime = detect_regime(draws, config=AppConfig())
    assert regime["adjustment_strength"] <= 0.10


def test_regime_changes_adjusted_score_ordering() -> None:
    components_a = {
        "momentum_score": 14.0,
        "warm_skip_count": 0.0,
        "streak_count": 2.0,
        "pair_sum": 14.0,
        "tail_unique": 1.0,
        "tens_unique": 1.0,
    }
    components_b = {
        "momentum_score": 3.0,
        "warm_skip_count": 2.0,
        "streak_count": 1.0,
        "pair_sum": 3.0,
        "tail_unique": 3.0,
        "tens_unique": 3.0,
    }

    normal = {"regime": "normal", "adjustment_strength": 0.0}
    hot = {"regime": "hot_continuation", "adjustment_strength": 0.10}

    normal_a = _apply_regime_adjustment(100.0, components_a, normal)
    normal_b = _apply_regime_adjustment(100.0, components_b, normal)
    hot_a = _apply_regime_adjustment(100.0, components_a, hot)
    hot_b = _apply_regime_adjustment(100.0, components_b, hot)

    assert normal_a == normal_b
    assert hot_a > hot_b


def test_metadata_contains_new_detector_fields() -> None:
    result = predict_top3(
        _baseline_draws(),
        latest_period=5000,
        config=AppConfig(min_score_threshold=10),
    )
    metadata = result["metadata"]
    expected = {
        "dedup_enabled",
        "dedup_rule",
        "raw_top_candidates_considered",
        "fallback_used",
        "regime",
        "anomaly_flags",
        "regime_adjustment_enabled",
        "regime_window",
        "trigger_count",
        "consecutive_trigger_hits",
        "adjustment_strength",
        "fallback_to_normal",
        "regime_metrics",
        "regime_metrics_raw",
        "regime_metrics_zscore",
        "regime_metrics_percentile",
        "normal_oscillation_flags",
        "warning_flags",
        "detector_band",
    }
    assert expected.issubset(set(metadata.keys()))


def test_fail_fast_when_draw_count_not_enough() -> None:
    try:
        predict_top3(
            _baseline_draws()[:20],
            latest_period=10,
            config=AppConfig(recent_draws_count=50),
        )
    except Exception as exc:  # noqa: BLE001
        assert "Need >=" in str(exc)
    else:
        assert False, "expected fail-fast error"


def test_single_high_metric_does_not_trigger_regime() -> None:
    draws = _baseline_draws()
    # force only overlap high but keep structure otherwise mixed
    draws[-1] = draws[-2][:]
    regime = detect_regime(draws, config=AppConfig())
    assert regime["regime"] == "normal"


def test_multimetric_resonance_can_trigger_regime() -> None:
    draws = _baseline_draws()
    draws[-3] = _concentrated_draw(1)
    draws[-2] = _concentrated_draw(1)
    draws[-1] = _concentrated_draw(1)
    cfg = AppConfig(
        regime=RegimeConfig(
            warning_zscore=1.0,
            anomaly_zscore=1.3,
            warning_percentile=0.80,
            anomaly_percentile=0.90,
        )
    )
    regime = detect_regime(draws, config=cfg)
    assert regime["regime"] in {
        "hot_continuation",
        "concentrated",
        "warm_rebound",
        "dispersed",
    }
    assert regime["fallback_to_normal"] is False


def test_detector_band_exposes_normal_vs_anomaly() -> None:
    normal = detect_regime(_baseline_draws(), config=AppConfig())
    assert normal["detector_band"] == "normal_band"

    draws = _baseline_draws()
    dense = _concentrated_draw(1)
    draws[-2] = dense
    draws[-1] = dense
    cfg = AppConfig(
        regime=RegimeConfig(
            warning_zscore=1.0,
            anomaly_zscore=1.3,
            warning_percentile=0.80,
            anomaly_percentile=0.90,
        )
    )
    anomaly = detect_regime(draws, config=cfg)
    assert anomaly["detector_band"] in {"warm_band", "anomaly_band"}
