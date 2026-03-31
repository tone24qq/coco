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


def _baseline_draws_n(count: int) -> list[list[int]]:
    draws = _baseline_draws()
    return draws[:count]


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
        "top10",
        "top10_display",
        "kill_zone",
        "metadata",
    }
    assert isinstance(result["top3"], list)
    assert isinstance(result["top10"], list)
    assert len(result["top10"]) == 10
    assert isinstance(result["top10_display"], list)
    assert len(result["top10_display"]) == 10
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


def test_top10_rank_and_sort_contract() -> None:
    result = predict_top3(
        _baseline_draws(),
        latest_period=1001,
        config=AppConfig(min_score_threshold=10),
    )
    top10 = result["top10"]
    assert [entry["rank"] for entry in top10] == list(range(1, 11))
    scores = [entry["score"] for entry in top10]
    assert scores == sorted(scores, reverse=True)
    for entry in top10:
        assert set(entry.keys()) == {
            "rank",
            "numbers",
            "score",
            "confidence",
            "overlap_count_vs_previous",
            "high_confidence_overlap",
        }


def test_predict_with_48_draws_still_allowed() -> None:
    result = predict_top3(
        _baseline_draws_n(48),
        latest_period=2222,
        config=AppConfig(min_prediction_draws=10),
    )
    assert len(result["top3"]) == 3
    assert result["metadata"]["analyzed_draws"] == 3
    assert result["metadata"]["effective_draws_used"] == 3
    assert result["metadata"]["effective_recent_window"] == 3
    assert result["metadata"]["regime_window"] == 3


def test_diversified_strict_overlap() -> None:
    result = predict_top3(
        _baseline_draws(),
        latest_period=2000,
        config=AppConfig(min_score_threshold=10),
    )
    top3 = result["top3"]
    assert len(set(top3[0]) & set(top3[1])) <= 1
    assert len(set(top3[0]) & set(top3[2])) <= 1


def test_top10_overlap_guardrail() -> None:
    result = predict_top3(
        _baseline_draws(),
        latest_period=2000,
        config=AppConfig(min_score_threshold=10),
    )
    top10 = result["top10"]
    for idx, current in enumerate(top10):
        current_numbers = current["numbers"]
        for prev in top10[:idx]:
            overlap = len(set(current_numbers) & set(prev["numbers"]))
            if overlap > 1:
                assert current["high_confidence_overlap"] is True
                assert current["confidence"] > 0.9


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

    normal_a = _apply_regime_adjustment(
        100.0, components_a, normal, AppConfig()
    )
    normal_b = _apply_regime_adjustment(
        100.0, components_b, normal, AppConfig()
    )
    hot_a = _apply_regime_adjustment(100.0, components_a, hot, AppConfig())
    hot_b = _apply_regime_adjustment(100.0, components_b, hot, AppConfig())

    assert normal_a == normal_b
    assert hot_a > hot_b


def test_regime_adjustment_knobs_change_delta_size() -> None:
    components = {
        "momentum_score": 10.0,
        "warm_skip_count": 1.0,
        "streak_count": 2.0,
        "pair_sum": 9.0,
        "tail_unique": 2.0,
        "tens_unique": 2.0,
    }
    hot = {"regime": "hot_continuation", "adjustment_strength": 0.10}
    low = _apply_regime_adjustment(
        100.0,
        components,
        hot,
        AppConfig(
            regime_hot_momentum_weight=0.2,
            regime_hot_streak_weight=2.0,
        ),
    )
    high = _apply_regime_adjustment(
        100.0,
        components,
        hot,
        AppConfig(
            regime_hot_momentum_weight=1.0,
            regime_hot_streak_weight=6.0,
        ),
    )
    assert high > low


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
        "warning_flags",
        "detector_band",
        "available_draws",
        "effective_draws_used",
        "min_prediction_draws",
        "max_recent_draws_count",
        "regime_min_history",
        "regime_disabled_reason",
        "candidate_pool_before_trim",
        "candidate_pool_after_trim",
        "total_combinations_evaluated",
    }
    assert expected.issubset(set(metadata.keys()))


def test_regime_detailed_metrics_only_in_debug_mode() -> None:
    normal = predict_top3(
        _baseline_draws(),
        latest_period=6000,
        config=AppConfig(min_score_threshold=10),
    )
    debug = predict_top3(
        _baseline_draws(),
        latest_period=6000,
        config=AppConfig(min_score_threshold=10),
        include_regime_debug=True,
    )

    assert "regime_metrics_raw" not in normal["metadata"]
    assert "regime_metrics_raw" in debug["metadata"]


def test_lightweight_detector_path_when_debug_false() -> None:
    draws = _baseline_draws()
    light = detect_regime(
        draws,
        config=AppConfig(),
        include_debug_metrics=False,
    )
    heavy = detect_regime(
        draws,
        config=AppConfig(),
        include_debug_metrics=True,
    )
    assert light["regime_metrics_zscore"] == {}
    assert light["regime_metrics_percentile"] == {}
    assert heavy["regime_metrics_zscore"] != {}
    assert light["regime_window"] == len(draws)
    assert heavy["regime_window"] == len(draws)


def test_fail_fast_when_draw_count_not_enough() -> None:
    try:
        predict_top3(
            _baseline_draws()[:20],
            latest_period=10,
            config=AppConfig(min_prediction_draws=30),
        )
    except Exception as exc:  # noqa: BLE001
        assert "Need >=" in str(exc)
    else:
        assert False, "expected fail-fast error"


def test_recent_window_with_less_than_50_uses_available_draws() -> None:
    draws = _baseline_draws_n(48)
    result = predict_top3(
        draws,
        latest_period=10,
        config=AppConfig(
            min_prediction_draws=10,
            recent_draws_count=5,
        ),
    )
    assert result["metadata"]["available_draws"] == 48
    assert result["metadata"]["analyzed_draws"] == 5
    assert result["metadata"]["effective_draws_used"] == 5
    assert result["metadata"]["effective_recent_window"] == 5
    assert result["metadata"]["regime_window"] == 5


def test_recent_window_uses_latest_5_when_history_is_large() -> None:
    draws = (_baseline_draws() * 27)[:1348]
    result = predict_top3(
        draws,
        latest_period=10,
        config=AppConfig(
            min_prediction_draws=10,
            recent_draws_count=5,
        ),
    )
    assert result["metadata"]["available_draws"] == 1348
    assert result["metadata"]["analyzed_draws"] == 5
    assert result["metadata"]["effective_draws_used"] == 5
    assert result["metadata"]["effective_recent_window"] == 5
    assert result["metadata"]["regime_window"] == 5


def test_regime_disabled_when_insufficient_history_but_predict_ok() -> None:
    draws = _baseline_draws_n(12)
    result = predict_top3(
        draws,
        latest_period=10,
        config=AppConfig(
            min_prediction_draws=10,
            regime=RegimeConfig(min_history=20),
        ),
    )
    assert result["metadata"]["regime"] == "normal"
    assert result["metadata"]["regime_adjustment_enabled"] is False
    assert result["metadata"]["fallback_to_normal"] is True
    assert (
        result["metadata"]["regime_disabled_reason"]
        == "insufficient_history"
    )


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


def test_max_recent_draws_count_caps_effective_window() -> None:
    draws = (_baseline_draws() * 20)[:500]
    result = predict_top3(
        draws,
        latest_period=10,
        config=AppConfig(
            min_prediction_draws=10,
            recent_draws_count=5,
            max_recent_draws_count=4,
            min_score_threshold=10,
        ),
    )
    assert result["metadata"]["analyzed_draws"] == 4
    assert result["metadata"]["effective_draws_used"] == 4


def test_effective_recent_window_never_exceeds_five_even_for_long_inputs(
) -> None:
    draws = (_baseline_draws() * 5)[:120]
    result = predict_top3(
        draws,
        latest_period=10,
        config=AppConfig(
            min_prediction_draws=12,
            recent_draws_count=80,
            max_recent_draws_count=80,
            min_score_threshold=10,
        ),
    )
    assert result["metadata"]["analyzed_draws"] == 5
    assert result["metadata"]["effective_draws_used"] == 5
    assert result["metadata"]["effective_recent_window"] == 5
