from __future__ import annotations

from dataclasses import dataclass, field


RECENT_WINDOW_MIN = 1
RECENT_WINDOW_MAX = 5


@dataclass(frozen=True)
class ScoreWeights:
    streak_perfect: int = 15
    streak_ok: int = 5
    streak_bad: int = -20
    warm_perfect: int = 15
    warm_ok: int = 5
    warm_bad: int = -20
    tail_perfect: int = 20
    tail_bad: int = -15
    parity_balance: int = 10
    size_balance: int = 10
    dispersion: int = 10
    momentum_multiplier: int = 2
    momentum_cap: int = 30


@dataclass(frozen=True)
class RegimeConfig:
    min_history: int = 10
    min_core_hits: int = 2
    min_structural_hits: int = 1
    consecutive_confirmation: int = 2
    hold_periods: int = 2
    adjustment_cap: float = 0.10
    percentile_window: int = 25
    warning_zscore: float = 1.3
    anomaly_zscore: float = 1.8
    warning_percentile: float = 0.90
    anomaly_percentile: float = 0.97
    quick_overlap_prev_warning: float = 8.0
    quick_overlap_prev_anomaly: float = 9.0
    quick_overlap_prev_low_warning: float = 2.0
    quick_overlap_prev_low_anomaly: float = 1.0
    quick_max_consecutive_run_warning: float = 5.0
    quick_max_consecutive_run_anomaly: float = 6.0
    quick_hot_number_peak_warning: float = 24.0
    quick_hot_number_peak_anomaly: float = 25.0
    quick_cold_number_floor_warning: float = 4.0
    quick_cold_number_floor_anomaly: float = 3.0
    quick_skip_concentration_warning: float = 0.16
    quick_skip_concentration_anomaly: float = 0.20
    quick_small_large_drift_warning: float = 0.18
    quick_small_large_drift_anomaly: float = 0.22
    quick_pair_concentration_warning: float = 0.75
    quick_pair_concentration_anomaly: float = 0.82
    quick_tens_zone_concentration_warning: float = 0.30
    quick_tens_zone_concentration_anomaly: float = 0.35
    quick_tail_entropy_low_warning: float = 2.85
    quick_tail_entropy_low_anomaly: float = 2.60
    quick_tail_entropy_high_warning: float = 3.15
    quick_tail_entropy_high_anomaly: float = 3.25
    quick_tens_dispersion_warning: float = 8.0
    quick_tens_dispersion_anomaly: float = 9.0
    quick_odd_even_drift_warning: float = 0.20
    quick_odd_even_drift_anomaly: float = 0.25
    quick_streak_concentration_warning: float = 0.10
    quick_streak_concentration_anomaly: float = 0.14
    normal_overlap_prev_min: int = 2
    normal_overlap_prev_max: int = 8
    normal_odd_count_min: int = 7
    normal_odd_count_max: int = 13
    normal_small_count_min: int = 7
    normal_small_count_max: int = 13
    normal_max_streak_min: int = 2
    normal_max_streak_max: int = 4
    normal_tens_peak_min: int = 4
    normal_tens_peak_max: int = 6
    normal_hot_number_min: int = 4
    normal_hot_number_max: int = 23
    normal_cold_number_min: int = 4
    normal_cold_number_max: int = 23
    core_metrics: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "hot_continuation": (
                "overlap_prev",
                "max_consecutive_run",
                "hot_number_peak",
            ),
            "warm_rebound": (
                "skip_concentration",
                "cold_number_floor",
                "small_large_drift",
            ),
            "concentrated": (
                "pair_concentration",
                "tens_zone_concentration",
                "tail_entropy_low",
            ),
            "dispersed": (
                "tail_entropy_high",
                "tens_dispersion",
                "overlap_prev_low",
            ),
        }
    )
    structural_metrics: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "hot_continuation": ("pair_concentration", "odd_even_drift"),
            "warm_rebound": ("skip_concentration", "odd_even_drift"),
            "concentrated": (
                "tens_zone_concentration",
                "tail_entropy_low",
            ),
            "dispersed": ("tens_dispersion", "tail_entropy_high"),
        }
    )


@dataclass(frozen=True)
class AppConfig:
    source_url: str = "https://winwin.tw/Bingo"
    request_timeout: int = 15
    recent_draws_count: int = 3
    min_prediction_draws: int = 10
    max_recent_draws_count: int | None = 5
    history_lookback_days: int = 7
    skip_kill_threshold: int = 10
    streak_kill_threshold: int = 4
    warm_skip_min: int = 3
    warm_skip_max: int = 5
    streak_min: int = 1
    streak_max: int = 3
    min_score_threshold: int = 60
    score_weights: ScoreWeights = ScoreWeights()
    regime: RegimeConfig = RegimeConfig()
    prediction_cache_ttl_seconds: int = 30
    candidate_trim_size: int = 40
    streak_score_len1_hit1: float = 20.0
    streak_score_len1_hit2: float = 10.0
    streak_score_len1_hit3: float = 5.0
    streak_score_len2: float = 5.0
    streak_score_len3: float = -20.0
    warm_score_len1_skip3: float = 20.0
    warm_score_len1_skip4: float = 10.0
    warm_score_len1_skip5: float = 5.0
    warm_score_len2: float = 5.0
    warm_score_len3: float = -20.0
    momentum_score_cap: float = 15.0
    transition_score_multiplier: float = 0.35
    coarse_number_count_weight: float = 2.0
    coarse_skip_weight: float = 1.0
    coarse_streak_weight: float = 2.0
    coarse_streak_cap: int = 3
    coarse_transition_weight: float = 1.5
    regime_hot_momentum_weight: float = 0.5
    regime_hot_streak_weight: float = 4.0
    regime_warm_skip_weight: float = 5.0
    regime_concentrated_pair_weight: float = 0.6
    regime_concentrated_tail_weight: float = 4.0
    regime_concentrated_tens_weight: float = 4.0
    regime_dispersed_tail_weight: float = 5.0
    regime_dispersed_tens_weight: float = 5.0
    regime_signal_hot_multiplier: float = 1.0
    regime_signal_warm_multiplier: float = 1.0
    regime_signal_concentrated_multiplier: float = 1.0
    regime_signal_dispersed_multiplier: float = 1.0
    regime_delta_base_ratio: float = 0.10
    regime_delta_min_abs: float = 1.0


DEFAULT_CONFIG = AppConfig()


def clamp_recent_window_value(value: int | None) -> int | None:
    if value is None:
        return None
    return max(RECENT_WINDOW_MIN, min(RECENT_WINDOW_MAX, int(value)))


def normalize_recent_window(
    recent_draws_count: int,
    max_recent_draws_count: int | None,
) -> tuple[int, int | None]:
    normalized_recent = clamp_recent_window_value(recent_draws_count)
    assert normalized_recent is not None
    normalized_max = clamp_recent_window_value(max_recent_draws_count)
    if normalized_max is None:
        return normalized_recent, None
    return min(normalized_recent, normalized_max), normalized_max
