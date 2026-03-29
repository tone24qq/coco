from __future__ import annotations

from dataclasses import dataclass, field


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
    recent_draws_count: int = 50
    min_prediction_draws: int = 10
    max_recent_draws_count: int | None = 50
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


DEFAULT_CONFIG = AppConfig()
