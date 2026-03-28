from __future__ import annotations

from dataclasses import dataclass


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
class AppConfig:
    source_url: str = "https://winwin.tw/Bingo"
    request_timeout: int = 15
    recent_draws_count: int = 50
    history_lookback_days: int = 7
    skip_kill_threshold: int = 10
    streak_kill_threshold: int = 4
    warm_skip_min: int = 3
    warm_skip_max: int = 5
    streak_min: int = 1
    streak_max: int = 3
    min_score_threshold: int = 60
    score_weights: ScoreWeights = ScoreWeights()


DEFAULT_CONFIG = AppConfig()
