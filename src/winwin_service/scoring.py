from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict

from .config import AppConfig, DEFAULT_CONFIG, RegimeConfig


class PredictError(RuntimeError):
    """Raised when prediction cannot be produced."""


# NOTE: This detector is a heuristic short-term oscillation classifier.
# It separates "normal oscillation" vs "possible anomaly" bands only.
# It is not a probability guarantee, not a proven predictive model,
# and not betting advice.

REGIME_PRIORITY = [
    "hot_continuation",
    "warm_rebound",
    "concentrated",
    "dispersed",
]


def _calc_triplet_base_score(
    triplet: tuple[int, int, int],
    skips: dict[int, int],
    streaks: dict[int, int],
    pair_counts: dict[tuple[int, int], int],
    config: AppConfig,
) -> tuple[float, dict[str, float]]:
    score = 0.0
    n1, n2, n3 = triplet

    streak_nums = [
        n
        for n in triplet
        if config.streak_min <= streaks[n] <= config.streak_max
    ]
    if len(streak_nums) == 1:
        s_val = streaks[streak_nums[0]]
        if s_val == 1:
            score += 20
        elif s_val == 2:
            score += 10
        elif s_val == 3:
            score += 5
    elif len(streak_nums) == 2:
        score += 5
    elif len(streak_nums) == 3:
        score -= 20

    warm_nums = [
        n
        for n in triplet
        if config.warm_skip_min <= skips[n] <= config.warm_skip_max
    ]
    if len(warm_nums) == 1:
        w_val = skips[warm_nums[0]]
        if w_val == 3:
            score += 20
        elif w_val == 4:
            score += 10
        elif w_val == 5:
            score += 5
    elif len(warm_nums) == 2:
        score += 5
    elif len(warm_nums) == 3:
        score -= 20

    weights = config.score_weights

    tails = [n % 10 for n in triplet]
    unique_tails = len(set(tails))
    if unique_tails == 2:
        score += weights.tail_perfect
    elif unique_tails == 1:
        score += weights.tail_bad

    odds = sum(1 for n in triplet if n % 2 != 0)
    if odds in (1, 2):
        score += weights.parity_balance

    smalls = sum(1 for n in triplet if n <= 40)
    if smalls in (1, 2):
        score += weights.size_balance

    tens = [n // 10 for n in triplet]
    tens_unique = len(set(tens))
    if tens_unique == 3:
        score += weights.dispersion

    pair1 = pair_counts[(n1, n2)]
    pair2 = pair_counts[(n2, n3)]
    pair3 = pair_counts[(n1, n3)]
    momentum_score = min(15, (pair1 + pair2 + pair3))
    score += momentum_score

    components = {
        "momentum_score": float(momentum_score),
        "warm_skip_count": float(len(warm_nums)),
        "streak_count": float(len(streak_nums)),
        "pair_sum": float(pair1 + pair2 + pair3),
        "tail_unique": float(unique_tails),
        "tens_unique": float(tens_unique),
    }
    return score, components


def _max_consecutive_run(values: list[int]) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    best = 1
    run = 1
    for prev, curr in zip(sorted_values, sorted_values[1:]):
        if curr == prev + 1:
            run += 1
        else:
            run = 1
        if run > best:
            best = run
    return best


def _compute_metric_snapshot(
    draws: list[list[int]],
) -> dict[str, float]:
    last_draw = draws[-1]
    current = set(last_draw)
    prev = set(draws[-2])

    pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    num_counts: Counter[int] = Counter()
    skips = {i: 0 for i in range(1, 81)}
    streaks = {i: 0 for i in range(1, 81)}

    for draw in draws:
        sorted_draw = sorted(draw)
        num_counts.update(sorted_draw)
        for pair in itertools.combinations(sorted_draw, 2):
            pair_counts[pair] += 1
        draw_set = set(draw)
        for num in range(1, 81):
            if num in draw_set:
                skips[num] = 0
                streaks[num] += 1
            else:
                skips[num] += 1
                streaks[num] = 0

    pair_peak = max(pair_counts.values()) if pair_counts else 0
    pair_concentration = pair_peak / max(1, len(draws))

    overlap_prev = float(len(current & prev))

    skip_concentration = (
        sum(1 for val in skips.values() if val >= 6) / 80.0
    )
    streak_concentration = (
        sum(1 for val in streaks.values() if val >= 2) / 80.0
    )

    odd_count = sum(1 for n in current if n % 2 == 1)
    small_count = sum(1 for n in current if n <= 40)
    odd_even_drift = abs((odd_count / 20.0) - 0.5)
    small_large_drift = abs((small_count / 20.0) - 0.5)

    tens_counts: Counter[int] = Counter(n // 10 for n in current)
    tens_zone_concentration = max(tens_counts.values()) / 20.0
    tens_unique_zones = float(len(tens_counts))

    tail_counts: Counter[int] = Counter(n % 10 for n in current)
    total = sum(tail_counts.values())
    tail_entropy = 0.0
    for count in tail_counts.values():
        p = count / total
        tail_entropy -= p * math.log2(p)

    hot_number_peak = float(max(num_counts.values()))
    cold_number_floor = float(min(num_counts.values()))

    return {
        "pair_concentration": pair_concentration,
        "overlap_prev": overlap_prev,
        "skip_concentration": skip_concentration,
        "streak_concentration": streak_concentration,
        "odd_even_drift": odd_even_drift,
        "small_large_drift": small_large_drift,
        "tens_zone_concentration": tens_zone_concentration,
        "tens_unique_zones": tens_unique_zones,
        "tail_entropy": tail_entropy,
        "hot_number_peak": hot_number_peak,
        "cold_number_floor": cold_number_floor,
        "max_consecutive_run": float(_max_consecutive_run(last_draw)),
        "odd_count": float(odd_count),
        "small_count": float(small_count),
    }


def _metric_direction_value(
    metric_name: str,
    raw_metrics: dict[str, float],
) -> float:
    if metric_name == "tail_entropy_low":
        return -raw_metrics["tail_entropy"]
    if metric_name == "tail_entropy_high":
        return raw_metrics["tail_entropy"]
    if metric_name == "overlap_prev_low":
        return -raw_metrics["overlap_prev"]
    if metric_name == "tens_dispersion":
        return raw_metrics["tens_unique_zones"]
    return raw_metrics[metric_name]


def _rank_percentile(values: list[float], current: float) -> float:
    if not values:
        return 0.5
    smaller_or_equal = sum(1 for val in values if val <= current)
    return smaller_or_equal / float(len(values))


def _z_score(values: list[float], current: float) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return (current - mean) / std


def _band_from_stats(
    zscore: float,
    percentile: float,
    config: RegimeConfig,
) -> str:
    if (
        zscore >= config.anomaly_zscore
        or percentile >= config.anomaly_percentile
    ):
        return "anomaly"
    if (
        zscore >= config.warning_zscore
        or percentile >= config.warning_percentile
    ):
        return "warning"
    return "normal"


def _collect_detector_bands(
    metrics_raw: dict[str, float],
    history_values: dict[str, list[float]],
    config: RegimeConfig,
) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    zscores: dict[str, float] = {}
    percentiles: dict[str, float] = {}
    bands: dict[str, str] = {}

    metric_names = {
        "pair_concentration",
        "overlap_prev",
        "skip_concentration",
        "streak_concentration",
        "odd_even_drift",
        "small_large_drift",
        "tens_zone_concentration",
        "tens_unique_zones",
        "tail_entropy",
        "hot_number_peak",
        "cold_number_floor",
        "max_consecutive_run",
        "tail_entropy_low",
        "tail_entropy_high",
        "overlap_prev_low",
        "tens_dispersion",
    }

    for metric in sorted(metric_names):
        current = _metric_direction_value(metric, metrics_raw)
        window_values = history_values.get(metric, [])
        series = window_values[-config.percentile_window:]
        zscores[metric] = _z_score(series, current)
        percentiles[metric] = _rank_percentile(series, current)
        bands[metric] = _band_from_stats(
            zscores[metric],
            percentiles[metric],
            config,
        )

    return zscores, percentiles, bands


def _evaluate_regime_candidate(
    band_map: dict[str, str],
    config: RegimeConfig,
) -> tuple[str, int, int, list[str], list[str], str]:
    picked = "normal"
    picked_anomaly = 0
    picked_core = 0
    picked_anomaly_flags: list[str] = []
    picked_warning_flags: list[str] = []
    picked_band = "normal_band"

    for regime in REGIME_PRIORITY:
        core_metrics = config.core_metrics[regime]
        structural_metrics = config.structural_metrics[regime]

        core_anomaly = sum(1 for m in core_metrics if band_map[m] == "anomaly")
        structural_anomaly = sum(
            1 for m in structural_metrics if band_map[m] == "anomaly"
        )

        core_warning = sum(1 for m in core_metrics if band_map[m] == "warning")
        structural_warning = sum(
            1 for m in structural_metrics if band_map[m] == "warning"
        )

        anomaly_count = core_anomaly + structural_anomaly
        warning_count = core_warning + structural_warning

        if (
            core_anomaly >= config.min_core_hits
            and structural_anomaly >= config.min_structural_hits
        ):
            picked = regime
            picked_anomaly = anomaly_count
            picked_core = core_anomaly
            picked_anomaly_flags = [
                f"{regime}:{metric}"
                for metric in core_metrics + structural_metrics
                if band_map[metric] == "anomaly"
            ]
            picked_warning_flags = [
                f"{regime}:{metric}"
                for metric in core_metrics + structural_metrics
                if band_map[metric] == "warning"
            ]
            picked_band = "anomaly_band"
            break

        if (
            picked == "normal"
            and (core_warning + structural_warning) >= 2
            and warning_count >= anomaly_count
        ):
            picked = regime
            picked_anomaly = anomaly_count
            picked_core = core_anomaly
            picked_anomaly_flags = [
                f"{regime}:{metric}"
                for metric in core_metrics + structural_metrics
                if band_map[metric] == "anomaly"
            ]
            picked_warning_flags = [
                f"{regime}:{metric}"
                for metric in core_metrics + structural_metrics
                if band_map[metric] == "warning"
            ]
            picked_band = "warm_band"

    return (
        picked,
        picked_anomaly,
        picked_core,
        sorted(set(picked_anomaly_flags)),
        sorted(set(picked_warning_flags)),
        picked_band,
    )


def _normal_oscillation_flags(
    metrics_raw: dict[str, float],
    config: RegimeConfig,
) -> list[str]:
    flags: list[str] = []
    ranges = {
        "odd_count": (
            config.normal_odd_count_min,
            config.normal_odd_count_max,
        ),
        "small_count": (
            config.normal_small_count_min,
            config.normal_small_count_max,
        ),
        "overlap_prev": (
            config.normal_overlap_prev_min,
            config.normal_overlap_prev_max,
        ),
        "max_consecutive_run": (
            config.normal_max_streak_min,
            config.normal_max_streak_max,
        ),
        "tens_zone_peak": (
            config.normal_tens_peak_min,
            config.normal_tens_peak_max,
            metrics_raw["tens_zone_concentration"] * 20.0,
        ),
        "hot_number_peak": (
            config.normal_hot_number_min,
            config.normal_hot_number_max,
        ),
        "cold_number_floor": (
            config.normal_cold_number_min,
            config.normal_cold_number_max,
        ),
    }

    for metric, values in ranges.items():
        if metric == "tens_zone_peak":
            lower, upper, current = values
        else:
            lower, upper = values
            lookup = metric if metric in metrics_raw else "overlap_prev"
            current = metrics_raw[lookup]

        if lower <= current <= upper:
            flags.append(f"{metric}:normal")
        else:
            flags.append(f"{metric}:out_of_normal_band")

    return flags


def _quick_detector_bands(metrics_raw: dict[str, float]) -> dict[str, str]:
    bands: dict[str, str] = {}

    def _set(metric: str, warning: bool, anomaly: bool) -> None:
        if anomaly:
            bands[metric] = "anomaly"
        elif warning:
            bands[metric] = "warning"
        else:
            bands[metric] = "normal"

    _set(
        "overlap_prev",
        warning=metrics_raw["overlap_prev"] >= 8,
        anomaly=metrics_raw["overlap_prev"] >= 9,
    )
    _set(
        "overlap_prev_low",
        warning=metrics_raw["overlap_prev"] <= 2,
        anomaly=metrics_raw["overlap_prev"] <= 1,
    )
    _set(
        "max_consecutive_run",
        warning=metrics_raw["max_consecutive_run"] >= 5,
        anomaly=metrics_raw["max_consecutive_run"] >= 6,
    )
    _set(
        "hot_number_peak",
        warning=metrics_raw["hot_number_peak"] >= 24,
        anomaly=metrics_raw["hot_number_peak"] >= 25,
    )
    _set(
        "cold_number_floor",
        warning=metrics_raw["cold_number_floor"] <= 4,
        anomaly=metrics_raw["cold_number_floor"] <= 3,
    )
    _set(
        "skip_concentration",
        warning=metrics_raw["skip_concentration"] >= 0.16,
        anomaly=metrics_raw["skip_concentration"] >= 0.20,
    )
    _set(
        "small_large_drift",
        warning=metrics_raw["small_large_drift"] >= 0.18,
        anomaly=metrics_raw["small_large_drift"] >= 0.22,
    )
    _set(
        "pair_concentration",
        warning=metrics_raw["pair_concentration"] >= 0.75,
        anomaly=metrics_raw["pair_concentration"] >= 0.82,
    )
    _set(
        "tens_zone_concentration",
        warning=metrics_raw["tens_zone_concentration"] >= 0.30,
        anomaly=metrics_raw["tens_zone_concentration"] >= 0.35,
    )
    _set(
        "tail_entropy_low",
        warning=metrics_raw["tail_entropy"] <= 2.85,
        anomaly=metrics_raw["tail_entropy"] <= 2.60,
    )
    _set(
        "tail_entropy_high",
        warning=metrics_raw["tail_entropy"] >= 3.15,
        anomaly=metrics_raw["tail_entropy"] >= 3.25,
    )
    _set(
        "tens_dispersion",
        warning=metrics_raw["tens_unique_zones"] >= 8,
        anomaly=metrics_raw["tens_unique_zones"] >= 9,
    )
    _set(
        "odd_even_drift",
        warning=metrics_raw["odd_even_drift"] >= 0.20,
        anomaly=metrics_raw["odd_even_drift"] >= 0.25,
    )
    _set(
        "streak_concentration",
        warning=metrics_raw["streak_concentration"] >= 0.10,
        anomaly=metrics_raw["streak_concentration"] >= 0.14,
    )
    return bands


def detect_regime(
    recent_draws: list[list[int]],
    config: AppConfig,
    include_debug_metrics: bool = True,
) -> dict[str, object]:
    detector_cfg = config.regime
    draws = recent_draws[:]

    if len(draws) < detector_cfg.min_history:
        return {
            "regime": "normal",
            "anomaly_flags": [],
            "trigger_count": 0,
            "consecutive_trigger_hits": 0,
            "adjustment_strength": 0.0,
            "fallback_to_normal": True,
            "regime_window": len(draws),
            "regime_adjustment_enabled": False,
            "regime_disabled_reason": "insufficient_history",
            "metrics": {},
            "regime_metrics_raw": {},
            "regime_metrics_zscore": {},
            "regime_metrics_percentile": {},
            "normal_oscillation_flags": [],
            "warning_flags": [],
            "detector_band": "normal_band",
        }

    now_raw = _compute_metric_snapshot(draws)
    prev_raw = _compute_metric_snapshot(draws[:-1])
    if include_debug_metrics:
        metric_history: defaultdict[str, list[float]] = defaultdict(list)
        for idx in range(detector_cfg.min_history - 1, len(draws)):
            snapshot = _compute_metric_snapshot(draws[: idx + 1])
            metric_history["pair_concentration"].append(
                snapshot["pair_concentration"]
            )
            metric_history["overlap_prev"].append(snapshot["overlap_prev"])
            metric_history["skip_concentration"].append(
                snapshot["skip_concentration"]
            )
            metric_history["streak_concentration"].append(
                snapshot["streak_concentration"]
            )
            metric_history["odd_even_drift"].append(
                snapshot["odd_even_drift"]
            )
            metric_history["small_large_drift"].append(
                snapshot["small_large_drift"]
            )
            metric_history["tens_zone_concentration"].append(
                snapshot["tens_zone_concentration"]
            )
            metric_history["tens_unique_zones"].append(
                snapshot["tens_unique_zones"]
            )
            metric_history["tail_entropy"].append(snapshot["tail_entropy"])
            metric_history["hot_number_peak"].append(
                snapshot["hot_number_peak"]
            )
            metric_history["cold_number_floor"].append(
                snapshot["cold_number_floor"]
            )
            metric_history["max_consecutive_run"].append(
                snapshot["max_consecutive_run"]
            )
            metric_history["tail_entropy_low"].append(
                -snapshot["tail_entropy"]
            )
            metric_history["tail_entropy_high"].append(
                snapshot["tail_entropy"]
            )
            metric_history["overlap_prev_low"].append(
                -snapshot["overlap_prev"]
            )
            metric_history["tens_dispersion"].append(
                snapshot["tens_unique_zones"]
            )

        now_z, now_pct, now_bands = _collect_detector_bands(
            now_raw,
            metric_history,
            detector_cfg,
        )
        _, _, prev_bands = _collect_detector_bands(
            prev_raw,
            metric_history,
            detector_cfg,
        )
    else:
        now_z = {}
        now_pct = {}
        now_bands = _quick_detector_bands(now_raw)
        prev_bands = _quick_detector_bands(prev_raw)

    (
        now_regime,
        now_anomaly_count,
        _,
        now_anomaly_flags,
        now_warning_flags,
        now_detector_band,
    ) = _evaluate_regime_candidate(now_bands, detector_cfg)
    (
        prev_regime,
        prev_anomaly_count,
        _,
        prev_anomaly_flags,
        _,
        prev_detector_band,
    ) = _evaluate_regime_candidate(prev_bands, detector_cfg)

    consecutive_hits = 0
    if (
        now_regime != "normal"
        and now_regime == prev_regime
        and now_detector_band == "anomaly_band"
        and prev_detector_band == "anomaly_band"
    ):
        consecutive_hits = detector_cfg.consecutive_confirmation

    hold_kept = False
    if (
        consecutive_hits == 0
        and now_regime == prev_regime
        and prev_detector_band == "anomaly_band"
        and now_detector_band == "warm_band"
        and detector_cfg.hold_periods >= 2
    ):
        hold_kept = True

    regime = now_regime
    if not (
        consecutive_hits >= detector_cfg.consecutive_confirmation
        or hold_kept
    ):
        regime = "normal"

    fallback_to_normal = regime == "normal"

    trigger_count = now_anomaly_count
    if hold_kept:
        trigger_count = max(now_anomaly_count, prev_anomaly_count)

    adjustment_strength = 0.0
    if not fallback_to_normal:
        raw_strength = 0.02 * trigger_count
        adjustment_strength = min(detector_cfg.adjustment_cap, raw_strength)

    normal_flags = _normal_oscillation_flags(now_raw, detector_cfg)

    anomaly_flags = now_anomaly_flags
    if hold_kept:
        anomaly_flags = sorted(set(now_anomaly_flags + prev_anomaly_flags))

    return {
        "regime": regime,
        "anomaly_flags": anomaly_flags,
        "trigger_count": trigger_count,
        "consecutive_trigger_hits": consecutive_hits,
        "adjustment_strength": adjustment_strength,
        "fallback_to_normal": fallback_to_normal,
        "regime_window": len(draws),
        "regime_adjustment_enabled": not fallback_to_normal,
        "regime_disabled_reason": "",
        "metrics": now_raw,
        "regime_metrics_raw": now_raw,
        "regime_metrics_zscore": now_z,
        "regime_metrics_percentile": now_pct,
        "normal_oscillation_flags": (
            normal_flags if include_debug_metrics else []
        ),
        "warning_flags": now_warning_flags,
        "detector_band": now_detector_band,
    }


def _apply_regime_adjustment(
    base_score: float,
    components: dict[str, float],
    regime_info: dict[str, object],
) -> float:
    regime = str(regime_info.get("regime", "normal"))
    strength = float(regime_info.get("adjustment_strength", 0.0))
    if regime == "normal" or strength <= 0:
        return base_score

    signal = 0.0
    if regime == "hot_continuation":
        signal = (
            components["momentum_score"] * 0.5
            + components["streak_count"] * 4
        )
    elif regime == "warm_rebound":
        signal = components["warm_skip_count"] * 5
    elif regime == "concentrated":
        signal = (
            components["pair_sum"] * 0.6
            + (4 - components["tail_unique"]) * 4
            + (4 - components["tens_unique"]) * 4
        )
    elif regime == "dispersed":
        signal = (
            components["tail_unique"] * 5
            + components["tens_unique"] * 5
        )

    raw_delta = signal * strength
    max_delta = abs(base_score) * 0.10
    if max_delta == 0:
        max_delta = 1.0
    delta = max(-max_delta, min(max_delta, raw_delta))
    return base_score + delta


def _shared_numbers(a: list[int], b: list[int]) -> int:
    return len(set(a) & set(b))


def _select_diversified_top3(
    candidates: list[dict[str, object]],
) -> tuple[list[list[int]], bool]:
    if not candidates:
        return [], False

    selected: list[list[int]] = [list(candidates[0]["triplet"])]

    for entry in candidates[1:]:
        candidate_triplet = list(entry["triplet"])
        if all(
            _shared_numbers(candidate_triplet, picked) <= 1
            for picked in selected
        ):
            selected.append(candidate_triplet)
        if len(selected) == 3:
            return selected, False

    fallback_used = len(selected) < 3
    if fallback_used:
        for entry in candidates[1:]:
            candidate_triplet = list(entry["triplet"])
            if candidate_triplet in selected:
                continue
            if all(
                _shared_numbers(candidate_triplet, picked) <= 2
                for picked in selected
            ):
                selected.append(candidate_triplet)
            if len(selected) == 3:
                break

    if len(selected) < 3:
        for entry in candidates:
            candidate_triplet = list(entry["triplet"])
            if candidate_triplet not in selected:
                selected.append(candidate_triplet)
            if len(selected) == 3:
                break

    return selected[:3], fallback_used


def predict_top3(
    past_draws: list[list[int]],
    latest_period: int,
    config: AppConfig = DEFAULT_CONFIG,
    include_regime_debug: bool = False,
) -> dict[str, object]:
    available_draws = len(past_draws)
    if available_draws < config.min_prediction_draws:
        raise PredictError(
            "Need >= "
            f"{config.min_prediction_draws} draws, got {available_draws}"
        )

    recent_window = max(config.min_prediction_draws, config.recent_draws_count)
    if config.max_recent_draws_count is not None:
        recent_window = min(recent_window, config.max_recent_draws_count)
        recent_window = max(recent_window, config.min_prediction_draws)
    recent_draws = past_draws[-recent_window:]
    effective_draws_used = len(recent_draws)

    skips = {i: 0 for i in range(1, 81)}
    streaks = {i: 0 for i in range(1, 81)}

    for draw in recent_draws:
        draw_set = set(draw)
        for num in range(1, 81):
            if num in draw_set:
                skips[num] = 0
                streaks[num] += 1
            else:
                skips[num] += 1
                streaks[num] = 0

    pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    number_counts: Counter[int] = Counter()
    for draw in recent_draws:
        number_counts.update(draw)
        for pair in itertools.combinations(sorted(draw), 2):
            pair_counts[pair] += 1

    kill_zone = sorted(
        num
        for num in range(1, 81)
        if skips[num] >= config.skip_kill_threshold
        or streaks[num] >= config.streak_kill_threshold
    )
    valid_pool = sorted(set(range(1, 81)) - set(kill_zone))

    if len(valid_pool) < 3:
        raise PredictError(
            "Valid number pool below 3 after applying kill-zone "
            f"(valid_pool_size={len(valid_pool)}, "
            f"kill_zone_size={len(kill_zone)})"
        )

    regime_info = detect_regime(
        recent_draws,
        config=config,
        include_debug_metrics=include_regime_debug,
    )

    candidate_pool_before_trim = len(valid_pool)
    coarse_ranked_pool = sorted(
        valid_pool,
        key=lambda num: (
            number_counts[num] * 2
            - skips[num]
            + min(streaks[num], 3) * 2
        ),
        reverse=True,
    )
    trim_size = max(
        3,
        min(config.candidate_trim_size, len(coarse_ranked_pool)),
    )
    trimmed_pool = sorted(coarse_ranked_pool[:trim_size])
    candidate_pool_after_trim = len(trimmed_pool)
    candidates: list[dict[str, object]] = []

    for triplet in itertools.combinations(trimmed_pool, 3):
        base_score, components = _calc_triplet_base_score(
            triplet,
            skips=skips,
            streaks=streaks,
            pair_counts=pair_counts,
            config=config,
        )
        adjusted_score = _apply_regime_adjustment(
            base_score=base_score,
            components=components,
            regime_info=regime_info,
        )

        if adjusted_score >= config.min_score_threshold:
            candidates.append(
                {
                    "triplet": list(triplet),
                    "score": adjusted_score,
                    "raw_score": base_score,
                }
            )

    if not candidates:
        raise PredictError(
            "No combinations exceed min_score_threshold "
            f"(min_score_threshold={config.min_score_threshold}, "
            "qualified_combinations=0)"
        )

    candidates.sort(key=lambda c: float(c["score"]), reverse=True)
    top3, fallback_used = _select_diversified_top3(candidates)

    result = {
        "target_period": latest_period + 1,
        "latest_period": latest_period,
        "top3": top3,
        "kill_zone": kill_zone,
        "metadata": {
            "analyzed_draws": len(recent_draws),
            "available_draws": available_draws,
            "effective_draws_used": effective_draws_used,
            "min_prediction_draws": config.min_prediction_draws,
            "max_recent_draws_count": config.max_recent_draws_count,
            "regime_min_history": config.regime.min_history,
            "regime_disabled_reason": regime_info["regime_disabled_reason"],
            "valid_pool_size": len(valid_pool),
            "candidate_pool_before_trim": candidate_pool_before_trim,
            "candidate_pool_after_trim": candidate_pool_after_trim,
            "total_combinations": math.comb(len(valid_pool), 3),
            "total_combinations_evaluated": math.comb(
                len(trimmed_pool), 3
            ),
            "qualified_combinations": len(candidates),
            "min_score_threshold": config.min_score_threshold,
            "dedup_enabled": True,
            "dedup_rule": "shared<=1_then_shared<=2_fallback",
            "raw_top_candidates_considered": len(candidates),
            "fallback_used": fallback_used,
            "regime": regime_info["regime"],
            "anomaly_flags": regime_info["anomaly_flags"],
            "regime_adjustment_enabled": regime_info[
                "regime_adjustment_enabled"
            ],
            "regime_window": regime_info["regime_window"],
            "trigger_count": regime_info["trigger_count"],
            "consecutive_trigger_hits": regime_info[
                "consecutive_trigger_hits"
            ],
            "adjustment_strength": regime_info["adjustment_strength"],
            "fallback_to_normal": regime_info["fallback_to_normal"],
            "warning_flags": regime_info["warning_flags"],
            "detector_band": regime_info["detector_band"],
        },
    }

    if include_regime_debug:
        result["metadata"].update(
            {
                "regime_metrics": regime_info["metrics"],
                "regime_metrics_raw": regime_info["regime_metrics_raw"],
                "regime_metrics_zscore": regime_info[
                    "regime_metrics_zscore"
                ],
                "regime_metrics_percentile": regime_info[
                    "regime_metrics_percentile"
                ],
                "normal_oscillation_flags": regime_info[
                    "normal_oscillation_flags"
                ],
            }
        )

    return result
