from __future__ import annotations

import json
from statistics import mean
from typing import Sequence

import numpy as np

from src.utils import build_recent_report

PRIMES_UNDER_80 = {
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
}


def _zone_counts(numbers: Sequence[int]) -> dict[str, int]:
    return {
        "A": int(sum(1 for n in numbers if n <= 20)),
        "B": int(sum(1 for n in numbers if 21 <= n <= 40)),
        "C": int(sum(1 for n in numbers if 41 <= n <= 60)),
        "D": int(sum(1 for n in numbers if 61 <= n <= 80)),
    }


def _window(values: list[float], size: int) -> list[float]:
    if not values:
        return []
    return values[-min(size, len(values)) :]


def _draw_metrics(draw: Sequence[int]) -> dict[str, float | int | dict[str, int]]:
    arr = sorted(int(x) for x in draw)
    odd = int(sum(1 for n in arr if n % 2 == 1))
    small = int(sum(1 for n in arr if n <= 40))
    s = int(sum(arr))
    mn = int(min(arr))
    mx = int(max(arr))
    span = int(mx - mn)
    prime = int(sum(1 for n in arr if n in PRIMES_UNDER_80))
    return {
        "odd_count": odd,
        "even_count": 20 - odd,
        "small_count": small,
        "big_count": 20 - small,
        "zone_counts": _zone_counts(arr),
        "issue_sum": s,
        "issue_average": float(mean(arr)),
        "issue_span": span,
        "issue_min": mn,
        "issue_max": mx,
        "prime_count": prime,
        "composite_count": 20 - prime,
    }


def build_local_analysis_bundle(recent_draws: Sequence[Sequence[int]]) -> dict:
    if not recent_draws:
        return {
            "comprehensive": {},
            "locations": {},
            "shape_oe": {},
            "shape_bs": {},
            "shape_po": {},
            "sumvalue": {},
            "span": {},
            "average": {},
            "total_reduce_mantissa": {},
            "max_min_mantissa": {},
            "total_analysis": {},
        }

    metrics = [_draw_metrics(d) for d in recent_draws]
    latest = metrics[-1]

    sums = [float(m["issue_sum"]) for m in metrics]
    spans = [float(m["issue_span"]) for m in metrics]
    avgs = [float(m["issue_average"]) for m in metrics]
    mins = [float(m["issue_min"]) for m in metrics]
    maxs = [float(m["issue_max"]) for m in metrics]

    w5_sum = _window(sums, 5)
    w20_sum = _window(sums, 20)
    w5_span = _window(spans, 5)
    w20_span = _window(spans, 20)

    comprehensive_components = {
        "sum_zscore_20": float(
            (sums[-1] - np.mean(w20_sum)) / max(float(np.std(w20_sum) or 1.0), 1.0)
        ),
        "span_zscore_20": float(
            (spans[-1] - np.mean(w20_span)) / max(float(np.std(w20_span) or 1.0), 1.0)
        ),
        "odd_even_balance": float(1.0 - abs(int(latest["odd_count"]) - 10) / 10.0),
        "big_small_balance": float(1.0 - abs(int(latest["big_count"]) - 10) / 10.0),
    }
    comprehensive_score = float(np.mean(list(comprehensive_components.values())))

    total_reduce = [
        {
            "issue_index": idx,
            "sum": int(sums[idx]),
            "span": int(spans[idx]),
            "sum_minus_span": int(sums[idx] - spans[idx]),
            "sum_mod_10": int(sums[idx] % 10),
            "span_mod_10": int(spans[idx] % 10),
        }
        for idx in range(len(sums))
    ]

    return {
        "comprehensive": {
            "score": comprehensive_score,
            "components": comprehensive_components,
            "window_size": min(20, len(metrics)),
        },
        "locations": {
            "latest_zone_counts": latest["zone_counts"],
            "zone_trend_5": {
                z: float(np.mean([float(m["zone_counts"][z]) for m in metrics[-5:]]))
                for z in ["A", "B", "C", "D"]
            },
        },
        "shape_oe": {
            "latest": {
                "odd": int(latest["odd_count"]),
                "even": int(latest["even_count"]),
            },
            "odd_mean_5": float(np.mean([m["odd_count"] for m in metrics[-5:]])),
            "odd_mean_20": float(np.mean([m["odd_count"] for m in metrics[-20:]])),
        },
        "shape_bs": {
            "latest": {
                "big": int(latest["big_count"]),
                "small": int(latest["small_count"]),
            },
            "big_mean_5": float(np.mean([m["big_count"] for m in metrics[-5:]])),
            "big_mean_20": float(np.mean([m["big_count"] for m in metrics[-20:]])),
        },
        "shape_po": {
            "latest": {
                "prime": int(latest["prime_count"]),
                "composite": int(latest["composite_count"]),
            },
            "prime_mean_20": float(np.mean([m["prime_count"] for m in metrics[-20:]])),
        },
        "sumvalue": {
            "latest": int(sums[-1]),
            "rolling_mean_5": float(np.mean(w5_sum)),
            "rolling_mean_20": float(np.mean(w20_sum)),
            "series": [int(x) for x in sums[-20:]],
        },
        "span": {
            "latest": int(spans[-1]),
            "rolling_mean_5": float(np.mean(w5_span)),
            "rolling_mean_20": float(np.mean(w20_span)),
            "series": [int(x) for x in spans[-20:]],
        },
        "average": {
            "latest": float(avgs[-1]),
            "rolling_mean_5": float(np.mean(_window(avgs, 5))),
            "rolling_mean_20": float(np.mean(_window(avgs, 20))),
            "series": [float(x) for x in avgs[-20:]],
        },
        "total_reduce_mantissa": {
            "latest": total_reduce[-1],
            "series": total_reduce[-20:],
        },
        "max_min_mantissa": {
            "latest": {
                "issue_max": int(maxs[-1]),
                "issue_min": int(mins[-1]),
                "max_minus_min": int(maxs[-1] - mins[-1]),
                "max_mod_10": int(maxs[-1] % 10),
                "min_mod_10": int(mins[-1] % 10),
            },
            "max_series": [int(x) for x in maxs[-20:]],
            "min_series": [int(x) for x in mins[-20:]],
        },
        "total_analysis": {
            "sample_size": len(metrics),
            "latest_metrics": latest,
            "recent_report_compatible": build_recent_report(recent_draws),
            "diagnostic": {
                "sum_std_20": float(np.std(w20_sum)) if w20_sum else 0.0,
                "span_std_20": float(np.std(w20_span)) if w20_span else 0.0,
            },
        },
    }


def parse_numbers_column(value: str) -> list[int]:
    raw = json.loads(value)
    return sorted(int(x) for x in raw)
