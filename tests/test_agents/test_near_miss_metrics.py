import numpy as np

from src.backtest import _compute_extended_metrics


def _scores_from_order(order: list[int]) -> np.ndarray:
    scores = np.zeros(80, dtype=float)
    for rank, n in enumerate(order):
        scores[n - 1] = 80 - rank
    return scores


def test_exact_and_adj_and_strict_metrics() -> None:
    order = [10, 11, 12] + [n for n in range(1, 81) if n not in {10, 11, 12}]
    actual = {10, 13, 40}
    m = _compute_extended_metrics(_scores_from_order(order), actual)

    assert m["exact_hit@3"] == 1 / 3
    assert m["adj_hit_pm1@3"] == 2 / 3
    assert m["strict_adj_only_pm1@3"] == 2 / 3
    assert m["top3_at_least_one_exact"] == 1.0
    assert m["top3_at_least_one_adj_pm1"] == 1.0
    assert m["top3_at_least_one_strict_adj_only_pm1"] == 1.0


def test_one_to_one_match_prevents_double_counting() -> None:
    order = [10, 12, 30] + [n for n in range(1, 81) if n not in {10, 12, 30}]
    actual = {11, 50, 60}
    m = _compute_extended_metrics(_scores_from_order(order), actual)

    assert m["adj_hit_pm1@3"] == 1 / 3
    assert m["strict_adj_only_pm1@3"] == 1 / 3


def test_boundary_numbers_handle_pm1_direction() -> None:
    order = [1, 80, 2, 79] + [n for n in range(1, 81) if n not in {1, 2, 79, 80}]
    actual = {2, 79, 40}
    m = _compute_extended_metrics(_scores_from_order(order), actual)

    assert m["near_miss_minus1_count"] >= 1
    assert m["near_miss_plus1_count"] >= 1
    assert m["strict_adj_only_pm1@3"] == 2 / 3


def test_distance_metrics_fixture() -> None:
    order = [9, 20, 31, 45, 60, 61, 62, 63, 64, 65] + [
        n for n in range(1, 81) if n not in {9, 20, 31, 45, 60, 61, 62, 63, 64, 65}
    ]
    actual = {10, 21, 30}
    m = _compute_extended_metrics(_scores_from_order(order), actual)

    assert m["mean_min_distance_at_3"] == 1.0
    assert m["median_min_distance_at_3"] == 1.0
    assert m["mean_min_distance_at_10"] > 1.0
