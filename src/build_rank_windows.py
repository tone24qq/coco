"""Feature builder and tensor contract for transformer ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

FEATURE_VERSION = "rank_window_v2"
TENSOR_CONTRACT = {
    "raw_tensor": "[batch, 80, feature_dim]",
    "model_input_tensor": "[batch, 80, d_model]",
    "attention_axis": "candidate_to_candidate",
}
FEATURE_NAMES = [
    "number_norm",
    "gap_last_seen",
    "gap_zscore",
    "hit_count_5",
    "hit_count_10",
    "hit_count_20",
    "hit_count_50",
    "hit_count_100",
    "momentum_slope",
    "ewma_hit_score",
    "neighbor_pm1_hits",
    "neighbor_pm2_hits",
    "neighbor_decay_score",
    "zone_id",
    "tail_digit",
    "is_odd",
    "is_big",
    "prev_zone_low_ratio",
    "prev_odd_ratio",
    "prev_sum_norm",
    "prev_span_norm",
    "prev_consecutive_count",
    "retrieval_similarity",
    "retrieval_next_hit_freq",
]


@dataclass
class RankWindow:
    issue: str
    target_issue: str
    number_ids: np.ndarray
    features: np.ndarray
    feature_names: List[str]
    feature_version: str
    tensor_contract: Dict[str, str]


def _number_cols() -> List[str]:
    return [f"n{i}" for i in range(1, 21)]


def _hits_for_number(frame: pd.DataFrame, number: int) -> int:
    return int((frame[_number_cols()] == number).sum().sum())


def _gap_last_seen(context: pd.DataFrame, number: int) -> int:
    for idx, (_, row) in enumerate(context.iloc[::-1].iterrows(), start=1):
        numbers = {int(row[col]) for col in _number_cols()}
        if number in numbers:
            return idx
    return len(context) + 1


def _rolling_counts(context: pd.DataFrame, number: int) -> Dict[str, float]:
    windows = [5, 10, 20, 50, 100]
    result = {}
    for window in windows:
        scoped = context.tail(window)
        result[f"hit_count_{window}"] = float(_hits_for_number(scoped, number))
    return result


def _momentum(context: pd.DataFrame, number: int) -> Tuple[float, float]:
    counts = [_hits_for_number(context.tail(window), number) for window in [5, 20, 100]]
    slope = float((counts[0] - counts[1]) + (counts[1] - counts[2]))

    ewma = 0.0
    alpha = 0.2
    for _, row in context.iterrows():
        hit = 1.0 if number in {int(row[col]) for col in _number_cols()} else 0.0
        ewma = alpha * hit + (1 - alpha) * ewma
    return slope, ewma


def _neighbor_stats(context: pd.DataFrame, number: int) -> Tuple[float, float, float]:
    pm1 = {
        _hits_for_number(context.tail(20), n)
        for n in [number - 1, number + 1]
        if 1 <= n <= 80
    }
    pm2 = {
        _hits_for_number(context.tail(20), n)
        for n in [number - 2, number + 2]
        if 1 <= n <= 80
    }
    neighbor_decay = 0.0
    for delta in [1, 2]:
        for n in [number - delta, number + delta]:
            if 1 <= n <= 80:
                neighbor_decay += _hits_for_number(context.tail(20), n) / float(delta)
    return float(sum(pm1)), float(sum(pm2)), float(neighbor_decay)


def _prev_draw_context(context: pd.DataFrame) -> Dict[str, float]:
    prev = context.iloc[-1]
    numbers = [int(prev[col]) for col in _number_cols()]
    zone_low_ratio = sum(1 for n in numbers if n <= 40) / 20.0
    odd_ratio = sum(1 for n in numbers if n % 2 == 1) / 20.0
    draw_sum = sum(numbers) / (80.0 * 20.0)
    span = (max(numbers) - min(numbers)) / 79.0
    consecutive = sum(
        1 for a, b in zip(sorted(numbers), sorted(numbers)[1:]) if b - a == 1
    )
    return {
        "prev_zone_low_ratio": zone_low_ratio,
        "prev_odd_ratio": odd_ratio,
        "prev_sum_norm": draw_sum,
        "prev_span_norm": span,
        "prev_consecutive_count": float(consecutive),
    }


def _retrieval_features(
    context: pd.DataFrame, number: int, top_k: int = 5
) -> Tuple[float, float]:
    if len(context) < 6:
        return 0.0, 0.0

    current = context.iloc[-1]
    current_set = {int(current[col]) for col in _number_cols()}
    sims: List[Tuple[float, int]] = []
    for idx in range(len(context) - 2):
        hist = context.iloc[idx]
        hist_set = {int(hist[col]) for col in _number_cols()}
        inter = len(current_set & hist_set)
        union = len(current_set | hist_set)
        sim = inter / float(union) if union else 0.0
        next_row = context.iloc[idx + 1]
        hit = 1 if number in {int(next_row[col]) for col in _number_cols()} else 0
        sims.append((sim, hit))

    sims.sort(key=lambda x: x[0], reverse=True)
    top = sims[:top_k]
    if not top:
        return 0.0, 0.0
    similarity = float(sum(item[0] for item in top) / len(top))
    next_hit_freq = float(sum(item[1] for item in top) / len(top))
    return similarity, next_hit_freq


def _build_features(
    context: pd.DataFrame, target_issue: str
) -> Tuple[np.ndarray, np.ndarray]:
    number_ids = np.arange(1, 81, dtype=np.int64)
    prev_stats = _prev_draw_context(context)

    gaps = np.array(
        [_gap_last_seen(context, int(n)) for n in number_ids], dtype=np.float64
    )
    gap_mean, gap_std = float(gaps.mean()), float(gaps.std() + 1e-8)

    rows: List[List[float]] = []
    for idx, number in enumerate(number_ids):
        rolling = _rolling_counts(context, int(number))
        slope, ewma = _momentum(context, int(number))
        n1, n2, ndec = _neighbor_stats(context, int(number))
        rsim, rhit = _retrieval_features(context, int(number))
        rows.append(
            [
                float(number) / 80.0,
                float(gaps[idx]),
                float((gaps[idx] - gap_mean) / gap_std),
                rolling["hit_count_5"],
                rolling["hit_count_10"],
                rolling["hit_count_20"],
                rolling["hit_count_50"],
                rolling["hit_count_100"],
                slope,
                ewma,
                n1,
                n2,
                ndec,
                0.0 if number <= 40 else 1.0,
                float(number % 10),
                1.0 if number % 2 else 0.0,
                1.0 if number > 40 else 0.0,
                prev_stats["prev_zone_low_ratio"],
                prev_stats["prev_odd_ratio"],
                prev_stats["prev_sum_norm"],
                prev_stats["prev_span_norm"],
                prev_stats["prev_consecutive_count"],
                rsim,
                rhit,
            ]
        )

    feature_matrix = np.asarray(rows, dtype=np.float32)
    if feature_matrix.shape != (80, len(FEATURE_NAMES)):
        raise RuntimeError(f"Feature shape mismatch: {feature_matrix.shape}")
    return number_ids, feature_matrix


def build_inference_window(history: pd.DataFrame, window_size: int) -> RankWindow:
    if len(history) < 2:
        raise ValueError("Need at least 2 rows to build inference window")

    context = history.tail(window_size)
    latest_issue = str(context.iloc[-1]["issue"])
    target_issue = str(int(latest_issue) + 1)

    number_ids, features = _build_features(context, target_issue)
    return RankWindow(
        issue=latest_issue,
        target_issue=target_issue,
        number_ids=number_ids,
        features=features,
        feature_names=FEATURE_NAMES,
        feature_version=FEATURE_VERSION,
        tensor_contract=TENSOR_CONTRACT,
    )


def build_training_samples(
    history: pd.DataFrame, window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    if len(history) <= window_size:
        raise ValueError("Insufficient rows for training windows")

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for idx in range(window_size, len(history)):
        context = history.iloc[idx - window_size : idx]
        target = history.iloc[idx]
        number_ids, features = _build_features(context, str(target["issue"]))
        labels = np.zeros(80, dtype=np.float32)
        target_numbers = {int(target[col]) for col in _number_cols()}
        for i, number in enumerate(number_ids):
            if int(number) in target_numbers:
                labels[i] = 1.0
        xs.append(features)
        ys.append(labels)

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
