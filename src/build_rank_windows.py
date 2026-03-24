"""Build deterministic ranking windows for transformer inference/training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class RankWindow:
    issue: str
    target_issue: str
    number_ids: np.ndarray
    features: np.ndarray


def _number_frequency(context: pd.DataFrame, number: int, lookback: int) -> float:
    number_cols = [f"n{i}" for i in range(1, 21)]
    scoped = context.tail(lookback)
    if scoped.empty:
        return 0.0
    hits = (scoped[number_cols] == number).sum().sum()
    return float(hits) / float(len(scoped) * 20)


def _omission_gap(context: pd.DataFrame, number: int) -> float:
    number_cols = [f"n{i}" for i in range(1, 21)]
    for reverse_idx, (_, row) in enumerate(context.iloc[::-1].iterrows(), start=1):
        if number in set(int(row[col]) for col in number_cols):
            return float(reverse_idx)
    return float(len(context) + 1)


def _build_features(
    context: pd.DataFrame, target_issue: str
) -> Tuple[np.ndarray, np.ndarray]:
    number_ids = np.arange(1, 81, dtype=np.int64)
    target_mod = int(target_issue) % 10
    feature_rows: List[List[float]] = []

    for number in number_ids:
        omission = _omission_gap(context, int(number))
        feature_rows.append(
            [
                float(number) / 80.0,
                _number_frequency(context, int(number), 5),
                _number_frequency(context, int(number), 20),
                _number_frequency(context, int(number), 100),
                omission / float(max(len(context), 1)),
                math.sin((target_mod / 10.0) * math.tau),
                math.cos((target_mod / 10.0) * math.tau),
            ]
        )

    features = np.asarray(feature_rows, dtype=np.float64)
    if features.shape != (80, 7):
        raise RuntimeError(
            f"Feature shape mismatch: expected (80, 7), got {features.shape}"
        )

    return number_ids, features


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
    )


def build_training_samples(
    history: pd.DataFrame, window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    if len(history) <= window_size:
        raise ValueError("Insufficient rows for training windows")

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    number_cols = [f"n{i}" for i in range(1, 21)]

    for idx in range(window_size, len(history)):
        context = history.iloc[idx - window_size : idx]
        target = history.iloc[idx]
        number_ids, features = _build_features(context, str(target["issue"]))

        labels = np.zeros(80, dtype=np.float64)
        target_numbers = set(int(target[col]) for col in number_cols)
        for i, number in enumerate(number_ids):
            if int(number) in target_numbers:
                labels[i] = 1.0

        xs.append(features)
        ys.append(labels)

    x_array = np.asarray(xs, dtype=np.float64)
    y_array = np.asarray(ys, dtype=np.float64)
    return x_array, y_array
