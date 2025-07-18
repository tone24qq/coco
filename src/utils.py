"""Utility helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def to_feature_matrix(boards: Iterable[np.ndarray]) -> np.ndarray:
    """Convert iterable of boards to feature matrix."""
    from .features import _board_features

    return np.stack([_board_features(b) for b in boards])
