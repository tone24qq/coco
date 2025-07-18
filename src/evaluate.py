"""Evaluation utilities for model training."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import auc, roc_curve


def make_hit_rate_metric(k: int = 3):
    """Create LightGBM-compatible hit rate metric."""

    def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
        y_pred = (
            y_pred.reshape(len(np.unique(y_true)), -1).T if y_pred.ndim == 1 else y_pred
        )
        top_k = np.argsort(y_pred, axis=1)[:, -k:]
        hits = [t in row for row, t in zip(top_k, y_true)]
        return f"hit_rate@{k}", float(np.mean(hits)), True

    return hit_rate


def calc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate AUC for binary problems."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return auc(fpr, tpr)
