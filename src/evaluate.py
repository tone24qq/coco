"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def hit_rate_score(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 3) -> float:
    """Compute Top-K hit rate."""
    top_k = np.argsort(y_pred_proba, axis=1)[:, -k:]
    hits = [true in preds for true, preds in zip(y_true, top_k)]
    return float(np.mean(hits))


def make_hit_rate_metric(k: int = 3):
    """Create LightGBM-compatible evaluation metric for hit rate."""

    def _metric(y_true: np.ndarray, y_pred: np.ndarray):
        y_pred = y_pred.reshape(len(np.unique(y_true)), -1).T
        score = hit_rate_score(y_true, y_pred, k=k)
        return f"hit_rate@{k}", score, True

    return _metric


def evaluate_model(
    y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 3
) -> dict[str, float]:
    """Evaluate model performance."""
    metrics = {
        f"hit_rate@{k}": hit_rate_score(y_true, y_pred_proba, k=k),
    }
    if y_pred_proba.shape[1] == 2:
        metrics["auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
    return metrics
