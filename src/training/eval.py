"""Evaluation utilities."""

from __future__ import annotations

import torch


def top1_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute per-cell top-1 accuracy."""
    correct = (pred == target).float().mean().item()
    return float(correct)
