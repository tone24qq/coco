"""Miscellaneous metric helpers."""

from __future__ import annotations

import torch


def exact_match(pred: torch.Tensor, target: torch.Tensor) -> bool:
    """Return True if all tokens match."""
    return bool(torch.all(pred == target))
