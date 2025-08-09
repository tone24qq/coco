"""Training loss functions."""

from __future__ import annotations

import torch
from torch import nn


def masked_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Cross entropy computed only on masked positions."""
    loss = nn.functional.cross_entropy(logits[mask], target[mask])
    return loss
