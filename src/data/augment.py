"""Masking and augmentation utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def mask_board(
    board: torch.Tensor,
    ratio: Optional[float] = None,
    *,
    min_ratio: float = 0.03,
    max_ratio: float = 0.70,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask a board by replacing some entries with 0.

    Returns the masked board and a boolean mask of the same shape indicating
    which positions were masked.
    """
    if board.ndim != 2:
        raise ValueError("board must be 2D")
    rng = rng or np.random.default_rng()
    total = board.numel()
    if ratio is None:
        ratio = float(rng.uniform(min_ratio, max_ratio))
    ratio = min(max(ratio, 0.0), 1.0)
    k = max(1, min(total, int(round(total * ratio))))
    flat_indices = rng.choice(total, k, replace=False)
    mask = torch.zeros(total, dtype=torch.bool)
    mask[flat_indices] = True
    tokens = board.clone().flatten()
    tokens[mask] = 0
    return tokens.view_as(board), mask.view_as(board)
