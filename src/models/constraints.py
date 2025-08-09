"""Uniqueness projection enforcing 1..N usage."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def uniqueness_projection(
    logits: torch.Tensor, mask: torch.Tensor, N: int
) -> torch.Tensor:
    """Project masked positions to ensure numbers 1..N are used exactly once."""
    B, M, V = logits.shape
    out = logits.argmax(-1).clone()
    for b in range(B):
        idx = torch.where(mask[b])[0].cpu().numpy()
        if idx.size == 0:
            continue
        with torch.no_grad():
            p = torch.softmax(logits[b, idx, 1 : N + 1], dim=-1).cpu().numpy() + 1e-9
            cost = -np.log(p)
        nums = np.arange(1, N + 1)
        r, c = linear_sum_assignment(cost)
        assign = nums[c]
        out[b, idx] = torch.tensor(assign, device=out.device)
    return out
