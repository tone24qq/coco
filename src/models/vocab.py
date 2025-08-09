"""Dynamic vocabulary masking."""

from __future__ import annotations

import torch


def masked_logits_clip(logits: torch.Tensor, N: int) -> torch.Tensor:
    """Set logits beyond `N` to ``-inf`` so softmax ignores them."""
    if logits.size(-1) > N + 1:
        logits = logits.clone()
        logits[..., N + 1 :] = -float("inf")
    return logits
