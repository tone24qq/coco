"""Iterative decoding loop."""

from __future__ import annotations

import torch

from ..models import constraints
from ..models.vocab import masked_logits_clip


def select_topk(
    masked: torch.Tensor, conf: torch.Tensor, fill_ratio: float
) -> torch.Tensor:
    sel = torch.zeros_like(masked)
    for b in range(masked.size(0)):
        idx = torch.where(masked[b])[0]
        if idx.numel() == 0:
            continue
        k = max(1, int(round(idx.numel() * fill_ratio)))
        topk = conf[b, idx].topk(k).indices
        sel[b, idx[topk]] = True
    return sel


@torch.no_grad()
def iterative_decode(
    model,
    tokens: torch.Tensor,
    attn_mask: torch.Tensor | None,
    N: int,
    *,
    steps: int = 8,
    fill_ratio: float = 0.3,
) -> torch.Tensor:
    """Iteratively fill in masked tokens using model predictions."""
    for _ in range(steps):
        logits = model(tokens, attn_mask)
        logits = masked_logits_clip(logits, N)
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(-1)
        masked = tokens.eq(0)
        sel = select_topk(masked, conf, fill_ratio)
        tokens = tokens.clone()
        tokens[sel] = pred[sel]
        tokens = constraints.uniqueness_projection(logits, masked, N)
        if not tokens.eq(0).any():
            break
    tokens = constraints.uniqueness_projection(logits, tokens.eq(0), N)
    return tokens
