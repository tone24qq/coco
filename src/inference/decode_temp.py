from __future__ import annotations

import torch

from src.models import constraints
from src.models.vocab import masked_logits_clip

from .decode import select_topk


@torch.no_grad()
def iterative_decode_temp(
    model,
    tokens: torch.Tensor,
    attn_mask: torch.Tensor | None,
    N: int,
    *,
    steps: int = 8,
    fill_ratio: float = 0.3,
    temperature: float = 1.0,
    topk: int | None = None,
    topp: float | None = None,
) -> torch.Tensor:
    """Iterative decode with optional sampling."""

    B, L = tokens.shape
    for _ in range(steps):
        logits = model(tokens, attn_mask)
        logits = masked_logits_clip(logits, N)
        if temperature != 1.0:
            logits = logits / max(1e-6, temperature)
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(-1)
        masked = tokens.eq(0)
        sel = select_topk(masked, conf, fill_ratio)

        if topk is not None or (topp is not None and 0 < topp < 1):
            sampled = pred.clone()
            for b in range(B):
                idxs = torch.where(sel[b])[0]
                if idxs.numel() == 0:
                    continue
                for i in idxs.tolist():
                    p = probs[b, i, 1 : N + 1]
                    if topk is not None:
                        k = min(topk, p.numel())
                        vals, ind = torch.topk(p, k)
                        p_cut = torch.zeros_like(p)
                        p_cut[ind] = vals
                        p = p_cut
                    if topp is not None and 0 < topp < 1:
                        sorted_p, sorted_idx = torch.sort(p, descending=True)
                        csum = torch.cumsum(sorted_p, dim=-1)
                        keep = csum <= topp
                        if not torch.any(keep):
                            keep[0] = True
                        p_cut = torch.zeros_like(p)
                        p_cut[sorted_idx[keep]] = p[sorted_idx[keep]]
                        p = p_cut
                    if p.sum() <= 0:
                        choice = torch.argmax(p).item() + 1
                    else:
                        p = p / p.sum()
                        choice = torch.multinomial(p, num_samples=1).item() + 1
                    sampled[b, i] = choice
            pred = sampled

        tokens = tokens.clone()
        tokens[sel] = pred[sel]
        tokens = constraints.uniqueness_projection(logits, masked, N)
        if not tokens.eq(0).any():
            break

    if tokens.eq(0).any():
        tokens = constraints.uniqueness_projection(logits, tokens.eq(0), N)
    return tokens
