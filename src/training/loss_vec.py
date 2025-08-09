from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_logits_by_N_batched(logits: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
    """Mask logits so that values beyond each sample's N are ignored.

    Args:
        logits: Tensor of shape [B, L, V].
        N: Tensor of shape [B] containing the maximum valid token per sample.
    """

    B, L, V = logits.shape
    ar = torch.arange(V, device=logits.device).view(1, 1, V)
    Nv = N.view(B, 1, 1)
    valid = ar <= Nv
    return torch.where(valid, logits, torch.full_like(logits, -1e9))


def compute_loss_vectorized(
    logits: torch.Tensor, target: torch.Tensor, N: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy loss over variable-N vocabularies without Python loops."""

    logits = clip_logits_by_N_batched(logits, N)
    B, L, V = logits.shape
    loss = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1), ignore_index=0)
    return loss
