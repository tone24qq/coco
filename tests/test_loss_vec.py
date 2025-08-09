"""Tests for vectorized loss functions."""

# isort: skip_file

import torch
import torch.nn.functional as F

from src.training.loss_vec import clip_logits_by_N_batched, compute_loss_vectorized


def test_clip_logits_by_n_batched_masks_invalid():
    logits = torch.zeros(1, 1, 5)
    N = torch.tensor([2])
    clipped = clip_logits_by_N_batched(logits, N)
    assert clipped[0, 0, 3:].eq(torch.full((2,), -1e9)).all()


def test_compute_loss_vectorized_matches_loop():
    torch.manual_seed(0)
    B, L, V = 2, 3, 5
    logits = torch.randn(B, L, V, dtype=torch.float32)
    target = torch.tensor([[1, 2, 3], [2, 0, 1]])
    N = torch.tensor([4, 2])

    loss_vec = compute_loss_vectorized(logits, target, N)

    # manual loop version for verification
    clipped = []
    for b in range(B):
        n = int(N[b].item())
        mask = torch.arange(V) <= n
        logits_b = logits[b : b + 1].clone()
        logits_b[..., ~mask] = -1e9
        clipped.append(logits_b)
    logits_clipped = torch.cat(clipped, dim=0)
    loss_manual = F.cross_entropy(
        logits_clipped.reshape(-1, V), target.reshape(-1), ignore_index=0
    )

    assert torch.isclose(loss_vec, loss_manual)
