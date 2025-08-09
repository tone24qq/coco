"""Batch collation utilities."""

from __future__ import annotations

from typing import List, Tuple

import torch


def collate_boards(boards: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad boards in a batch to the same size and build attention masks.

    Returns a tuple of `(tokens, attn_mask)` where `tokens` is of shape
    `(B, M)` and `attn_mask` is a boolean tensor of the same shape marking
    valid positions.
    """
    max_len = max(b.numel() for b in boards)
    batch = torch.zeros(len(boards), max_len, dtype=torch.long)
    attn_mask = torch.zeros(len(boards), max_len, dtype=torch.bool)
    for i, board in enumerate(boards):
        flat = board.flatten()
        batch[i, : flat.numel()] = flat
        attn_mask[i, : flat.numel()] = True
    return batch, attn_mask
