import numpy as np
import torch

from src.data.augment import mask_board


def test_mask_board_ratio_bounds() -> None:
    board = torch.arange(1, 17).view(4, 4)
    masked, _ = mask_board(board, rng=np.random.default_rng(0))
    ratio = masked.eq(0).float().mean().item()
    assert 0.03 <= ratio <= 0.70


def test_mask_board_exact_ratio() -> None:
    board = torch.arange(1, 10).view(3, 3)
    masked, mask = mask_board(board, ratio=0.5, rng=np.random.default_rng(1))
    assert masked.eq(0).sum() == mask.sum()
    assert mask.sum().item() in {4, 5}
