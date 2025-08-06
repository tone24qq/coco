"""Tests for :func:`filter_by_target` mask pattern filtering."""

import numpy as np

from app import filter_by_target
from dataset import BLANK_VALUE
from memory_loader import MEMORY_CACHE


def test_filter_by_target_mask_overlap() -> None:
    rows, cols = 2, 2
    board = np.array([[BLANK_VALUE, 1], [2, BLANK_VALUE]], dtype=int)
    keys = np.zeros((2, rows * cols), dtype=np.float32)
    vals = np.zeros((2, rows * cols), dtype=np.float32)
    targets = np.array([5, 5], dtype=np.int16)
    boards = np.array(
        [
            [BLANK_VALUE, 1, 2, BLANK_VALUE],
            [BLANK_VALUE, BLANK_VALUE, 2, 3],
        ],
        dtype=np.int8,
    )
    original = MEMORY_CACHE.get((rows, cols))
    MEMORY_CACHE[(rows, cols)] = (keys, vals, targets, boards)
    try:
        idx = filter_by_target(rows, cols, 5, board=board, min_mask_overlap=0.75)
        assert idx == [0]
    finally:
        if original is None:
            MEMORY_CACHE.pop((rows, cols), None)
        else:
            MEMORY_CACHE[(rows, cols)] = original
