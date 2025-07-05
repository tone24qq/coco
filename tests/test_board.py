# tests/test_board.py
import numpy as np
import pytest

import brain
from modules import generate_unique_grid


def test_generate_unique_grid_basic():
    grid = generate_unique_grid(4, 4, hidden=(1, 2), rng=np.random.default_rng(0))
    assert grid.shape == (4, 4)
    values = [v for v in grid.ravel() if v != -1]
    assert len(values) == len(set(values))
    assert min(values) >= 1 and max(values) <= 16
    assert grid[1, 2] == -1


def test_generate_unique_grid_invalid_size():
    with pytest.raises(ValueError):
        generate_unique_grid(21, 5)


def test_generate_unique_grid_multi_hidden():
    grid = generate_unique_grid(
        4,
        5,
        hidden=[(0, 0), (1, 1)],
        rng=np.random.default_rng(1),
    )
    assert grid[0, 0] == -1 and grid[1, 1] == -1
    values = [v for v in grid.ravel() if v != -1]
    assert len(values) == len(set(values))


def test_fill_blanks_with_remaining_numbers():
    util = brain.BoardAnalyzerUtils()
    grid = np.array([[1, -1], [3, -1]])
    filled = util.fill_blanks_with_remaining_numbers(grid, rng=np.random.default_rng(0))
    assert -1 not in filled
    assert len(np.unique(filled)) == 4
    assert set(filled.ravel()) == {1, 2, 3, 4}
