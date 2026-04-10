import numpy as np

from src.multi_size_data_loader import _validate_complete_grid


def test_invalid_board_fail_fast_duplicate() -> None:
    grid = np.arange(1, 161).reshape(10, 16)
    grid[0, 0] = grid[0, 1]
    assert _validate_complete_grid(grid, "160") == "duplicate_values"


def test_invalid_board_fail_fast_out_of_range() -> None:
    grid = np.arange(1, 161).reshape(10, 16)
    grid[0, 0] = 999
    assert _validate_complete_grid(grid, "160") == "out_of_range_values"
