import numpy as np
import pytest

from dataset import BLANK_VALUE, validate_board


def test_validate_board_require_complete() -> None:
    board = np.array([[1, 2], [3, BLANK_VALUE]])
    with pytest.raises(ValueError):
        validate_board(board, allow_blank=True, require_complete=True)
