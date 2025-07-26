import numpy as np
import pytest

from agents.scratch_solver_agent import predict


def _make_unique_board(rows: int, cols: int, blanks: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    numbers = np.arange(1, rows * cols + 1)
    rng.shuffle(numbers)
    board = numbers.reshape(rows, cols)
    blank_indices = rng.choice(rows * cols, size=blanks, replace=False)
    for idx in blank_indices:
        r, c = divmod(idx, cols)
        board[r, c] = -1
    return board


def test_predict_interface():
    board = _make_unique_board(4, 5, blanks=6, seed=42)
    target = 3
    result = predict(board.copy(), target=target)
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, dict)
        assert {"row", "col", "score"} <= item.keys()
        assert (
            board[item["row"], item["col"]] == -1
            or board[item["row"], item["col"]] == target
        )


def test_duplicate_numbers_error():
    board = np.array([[1, 2], [2, -1]])
    with pytest.raises(ValueError):
        predict(board, target=1)
