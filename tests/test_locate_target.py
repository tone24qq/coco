import numpy as np

from modules import generate_excel_style_card, locate_target_by_partial_grid


def test_locate_target_visible():
    rng = np.random.default_rng(0)
    board = generate_excel_style_card(4, 4, rng)
    target = int(board[1, 2])
    row, col = locate_target_by_partial_grid(board.tolist(), target)
    assert (row, col) == (1, 2)


def test_locate_target_hidden_valid_range():
    rng = np.random.default_rng(1)
    board = generate_excel_style_card(5, 5, rng)
    target = int(board[2, 3])
    board[2, 3] = -1
    row, col = locate_target_by_partial_grid(board.tolist(), target)
    assert 0 <= row < 5 and 0 <= col < 5
