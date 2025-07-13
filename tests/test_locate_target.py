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
    assert (row, col) == (2, 3)


def test_locate_target_with_sample_library():
    rng = np.random.default_rng(2)
    board = generate_excel_style_card(3, 3, rng)
    target = int(board[0, 1])
    partial = board.copy()
    partial[0, 1] = -1
    library = [board.tolist()]
    row, col = locate_target_by_partial_grid(
        partial.tolist(), target, sample_library=library
    )
    assert (row, col) == (0, 1)
