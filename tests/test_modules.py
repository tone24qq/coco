import numpy as np

from modules import global_offset_cooccurrence, neighbor_value_distribution


def test_global_offset_cooccurrence_batch():
    boards = np.array(
        [
            [[1, -1], [2, 3]],
            [[2, -1], [1, 4]],
        ]
    )
    out = global_offset_cooccurrence(boards, target=1, offsets=[1, -1])
    assert out.shape == boards.shape
    assert np.all(out >= 0)


def test_global_offset_cooccurrence_single():
    board = np.array([[1, -1], [2, 3]])
    out = global_offset_cooccurrence(board, target=1, offsets=[1])
    assert out.shape == board.shape
    assert np.all(out >= 0)


def test_neighbor_value_distribution_fallback():
    board = np.array([[4, 10], [5, -1]])
    base = neighbor_value_distribution(board, target=7, tolerance=1, radius=1)
    near = neighbor_value_distribution(
        board, target=7, tolerance=1, radius=1, nearest_k=1
    )
    assert near.shape == board.shape
    assert not np.allclose(base, near)
