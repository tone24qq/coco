import numpy as np

from modules import global_offset_cooccurrence


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
