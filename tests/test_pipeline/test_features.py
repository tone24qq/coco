import numpy as np

from src.features import _board_features


def test_board_features_shape():
    rng = np.random.default_rng(0)
    board = rng.integers(0, 10, size=(4, 4))
    board[0, 0] = -1
    features = _board_features(board)
    assert features.shape == (13,)
