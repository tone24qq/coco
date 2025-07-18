import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # noqa: E402

import numpy as np  # noqa: E402

from src.features import _board_features  # noqa: E402


def test_board_features_shape():
    board = np.random.randint(0, 10, size=(5, 5))
    board[0, 0] = -1
    feats = _board_features(board)
    assert feats.ndim == 1
    assert feats.size > 0
