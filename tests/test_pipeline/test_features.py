import numpy as np
import pandas as pd

from src.features import _board_features, build_features


def test_board_features_shape():
    rng = np.random.default_rng(0)
    board = rng.integers(0, 10, size=(4, 4))
    board[0, 0] = -1
    features = _board_features(board)
    assert features.shape == (13,)


def test_build_features_parallel():
    rng = np.random.default_rng(1)
    boards = rng.integers(0, 10, size=(5, 16))
    df = pd.DataFrame(boards, columns=[f"cell_{i}" for i in range(16)])
    feat_df = build_features(df, (4, 4), n_jobs=2)
    assert feat_df.shape == (5, 13)
