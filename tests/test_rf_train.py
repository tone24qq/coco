from pathlib import Path

import numpy as np

from rf_infer.core import extract_features, infer_top3_for_target
from rf_infer.train import train_from_features


def test_train_and_infer(tmp_path: Path) -> None:
    features_dir = tmp_path / "features" / "2x2"
    features_dir.mkdir(parents=True)
    board_full = np.array([[1, 2], [3, 4]])
    X_list = []
    y_list = []
    board = board_full.copy()
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            board[r, c] = -1
            X_list.append(extract_features(board, r, c))
            y_list.append(board_full[r, c])
            board[r, c] = board_full[r, c]
    X = np.vstack(X_list)
    y = np.array(y_list)
    np.savez_compressed(features_dir / "2x2_features.npz", X=X, y=y)

    models_dir = tmp_path / "models"
    train_from_features(str(tmp_path / "features"), str(models_dir), n_estimators=5)

    model_path = models_dir / "2x2.pkl"
    assert model_path.exists()

    board = np.array([[-1, -1], [-1, -1]])
    top3 = infer_top3_for_target(board, 1, models_dir=str(models_dir))
    assert 1 <= len(top3) <= 3
