import json
import subprocess
from pathlib import Path

import numpy as np

from train_lgbm_pipeline import _board_features


def test_duplicate_and_range_features() -> None:
    board = np.arange(1, 21).reshape(4, 5)
    board_masked = board.copy()
    board_masked[0, 2] = -1  # target 3 hidden

    feats = _board_features(board_masked, 3, (0, 2))
    assert feats[-2] == 0.0  # no duplicate
    assert feats[-1] == 1.0  # within range

    board_dup = board_masked.copy()
    board_dup[1, 1] = 3  # introduce duplicate value
    feats_dup = _board_features(board_dup, 3, (0, 2))
    assert feats_dup[-2] >= 1.0

    feats_out = _board_features(board_masked, 21, (0, 2))
    assert feats_out[-1] == 0.0


def test_features_saved(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    board = {"board": [[1, -1], [3, 4]], "target": 2}
    with open(data_dir / "sample.json", "w", encoding="utf-8") as f:
        json.dump(board, f)

    out_feat = tmp_path / "features"
    out_model = tmp_path / "models"

    subprocess.run(
        [
            "python",
            "train_lgbm_pipeline.py",
            "--root",
            str(data_dir),
            "--shard-size",
            "2",
            "--trees-per-shard",
            "2",
            "--workers",
            "1",
            "--out-feat",
            str(out_feat),
            "--out-model",
            str(out_model),
        ],
        check=True,
    )

    parts = list(out_feat.rglob("part_*.npz"))
    assert parts
    data = np.load(parts[0])
    X = data["X"]
    y = data["y"]
    assert X.shape[1] >= 27

    board_np = np.array(board["board"], dtype=int)
    expected = _board_features(board_np, 2, (0, 1))
    pos_row = X[y == 1][0]
    assert pos_row[-2] == expected[-2]
    assert pos_row[-1] == expected[-1]
