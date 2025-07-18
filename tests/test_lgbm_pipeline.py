import json
import subprocess
from pathlib import Path

import numpy as np


def test_train_pipeline(tmp_path: Path) -> None:
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
    model_file = out_model / "2x2.pkl"
    assert model_file.exists()
    data = np.load(parts[0])
    assert "bid" in data
    assert data["bid"].shape[0] == data["y"].shape[0]
    assert data["X"].shape[1] >= 25


def test_plain_text_board(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    text = "5 14 1 10 17\n12 19 7 13 20\n9 15 2 6 18\n8 11 4 16 3\n"
    (data_dir / "plain.json").write_text(text, encoding="utf-8")

    out_feat = tmp_path / "features"
    out_model = tmp_path / "models"

    subprocess.run(
        [
            "python",
            "train_lgbm_pipeline.py",
            "--root",
            str(data_dir),
            "--shard-size",
            "10",
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
    model_file = out_model / "4x5.pkl"
    assert model_file.exists()
    data = np.load(parts[0])
    assert "bid" in data


def test_mask_ratio(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    board = {"board": [[1, 2], [3, 4]]}
    with open(data_dir / "full.json", "w", encoding="utf-8") as f:
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
            "--mask-ratio",
            "0.5",
            "--out-feat",
            str(out_feat),
            "--out-model",
            str(out_model),
        ],
        check=True,
    )

    parts = list(out_feat.rglob("part_*.npz"))
    assert parts
    model_file = out_model / "2x2.pkl"
    assert model_file.exists()
    data = np.load(parts[0])
    assert "bid" in data


def test_mask_range(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    board = {"board": [[1, 2], [3, 4]]}
    with open(data_dir / "full.json", "w", encoding="utf-8") as f:
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
            "--mask-range",
            "0.1",
            "0.5",
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
    assert "bid" in data
    model_file = out_model / "2x2.pkl"
    assert model_file.exists()
