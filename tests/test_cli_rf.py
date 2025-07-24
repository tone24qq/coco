import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier

from rf_infer.core import extract_features


def _train_simple_model() -> LGBMClassifier:
    board_full = np.array([[1, 2], [3, 4]])
    board_masked = np.array([[1, -1], [3, -1]])

    feats = []
    labels = []
    for r in range(board_full.shape[0]):
        for c in range(board_full.shape[1]):
            feats.append(extract_features(board_masked, r, c))
            labels.append(board_full[r, c])

    clf = LGBMClassifier(n_estimators=10, random_state=0)
    clf.fit(np.vstack(feats), np.array(labels))
    return clf


def test_cli_help() -> None:
    result = subprocess.run(
        [
            "python",
            "-m",
            "rf_infer.cli",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "LightGBM board inference" in result.stdout


def test_cli_run(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump({"board": [[-1, -1], [-1, -1]], "target": 3}, f)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_file = models_dir / "2x2.pkl"
    joblib.dump(_train_simple_model(), model_file)
    out_file = tmp_path / "out.json"

    subprocess.run(
        [
            "python",
            "-m",
            "rf_infer.cli",
            "--input",
            str(input_path),
            "--output",
            str(out_file),
            "--k",
            "2",
            "--models-dir",
            str(models_dir),
        ],
        check=True,
    )

    data = json.loads(out_file.read_text())
    assert data[0]["rows"] == 2
    assert data[0]["target"] == 3
    assert 1 <= len(data[0]["predictions"]) <= 2
