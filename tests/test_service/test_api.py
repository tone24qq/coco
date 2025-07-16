from pathlib import Path

import joblib
import numpy as np
from fastapi.testclient import TestClient
from lightgbm import LGBMClassifier

from coco_service.main import app
from rf_infer.core import extract_features


def _train_simple_model(
    board_full: np.ndarray, board_masked: np.ndarray
) -> LGBMClassifier:
    feats = []
    labels = []
    for r in range(board_full.shape[0]):
        for c in range(board_full.shape[1]):
            feats.append(extract_features(board_masked, r, c))
            labels.append(board_full[r, c])
    clf = LGBMClassifier(n_estimators=10, random_state=0)
    clf.fit(np.vstack(feats), np.array(labels))
    return clf


def test_predict_endpoint_basic(tmp_path: Path, monkeypatch) -> None:
    client = TestClient(app)
    board_full = np.array([[1, 2], [3, 4]])
    board_masked = np.array([[1, -1], [3, -1]])
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    joblib.dump(_train_simple_model(board_full, board_masked), model_dir / "2x2.pkl")
    monkeypatch.setenv("MODELS_DIR", str(model_dir))

    response = client.post(
        "/predict",
        json={"board": board_masked.tolist(), "target": 4},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert 0 < len(data) <= 3
    for item in data:
        assert {"row", "col", "score"} <= set(item)


def test_root_endpoint():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
