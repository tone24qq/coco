import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from inference_rf import extract_features, predict_top_k


def test_predict_top_k_simple(tmp_path: Path) -> None:
    board_full = np.array([[1, 2], [3, 4]])
    board_masked = np.array([[1, -1], [3, -1]])

    feats = []
    labels = []
    for r in range(board_full.shape[0]):
        for c in range(board_full.shape[1]):
            feats.append(extract_features(board_masked, r, c))
            labels.append(board_full[r, c])

    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(np.vstack(feats), np.array(labels))

    model_path = tmp_path / "model.pkl"
    joblib.dump(clf, model_path)

    board_path = tmp_path / "board.json"
    with open(board_path, "w", encoding="utf-8") as f:
        json.dump({"board": board_masked.tolist(), "target": 4}, f)

    res = predict_top_k(model_path, board_path, k=1)

    assert res["target"] == 4
    assert res["predictions"]
    pred = res["predictions"][0]
    assert (pred["r"], pred["c"]) == (1, 1)
