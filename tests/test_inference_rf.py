import json
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier

from rf_infer.core import extract_features, predict_top_k


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


def test_predict_top_k_simple(tmp_path: Path) -> None:
    board_full = np.array([[1, 2], [3, 4]])
    board_masked = np.array([[1, -1], [3, -1]])
    clf = _train_simple_model(board_full, board_masked)

    model_path = tmp_path / "model.pkl"
    joblib.dump(clf, model_path)

    board_path = tmp_path / "board.json"
    with open(board_path, "w", encoding="utf-8") as f:
        json.dump({"board": board_masked.tolist(), "target": 4}, f)

    model = joblib.load(model_path)
    res = predict_top_k(model, board_masked, 4, k=1)

    assert res["target"] == 4
    assert res["predictions"]
    pred = res["predictions"][0]
    assert (pred["r"], pred["c"]) == (1, 1)


def test_predict_no_blanks(tmp_path: Path) -> None:
    board_full = np.array([[1, 2], [3, 4]])
    board_masked = board_full.copy()
    clf = _train_simple_model(board_full, board_masked)
    model = clf

    res = predict_top_k(model, board_masked, 2, k=3)
    assert res["predictions"] == []


def test_predict_all_blanks_large_k(tmp_path: Path) -> None:
    board_full = np.array([[1, 2], [3, 4]])
    board_masked = np.full_like(board_full, -1)
    clf = _train_simple_model(board_full, board_masked)
    model = clf

    res = predict_top_k(model, board_masked, 1, k=10)
    assert len(res["predictions"]) == board_masked.size


def test_predict_target_missing(tmp_path: Path) -> None:
    board_full = np.array([[1, 2], [3, 4]])
    board_masked = np.full_like(board_full, -1)
    clf = _train_simple_model(board_full, board_masked)
    model = clf

    res = predict_top_k(model, board_masked, 99, k=2)
    assert res["predictions"] == []


def test_predict_enforce_unique(tmp_path: Path) -> None:
    board_full = np.array([[1, 2], [3, 4]])
    board_masked = np.array([[1, -1], [3, -1]])
    clf = _train_simple_model(board_full, board_masked)
    model = clf

    res = predict_top_k(model, board_masked, 2, k=2, enforce_unique=True)
    coords = {(p["r"], p["c"]) for p in res["predictions"]}
    assert coords == {(0, 1), (1, 1)}
