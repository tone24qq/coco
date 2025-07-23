import json
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier

from coco_common.scalers import Float32StandardScaler
# fmt: off
from rf_infer.core import (_load_model, _select_model, extract_features,
                           infer_top3_for_target, predict_top_k)

# fmt: on


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
    assert res["status"] == "multiple"
    assert res["num_solutions"] == 2
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
    assert res["status"] == "unique"


def test_predict_all_blanks_large_k(tmp_path: Path) -> None:
    board_full = np.array([[1, 2], [3, 4]])
    board_masked = np.full_like(board_full, -1)
    clf = _train_simple_model(board_full, board_masked)
    model = clf

    res = predict_top_k(model, board_masked, 1, k=10)
    assert res["predictions"] == []
    assert res["status"] == "no_valid_solution"


def test_predict_target_missing(tmp_path: Path) -> None:
    board_full = np.array([[1, 2], [3, 4]])
    board_masked = np.full_like(board_full, -1)
    clf = _train_simple_model(board_full, board_masked)
    model = clf

    res = predict_top_k(model, board_masked, 99, k=2)
    assert res["predictions"] == []
    assert res["status"] == "no_valid_solution"


def test_predict_enforce_unique(tmp_path: Path) -> None:
    board_full = np.array([[1, 2], [3, 4]])
    board_masked = np.array([[1, -1], [3, -1]])
    clf = _train_simple_model(board_full, board_masked)
    model = clf

    res = predict_top_k(model, board_masked, 2, k=2, enforce_unique=True)
    coords = {(p["r"], p["c"]) for p in res["predictions"]}
    assert coords == {(0, 1), (1, 1)}
    assert res["unique"] is False


def test_invalid_board_status(tmp_path: Path) -> None:
    board_full = np.array([[1, 2], [3, 4]])
    board_masked = np.array([[1, 1], [3, -1]])
    clf = _train_simple_model(board_full, board_masked)
    model = clf

    res = predict_top_k(model, board_masked, 2, k=2)
    assert res["predictions"] == []
    assert res["status"] == "no_valid_solution"


def test_filter_invalid_prediction(tmp_path: Path) -> None:
    board_full = np.array([[1, 2], [3, 4]])
    board_masked = np.array([[1, -1], [3, -1]])
    clf = _train_simple_model(board_full, board_masked)
    model = clf

    res = predict_top_k(model, board_masked, 1, k=2)
    assert res["predictions"] == []
    assert res["status"] == "no_valid_solution"


def test_load_model_dict(tmp_path: Path) -> None:
    board_full = np.array([[1, 2], [3, 4]])
    board_masked = np.array([[1, -1], [3, -1]])
    clf = _train_simple_model(board_full, board_masked)
    feats = [
        extract_features(board_masked, r, c)
        for r in range(board_full.shape[0])
        for c in range(board_full.shape[1])
    ]
    scaler = Float32StandardScaler().fit(np.vstack(feats))
    model_path = tmp_path / "2x2.pkl"
    joblib.dump(
        {
            "model": clf.booster_,
            "scaler": scaler,
            "n_features_in_": len(feats[0]),
        },
        model_path,
    )

    coords = infer_top3_for_target(board_masked, 4, models_dir=str(tmp_path))
    assert coords and coords[0] == (1, 1)
    m = _load_model(str(model_path))
    assert getattr(m, "n_features_in_", None) == len(feats[0])


def test_select_model_with_suffix(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / "4x10_lgbm_best.pkl"
    model_path.touch()
    assert _select_model(str(models_dir), 4, 10) == str(model_path)


def test_select_model_prefers_exact(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    exact = models_dir / "4x10.pkl"
    with_suffix = models_dir / "4x10_lgbm_best.pkl"
    exact.touch()
    with_suffix.touch()
    assert _select_model(str(models_dir), 4, 10) == str(exact)
