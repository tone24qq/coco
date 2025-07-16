import json
import logging
import os
import time
from glob import glob
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
from sklearn.base import ClassifierMixin

logger = logging.getLogger(__name__)


def extract_features(board: np.ndarray, r: int, c: int) -> np.ndarray:
    """Return feature vector for position (r, c)."""
    rows, cols = board.shape
    feats: List[float] = []

    feats += [r, c, rows, cols]

    known = board[board >= 0]
    feats += [
        known.size,
        float(known.mean()) if known.size else 0.0,
        float(known.std()) if known.size else 0.0,
    ]

    row_vals = board[r, :]
    row_known = row_vals[row_vals >= 0]
    feats += [
        row_known.size,
        float(row_known.mean()) if row_known.size else 0.0,
        float(row_known.std()) if row_known.size else 0.0,
    ]

    col_vals = board[:, c]
    col_known = col_vals[col_vals >= 0]
    feats += [
        col_known.size,
        float(col_known.mean()) if col_known.size else 0.0,
        float(col_known.std()) if col_known.size else 0.0,
    ]

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                feats.append(int(board[rr, cc]))
            else:
                feats.append(-1)

    return np.array(feats, dtype=float)


def _validate_path(
    path: str, *, must_exist: bool = True, suffix: Optional[str] = None
) -> None:
    if must_exist and not os.path.exists(path):
        raise FileNotFoundError(path)
    if suffix and not path.endswith(suffix):
        raise ValueError(f"{path} must end with {suffix}")


def _load_model(path: str) -> ClassifierMixin:
    _validate_path(path, suffix=".pkl")
    logger.info("Loading model %s", path)
    return joblib.load(path)


def _load_board_file(path: str) -> List[Dict[str, Any]]:
    _validate_path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        boards = data
    else:
        boards = [data]
    for b in boards:
        if "board" not in b or "target" not in b:
            raise ValueError("Input JSON missing 'board' or 'target'")
    return boards


def load_boards(pattern: str) -> Iterable[Dict[str, Any]]:
    paths = glob(pattern)
    if not paths:
        raise FileNotFoundError(pattern)
    for p in paths:
        for item in _load_board_file(p):
            item["__source__"] = p
            yield item


def _select_model(models_dir: str, rows: int, cols: int) -> str:
    cand = os.path.join(models_dir, f"{rows}x{cols}.pkl")
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"No model for {rows}x{cols}")


def predict_top_k(
    model: ClassifierMixin, board: np.ndarray, target: int, k: int
) -> Dict[str, Any]:
    rows, cols = board.shape
    feats_list: List[np.ndarray] = []
    coords: List[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == -1:
                feats_list.append(extract_features(board, r, c))
                coords.append((r, c))
    if not feats_list:
        return {"rows": rows, "cols": cols, "target": target, "predictions": []}
    X = np.vstack(feats_list)
    probs = model.predict_proba(X)
    try:
        idx = list(model.classes_).index(target)
    except ValueError:
        logger.warning("target %s not in model classes", target)
        return {"rows": rows, "cols": cols, "target": target, "predictions": []}
    target_probs = probs[:, idx]
    top_idx = np.argsort(target_probs)[-k:][::-1]
    results = [
        {
            "r": int(coords[i][0]),
            "c": int(coords[i][1]),
            "prob": float(round(target_probs[i], 4)),
        }
        for i in top_idx
    ]
    return {"rows": rows, "cols": cols, "target": target, "predictions": results}


def batch_predict(
    model_path: str, input_pattern: str, k: int, models_dir: str = "models"
) -> List[Dict[str, Any]]:
    boards = list(load_boards(input_pattern))
    if not boards:
        raise RuntimeError("no input boards")

    sample_board = np.array(boards[0]["board"], dtype=int)
    if model_path:
        model_file = model_path
    else:
        model_file = _select_model(
            models_dir, sample_board.shape[0], sample_board.shape[1]
        )

    model = _load_model(model_file)

    results = []
    durations: List[float] = []
    failures = 0
    for data in boards:
        board = np.array(data["board"], dtype=int)
        target = int(data["target"])
        start = time.perf_counter()
        try:
            res = predict_top_k(model, board, target, k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed inference for %s: %s", data.get("__source__"), exc)
            failures += 1
            res = {
                "rows": board.shape[0],
                "cols": board.shape[1],
                "target": target,
                "predictions": [],
            }
        duration = time.perf_counter() - start
        durations.append(duration)
        results.append(res)
    if durations:
        logger.info(
            "Processed %d boards, avg_time=%.3fs failures=%d",
            len(durations),
            sum(durations) / len(durations),
            failures,
        )
    return results


def infer_top3_for_target(
    board: np.ndarray, target: int, models_dir: str = "models"
) -> List[tuple[int, int]]:
    """Return the top-3 coordinates most likely to contain ``target``."""
    rows, cols = board.shape
    model_path = _select_model(models_dir, rows, cols)
    model = _load_model(model_path)
    res = predict_top_k(model, board, target, 3)
    return [(p["r"], p["c"]) for p in res["predictions"]]
