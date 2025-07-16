import json
import logging
import os
import time
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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


def _solve_boards(
    board: np.ndarray,
    row_sets: List[Set[int]],
    col_sets: List[Set[int]],
    digits: Set[int],
    blanks: List[Tuple[int, int]],
    solutions: List[np.ndarray],
    limit: int,
) -> None:
    """Recursive helper to enumerate board solutions."""
    if len(solutions) >= limit:
        return
    if not blanks:
        if not digits:
            solutions.append(board.copy())
        return
    # choose the most constrained blank
    blanks.sort(key=lambda rc: len(digits - row_sets[rc[0]] - col_sets[rc[1]]))
    r, c = blanks.pop(0)
    allowed = digits - row_sets[r] - col_sets[c]
    for num in sorted(allowed):
        board[r, c] = num
        digits.remove(num)
        row_sets[r].add(num)
        col_sets[c].add(num)
        _solve_boards(board, row_sets, col_sets, digits, blanks, solutions, limit)
        row_sets[r].remove(num)
        col_sets[c].remove(num)
        digits.add(num)
        board[r, c] = -1
        if len(solutions) >= limit:
            break
    blanks.insert(0, (r, c))


def find_solutions(board: np.ndarray, limit: int = 2) -> List[np.ndarray]:
    """Return up to ``limit`` complete boards satisfying uniqueness rules."""
    rows, cols = board.shape
    digits: Set[int] = set(range(1, rows * cols + 1))
    row_sets: List[Set[int]] = [set() for _ in range(rows)]
    col_sets: List[Set[int]] = [set() for _ in range(cols)]
    blanks: List[Tuple[int, int]] = []

    for r in range(rows):
        for c in range(cols):
            val = int(board[r, c])
            if val == -1:
                blanks.append((r, c))
                continue
            if val in row_sets[r] or val in col_sets[c]:
                return []
            row_sets[r].add(val)
            col_sets[c].add(val)
            digits.discard(val)

    solutions: List[np.ndarray] = []
    _solve_boards(board.copy(), row_sets, col_sets, digits, blanks, solutions, limit)
    return solutions


def _is_valid_board(board: np.ndarray) -> bool:
    """Return True if board has no duplicate or out-of-range numbers."""
    rows, cols = board.shape
    row_sets: List[Set[int]] = [set() for _ in range(rows)]
    col_sets: List[Set[int]] = [set() for _ in range(cols)]
    max_val = rows * cols
    for r in range(rows):
        for c in range(cols):
            val = int(board[r, c])
            if val == -1:
                continue
            if val < 1 or val > max_val:
                return False
            if val in row_sets[r] or val in col_sets[c]:
                return False
            row_sets[r].add(val)
            col_sets[c].add(val)
    return True


def _is_unique_board(board: np.ndarray) -> bool:
    """Return True if the board is valid and solvable."""
    if not _is_valid_board(board):
        return False
    return bool(find_solutions(board, limit=1))


def _filter_unique_candidates(
    board: np.ndarray, predictions: List[Dict[str, Any]], target: int
) -> List[Dict[str, Any]]:
    """Filter candidate cells to those keeping the board solvable and unique."""
    valid: List[Dict[str, Any]] = []
    for p in predictions:
        r, c = p["r"], p["c"]
        tmp = board.copy()
        tmp[r, c] = target
        if _is_unique_board(tmp):
            valid.append(p)
    return valid


def predict_top_k(
    model: ClassifierMixin,
    board: np.ndarray,
    target: int,
    k: int,
    *,
    enforce_unique: bool = False,
) -> Dict[str, Any]:
    rows, cols = board.shape
    solutions = find_solutions(board, limit=2)
    num_solutions = len(solutions)
    status = "no_valid_solution"
    if num_solutions == 1:
        status = "unique"
    elif num_solutions > 1:
        status = "multiple"

    feats_list: List[np.ndarray] = []
    coords: List[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == -1:
                feats_list.append(extract_features(board, r, c))
                coords.append((r, c))
    if not feats_list:
        return {
            "rows": rows,
            "cols": cols,
            "target": target,
            "predictions": [],
            "unique": num_solutions == 1,
            "num_solutions": num_solutions,
            "status": status,
        }
    X = np.vstack(feats_list)
    probs = model.predict_proba(X)
    try:
        idx = list(model.classes_).index(target)
    except ValueError:
        logger.warning("target %s not in model classes", target)
        return {
            "rows": rows,
            "cols": cols,
            "target": target,
            "predictions": [],
            "unique": num_solutions == 1,
            "num_solutions": num_solutions,
            "status": "no_valid_solution",
        }
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
    results = _filter_unique_candidates(board, results, target)
    if enforce_unique:
        solutions = find_solutions(board, limit=k + 1)
        valid_coords = {
            (int(r), int(c)) for sol in solutions for r, c in np.argwhere(sol == target)
        }
        if valid_coords:
            results = [p for p in results if (p["r"], p["c"]) in valid_coords]
            if not results:
                results = [
                    {"r": r, "c": c, "prob": 1.0} for r, c in sorted(valid_coords)
                ]
        else:
            results = []

    if not results:
        status = "no_valid_solution"
    return {
        "rows": rows,
        "cols": cols,
        "target": target,
        "predictions": results,
        "unique": num_solutions == 1,
        "num_solutions": num_solutions,
        "status": status,
    }


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
