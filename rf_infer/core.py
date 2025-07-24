import json
import logging
import os
import sys
import time
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import joblib
import numpy as np
from sklearn.base import ClassifierMixin

from coco_common.scalers import Float32StandardScaler  # noqa: F401

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


def _extract_features(board: np.ndarray, target: int) -> np.ndarray:
    """Return feature matrix for all cells on ``board`` for ``target``.

    \u6703\u5617\u8a66\u7528\u8a13\u7df4\u6d41\u7a0b\u7684 _board_features\uff1b\u5931\u6557\u5247\u9000\u56de extract_features\u3002
    \uff08\u9019\u88e1\u4e0d\u53ea\u9650 blank\uff0c\u4fdd\u6301\u548c\u820a\u884c\u70ba\u4e00\u81f4\uff1b\u5982\u8981\u53ea\u5c0d blank\uff0c\u53ef\u81ea\u884c\u8abf\u6574\uff09
    """
    feats_list: List[np.ndarray] = []
    rows, cols = board.shape
    try:
        from train_lgbm_pipeline import _board_features  # type: ignore

        use_pipeline = True
    except Exception:  # pragma: no cover - pipeline absent
        use_pipeline = False

    for r in range(rows):
        for c in range(cols):
            masked = board.copy()
            masked[r, c] = -1
            if use_pipeline:
                try:
                    feats = _board_features(masked, int(target), (r, c))
                except Exception:  # pragma: no cover
                    feats = extract_features(masked, r, c)
            else:
                feats = extract_features(masked, r, c)
            feats_list.append(np.asarray(feats, dtype=float))

    return np.vstack(feats_list)


def _predict_proba_any(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict class probabilities for ``X`` using diverse model types (sklearn / lgb.Booster / dict bundle)."""

    # unwrap dict bundle
    if isinstance(model, dict):
        scaler = model.get("scaler")
        if scaler is not None:
            X = scaler.transform(X)
        model = model.get("model")

    # \u5c0d\u9f4a\u7279\u5fb5\u7dad\u5ea6
    n_feat = getattr(model, "n_features_in_", None)
    if n_feat is None and hasattr(model, "num_feature"):
        try:
            n_feat = int(model.num_feature())
        except Exception:
            n_feat = None
    if n_feat is not None and X.shape[1] != n_feat:
        if X.shape[1] > n_feat:
            X = X[:, :n_feat]
        else:
            pad = np.zeros((X.shape[0], n_feat - X.shape[1]), dtype=X.dtype)
            X = np.column_stack([X, pad])

    # \u9996\u9078 predict_proba
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    # LightGBM Booster -> predict
    if hasattr(model, "predict"):
        probs = model.predict(
            X,
            num_iteration=getattr(model, "best_iteration", None),
        )
        probs = np.asarray(probs)
        if probs.ndim == 1:  # binary \u6a5f\u7387(\u6b63\u985e)
            return np.column_stack([1 - probs, probs])
        return probs

    raise AttributeError("model object lacks predict methods")


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
    sys.modules["__main__"].Float32StandardScaler = Float32StandardScaler
    obj = joblib.load(path)

    class _ModelWrapper(ClassifierMixin):
        def __init__(
            self,
            model: Any,
            scaler: Any | None,
            classes: List[int] | None,
            n_features: int | None = None,
        ) -> None:
            self.model = model
            self.scaler = scaler
            if classes is not None:
                self.classes_ = np.array(classes)
            else:
                self.classes_ = getattr(model, "classes_", None)
            self.n_features_in_ = (
                int(n_features)
                if n_features is not None
                else getattr(model, "n_features_in_", None)
            )

        def predict_proba(self, X: np.ndarray) -> np.ndarray:  # type: ignore[override]
            if self.scaler is not None:
                X = self.scaler.transform(X)
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)
            probs = self.model.predict(
                X, num_iteration=getattr(self.model, "best_iteration", None)
            )
            if probs.ndim == 1:
                return np.column_stack([1 - probs, probs])
            return probs

    def _classes_from_path(p: str) -> List[int] | None:
        base = os.path.splitext(os.path.basename(p))[0]
        if "x" in base:
            a, b = base.split("x", 1)
            if a.isdigit() and b.isdigit():
                rows, cols = int(a), int(b)
                return list(range(1, rows * cols + 1))
        return None

    classes = _classes_from_path(path)

    if isinstance(obj, dict) and "model" in obj:
        return _ModelWrapper(
            obj["model"],
            obj.get("scaler"),
            classes,
            obj.get("n_features_in_"),
        )
    if isinstance(obj, tuple) and len(obj) == 2:
        return _ModelWrapper(obj[1], obj[0], classes)
    return obj


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
    """Return a model path for the given board size.

    Priority:
    1. exact:   "<rows>x<cols>.pkl"
    2. fallback "<rows>x<cols>_*.pkl" (first sorted hit)

    Rationale:
    - keep backward compatibility
    - deterministic choice
    """

    base = f"{rows}x{cols}"
    exact = os.path.join(models_dir, f"{base}.pkl")
    if os.path.exists(exact):
        return exact

    pattern = os.path.join(models_dir, f"{base}_*.pkl")
    matches = sorted(glob(pattern))
    if matches:
        return matches[0]

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
    """Filter cells that keep the board valid and uniquely solvable."""
    valid: List[Dict[str, Any]] = []
    for p in predictions:
        r, c = p["r"], p["c"]
        tmp = board.copy()
        tmp[r, c] = target
        if not _is_valid_board(tmp):
            continue
        if len(find_solutions(tmp, limit=2)) == 1:
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

    max_val = rows * cols
    if not _is_valid_board(board) or not (1 <= target <= max_val):
        return {
            "rows": rows,
            "cols": cols,
            "target": target,
            "predictions": [],
            "unique": False,
            "num_solutions": 0,
            "status": "no_valid_solution",
        }

    solutions = find_solutions(board, limit=2)
    num_solutions = len(solutions)
    status = "no_valid_solution"
    if num_solutions == 1:
        status = "unique"
    elif num_solutions > 1:
        status = "multiple"

    feats_list: List[np.ndarray] = []
    coords: List[tuple[int, int]] = []
    n_features = getattr(model, "n_features_in_", None)
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == -1:
                if n_features and n_features > 22:
                    try:
                        from train_lgbm_pipeline import _board_features

                        feats = _board_features(board, target, (r, c))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("failed advanced feature extraction: %s", exc)
                        feats = extract_features(board, r, c)
                else:
                    feats = extract_features(board, r, c)
                feats_list.append(np.asarray(feats, dtype=float))
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
    # ---- NEW: robust probability prediction ----
    probs = _predict_proba_any(model, X)

    # map to target column
    def _target_probs_from(probs: np.ndarray) -> np.ndarray:
        # sklearn-style
        if hasattr(model, "classes_"):
            try:
                idx = list(model.classes_).index(target)
                return probs[:, idx]
            except Exception:
                logger.warning("target %s not in model classes", target)
                return np.array([])

        # Booster w/o classes_: try common layouts
        if probs.shape[1] == 2:
            # binary -> assume col 1 is positive class
            return probs[:, 1]

        total = rows * cols
        if probs.shape[1] == total and 1 <= target <= total:
            # classes are 1..total
            return probs[:, target - 1]

        logger.error(
            "Cannot map target %s to prob column (probs.shape=%s)", target, probs.shape
        )
        return np.array([])

    target_probs = _target_probs_from(probs)
    if target_probs.size == 0:
        return {
            "rows": rows,
            "cols": cols,
            "target": target,
            "predictions": [],
            "unique": num_solutions == 1,
            "num_solutions": num_solutions,
            "status": "no_valid_solution",
        }
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
