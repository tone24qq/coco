import json
import logging
import os
import re
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
            if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
                if X.shape[1] < self.n_features_in_:
                    pad = self.n_features_in_ - X.shape[1]
                    X = np.pad(X, ((0, 0), (0, pad)), constant_values=0)
                else:
                    X = X[:, : self.n_features_in_]
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
        m = re.search(r"(\d+)x(\d+)", base)
        if m:
            rows, cols = int(m.group(1)), int(m.group(2))
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
    try:
        from lightgbm import Booster  # type: ignore

        if isinstance(obj, Booster):
            return _ModelWrapper(obj, None, classes, obj.num_feature())
    except Exception:  # noqa: BLE001
        pass
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
    import time

    t0 = time.time()
    logger.info("[PRED] start predict_top_k")
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

    num_solutions = None
    status = "skipped_check"
    if enforce_unique:
        t_uni = time.time()
        solutions = find_solutions(board, limit=2)
        num_solutions = len(solutions)
        status = (
            "unique"
            if num_solutions == 1
            else ("multiple" if num_solutions > 1 else "no_valid_solution")
        )
        logger.info(
            "[PRED] uniqueness check done in %.3fs (status=%s, num=%s)",
            time.time() - t_uni,
            status,
            num_solutions,
        )

    logger.info("[PRED] building feature matrix …")
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
    # ――― 推理 ―――――――――――――――――――――――――――――――――――――――――――――――――
    X = np.vstack(feats_list)
    t_pred = time.time()
    probs = model.predict_proba(X)  # shape (n, C) or (n,)
    logger.info("[PRED] model.predict done in %.3fs", time.time() - t_pred)

    # 1）多類模型: 機率列數與 classes_ 一致
    if probs.ndim == 2 and probs.shape[1] == len(getattr(model, "classes_", [])):
        try:
            idx = list(model.classes_).index(target)
            target_probs = probs[:, idx]
        except ValueError:
            logger.warning("target %s not in model classes", target)
            return {
                "rows": rows,
                "cols": cols,
                "target": target,
                "predictions": [],
                "unique": num_solutions == 1,
                "num_solutions": num_solutions,
                "status": status,
            }
    # 2）二元(OVA)或 shape 不相符: 取最後一欄/一維向量當正類機率
    else:
        if probs.ndim == 1:
            target_probs = probs
        else:
            target_probs = probs[:, -1]
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
    logger.info("[PRED] total %.3fs", time.time() - t0)
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
    logger.info(
        "infer_top3_for_target: board=%dx%d target=%s",
        rows,
        cols,
        target,
    )
    model_path = _select_model(models_dir, rows, cols)
    logger.info("Selected model path: %s", model_path)
    model = _load_model(model_path)
    logger.info("Model loaded, running predict_top_k …")
    res = predict_top_k(model, board, target, 3)
    logger.info(
        "predict_top_k returned %d candidates",
        len(res.get("predictions", [])),
    )
    return [(p["r"], p["c"]) for p in res["predictions"]]
