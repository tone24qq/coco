import inspect
import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import modern_predictor
from modules import STRATEGY_REGISTRY, fuse_scores
from weights import AGG_WEIGHTS as DEFAULT_AGG_WEIGHTS

logger = logging.getLogger(__name__)


def safe_call(func: Callable, *args: Any, **kwargs: Any) -> Any:
    sig = inspect.signature(func)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(*args, **allowed)


class BoardAnalyzerUtils:
    """Simple board utilities."""

    def get_card_max_value_from_gridDimensions(
        self, grid_shape: Tuple[int, int]
    ) -> int:
        rows, cols = grid_shape
        return rows * cols if rows and cols else 0

    def get_legal_values_for_placement(self, grid: np.ndarray) -> set[int]:
        rows, cols = grid.shape
        all_vals = set(
            range(1, self.get_card_max_value_from_gridDimensions((rows, cols)) + 1)
        )
        used = set(int(v) for v in grid.flatten() if v != -1 and v > 0)
        return all_vals - used


DTYPE_DEFAULT = np.int32
ITEMSIZE = np.dtype(DTYPE_DEFAULT).itemsize


def bytes_to_grid(grid_bytes: bytes, shape):
    arr = np.frombuffer(grid_bytes, dtype=DTYPE_DEFAULT)
    return arr.reshape(shape)


REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    name: strat.func for name, strat in STRATEGY_REGISTRY.items()
}

# predictor registry for ranking strategies
REGISTERED_MODULES: Dict[str, Callable] = {
    "modern": modern_predictor.predict_location,
    "legacy": fuse_scores,
}


def _read_performance(file_path: str) -> Dict[str, float]:
    acc: Dict[str, float] = {}
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        acc[parts[0]] = float(parts[1])
                    except ValueError:
                        continue
    return acc


def _load_weights() -> Dict[str, float]:
    w = {name: strat.weight for name, strat in STRATEGY_REGISTRY.items()}
    w.update(DEFAULT_AGG_WEIGHTS)
    perf_file = os.getenv("PERFORMANCE_FILE", "module_performance.txt")
    perf = _read_performance(perf_file)
    if perf:
        total = sum(perf.get(m, 0.0) for m in w) or 1.0
        w = {m: perf.get(m, 0.0) / total for m in w}
    for name in w:
        env_key = f"WEIGHT_{name.upper()}"
        env_val = os.getenv(env_key)
        if env_val is not None:
            try:
                w[name] = float(env_val)
            except ValueError:
                logger.warning("Invalid weight for %s: %s", env_key, env_val)
    total = sum(w.values())
    if not math.isclose(total, 1.0, rel_tol=1e-3):
        logger.info("Normalizing module weights (sum %.3f)", total)
        for k in w:
            w[k] /= total or 1.0
    return w


AGG_WEIGHTS = _load_weights()


def get_core_modules(limit: Optional[int] = None) -> List[str]:
    limit_env_str = os.getenv("CORE_LIMIT", "6")
    try:
        limit_env = int(limit_env_str)
    except ValueError:  # FIXME invalid env value
        logger.warning("Invalid CORE_LIMIT '%s', using default 6", limit_env_str)
        limit_env = 6
    if limit is None:
        limit = limit_env
    names = list(REGISTERED_MODULES_BRAIN)
    limit = max(1, min(len(names), limit))
    sorted_mods = sorted(AGG_WEIGHTS.items(), key=lambda kv: kv[1], reverse=True)
    return [m for m, _ in sorted_mods[:limit]]


def get_module_score(
    module_name: str, grid: np.ndarray, target: Optional[int] = None, **kwargs
) -> np.ndarray:
    if module_name not in REGISTERED_MODULES_BRAIN:
        logger.error("Module %s not found in REGISTERED_MODULES_BRAIN.", module_name)
        rows, cols = grid.shape
        return np.zeros((rows, cols), dtype=float)
    func = REGISTERED_MODULES_BRAIN[module_name]
    kwargs["target"] = target
    return safe_call(func, grid, **kwargs)


def aggregate_scores(
    stack: np.ndarray, weights: np.ndarray, names: Optional[List[str]] | None = None
) -> np.ndarray:
    mu = stack.mean(axis=(1, 2), keepdims=True)
    sigma = stack.std(axis=(1, 2), keepdims=True) + 1e-6
    stack_z = (stack - mu) / sigma

    weights = np.asarray(weights, dtype=float)
    base = weights / (weights.sum() + 1e-10)

    conf = stack_z.max(axis=(1, 2)) - stack_z.mean(axis=(1, 2))
    conf = np.clip(conf, 0.0, None)
    if conf.sum() > 0:
        conf /= conf.sum()
        alpha = float(os.getenv("WEIGHT_ALPHA", "0.8"))
        weights = alpha * base + (1.0 - alpha) * conf
    else:
        weights = base

    final = np.tensordot(weights, stack_z, axes=(0, 0))
    return final


def compute_nearest_value_heatmap(
    grid: np.ndarray,
    *,
    target: int,
    cooc_prob: Dict[Tuple[int, int], Dict[int, Dict[int, float]]],
    k: int,
) -> np.ndarray:
    score = np.zeros_like(grid, dtype=float)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in offsets:
        prob = cooc_prob.get((dr, dc), {}).get(target, {})
        if not prob:
            continue
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] != -1:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    v = grid[nr, nc]
                    if v > 0:
                        score[r, c] += prob.get(v, 0.0)
    if score.max() > 0:
        score /= float(score.max())
    return score
