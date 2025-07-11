import inspect
import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import modern_predictor
from modules import STRATEGY_REGISTRY, fuse_scores
from weights import AGG_WEIGHTS as DEFAULT_AGG_WEIGHTS

# Apply centralized weight overrides if present
try:  # noqa: WPS501
    from weights_config import WEIGHTS as USER_WEIGHTS

    DEFAULT_AGG_WEIGHTS.update(USER_WEIGHTS)
except Exception:
    pass

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

    def fill_blanks_with_remaining_numbers(
        self,
        grid: np.ndarray,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Fill ``-1`` cells with unused numbers from ``1`` to ``N``.

        Parameters
        ----------
        grid : np.ndarray
            Board matrix with ``-1`` marking unknown cells.
        rng : np.random.Generator, optional
            RNG used for shuffling remaining numbers.

        Returns
        -------
        np.ndarray
            New grid with blanks replaced by unique numbers.
        """

        arr = np.asarray(grid, dtype=int).copy()
        blanks = np.argwhere(arr == -1)
        if blanks.size == 0:
            return arr

        if rng is None:
            rng = np.random.default_rng()

        legal = list(self.get_legal_values_for_placement(arr))
        if len(legal) < blanks.shape[0]:
            raise ValueError("Not enough numbers to fill blanks")

        rng.shuffle(legal)
        arr[blanks[:, 0], blanks[:, 1]] = legal[: blanks.shape[0]]
        return arr

    @staticmethod
    def ring_index(rows: int, cols: int) -> np.ndarray:
        """Return each cell's distance from the outer frame (0=outermost)."""

        return np.fromfunction(
            lambda r, c: np.minimum.reduce([r, c, rows - 1 - r, cols - 1 - c]),
            (rows, cols),
            dtype=int,
        )


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
                logger.warning(
                    "Invalid weight for %s: %s - 無效權重值",
                    env_key,
                    env_val,
                )
                # 中文說明：環境變數提供的權重值無法解析，將忽略
    total = sum(w.values())
    if not math.isclose(total, 1.0, rel_tol=1e-3):
        logger.info(
            "Normalizing module weights (sum %.3f) - 權重總和調整",
            total,
        )
        # 中文說明：權重總和不為 1，將自動正規化
        for k in w:
            w[k] /= total or 1.0
    return w


AGG_WEIGHTS = _load_weights()


def get_core_modules(limit: Optional[int] = None) -> List[str]:
    """Return top modules sorted by weight.

    The ``limit`` parameter or ``CORE_LIMIT`` environment variable controls how
    many modules are returned. Invalid values fall back to defaults.
    """

    limit_env_str = os.getenv("CORE_LIMIT", "6")
    try:
        limit_env = int(limit_env_str)
    except ValueError:  # FIXME invalid env value
        logger.warning(
            "Invalid CORE_LIMIT '%s', using default 6 - 無效 CORE_LIMIT，採預設 6",
            limit_env_str,
        )
        # 中文說明：環境變數 CORE_LIMIT 不是整數，改用預設值
        limit_env = 6
    if limit is None:
        limit_final = limit_env
    else:
        try:
            limit_final = int(limit)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid limit '%s', using CORE_LIMIT - 非法 limit 值，改用環境設定",
                limit,
            )
            # 中文說明：函式引數 limit 非法，退回使用環境變數值
            limit_final = limit_env

    if limit_final < 1:
        logger.warning("Limit must be >=1, using 1 - 限制值至少 1")
        # 中文說明：限制值過小，自動修正為 1
        limit_final = 1

    names = list(REGISTERED_MODULES_BRAIN)
    limit_final = min(len(names), limit_final)
    sorted_mods = sorted(AGG_WEIGHTS.items(), key=lambda kv: kv[1], reverse=True)
    return [m for m, _ in sorted_mods[:limit_final]]


def get_module_score(
    module_name: str, grid: np.ndarray, target: Optional[int] = None, **kwargs
) -> np.ndarray:
    if module_name not in REGISTERED_MODULES_BRAIN:
        logger.error(
            "Module %s not found in REGISTERED_MODULES_BRAIN. - 模組未註冊",
            module_name,
        )
        # 中文說明：指定的模組名稱不存在，回傳全零矩陣避免崩潰
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
