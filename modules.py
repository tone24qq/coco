import numpy as np
from typing import Dict, Callable, Tuple
from collections import Counter
import logging
import os
import json
import random
from numba import njit

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Formula registry for Monte Carlo simulation
FORMULA_REGISTRY: Dict[str, Callable[[int, int, np.random.Generator], np.ndarray]] = {}

def register_formula(name: str) -> Callable:
    """Decorator to register formula functions for generating scratch card grids."""
    def _decorator(fn: Callable) -> Callable:
        FORMULA_REGISTRY[name] = fn
        return fn
    return _decorator

@register_formula("excel")
@njit
def gen_excel(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """Generate grid using random permutation of numbers 1 to N."""
    nums = rng.permutation(np.arange(1, rows * cols + 1, dtype=np.int16))
    return nums.reshape(rows, cols)

@register_formula("shuffle")
@njit
def gen_shuffle(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """Generate grid by shuffling numbers within each row."""
    nums = np.arange(1, rows * cols + 1, dtype=np.int16)
    board = nums.reshape(rows, cols)
    for r in range(rows):
        rng.shuffle(board[r])
    return board

@register_formula("random_entropy")
@njit
def gen_random_entropy(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """Generate grid with entropy-based random dispersion."""
    grid = np.zeros((rows, cols), dtype=np.int16)
    legal = np.arange(1, rows * cols + 1, dtype=np.int16)
    rng.shuffle(legal)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        grid[r, c] = legal[i]
    return grid

class AdaptiveWeights:
    """Manages dynamic weight adjustments for formulas based on performance."""
    def __init__(self, initial_weights: Dict[str, float]):
        self.weights = initial_weights.copy()
        self.history: Dict[str, float] = {name: 0.0 for name in initial_weights}
        self.total_trials = 0

    def update(self, success_rate: float, module_scores: Dict[str, float]) -> None:
        """Update weights based on success rate and module scores."""
        self.total_trials += 1
        for name in self.weights:
            score = module_scores.get(name, success_rate)
            self.history[name] = (self.history[name] * (self.total_trials - 1) + score) / self.total_trials
            self.weights[name] = max(0.1, min(0.9, self.history[name] * 1.05))
        total = sum(self.weights.values()) or 1e-10
        for name in self.weights:
            self.weights[name] /= total

    def save_history(self, path: str) -> None:
        """Save weight history to file."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False)
        except OSError as e:
            logging.error(f"Failed to save weights history: {e}")

# ---------- FIX for Numba boolean indexing ----------
@njit(cache=True, fastmath=True)
def compute_global_features(grid: np.ndarray) -> Tuple[float, float]:
    """
    Compute global statistical features of the grid.
    Returns (mean, entropy) of known values.  -1 代表未開格。
    """
    flat = grid.ravel()                 # 先展平成 1-D
    mask = flat != -1
    if mask.sum() == 0:                 # 全空盤保險
        return 0.0, 0.0

    known_vals = flat[mask].astype(np.float32)  # 1-D ⇐ 1-D 布林遮罩 ✅
    mean_val = known_vals.mean()

    # Shannon entropy（簡單版本）
    hist = np.bincount(known_vals.astype(np.int32))
    probs = hist[hist > 0] / hist.sum()
    entropy = -np.sum(probs * np.log2(probs))

    return mean_val, entropy
# ---------- END FIX ---------------------------------