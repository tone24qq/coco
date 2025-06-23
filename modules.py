import numpy as np
from typing import Dict, Callable
import logging
import json

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
def gen_excel(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """Generate grid using random permutation of numbers 1 to N."""
    nums = rng.permutation(rows * cols) + 1
    return nums.reshape(rows, cols)

@register_formula("shuffle")
def gen_shuffle(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """Generate grid by shuffling numbers within each row."""
    nums = np.arange(1, rows * cols + 1)
    board = nums.reshape(rows, cols)
    shuffle_idx = rng.random((rows, cols)).argsort(axis=1)
    board = np.take_along_axis(board, shuffle_idx, axis=1)
    return board

@register_formula("random_entropy")
def gen_random_entropy(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """Generate grid with entropy-based random dispersion."""
    nums = rng.permutation(rows * cols) + 1
    return nums.reshape(rows, cols)

@register_formula("spatial_entropy")
def gen_spatial_entropy(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """Generate grid placing larger numbers near the center based on a spatial entropy map."""
    n = rows * cols
    nums = rng.permutation(n) + 1
    base_grid = nums.reshape(rows, cols)
    r_idx = np.linspace(-1.0, 1.0, rows)[:, None]
    c_idx = np.linspace(-1.0, 1.0, cols)
    dist = np.sqrt(r_idx ** 2 + c_idx ** 2)
    order = np.argsort(dist.ravel())
    flat = base_grid.ravel()
    flat = flat[order]
    return flat.reshape(rows, cols)

@register_formula("tail_cluster")
def gen_tail_cluster(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """Generate grid clustering larger numbers near the bottom-right corner."""
    n = rows * cols
    high_rows = max(1, int(rows * 0.3))
    high_cols = max(1, int(cols * 0.3))
    mask = np.zeros((rows, cols), dtype=bool)
    mask[-high_rows:, -high_cols:] = True
    high_idx = np.flatnonzero(mask.ravel())
    low_idx = np.flatnonzero(~mask.ravel())

    nums = np.arange(1, n + 1)
    rng.shuffle(nums)
    nums.sort()
    high_nums = nums[-len(high_idx):]
    low_nums = nums[:-len(high_idx)]
    rng.shuffle(high_nums)
    rng.shuffle(low_nums)

    board = np.empty(n, dtype=np.int64)
    board[low_idx] = low_nums
    board[high_idx] = high_nums
    return board.reshape(rows, cols)

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
