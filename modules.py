import numpy as np
from typing import Dict, Callable, Tuple
from collections import Counter
import logging
import os

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

FORMULA_REGISTRY: Dict[str, Callable[[int, int, np.random.Generator], np.ndarray]] = {}

def register_formula(name: str) -> Callable:
    def _decorator(fn: Callable) -> Callable:
        FORMULA_REGISTRY[name] = fn
        return fn
    return _decorator

@register_formula("excel")
def gen_excel(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate grid using random permutation of numbers 1 to N.
    """
    return rng.permutation(rows * cols).reshape(rows, cols) + 1

@register_formula("shuffle")
def gen_shuffle(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate grid by shuffling numbers within each row.
    """
    nums = np.arange(1, rows * cols + 1)
    board = nums.reshape(rows, cols).copy()
    for r in range(rows):
        rng.shuffle(board[r])
    return board

class AdaptiveWeights:
    """
    Manages dynamic weight adjustments for formulas based on performance.
    """
    def __init__(self, initial_weights: Dict[str, float]):
        self.weights = initial_weights.copy()
        self.history: Dict[str, float] = {name: 0.0 for name in initial_weights}
        self.total_trials = 0

    def update(self, success_rate: float, module_scores: Dict[str, float]) -> None:
        self.total_trials += 1
        for name in self.weights:
            score = module_scores.get(name, success_rate)
            self.history[name] = (self.history[name] * (self.total_trials - 1) + score) / self.total_trials
            self.weights[name] = max(0.1, min(0.9, self.history[name]))
        total = sum(self.weights.values())
        if total > 0:
            for name in self.weights:
                self.weights[name] /= total

    def save_history(self, path: str) -> None:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                import json
                json.dump(self.history, f)
        except OSError as e:
            logging.error(f"Failed to save weights history: {e}")

def compute_global_features(grid: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean and std deviation of known values in the grid.
    """
    known_vals = grid[grid != -1].astype(np.float32)
    if known_vals.size == 0:
        return 0.0, 1.0
    mean_val = np.mean(known_vals)
    std_val = np.std(known_vals)
    return mean_val, std_val if std_val > 0 else 1.0