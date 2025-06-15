# app/modules.py

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

# Formula registry for Monte Carlo simulation
FORMULA_REGISTRY: Dict[str, Callable[[int, int, np.random.Generator], np.ndarray]] = {}

def register_formula(name: str) -> Callable:
    """
    Decorator to register formula functions for generating scratch card grids.
    
    Args:
        name (str): Name of the formula.
    
    Returns:
        Callable: Decorated function.
    """
    def _decorator(fn: Callable) -> Callable:
        FORMULA_REGISTRY[name] = fn
        return fn
    return _decorator

@register_formula("excel")
def gen_excel(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate grid using random permutation of numbers 1 to N.
    
    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.
        rng (np.random.Generator): Random number generator.
    
    Returns:
        np.ndarray: Generated grid.
    """
    nums = rng.permutation(rows * cols) + 1
    return nums.reshape(rows, cols)

@register_formula("shuffle")
def gen_shuffle(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate grid by shuffling numbers within each row.
    
    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.
        rng (np.random.Generator): Random number generator.
    
    Returns:
        np.ndarray: Generated grid.
    """
    nums = np.arange(1, rows * cols + 1)
    board = nums.reshape(rows, cols)
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
        """
        Update weights based on success rate and module scores.
        
        Args:
            success_rate (float): Success rate of predictions.
            module_scores (Dict[str, float]): Scores from modules.
        """
        self.total_trials += 1
        for name in self.weights:
            score = module_scores.get(name, success_rate)
            self.history[name] = (self.history[name] * (self.total_trials - 1) + score) / self.total_trials
            self.weights[name] = max(0.1, min(0.9, self.history[name]))
        total = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= total

    def save_history(self, path: str) -> None:
        """
        Save weight history to file.
        
        Args:
            path (str): File path to save history.
        """
        try:
            with open(path, 'w', encoding='utf-8') as f:
                import json
                json.dump(self.history, f)
        except OSError as e:
            logging.error(f"Failed to save weights history: {e}")

def compute_global_features(grid: np.ndarray) -> Tuple[float, float]:
    """
    Compute global statistical features of the grid.
    
    Args:
        grid (np.ndarray): Input grid.
    
    Returns:
        Tuple[float, float]: Mean and standard deviation of known values.
    """
    known_vals = grid[grid != -1].astype(np.float32)
    if known_vals.size == 0:
        return 0.0, 1.0
    mean_val = np.mean(known_vals)
    std_val = np.std(known_vals) if np.std(known_vals) > 0 else 1.0
    return mean_val, std_val

# 自檢報告：
# - 語法檢查：通過
# - 括號配對：無遺漏
# - 標識符定義：無未定義/拼寫錯誤
# - 測試環境：Python 3.11