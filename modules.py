import numpy as np
from typing import Dict, Callable
import logging
import json

# 配置日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FORMULA_REGISTRY: Dict[str, Callable[[int, int, np.random.Generator], np.ndarray]] = {}

def register_formula(name: str) -> Callable:
    """註冊盤面生成公式"""
    def _decorator(fn: Callable) -> Callable:
        FORMULA_REGISTRY[name] = fn
        return fn
    return _decorator

@register_formula("excel")
def gen_excel(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """隨機排列生成盤面"""
    nums = rng.permutation(rows * cols) + 1
    return nums.reshape(rows, cols)

@register_formula("shuffle")
def gen_shuffle(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """按行隨機打亂生成盤面"""
    nums = np.arange(1, rows * cols + 1)
    board = nums.reshape(rows, cols)
    for r in range(rows):
        rng.shuffle(board[r])
    return board

@register_formula("random_entropy")
def gen_random_entropy(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """基於熵的隨機分散生成盤面"""
    grid = np.zeros((rows, cols), dtype=np.int64)
    legal = list(range(1, rows * cols + 1))
    rng.shuffle(legal)
    for i, val in enumerate(legal):
        r, c = divmod(i, cols)
        grid[r, c] = val
    return grid

class AdaptiveWeights:
    """動態調整權重"""
    def __init__(self, initial_weights: Dict[str, float]):
        self.weights = initial_weights.copy()
        self.history: Dict[str, float] = {name: 0.0 for name in initial_weights}
        self.total_trials = 0

    def update(self, success_rate: float, module_scores: Dict[str, float]) -> None:
        """根據成功率更新權重"""
        self.total_trials += 1
        for name in self.weights:
            score = module_scores.get(name, success_rate)
            self.history[name] = (self.history[name] * (self.total_trials - 1) + score) / self.total_trials
            self.weights[name] = max(0.1, min(0.9, self.history[name]))
        total = sum(self.weights.values()) or 1e-10
        for name in self.weights:
            self.weights[name] /= total

    def save_history(self, path: str) -> None:
        """保存歷史數據"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f)
        except OSError as e:
            logging.error(f"保存權重歷史失敗: {e}")