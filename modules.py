# modules.py

import numpy as np
from typing import Dict, Callable, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ── 公式註冊 ───────────────────────────────────────────
# Monte Carlo 生成盤面的兩種公式：'shuffle'、'excel'
FORMULA_REGISTRY: Dict[str, Callable[[int, int, np.random.Generator], np.ndarray]] = {}

def gen_shuffle(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """
    隨機排列所有 1..rows*cols 的數字，並重塑成盤面。
    """
    card = np.arange(1, rows * cols + 1, dtype=np.int64)
    rng.shuffle(card)
    board = card.reshape(rows, cols)
    logger.debug(f"gen_shuffle: generated board {rows}x{cols}")
    return board

def gen_excel(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """
    模擬 Excel RAND 排序：對數字打隨機標籤後排序。
    """
    values = np.arange(1, rows * cols + 1, dtype=np.int64)
    rand_vals = rng.random(rows * cols)
    idx = np.argsort(rand_vals)
    board = values[idx].reshape(rows, cols)
    logger.debug(f"gen_excel: generated board {rows}x{cols}")
    return board

FORMULA_REGISTRY['shuffle'] = gen_shuffle
FORMULA_REGISTRY['excel'] = gen_excel

# ── 全域特徵計算 ────────────────────────────────────────
def compute_global_features(grid: np.ndarray) -> Tuple[float, float]:
    """
    回傳已知格子數值的平均與標準差，用於加權。
    空時預設為 (0.0, 1.0)。
    """
    known = grid[grid != -1].astype(np.float32)
    if known.size == 0:
        logger.debug("compute_global_features: no known values, returning (0.0, 1.0)")
        return 0.0, 1.0
    mean = float(np.mean(known))
    std  = float(np.std(known)) or 1.0
    logger.debug(f"compute_global_features: mean={mean:.3f}, std={std:.3f}")
    return mean, std

# ── 自適應權重管理 ───────────────────────────────────────
class AdaptiveWeights:
    """
    管理公式混合權重，可根據回饋動態調整。
    """

    def __init__(self, initial: Optional[Dict[str, float]] = None):
        if initial:
            self.weights = initial.copy()
        else:
            n = len(FORMULA_REGISTRY)
            self.weights = {name: 1.0 / n for name in FORMULA_REGISTRY}
        self.normalize()

    def normalize(self) -> None:
        total = sum(self.weights.values()) or 1.0
        for name in self.weights:
            self.weights[name] /= total
        logger.debug(f"AdaptiveWeights.normalize -> {self.weights}")

    def update(self, feedback: Dict[str, float]) -> None:
        """
        根據 feedback（公式名->正向分數），放大對應權重，並重新正規化。
        """
        for name, score in feedback.items():
            if name in self.weights:
                self.weights[name] *= (1 + score)
        self.normalize()
        logger.debug(f"AdaptiveWeights.update with feedback {feedback} -> {self.weights}")