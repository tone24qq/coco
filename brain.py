import numpy as np
from typing import Dict, Callable, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}

class BoardAnalyzerUtils:
    """盤面分析工具"""
    def get_neighborhood_values(self, grid: np.ndarray, r: int, c: int, radius: int = 2) -> list:
        """獲取鄰域值"""
        neighbors = []
        rows, cols = grid.shape
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                    neighbors.append(grid[nr, nc])
        return neighbors

    def get_legal_values(self, grid: np.ndarray) -> set:
        """獲取可用數字"""
        rows, cols = grid.shape
        all_vals = set(range(1, rows * cols + 1))
        used = set(grid.flatten()[grid.flatten() != -1])
        return all_vals - used

def get_module_score(module_name: str, grid: np.ndarray) -> np.ndarray:
    """執行指定模組"""
    if module_name not in REGISTERED_MODULES_BRAIN:
        return np.zeros(grid.shape, dtype=float)
    return REGISTERED_MODULES_BRAIN[module_name](grid)

def EXT_M1_Tail_Pattern(grid: np.ndarray) -> np.ndarray:
    """基於尾數模式的評分"""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    utils = BoardAnalyzerUtils()
    legal = utils.get_legal_values(grid)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            neighbors = utils.get_neighborhood_values(grid, r, c)
            if not neighbors:
                continue
            tails = [n % 10 for n in neighbors]
            tail_count = max([tails.count(t) for t in set(tails)] or [1])
            scores[r, c] = tail_count / len(neighbors) if neighbors else 0.5
    return scores

def EXT_M3_Local_Focus(grid: np.ndarray) -> np.ndarray:
    """基於鄰域均值和方差的評分"""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    utils = BoardAnalyzerUtils()
    legal = utils.get_legal_values(grid)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            neighbors = utils.get_neighborhood_values(grid, r, c)
            if not neighbors:
                continue
            mean = np.mean(neighbors)
            std = np.std(neighbors) or 1
            scores[r, c] = 1 - min([abs(v - mean) / std for v in legal]) if legal else 0.5
    return scores

REGISTERED_MODULES_BRAIN.update({
    "EXT_M1_Tail_Pattern": EXT_M1_Tail_Pattern,
    "EXT_M3_Local_Focus": EXT_M3_Local_Focus
})