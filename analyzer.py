import numpy as np
from typing import Dict, List, Any, Optional
from scipy.optimize import linear_sum_assignment
from modules import FORMULA_REGISTRY
from brain import get_module_score
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def simulate_full_board(grid: np.ndarray, target_num: Optional[int], n_iter: int) -> Dict[tuple, Dict[int, float]]:
    """模擬完整盤面"""
    rows, cols = grid.shape
    blanks = [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == -1]
    legal = set(range(1, rows * cols + 1)) - set(grid.flatten()[grid.flatten() != -1])
    rng = np.random.default_rng()
    counts = {pos: {} for pos in blanks}
    
    for _ in range(n_iter):
        board = FORMULA_REGISTRY["excel"](rows, cols, rng)
        for r, c in blanks:
            num = board[r, c]
            if num in legal:
                counts[(r, c)][num] = counts[(r, c)].get(num, 0) + 1
    
    prob_map = {}
    for pos in blanks:
        total = sum(counts[pos].values()) or 1
        prob_map[pos] = {n: c / total for n, c in counts[pos].items()}
    return prob_map

def weight_prob_by_modules(grid: np.ndarray, prob_map: Dict[tuple, Dict[int, float]]) -> Dict[tuple, Dict[int, float]]:
    """根據模組加權概率"""
    modules = ["EXT_M1_Tail_Pattern", "EXT_M3_Local_Focus"]
    module_scores = np.mean([get_module_score(mod, grid) for mod in modules], axis=0)
    result = {}
    for pos, probs in prob_map.items():
        r, c = pos
        weight = module_scores[r, c]
        total = sum(p * weight for p in probs.values()) or 1
        result[pos] = {n: (p * weight) / total for n, p in probs.items()}
    return result

def global_unique(prob_map: Dict[tuple, Dict[int, float]], blanks: List[tuple]) -> Dict[tuple, tuple]:
    """全局唯一分配"""
    nums = sorted({n for d in prob_map.values() for n in d})
    cost = np.zeros((len(blanks), len(nums)))
    for i, pos in enumerate(blanks):
        for j, n in enumerate(nums):
            cost[i, j] = -np.log(prob_map[pos].get(n, 1e-10))
    row, col = linear_sum_assignment(cost)
    return {blanks[r]: (nums[c], prob_map[blanks[r]].get(nums[c], 0.0)) for r, c in zip(row, col)}

def predict_scratch_card(grid: List[List[int]], target_num: Optional[int] = None, iterations: int = 10000) -> Dict[str, Any]:
    """預測刮刮樂盤面"""
    grid_np = np.array(grid, dtype=np.int64)
    blanks = [(r, c) for r in range(grid_np.shape[0]) for c in range(grid_np.shape[1]) if grid_np[r, c] == -1]
    prob_map = simulate_full_board(grid_np, target_num, iterations)
    prob_map = weight_prob_by_modules(grid_np, prob_map)
    
    if target_num:
        rank = [{"row": r, "col": c, "candidates": [target_num], "probability": prob_map[(r, c)].get(target_num, 0.0) * 100}
                for r, c in blanks]
        rank.sort(key=lambda x: x["probability"], reverse=True)
        return {"mode": "target", "predictions": rank[:3], "full_probabilities": prob_map}
    
    assign = global_unique(prob_map, blanks)
    preds = [{"row": r, "col": c, "candidates": [n], "probability": p * 100} for (r, c), (n, p) in assign.items()]
    return {"mode": "unique", "predictions": preds, "full_probabilities": prob_map}