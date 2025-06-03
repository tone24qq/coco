# analyzer11.py
"""
analyzer11.py：實作分數蒐集、正規化與融合，並選取 Top‐K 位置。
依據 new_module3.py 中 REGISTERED_MODULES_BRAIN 動態呼叫所有 EXT_*_Vec 函式。
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Callable, Any
import new_module3

logger = logging.getLogger(__name__)

def collect_all_scores(grid: np.ndarray, request_id: str = "API") -> np.ndarray:
    """
    執行所有被註冊的 EXT_*_Vec 模組，收集分數矩陣。
    回傳 tensor shape = (num_modules, rows, cols)。
    """
    score_list = []
    for name, func in new_module3.REGISTERED_MODULES_BRAIN.items():
        try:
            # 所有函式簽名皆為 (grid: np.ndarray, request_id: Optional[str]) -> np.ndarray
            score = func(grid, request_id=request_id)
            if not isinstance(score, np.ndarray):
                score = np.array(score, dtype=float)
            if score.shape != grid.shape:
                logger.warning(f"Module {name} 產出 shape {score.shape} != grid shape {grid.shape}")
            score_list.append(score.astype(float))
        except Exception as e:
            logger.error(f"Error executing module {name}: {e}", exc_info=True)
            # 若某模組失敗，直接填 0 矩陣，避免整個流程中斷
            rows, cols = grid.shape
            score_list.append(np.zeros((rows, cols), dtype=float))
    if not score_list:
        logger.error("No scoring modules found in REGISTERED_MODULES_BRAIN.")
        return np.empty((0, *grid.shape))
    tensor = np.stack(score_list, axis=0)
    logger.info(f"Collected scores from {tensor.shape[0]} modules.")
    return tensor

def normalize_tensor(tensor: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    對 tensor 的每個模組（第一維度）做正規化 (minmax 或 zscore)。
    返回 shape (=tensor.shape) 的新陣列。
    """
    normalized = tensor.astype(float).copy()
    num_mod = normalized.shape[0]
    if method == "minmax":
        for i in range(num_mod):
            arr = normalized[i]
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-9:
                normalized[i] = 0
            else:
                normalized[i] = (arr - mn) / (mx - mn)
    elif method == "zscore":
        for i in range(num_mod):
            arr = normalized[i]
            mean, std = arr.mean(), arr.std()
            if std < 1e-9:
                normalized[i] = 0
            else:
                normalized[i] = (arr - mean) / std
    else:
        logger.warning(f"Unknown normalization method: {method}. Skipping.")
    return normalized

def fuse_scores(tensor: np.ndarray, weights: List[float] = None) -> np.ndarray:
    """
    融合多個模組分數：
    - 若 weights 為 None，則等權平均；否則使用 weights 做加權和。
    返回 fused shape = (rows, cols)。
    """
    num_mod, rows, cols = tensor.shape
    if weights is None:
        return tensor.mean(axis=0)
    w = np.array(weights, dtype=float)
    if w.shape[0] != num_mod or abs(w.sum()) < 1e-9:
        logger.error("Weights invalid or mismatched. Using equal weights.")
        return tensor.mean(axis=0)
    w_norm = w / w.sum()
    return np.tensordot(w_norm, tensor, axes=([0], [0]))

def get_topk_positions(fused: np.ndarray, grid: np.ndarray, k: int = 3) -> List[Tuple[int,int,float]]:
    """
    從 fused 分數矩陣中，僅選取 grid 中值為 -1（被遮蔽）的索引，找出前 k 高分位置。
    回傳列表 [ ((r0,c0), score0), ((r1,c1), score1), ... ]，r,c 為 0-based index。
    """
    blank_idx = np.argwhere(grid == -1)
    if blank_idx.size == 0:
        logger.warning("Grid has no masked cells.")
        return []
    # 取出 fused[mask] 的 values
    scores = fused[grid == -1]
    k = min(k, scores.size)
    top_idxs = np.argsort(scores)[::-1][:k]
    top_pos = blank_idx[top_idxs]
    top_scores = scores[top_idxs]
    result = [((int(r), int(c)), float(s)) for (r, c), s in zip(top_pos, top_scores)]
    return result