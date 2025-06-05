"""
analyzer11_optimized.py - 優化的分析器，包含模組調用，無歷史先驗
"""
import numpy as np
import logging
import time
from typing import List, Tuple, Optional

from vectorized_brain_modules import VectorizedBrainModules
from vectorized_modules import SCORING_MODULES

logger = logging.getLogger(__name__)

DEFAULT_K = 3
NORMALIZATION_EPSILON = 1e-8

def collect_all_scores(grid: np.ndarray, brain: VectorizedBrainModules) -> np.ndarray:
    """從所有評分模組收集分數，生成 3D 張量。"""
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)):
        raise ValueError("Grid 必須是 2D 整數 numpy 陣列")
    
    try:
        rows, cols = grid.shape
        num_modules = len(SCORING_MODULES)
        tensor = np.zeros((num_modules, rows, cols), dtype=np.float32)
        
        for i, (module_name, module_func) in enumerate(SCORING_MODULES.items()):
            start_time = time.time()
            tensor[i] = module_func(grid)
            logger.debug(f"{module_name} 耗時 {time.time() - start_time:.4f} 秒")
        
        logger.debug("已收集所有模組的分數")
        return tensor
    except Exception as e:
        logger.error(f"分數收集失敗: {e}")
        raise

def normalize_tensor(tensor: np.ndarray) -> np.ndarray:
    """使用 min-max 縮放進行向量化的張量正規化。"""
    if not (isinstance(tensor, np.ndarray) and tensor.ndim == 3):
        raise ValueError("張量必須是 3D numpy 陣列")
    
    try:
        num_modules = tensor.shape[0]
        mins = tensor.reshape(num_modules, -1).min(axis=1, keepdims=True)
        maxs = tensor.reshape(num_modules, -1).max(axis=1, keepdims=True)
        
        ranges = maxs - mins
        ranges[ranges < NORMALIZATION_EPSILON] = 1.0
        
        normalized = (tensor.reshape(num_modules, -1) - mins) / ranges
        return normalized.reshape(tensor.shape)
    except Exception as e:
        logger.error(f"正規化失敗: {e}")
        raise

def fuse_scores(normed: np.ndarray, weights: Optional[List[float]] = None) -> np.ndarray:
    """向量化的分數融合，支持可選的加權組合。"""
    if not (isinstance(normed, np.ndarray) and normed.ndim == 3):
        raise ValueError("正規化張量必須是 3D numpy 陣列")
    
    try:
        num_modules = normed.shape[0]
        if weights is None:
            weights = np.array([1.0 / num_modules] * num_modules, dtype=np.float32)
        else:
            weights = np.array(weights, dtype=np.float32) / np.sum(weights)
        weights = weights.reshape(-1, 1, 1)
        return np.sum(normed * weights, axis=0)
    except Exception as e:
        logger.error(f"分數融合失敗: {e}")
        raise

def get_topk_positions(
    fused: np.ndarray, grid: np.ndarray, k: int = DEFAULT_K
) -> List[Tuple[int, int, float]]:
    """從融合分數中獲取前 k 個最高分數的位置。"""
    if not (
        isinstance(fused, np.ndarray)
        and fused.ndim == 2
        and isinstance(grid, np.ndarray)
        and grid.ndim == 2
    ):
        raise ValueError("融合分數和 grid 必須是 2D numpy 陣列")
    
    try:
        blank_mask = (grid == -1)
        masked_scores = np.where(blank_mask, fused, -np.inf)
        flat_scores = masked_scores.flatten()
        num_blanks = int(np.sum(blank_mask))
        
        if num_blanks == 0:
            logger.warning("無空白格可分析")
            return []
        
        k = min(k, num_blanks)
        top_k_indices = np.argpartition(flat_scores, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(flat_scores[top_k_indices])[::-1]]
        
        results = []
        total_score = float(np.sum(masked_scores[blank_mask]))
        for idx in top_k_indices:
            r = idx // grid.shape[1]
            c = idx % grid.shape[1]
            confidence = (fused[r, c] / total_score) if total_score > 0 else (1.0 / k)
            results.append((r, c, confidence))
        
        return results
    except Exception as e:
        logger.error(f"Top-K 選擇失敗: {e}")
        raise

def analyze_with_prior(
    grid: np.ndarray, target: int, request_id: str = "API", weights: Optional[List[float]] = None
) -> List[Tuple[int, int, float]]:
    """主分析函數，調用所有模組，無歷史先驗。"""
    logger.info(f"[{request_id}] 開始分析: target={target}, grid={grid.shape}")
    
    try:
        if not (isinstance(grid, np.ndarray) and grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)):
            raise ValueError("Grid 必須是 2D 整數 numpy 陣列")
        if target < 0:
            raise ValueError("目標數字不能為負")
        if not np.any(grid == -1):
            raise ValueError("Grid 必須包含至少一個空白格 (-1)")
        
        start_time = time.time()
        
        brain = VectorizedBrainModules()
        tensor = collect_all_scores(grid, brain)
        normed = normalize_tensor(tensor)
        fused = fuse_scores(normed, weights)
        results = get_topk_positions(fused, grid, k=DEFAULT_K)
        
        logger.info(f"[{request_id}] 分析完成，耗時 {time.time() - start_time:.4f} 秒")
        return results
    except Exception as e:
        import traceback
        logger.error(f"[{request_id}] 分析失敗: {e}\n{traceback.format_exc()}")
        raise
