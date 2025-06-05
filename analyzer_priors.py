"""
analyzer11_optimized.py - 優化版分析器，整合4模組，無歷史先驗
"""
import numpy as np
import logging
import time
from typing import List, Tuple, Optional

from vectorized_brain_modules import VectorizedBrainModules

logger = logging.getLogger(__name__)

def collect_all_scores(grid: np.ndarray, brain: VectorizedBrainModules) -> np.ndarray:
    """收集4模組分數（向量化）"""
    try:
        rows, cols = grid.shape
        tensor = np.zeros((4, rows, cols), dtype=np.float32)
        
        tensor[0] = brain.edge_proximity_fusion(grid)
        tensor[1] = brain.sequence_tail_analyzer(grid)
        tensor[2] = brain.connectivity_heatmap(grid)
        tensor[3] = brain.entropy_risk_fusion(grid)
        
        logger.debug("4模組分數收集完成")
        return tensor
    except Exception as e:
        logger.error(f"分數收集失敗: {e}")
        raise

def normalize_tensor(tensor: np.ndarray) -> np.ndarray:
    """向量化張量正規化（minmax）"""
    try:
        num_modules = tensor.shape[0]
        mins = tensor.reshape(num_modules, -1).min(axis=1, keepdims=True)
        maxs = tensor.reshape(num_modules, -1).max(axis=1, keepdims=True)
        
        ranges = maxs - mins
        ranges[ranges < 1e-8] = 1.0
        
        normalized = (tensor.reshape(num_modules, -1) - mins) / ranges
        return normalized.reshape(tensor.shape)
    except Exception as e:
        logger.error(f"正規化失敗: {e}")
        raise

def fuse_scores(normed: np.ndarray) -> np.ndarray:
    """向量化分數融合（固定權重）"""
    try:
        weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32).reshape(-1, 1, 1)
        return np.sum(normed * weights, axis=0)
    except Exception as e:
        logger.error(f"分數融合失敗: {e}")
        raise

def get_topk_positions(fused: np.ndarray, grid: np.ndarray, k: int = 3) -> List[Tuple[int, int, float]]:
    """獲取前k個最高分位置"""
    try:
        blank_mask = (grid == -1)
        masked_scores = np.where(blank_mask, fused, -np.inf)
        flat_scores = masked_scores.flatten()
        num_blanks = np.sum(blank_mask)
        
        if num_blanks == 0:
            logger.warning("無空格可分析")
            return []
        
        k = min(k, num_blanks)
        top_k_indices = np.argpartition(flat_scores, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(flat_scores[top_k_indices])[::-1]]
        
        results = []
        total_score = np.sum(masked_scores[blank_mask])
        for idx in top_k_indices:
            r = idx // grid.shape[1]
            c = idx % grid.shape[1]
            confidence = fused[r, c] / total_score if total_score > 0 else 0
            results.append((r, c, confidence))
        
        return results
    except Exception as e:
        logger.error(f"獲取Top-K失敗: {e}")
        raise

def analyze_with_prior(grid: np.ndarray, target: int, request_id: str = "API") -> List[Tuple[int, int, float]]:
    """主分析函數，整合4模組，無歷史先驗"""
    logger.info(f"[{request_id}] 開始分析 target={target}, grid={grid.shape}")
    
    try:
        if not np.any(grid == -1):
            raise ValueError("網格中沒有空格可分析")
        if grid.size == 0:
            raise ValueError("網格為空")
        if target < 1 or target > grid.size:
            raise ValueError(f"目標數字 {target} 無效")
        
        start_time = time.time()
        
        brain = VectorizedBrainModules()
        tensor = collect_all_scores(grid, brain)
        normed = normalize_tensor(tensor)
        fused = fuse_scores(normed)
        results = get_topk_positions(fused, grid, k=3)
        
        process_time = time.time() - start_time
        logger.info(f"[{request_id}] 分析完成，耗時: {process_time:.4f}秒")
        
        return results
    except Exception as e:
        logger.error(f"[{request_id}] 分析失敗: {e}")
        raise