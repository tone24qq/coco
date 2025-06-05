"""
analyzer11_optimized.py - Optimized analyzer with 4 modules, no historical priors
"""
import numpy as np
import logging
import time
from typing import List, Tuple, Optional

from vectorized_brain_modules import VectorizedBrainModules

logger = logging.getLogger(__name__)

def collect_all_scores(grid: np.ndarray, brain: VectorizedBrainModules) -> np.ndarray:
    """Collect scores from 4 modules (vectorized)"""
    try:
        rows, cols = grid.shape
        tensor = np.zeros((4, rows, cols), dtype=np.float32)
        
        tensor[0] = brain.edge_proximity_fusion(grid)
        tensor[1] = brain.sequence_tail_analyzer(grid)
        tensor[2] = brain.connectivity_heatmap(grid)
        tensor[3] = brain.entropy_risk_fusion(grid)
        
        logger.debug("Collected scores from 4 modules")
        return tensor
    except Exception as e:
        logger.error(f"Score collection failed: {e}")
        raise

def normalize_tensor(tensor: np.ndarray) -> np.ndarray:
    """Vectorized tensor normalization (minmax)"""
    try:
        num_modules = tensor.shape[0]
        mins = tensor.reshape(num_modules, -1).min(axis=1, keepdims=True)
        maxs = tensor.reshape(num_modules, -1).max(axis=1, keepdims=True)
        
        ranges = maxs - mins
        ranges[ranges < 1e-8] = 1.0
        
        normalized = (tensor.reshape(num_modules, -1) - mins) / ranges
        return normalized.reshape(tensor.shape)
    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        raise

def fuse_scores(normed: np.ndarray) -> np.ndarray:
    """Vectorized score fusion (fixed weights)"""
    try:
        weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32).reshape(-1, 1, 1)
        return np.sum(normed * weights, axis=0)
    except Exception as e:
        logger.error(f"Score fusion failed: {e}")
        raise

def get_topk_positions(fused: np.ndarray, grid: np.ndarray, k: int = 3) -> List[Tuple[int, int, float]]:
    """Get top-k highest-scoring positions"""
    try:
        blank_mask = (grid == -1)
        masked_scores = np.where(blank_mask, fused, -np.inf)
        flat_scores = masked_scores.flatten()
        num_blanks = np.sum(blank_mask)
        
        if num_blanks == 0:
            logger.warning("No blank cells to analyze")
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
        logger.error(f"Top-K selection failed: {e}")
        raise

def analyze_with_prior(grid: np.ndarray, target: int, request_id: str = "API") -> List[Tuple[int, int, float]]:
    """Main analysis function with 4 modules, no historical priors"""
    logger.info(f"[{request_id}] Starting analysis: target={target}, grid={grid.shape}")
    
    try:
        if not np.any(grid == -1):
            raise ValueError("No blank cells in grid")
        if grid.size == 0:
            raise ValueError("Grid is empty")
        if target < 1 or target > grid.size:
            raise ValueError(f"Invalid target number: {target}")
        
        start_time = time.time()
        
        brain = VectorizedBrainModules()
        tensor = collect_all_scores(grid, brain)
        normed = normalize_tensor(tensor)
        fused = fuse_scores(normed)
        results = get_topk_positions(fused, grid, k=3)
        
        process_time = time.time() - start_time
        logger.info(f"[{request_id}] Analysis completed in {process_time:.4f} seconds")
        
        return results
    except Exception as e:
        logger.error(f"[{request_id}] Analysis failed: {e}")
        raise