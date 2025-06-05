"""
analyzer11_optimized.py - Optimized analyzer with 4 modules, no historical priors
"""
import numpy as np
import logging
import time
from typing import List, Tuple, Optional

from vectorized_brain_modules import VectorizedBrainModules
from vectorized_modules import SCORING_MODULES

logger = logging.getLogger(__name__)

# Module constants
DEFAULT_K = 3  # Default top-k value for position selection
NORMALIZATION_EPSILON = 1e-8  # Small epsilon for normalization to avoid division by zero

def collect_all_scores(grid: np.ndarray, brain: VectorizedBrainModules) -> np.ndarray:
    """Collect scores from all scoring modules into a 3D tensor.
    
    Args:
        grid: 2D integer array with -1 indicating blank cells.
        brain: Instance containing scoring modules.
        
    Returns:
        3D tensor of shape [num_modules, rows, cols] with scores.
        
    Raises:
        ValueError: If grid is invalid.
        Exception: If any scoring module fails.
    """
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)):
        raise ValueError("Grid must be a 2D integer numpy array")
    
    try:
        rows, cols = grid.shape
        num_modules = len(SCORING_MODULES)
        tensor = np.zeros((num_modules, rows, cols), dtype=np.float32)
        
        for i, (module_name, module_func) in enumerate(SCORING_MODULES.items()):
            start_time = time.time()
            tensor[i] = module_func(grid)
            logger.debug(f"{module_name} took {time.time() - start_time:.4f} seconds")
        
        logger.debug("Collected scores from all modules")
        return tensor
    except Exception as e:
        logger.error(f"Score collection failed: {e}")
        raise

def normalize_tensor(tensor: np.ndarray) -> np.ndarray:
    """Vectorized tensor normalization using min-max scaling.
    
    Args:
        tensor: 3D tensor of shape [num_modules, rows, cols] with raw scores.
        
    Returns:
        Normalized 3D tensor with values in [0, 1].
        
    Raises:
        ValueError: If tensor is invalid.
        Exception: If normalization fails.
    """
    if not (isinstance(tensor, np.ndarray) and tensor.ndim == 3):
        raise ValueError("Tensor must be a 3D numpy array")
    
    try:
        num_modules = tensor.shape[0]
        mins = tensor.reshape(num_modules, -1).min(axis=1, keepdims=True)
        maxs = tensor.reshape(num_modules, -1).max(axis=1, keepdims=True)
        
        ranges = maxs - mins
        ranges[ranges < NORMALIZATION_EPSILON] = 1.0
        
        normalized = (tensor.reshape(num_modules, -1) - mins) / ranges
        return normalized.reshape(tensor.shape)
    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        raise

def fuse_scores(normed: np.ndarray, weights: Optional[List[float]] = None) -> np.ndarray:
    """Vectorized score fusion with optional weighted combination.
    
    Args:
        normed: Normalized 3D tensor of shape [num_modules, rows, cols].
        weights: List of weights for each module, defaults to equal weights.
        
    Returns:
        2D heatmap with fused scores.
        
    Raises:
        ValueError: If inputs are invalid.
        Exception: If fusion fails.
    """
    if not (isinstance(normed, np.ndarray) and normed.ndim == 3):
        raise ValueError("Normed tensor must be a 3D numpy array")
    
    try:
        num_modules = normed.shape[0]
        if weights is None:
            weights = np.array([1.0 / num_modules] * num_modules, dtype=np.float32)
        else:
            weights = np.array(weights, dtype=np.float32) / np.sum(weights)
        weights = weights.reshape(-1, 1, 1)
        return np.sum(normed * weights, axis=0)
    except Exception as e:
        logger.error(f"Score fusion failed: {e}")
        raise

def get_topk_positions(fused: np.ndarray, grid: np.ndarray, k: int = DEFAULT_K) -> List[Tuple[int, int, float]]:
    """Get top-k highest-scoring positions from fused scores.
    
    Args:
        fused: 2D array of fused scores.
        grid: 2D integer array with -1 indicating blank cells.
        k: Number of top positions to return, defaults to 3.
        
    Returns:
        List of (row, col, confidence) tuples.
        
    Raises:
        ValueError: If inputs are invalid.
        Exception: If top-k selection fails.
    """
    if not (isinstance(fused, np.ndarray) and fused.ndim == 2 and isinstance(grid, np.ndarray) and grid.ndim == 2):
        raise ValueError("Fused and grid must be 2D numpy arrays")
    
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

def detect_skip_patterns(grid: np.ndarray) -> np.ndarray:
    """Detect row/column skip patterns and return a heatmap.
    
    Args:
        grid: 2D integer array with -1 indicating blank cells.
        
    Returns:
        2D heatmap with scores indicating likelihood based on skip patterns.
    """
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2):
        raise ValueError("Grid must be a 2D numpy array")
    
    rows, cols = grid.shape
    heatmap = np.zeros((rows, cols), dtype=np.float32)
    blank_mask = (grid == -1)
    
    for axis in range(2):  # 0 for rows, 1 for columns
        data = grid if axis == 0 else grid.T
        size = cols if axis == 0 else rows
        
        for i in range(size):
            row = data[i]
            filled_indices = np.where(row > 0)[0]
            if len(filled_indices) < 2:
                continue
            differences = np.diff(filled_indices)
            common_diff = np.median(differences) if len(differences) > 0 else 1
            
            for j in range(size):
                if (blank_mask[i, j] if axis == 0 else blank_mask[j, i]):
                    next_expected = filled_indices[-1] + common_diff if filled_indices.size > 0 else j
                    if abs(j - next_expected) <= 1:
                        if axis == 0:
                            heatmap[i, j] = 0.9
                        else:
                            heatmap[j, i] = 0.9
    return heatmap

def compute_focus_score(grid: np.ndarray) -> np.ndarray:
    """Compute focus score based on local density of known numbers in a 3x3 window.
    
    Args:
        grid: 2D integer array with -1 indicating blank cells.
        
    Returns:
        2D heatmap with scores based on local density.
    """
    from scipy.signal import convolve2d
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2):
        raise ValueError("Grid must be a 2D numpy array")
    
    kernel = np.ones((3, 3), dtype=np.float32)
    density = convolve2d((grid > 0).astype(np.float32), kernel, mode='same', boundary='symm')
    max_density = np.max(density)
    return np.where(grid == -1, density / (max_density + NORMALIZATION_EPSILON), 0)

def detect_mirror_sequences(grid: np.ndarray) -> np.ndarray:
    """Detect mirror sequences after horizontal/vertical mirroring.
    
    Args:
        grid: 2D integer array with -1 indicating blank cells.
        
    Returns:
        2D heatmap with scores for potential mirror sequence completions.
        
    Notes:
        Scores are assigned if mirroring suggests a consecutive number (e.g., 3,4,-1 -> 5).
    """
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2):
        raise ValueError("Grid must be a 2D numpy array")
    
    rows, cols = grid.shape
    heatmap = np.zeros((rows, cols), dtype=np.float32)
    blank_mask = (grid == -1)
    
    # Horizontal mirror
    h_mirrored = grid[:, ::-1]
    for i in range(rows):
        row = h_mirrored[i]
        filled = row[row > 0]
        if len(filled) >= 2:
            sorted_filled = np.sort(filled)
            for j in range(cols):
                if blank_mask[i, cols-1-j]:
                    expected = sorted_filled[-1] + 1 if sorted_filled[-1] < rows * cols else 0
                    if expected == sorted_filled[-2] + 2:
                        heatmap[i, cols-1-j] = 0.8
    
    # Vertical mirror
    v_mirrored = grid[::-1, :]
    for j in range(cols):
        col = v_mirrored,No newline at end of file
        filled = col[col > 0]
        if len(filled) >= 2:
            sorted_filled = np.sort(filled)
            for i in range(rows):
                if blank_mask[rows-1-i, j]:
                    expected = sorted_filled[-1] + 1 if sorted_filled[-1] < rows * cols else 0
                    if expected == sorted_filled[-2] + 2:
                        heatmap[rows-1-i, j] = 0.8
    
    return heatmap

def compute_difference_trend(grid: np.ndarray) -> np.ndarray:
    """Compute difference trend scores based on arithmetic progression likelihood.
    
    Args:
        grid: 2D integer array with -1 indicating blank cells.
        
    Returns:
        2D heatmap with scores based on arithmetic progression likelihood.
    """
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2):
        raise ValueError("Grid must be a 2D numpy array")
    
    rows, cols = grid.shape
    heatmap = np.zeros((rows, cols), dtype=np.float32)
    blank_mask = (grid == -1)
    
    for i in range(rows):
        for j in range(cols):
            if blank_mask[i, j]:
                neighbors = []
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and grid[ni, nj] > 0:
                        neighbors.append(grid[ni, nj])
                if len(neighbors) >= 2:
                    differences = np.diff(sorted(neighbors))
                    median_diff = np.median(differences)
                    expected = neighbors[0] + median_diff * (len(neighbors) + 1)
                    if 1 <= expected <= rows * cols:
                        heatmap[i, j] = 0.7 / (1 + abs(expected - np.mean(neighbors)))
    
    return heatmap

def analyze_with_prior(grid: np.ndarray, target: int, request_id: str = "API") -> List[Tuple[int, int, float]]:
    """Main analysis function with 4 modules, no historical priors.
    
    Args:
        grid: 2D integer array with -1 indicating blank cells.
        target: Target number to predict (non-negative).
        request_id: Identifier for logging, defaults to "API".
        
    Returns:
        List of top-k (row, col, confidence) positions.
        
    Raises:
        ValueError: If grid or target is invalid.
    """
    logger.info(f"[{request_id}] Starting analysis: target={target}, grid={grid.shape}")
    
    try:
        # Input validation
        if not (isinstance(grid, np.ndarray) and grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)):
            raise ValueError("Grid must be a 2D integer numpy array")
        if target < 0:
            raise ValueError("Target cannot be negative")
        if not np.any(grid == -1):
            raise ValueError("Grid must contain at least one blank cell (-1)")
        
        start_time = time.time()
        
        brain = VectorizedBrainModules()
        tensor = collect_all_scores(grid, brain)
        normed = normalize_tensor(tensor)
        fused = fuse_scores(normed)
        results = get_topk_positions(fused, grid, k=DEFAULT_K)
        
        logger.info(f"[{request_id}] Analysis completed in {time.time() - start_time:.4f} seconds")
        return results
    except Exception as e:
        logger.error(f"[{request_id}] Analysis failed: {e}")
        raise
