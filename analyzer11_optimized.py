```python
"""
analyzer11_optimized.py - 優化的分析器，包含 4 個模組，無歷史先驗
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
    """從所有評分模組收集分數，生成 3D 張量。
    
    參數:
        grid: 2D 整數陣列，-1 表示空白格。
        brain: 包含評分模組的實例。
        
    返回:
        形狀為 [模組數, 行數, 列數] 的 3D 張量，包含分數。
        
    異常:
        ValueError: 如果 grid 無效。
        Exception: 如果任何評分模組失敗。
    """
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)):
        raise ValueError("Grid 必須是 2D 整數 numpy 陣列")
    
    try:
        rows, cols = grid.shape
        num_modules = len(SCORING_MODULES)
        tensor = np.zeros((num_modules, rows, cols), dtype=np.float32)
        
        for i, (module_name, module_func) in enumerate(SCORING_MODULES.items()):
            start_time = time.time()
            tensor[i] = module_func(grid)
            logger.debug(f"{module_name} 耗時 {time.time - start_time:.4f} 秒")
        
        logger.debug("已收集所有模組的分數")
        return tensor
    except Exception as e:
        logger.error(f"分數收集失敗: {e}")
        raise

def normalize_tensor(tensor: np.ndarray) -> np.ndarray:
    """使用 min-max 縮放進行向量化的張量正規化。
    
    參數:
        tensor: 形狀為 [模組數, 行數, 列數] 的 3D 張量，包含原始分數。
        
    返回:
        正規化後的值在 [0, 1] 的 3D 張量。
        
    異常:
        ValueError: 如果張量無效。
        Exception: 如果正規化失敗。
    """
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
    """向量化的分數融合，支持可選的加權組合。
    
    參數:
        normed: 形狀為 [模組數, 行數, 列數] 的正規化 3D 張量。
        weights: 每個模組的權重列表，預設為均等權重。
        
    返回:
        2D 熱圖，包含融合後的分數。
        
    異常:
        ValueError: 如果輸入無效。
        Exception: 如果融合失敗。
    """
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

def get_topk_positions(fused: np.ndarray, grid: np.ndarray, k: int = DEFAULT_K) -> List[Tuple[int, int, float]]:
    """從融合分數中獲取前 k 個最高分數的位置。
    
    參數:
        fused: 2D 融合分數陣列。
        grid: 2D 整數陣列，-1 表示空白格。
        k: 返回的前 k 個位置數量，預設為 3。
        
    返回:
        包含 (行, 列, 置信度) 元組的列表。
        
    異常:
        ValueError: 如果輸入無效。
        Exception: 如果 top-k 選擇失敗。
    """
    if not (isinstance(fused, np.ndarray) and fused.ndim == 2 and isinstance(grid, np.ndarray) and grid.ndim == 2):
        raise ValueError("融合分數和 grid 必須是 2D numpy 陣列")
    
    try:
        blank_mask = (grid == -1)
        masked_scores = np.where(blank_mask, fused, -np.inf)
        flat_scores = masked_scores.flatten()
        num_blanks = np.sum(blank_mask)
        
        if num_blanks == 0:
            logger.warning("無空白格可分析")
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
        logger.error(f"Top-K 選擇失敗: {e}")
        raise

def detect_skip_patterns(grid: np.ndarray) -> np.ndarray:
    """檢測行/列跳躍模式並返回熱圖。
    
    參數:
        grid: 2D 整數陣列，-1 表示空白格。
        
    返回:
        2D 熱圖，包含基於跳躍模式可能性的分數。
    """
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2):
        raise ValueError("Grid 必須是 2D numpy 陣列")
    
    rows, cols = grid.shape
    heatmap = np.zeros((rows, cols), dtype=np.float32)
    blank_mask = (grid == -1)
    
    for axis in range(2):
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
    """基於 3x3 窗口內已知數字的局部密度計算焦點分數。
    
    參數:
        grid: 2D 整數陣列，-1 表示空白格。
        
    返回:
        2D 熱圖，包含基於局部密度的分數。
    """
    from scipy.signal import convolve2d
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2):
        raise ValueError("Grid 必須是 2D numpy 陣列")
    
    kernel = np.ones((3, 3), dtype=np.float32)
    density = convolve2d((grid > 0).astype(np.float32), kernel, mode='same', boundary='symm')
    max_density = np.max(density)
    return np.where(grid == -1, density / (max_density + NORMALIZATION_EPSILON), 0)

def detect_mirror_sequences(grid: np.ndarray) -> np.ndarray:
    """檢測水平/垂直鏡像後的序列模式。
    
    參數:
        grid: 2D 整數陣列，-1 表示空白格。
        
    返回:
        2D 熱圖，包含基於鏡像序列完成可能性的分數。
    """
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2):
        raise ValueError("Grid 必須是 2D numpy 陣列")
    
    rows, cols = grid.shape
    heatmap = np.zeros((rows, cols), dtype=np.float32)
    blank_mask = (grid == -1)
    
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
    
    v_mirrored = grid[::-1, :]
    for j in range(cols):
        col = v_mirrored[:, j]
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
    """基於相鄰已知數字的算術進展可能性計算差異趨勢分數。
    
    參數:
        grid: 2D 整數陣列，-1 表示空白格。
        
    返回:
        2D 熱圖，包含基於算術進展可能性的分數。
    """
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2):
        raise ValueError("Grid 必須是 2D numpy 陣列")
    
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
    """主分析函數，包含 4 個模組，無歷史先驗。
    
    參數:
        grid: 2D 整數陣列，-1 表示空白格。
        target: 預測的目標數字（非負）。
        request_id: 日誌識別符，預設為 "API"。
        
    返回:
        包含前 k 個 (行, 列, 置信度) 位置的列表。
        
    異常:
        ValueError: 如果 grid 或 target 無效。
    """
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
        fused = fuse_scores(normed)
        results = get_topk_positions(fused, grid, k=DEFAULT_K)
        
        logger.info(f"[{request_id}] 分析完成，耗時 {time.time() - start_time:.4f} 秒")
        return results
    except Exception as e:
        logger.error(f"[{request_id}] 分析失敗: {e}")
        raise
```

