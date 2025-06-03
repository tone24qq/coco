# analyzer11.py
"""
分析與融合模組：
包含將各模組分數蒐集、正規化、融合，以及選取 Top-K 位置等功能。
"""
import numpy as np
import logging
import inspect
import new_module3

def collect_all_scores(grid: np.ndarray, target: int) -> np.ndarray:
    """
    執行所有模組的 _score 函式，收集分數矩陣。
    回傳 shape = (num_modules, rows, cols) 的 tensor，其中 num_modules 為模組數。
    """
    score_matrices = []
    for name, func in inspect.getmembers(new_module3, inspect.isfunction):
        if name.endswith('_score'):
            try:
                result = func(grid, target)
                if not isinstance(result, np.ndarray):
                    result = np.array(result, dtype=float)
                if result.shape != grid.shape:
                    logging.warning(f"Module {name} output shape {result.shape} != grid shape {grid.shape}.")
                score_matrices.append(result.astype(float))
            except Exception as e:
                logging.error(f"Error in module {name}: {e}")
    if not score_matrices:
        logging.error("No score modules executed.")
        return np.empty((0, *grid.shape))
    tensor = np.stack(score_matrices, axis=0)
    logging.info(f"Collected scores from {tensor.shape[0]} modules.")
    return tensor

def normalize_tensor(tensor: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    對模組分數 tensor 在每一模組維度上進行正規化。
    支援 'minmax'（0-1線性縮放）與 'zscore'（標準化）兩種方法。
    """
    normalized = tensor.astype(float).copy()
    num_modules = normalized.shape[0]
    if method == "minmax":
        for i in range(num_modules):
            min_val = normalized[i].min()
            max_val = normalized[i].max()
            if max_val - min_val < 1e-9:
                normalized[i] = 0  # 所有值相等，直接設為0矩陣
            else:
                normalized[i] = (normalized[i] - min_val) / (max_val - min_val)
    elif method == "zscore":
        for i in range(num_modules):
            mean = normalized[i].mean()
            std = normalized[i].std()
            if std < 1e-9:
                normalized[i] = 0
            else:
                normalized[i] = (normalized[i] - mean) / std
    else:
        logging.warning(f"Unknown normalization method: {method}. Skipping normalization.")
    return normalized

def fuse_scores(tensor: np.ndarray, weights: list = None) -> np.ndarray:
    """
    融合多個模組的分數。
    可等權重（預設）或使用提供的權重列表。
    回傳融合後的單一分數矩陣，shape = (rows, cols)。
    """
    num_modules, rows, cols = tensor.shape
    if weights is None:
        fused = tensor.mean(axis=0)
    else:
        w = np.array(weights, dtype=float)
        if w.shape[0] != num_modules:
            logging.error("Weights length mismatch. Using equal weights.")
            fused = tensor.mean(axis=0)
        else:
            if abs(w.sum()) < 1e-9:
                logging.error("Weights sum to zero. Using equal weights.")
                fused = tensor.mean(axis=0)
            else:
                w_norm = w / w.sum()
                fused = np.tensordot(w_norm, tensor, axes=([0], [0]))
    return fused

def get_topk_positions(fused_scores: np.ndarray, grid: np.ndarray, k: int = 3) -> list:
    """
    根據融合後的分數矩陣，從原 grid 的遮蔽區域中選出分數最高的 k 個位置。
    回傳包含 ((row_index, col_index), score) 的列表（使用0-based索引）。
    """
    blank_indices = np.argwhere(grid == -1)
    if blank_indices.size == 0:
        logging.warning("No masked cells in grid.")
        return []
    scores = fused_scores[grid == -1]
    k = min(k, scores.size)
    top_idx = np.argsort(scores)[::-1][:k]
    top_positions = blank_indices[top_idx]
    top_scores = scores[top_idx]
    result = [((int(r), int(c)), float(s)) for (r, c), s in zip(top_positions, top_scores)]
    return result