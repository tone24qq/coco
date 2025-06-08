import numpy as np
import logging
from modules import ScratchSolver
import asyncio
import json
import os
from numpy.lib.stride_tricks import sliding_window_view
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_board(
    grid: np.ndarray,
    target_num: int,
    knowledge_base: List[Dict] = None,
    heatmap_data: Dict[str, Any] = None
) -> Dict:
    solver = ScratchSolver()
    solver.update_tree(grid)

    # 驗證網格大小（4x4 至 20x20）
    if grid.shape[0] < 4 or grid.shape[1] < 4 or grid.shape[0] > 20 or grid.shape[1] > 20:
        logger.error("網格尺寸超出 4x4 至 20x20 限制")
        return {"recommendations": [], "confidence": []}
    
    N = grid.size
    opened_nums = set(grid[grid != -1])
    if len(opened_nums) != len(set(opened_nums)) or max(opened_nums, default=0) > N:
        logger.error("數字不滿足 1~N 不重複規則")
        return {"recommendations": [], "confidence": []}
    
    if target_num is not None and target_num in opened_nums:
        logger.warning(f"目標數字 {target_num} 已出現在盤面")
        return {"recommendations": [], "confidence": []}

    # 記錄知識庫和熱力圖載入
    if knowledge_base:
        logger.info(f"Loaded {len(knowledge_base)} KB concepts")
    if heatmap_data:
        logger.info(f"Loaded {len(heatmap_data)} heatmap files")

    # 讀取外部熱力圖數據
    initial_scores = None
    if heatmap_data:
        # 選擇與當前盤面尺寸匹配的熱力圖
        for name, data in heatmap_data.items():
            for sheet, heatmap in data.items():
                if np.array(heatmap).shape == grid.shape:
                    initial_scores = np.array(heatmap, dtype=float)
                    logger.info(f"Using heatmap: {name}, {sheet}")
                    break
            if initial_scores is not None:
                break
        if initial_scores is None:
            logger.warning("未找到匹配的熱力圖數據，將使用預設零分數")
            initial_scores = np.zeros_like(grid, dtype=float)

    # 模組計算
    mod_scores = {}
    for mod_name, mod_func in solver.MODULE_REGISTRY.items():
        try:
            result = mod_func(grid)
            if isinstance(result, tuple):
                mod_scores[mod_name] = result[0]  # 取分數
            else:
                mod_scores[mod_name] = result
            if mod_scores[mod_name].size != np.count_nonzero(grid == -1):
                logger.warning(f"{mod_name} 返回分數大小 {mod_scores[mod_name].size} 與未開格數 {np.count_nonzero(grid == -1)} 不匹配")
                mod_scores[mod_name] = np.zeros(np.count_nonzero(grid == -1))
        except Exception as e:
            logger.error(f"{mod_name} 失敗: {e}")
            mod_scores[mod_name] = np.zeros(np.count_nonzero(grid == -1))

    # 應用知識庫（例如增強等差數列檢測）
    if knowledge_base:
        for concept in knowledge_base:
            if concept.get("term") == "Arithmetic Progression":
                patterns = solver.analyze_number_patterns(grid)
                for (idx, direction), pattern in patterns.items():
                    if pattern['type'] == 'arithmetic':
                        mod_scores['analyze_number_patterns'] *= 1.5  # 增強等差數列分數
                        logger.info(f"Applied Arithmetic Progression enhancement for {direction} pattern")

    # 盤面分類
    board_type = solver.classify_board_type(mod_scores.get('compute_dynamic_hot_cold_vectorized', np.zeros_like(list(mod_scores.values())[0])))

    # 使用自適應權重
    solver.adaptive_weights.update(success_rate=np.random.random(), module_scores=mod_scores)
    final_score = solver.fuse_scores_vectorized(mod_scores, board_type, solver.adaptive_weights.weights)

    # 整合外部熱力圖分數
    if initial_scores is not None:
        heatmap_scores = initial_scores[grid == -1]
        if heatmap_scores.size == final_score.size:
            final_score = 0.7 * final_score + 0.3 * heatmap_scores  # 加權融合
            logger.info("Integrated external heatmap scores into final score")

    # 預測整合
    patterns = solver.analyze_number_patterns(grid)
    predictions, confidence = solver.integrate_predictions(grid, final_score, patterns)

    # 確保非空 Top-3
    empty_yx = np.argwhere(grid == -1)
    if not empty_yx.size:
        top3 = [(0, 0, 0.1, {"default": 0.1})]
    else:
        top3 = solver.predict_top3_vectorized(final_score, empty_yx)
        top3 = filter_global_rules(top3, grid)
        if not top3 or len(top3) < 3:
            valid_empty_yx = [(int(y), int(x), 0.1, {"default": 0.1}) for y, x in empty_yx if grid[y, x] == -1]
            top3 = valid_empty_yx[:3]
            top3.extend([(0, 0, 0.1, {"default": 0.1})] * (3 - len(top3)))

    # 回傳包含細部加權說明
    weights = solver.adaptive_weights.weights
    top3_with_weights = [
        {
            "row": pos[0],
            "col": pos[1],
            "value": pos[2] if isinstance(pos[2], (int, float)) else pos[2].get("default", 0.1),
            "confidence": max(float(pos[2]), 0.1) if not isinstance(pos[2], dict) else max(float(pos[2].get("default", 0.1)), 0.1),
            "weights": {k: weights.get(k, 0.0) for k in solver.MODULE_REGISTRY.keys()}
        } for pos in top3
    ]
    return {"recommendations": top3_with_weights, "confidence": confidence.tolist()}

def filter_global_rules(suggestions, grid):
    """
    過濾建議：若建議值在全盤、行或列已存在，則刪除該建議。

    Args:
        suggestions: 包含 (row, col, value, confidence) 的建議列表。
        grid: 當前盤面，-1 表示未開格。

    Returns:
        過濾後的建議列表。
    """
    h, w = grid.shape
    filtered = []
    existing = {v for row in grid for v in row if v != -1}
    for row, col, value, confidence in suggestions:
        if value not in existing:
            row_vals = set(grid[row])
            col_vals = {grid[r][col] for r in range(h)}
            if value not in row_vals and value not in col_vals:
                filtered.append((row, col, value, confidence))
    return filtered