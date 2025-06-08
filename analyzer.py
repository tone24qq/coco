# analyzer.py
import numpy as np
import logging
from modules import ScratchSolver
import asyncio
import json
import os
from numpy.lib.stride_tricks import sliding_window_view

# 日誌設置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_board(grid: np.ndarray, weights: dict, return_predictions: bool = False, target_num: int = None, json_heatmap_path: str = None):
    solver = ScratchSolver()
    solver.update_tree(grid)

    # 驗證網格大小，最大 20x20
    if grid.shape[0] < 4 or grid.shape[1] < 5 or grid.shape[0] > 20 or grid.shape[1] > 20:
        logger.error("網格尺寸超出 4x5 至 20x20 限制")
        raise ValueError("網格尺寸超出 4x5 至 20x20 限制")
    N = grid.size
    opened_nums = set(grid[grid != -1])
    if len(opened_nums) != len(set(opened_nums)) or max(opened_nums, default=0) > N:
        logger.error("數字不滿足 1~N 不重複規則")
        raise ValueError("數字不滿足 1~N 不重複規則")

    if target_num is not None and target_num in opened_nums:
        logger.error(f"目標數字 {target_num} 已出現在盤面")
        raise ValueError(f"目標數字 {target_num} 已出現在盤面")

    # 讀取 JSON 熱力圖
    initial_scores = None
    if json_heatmap_path and os.path.exists(json_heatmap_path):
        try:
            with open(json_heatmap_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            initial_scores = np.array(data.get('heatmap', np.zeros_like(grid, dtype=float)))
            if initial_scores.shape != grid.shape:
                logger.warning(f"JSON 熱力圖形狀 {initial_scores.shape} 與網格 {grid.shape} 不匹配")
                initial_scores = None
        except Exception as e:
            logger.error(f"無法讀取 JSON 熱力圖: {e}")
            initial_scores = None

    # 模組計算
    mod_scores = {}
    for mod_name, mod_func in solver.MODULE_REGISTRY.items():
        try:
            mod_scores[mod_name] = mod_func(grid)
        except Exception as e:
            logger.error(f"{mod_name} 失敗: {e}")
            mod_scores[mod_name] = np.zeros(np.count_nonzero(grid == -1))

    # 盤面分類
    board_type = solver.classify_board_type(mod_scores.get('compute_dynamic_hot_cold_vectorized', np.zeros_like(list(mod_scores.values())[0])))

    # 使用自適應權重
    solver.adaptive_weights.update(success_rate=np.random.random(), module_scores=mod_scores)  # 模擬成功率
    final_score = solver.fuse_scores_vectorized(mod_scores, board_type, solver.adaptive_weights.weights)

    # 預測整合
    patterns = solver.analyze_number_patterns(grid)
    predictions, confidence = solver.integrate_predictions(grid, final_score, patterns)

    # 針對目標數字的推論
    empty_yx = np.argwhere(grid == -1)
    top3 = solver.predict_top3_vectorized(final_score, empty_yx)
    
    # 確保有最終答案，即使信心分數低
    if target_num is not None:
        if not empty_yx.size:
            logger.warning("無未開格，無法推測目標數字")
            best_pos = [(0, 0, 0.1, {"default": 0.1})]  # 最低信心分數
        else:
            best_idx = np.argmax(final_score)
            best_pos = [(int(empty_yx[best_idx][0]), int(empty_yx[best_idx][1]), float(final_score[best_idx]), 
                         {name: float(mod_scores[name][best_idx]) for name in mod_scores})]
            if final_score[best_idx] < 0.1:
                best_pos[0] = (best_pos[0][0], best_pos[0][1], 0.1, best_pos[0][3])  # 最低信心分數 0.1

    # 評估預測
    true_values = grid.copy()
    remaining_nums = list(set(range(1, N + 1)) - set(opened_nums))
    np.random.shuffle(remaining_nums)
    for (i, j), num in zip(empty_yx, remaining_nums):
        true_values[i, j] = num
    metrics = solver.evaluate_prediction(grid, predictions, true_values)

    return final_score, predictions, top3, metrics