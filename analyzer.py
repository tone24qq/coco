# analyzer.py
import numpy as np
import logging
from modules import ScratchSolver
import asyncio
import json
import os
from numpy.lib.stride_tricks import sliding_window_view

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_board(grid: np.ndarray, weights: dict, return_predictions: bool = False, target_num: int = None, json_heatmap_path: str = None):
    solver = ScratchSolver()
    solver.update_tree(grid)

    # 驗證網格大小
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
        return None, None, {"error": f"目標數字 {target_num} 已出現在盤面"}

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
    try:
        mod_scores['compute_dynamic_hot_cold_vectorized'] = solver.compute_dynamic_hot_cold_vectorized(grid)
    except Exception as e:
        logger.error(f"compute_dynamic_hot_cold_vectorized 失敗: {e}")
        mod_scores['compute_dynamic_hot_cold_vectorized'] = np.zeros(np.count_nonzero(grid == -1)) / np.count_nonzero(grid == -1)

    try:
        mod_scores['compute_block_heatmap_vectorized'] = solver.compute_block_heatmap_vectorized(grid)
    except Exception as e:
        logger.error(f"compute_block_heatmap_vectorized 失敗: {e}")
        mod_scores['compute_block_heatmap_vectorized'] = np.zeros(np.count_nonzero(grid == -1)) / np.count_nonzero(grid == -1)

    try:
        mod_scores['idw_vectorized'] = solver.idw_vectorized(grid)
    except Exception as e:
        logger.error(f"idw_vectorized 失敗: {e}")
        mod_scores['idw_vectorized'] = np.zeros(np.count_nonzero(grid == -1)) / np.count_nonzero(grid == -1)

    try:
        mod_scores['compute_global_diff_heatmap'] = solver.compute_global_diff_heatmap(grid)
    except Exception as e:
        logger.error(f"compute_global_diff_heatmap 失敗: {e}")
        mod_scores['compute_global_diff_heatmap'] = np.zeros(np.count_nonzero(grid == -1)) / np.count_nonzero(grid == -1)

    try:
        mod_scores['compute_focus_score'] = solver.compute_focus_score(grid)[0]  # 只取分數
    except Exception as e:
        logger.error(f"compute_focus_score 失敗: {e}")
        mod_scores['compute_focus_score'] = np.zeros(np.count_nonzero(grid == -1)) / np.count_nonzero(grid == -1)

    try:
        mod_scores['detect_skip_patterns'] = solver.detect_skip_patterns(grid)[0]
    except Exception as e:
        logger.error(f"detect_skip_patterns 失敗: {e}")
        mod_scores['detect_skip_patterns'] = np.zeros(np.count_nonzero(grid == -1)) / np.count_nonzero(grid == -1)

    try:
        mod_scores['compute_difference_trend'] = solver.compute_difference_trend(grid)[0]
    except Exception as e:
        logger.error(f"compute_difference_trend 失敗: {e}")
        mod_scores['compute_difference_trend'] = np.zeros(np.count_nonzero(grid == -1)) / np.count_nonzero(grid == -1)

    try:
        mod_scores['detect_mirror_sequences'] = solver.detect_mirror_sequences(grid)[0]
    except Exception as e:
        logger.error(f"detect_mirror_sequences 失敗: {e}")
        mod_scores['detect_mirror_sequences'] = np.zeros(np.count_nonzero(grid == -1)) / np.count_nonzero(grid == -1)

    try:
        mod_scores['connectivity_heatmap'] = solver.connectivity_heatmap(grid)[0]
    except Exception as e:
        logger.error(f"connectivity_heatmap 失敗: {e}")
        mod_scores['connectivity_heatmap'] = np.zeros(np.count_nonzero(grid == -1)) / np.count_nonzero(grid == -1)

    try:
        mod_scores['sequence_tail_analyzer'] = solver.sequence_tail_analyzer(grid)[0]
    except Exception as e:
        logger.error(f"sequence_tail_analyzer 失敗: {e}")
        mod_scores['sequence_tail_analyzer'] = np.zeros(np.count_nonzero(grid == -1)) / np.count_nonzero(grid == -1)

    # 盤面分類
    board_type = solver.classify_board_type(mod_scores['compute_dynamic_hot_cold_vectorized'])

    # 融合分數
    final_score = solver.fuse_scores_vectorized(mod_scores, board_type, weights)

    # 準備空缺位置
    empty_yx = np.argwhere(grid == -1)

    if return_predictions:
        # Top-3 推論
        top3 = solver.predict_top3_vectorized(final_score, empty_yx)
        return final_score, None, top3
    else:
        return final_score, None, None