import numpy as np
from modules import ScratchSolver
import json
import os
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_board(grid: np.ndarray, weights: dict, return_predictions: bool = False, target_num: int = None, json_heatmap_path: str = None):
    solver = ScratchSolver()

    # 驗證網格大小
    if grid.shape[0] > 20 or grid.shape[1] > 20:
        raise ValueError("網格超過 20x20 限制")
    N = grid.size
    opened_nums = set(grid[grid != -1])
    if len(opened_nums) != len(set(opened_nums)) or max(opened_nums, default=0) > N:
        raise ValueError("數字不滿足 1~N 不重複規則")

    # 讀取JSON熱力圖（若提供）
    initial_scores = None
    if json_heatmap_path and os.path.exists(json_heatmap_path):
        with open(json_heatmap_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            initial_scores = np.array(data.get('heatmap', np.zeros_like(grid, dtype=float)))
            if initial_scores.shape != grid.shape:
                logger.warning(f"JSON熱力圖形狀 {initial_scores.shape} 與網格 {grid.shape} 不匹配，使用零陣列")
                initial_scores = None

    # 計算各子模組分數與預測值
    score_focus, pred_focus = solver.compute_focus_score(grid)
    score_skip, pred_skip = solver.detect_skip_patterns(grid)
    score_diff, pred_diff = solver.compute_difference_trend(grid)
    score_mirror, pred_mirror = solver.detect_mirror_sequences(grid)
    score_conn, pred_conn = solver.connectivity_heatmap(grid)
    score_tail, pred_tail = solver.sequence_tail_analyzer(grid)

    # 加入約束求解和張量打分
    constraint_score = solver.constraint_solver(grid, target_num) if target_num else np.zeros_like(grid, dtype=float)
    tensor_score = solver.tensor_full_score(grid)

    # 動態權重計算
    dynamic_weights = solver.dynamic_weights(grid, {
        'focus': score_focus, 'skip': score_skip, 'diff': score_diff,
        'mirror': score_mirror, 'conn': score_conn, 'tail': score_tail,
        'constraint': constraint_score, 'tensor': tensor_score
    }, weights, initial_scores)

    # 收集到字典
    gridscores = {
        'focus': score_focus,
        'skip': score_skip,
        'diff': score_diff,
        'mirror': score_mirror,
        'conn': score_conn,
        'tail': score_tail,
        'constraint': constraint_score,
        'tensor': tensor_score,
        '_weights': dynamic_weights
    }
    gridpreds = {
        'focus': pred_focus,
        'skip': pred_skip,
        'diff': pred_diff,
        'mirror': pred_mirror,
        'conn': pred_conn,
        'tail': pred_tail
    }

    # 融合分數
    final_score, final_pred = solver.fuse_scores(gridscores, grid, gridpreds, target_num)

    if return_predictions:
        # 推測指定數字位置
        best_pos = solver.predict_specific_number(grid, final_score, target_num, dynamic_weights)
        return final_score, final_pred, best_pos
    else:
        return final_score, None, None