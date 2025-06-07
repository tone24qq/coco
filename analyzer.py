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
        logger.error("網格超過 20x20 限制")
        raise ValueError("網格超過 20x20 限制")

    # 驗證數字合法性
    valid = (grid[grid != -1] >= 1).all()
    if not valid:
        logger.error("存在不合法格位數字")
        raise ValueError("存在不合法格位數字")

    # 檢查目標數字是否已開
    if target_num is not None:
        opened_nums = grid[grid != -1]
        if target_num in opened_nums:
            count = np.sum(opened_nums == target_num)
            logger.error(f"目標數字 {target_num} 已出現在盤面 {count} 次")
            return None, None, {"error": f"目標數字 {target_num} 已出現在盤面 {count} 次"}

    # 讀取JSON熱力圖
    initial_scores = None
    if json_heatmap_path and os.path.exists(json_heatmap_path):
        try:
            with open(json_heatmap_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            initial_scores = np.array(data.get('heatmap', np.zeros_like(grid, dtype=float)))
            if initial_scores.shape != grid.shape:
                logger.warning(f"JSON熱力圖形狀 {initial_scores.shape} 與網格 {grid.shape} 不匹配")
                initial_scores = None
        except (FileNotFoundError, PermissionError, json.JSONDecodeError) as e:
            logger.error(f"無法讀取JSON熱力圖: {e}")
            initial_scores = None

    # 計算模組分數與預測
    score_focus, pred_focus = solver.compute_focus_score(grid)
    score_skip, pred_skip = solver.detect_skip_patterns(grid)
    score_diff, pred_diff = solver.compute_difference_trend(grid)
    score_mirror, pred_mirror = solver.detect_mirror_sequences(grid)
    score_conn, pred_conn = solver.connectivity_heatmap(grid)
    score_tail, pred_tail = solver.sequence_tail_analyzer(grid)
    score_pattern = solver.pattern_mining(grid)
    constraint_score = solver.constraint_solver(grid, target_num) if target_num else np.zeros_like(grid, dtype=float)
    tensor_score = solver.tensor_full_score(grid)

    # 檢查模組分數
    scores = {
        'focus': score_focus,
        'skip': score_skip,
        'diff': score_diff,
        'mirror': score_mirror,
        'conn': score_conn,
        'tail': score_tail,
        'constraint': constraint_score,
        'tensor': tensor_score,
        'pattern': score_pattern,
        '_weights': weights
    }
    if all(np.all(s[grid == -1] == 0) for s in scores.values() if s is not None and s is not weights):
        logger.warning("所有模組分數為零，返回均勻分數")
        solver.log_module_failure(grid, target_num)
        scores = {k: np.ones_like(grid, dtype=float) / np.sum(grid == -1) if k != '_weights' else weights for k in scores}

    # 動態權重
    dynamic_weights = solver.dynamic_weights(grid, scores, weights, initial_scores)

    # 更新分數字典
    gridscores = scores.copy()
    gridscores['_weights'] = dynamic_weights
    if initial_scores is not None:
        gridscores['json'] = initial_scores
        dynamic_weights['json'] = dynamic_weights.get('json', 0.1)

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
        best_pos = solver.predict_specific_number(grid, final_score, target_num, dynamic_weights)
        if best_pos is None or len(best_pos) == 0:
            logger.warning(f"無法為目標數字 {target_num} 找到候選格，返回Top3均勻候選")
            solver.log_module_failure(grid, target_num)
            best_pos = solver.default_candidate(grid, target_num, dynamic_weights)
        return final_score, final_pred, best_pos
    else:
        return final_score, None, None