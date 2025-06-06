import numpy as np
from modules import ScratchSolver

def analyze_board(grid: np.ndarray, weights: dict, return_predictions: bool = False):
    solver = ScratchSolver()

    # 計算各子模組分數與預測值
    score_focus, pred_focus = solver.compute_focus_score(grid)
    score_skip, pred_skip = solver.detect_skip_patterns(grid)
    score_diff, pred_diff = solver.compute_difference_trend(grid)
    score_mirror, pred_mirror = solver.detect_mirror_sequences(grid)
    score_conn, pred_conn = solver.connectivity_heatmap(grid)
    score_tail, pred_tail = solver.sequence_tail_analyzer(grid)

    # 收集到字典
    gridscores = {
        'focus': score_focus,
        'skip': score_skip,
        'diff': score_diff,
        'mirror': score_mirror,
        'conn': score_conn,
        'tail': score_tail,
        '_weights': weights
    }
    gridpreds = {
        'focus': pred_focus,
        'skip': pred_skip,
        'diff': pred_diff,
        'mirror': pred_mirror,
        'conn': pred_conn,
        'tail': pred_tail
    }

    # 融合分數與預測值
    final_score, final_pred = solver.fuse_scores(gridscores, grid, gridpreds)

    if return_predictions:
        return final_score, final_pred
    else:
        return final_score, None