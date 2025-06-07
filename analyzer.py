import numpy as np
import json
import os
import logging
from modules import ScratchSolver  # 請確認 modules.py 裡有這個類別

# 設定日誌
logging.basicConfig(
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def analyze_board(
    grid: np.ndarray,
    weights: dict,
    return_predictions: bool = False,
    target_num: int = None,
    json_heatmap_path: str = None
):
    """
    grid: 二維 np.ndarray，-1 表示未揭露，0/正數表示已揭露
    weights: 外部傳入的模組權重 dict
    return_predictions: 是否要額外回傳 best_pos
    target_num: 要預測的指定數字（如不傳回 None）
    json_heatmap_path: 若有提供，用來載入初始熱力圖
    回傳 (final_score, final_pred, best_pos 或 None)
    """
    solver = ScratchSolver()

    # ===== 1. 驗證網格合法性 =====
    # 大小限制
    if grid.shape[0] > 20 or grid.shape[1] > 20:
        raise ValueError("網格超過 20x20 限制")

    # 已揭露數字範圍檢查（忽略 -1、0）
    N = grid.size
    opened = grid[grid > 0]
    if opened.size > 0:
        mn, mx = int(opened.min()), int(opened.max())
        if mn < 1 or mx > N:
            raise ValueError(f"數字不在 1~{N} 範圍內（min={mn}, max={mx}）")

    # ===== 2. 載入 JSON 熱力圖（可選） =====
    initial_scores = None
    if json_heatmap_path:
        if os.path.exists(json_heatmap_path):
            try:
                with open(json_heatmap_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                heat = data.get('heatmap')
                initial_scores = np.array(heat) if heat is not None else None
                if initial_scores is not None and initial_scores.shape != grid.shape:
                    logger.warning(
                        f"JSON 熱力圖形狀 {initial_scores.shape} 與網格 {grid.shape} 不匹配，忽略初始熱力圖"
                    )
                    initial_scores = None
            except Exception as e:
                logger.warning(f"讀取 JSON 熱力圖失敗：{e}")
        else:
            logger.warning(f"找不到 JSON 熱力圖檔案：{json_heatmap_path}")

    # ===== 3. 各模組計分與預測 =====
    score_focus, pred_focus   = solver.compute_focus_score(grid)
    score_skip, pred_skip     = solver.detect_skip_patterns(grid)
    score_diff, pred_diff     = solver.compute_difference_trend(grid)
    score_mirror, pred_mirror = solver.detect_mirror_sequences(grid)
    score_conn, pred_conn     = solver.connectivity_heatmap(grid)
    score_tail, pred_tail     = solver.sequence_tail_analyzer(grid)

    # 約束解算分數 + 張量全盤分數
    constraint_score = (
        solver.constraint_solver(grid, target_num)
        if target_num is not None else
        np.zeros_like(grid, dtype=float)
    )
    tensor_score = solver.tensor_full_score(grid)

    # ===== 4. 動態權重融合 =====
    dynamic_weights = solver.dynamic_weights(
        grid,
        {
            'focus':      score_focus,
            'skip':       score_skip,
            'diff':       score_diff,
            'mirror':     score_mirror,
            'conn':       score_conn,
            'tail':       score_tail,
            'constraint': constraint_score,
            'tensor':     tensor_score
        },
        weights,
        initial_scores
    )

    # 收集分數與預測
    gridscores = {
        'focus':      score_focus,
        'skip':       score_skip,
        'diff':       score_diff,
        'mirror':     score_mirror,
        'conn':       constraint_score,
        'tail':       score_tail,
        'constraint': constraint_score,
        'tensor':     tensor_score,
        '_weights':   dynamic_weights
    }
    gridpreds = {
        'focus':  pred_focus,
        'skip':   pred_skip,
        'diff':   pred_diff,
        'mirror': pred_mirror,
        'conn':   pred_conn,
        'tail':   pred_tail
    }

    # ===== 5. 融合最終分數 & 回傳 =====
    final_score, final_pred = solver.fuse_scores(
        gridscores, grid, gridpreds, target_num
    )

    if return_predictions:
        best_pos = solver.predict_specific_number(
            grid, final_score, target_num, dynamic_weights
        )
        return final_score, final_pred, best_pos
    else:
        return final_score, None, None