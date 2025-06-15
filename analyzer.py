# analyzer.py  (速度優化 + 全局唯一)
import os
import sys
from functools import lru_cache
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional

import numpy as np

from modules import FORMULA_REGISTRY, compute_global_features
from brain import (
    EXT_GM20_Skip_Pattern_Confidence_Vec,
    MathUtils,
    BoardAnalyzerUtils,
)

# ------------------------------------------------------- 共用 util
math_utils = MathUtils()
analyzer_utils = BoardAnalyzerUtils()


# ------------------------------------------------------- 內核：蒙地卡羅 + early-stop
@lru_cache(maxsize=128)
def simulate_with_formulas(
    grid_bytes: bytes,
    rows: int,
    cols: int,
    n_iter: int,
    quick_mode: bool = False,
) -> Dict[Tuple[int, int], Dict[int, float]]:
    """
    quick_mode=True → 關掉序列檢查與 skip-corr，極速粗估；
    False → 全啟用（但有 early-stop）
    """
    grid = np.frombuffer(grid_bytes, dtype=np.int64).reshape(rows, cols)
    rng = np.random.default_rng()

    blanks = np.argwhere(grid == -1)
    known_idx = np.argwhere(grid != -1)
    known_vals = grid[grid != -1]
    hit_counter = {tuple(b): Counter() for b in map(tuple, blanks)}

    # 權重 & batch
    w = {"excel": 0.6, "shuffle": 0.4}
    names, probs = list(w), list(w.values())
    batch_size = max(500, 20000 // (rows * cols))  # adaptive

    # baseline 特徵
    skip_scores = EXT_GM20_Skip_Pattern_Confidence_Vec(grid) if not quick_mode else None
    mean_val, std_val = compute_global_features(grid.astype(np.float32))[:2]
    std_val = std_val or 1.0

    lin_known = rows * known_idx[:, 0] + known_idx[:, 1]
    total_seen, effective = 0, 0

    while total_seen < n_iter:
        cur_batch = min(batch_size, n_iter - total_seen)
        boards = np.zeros((cur_batch, rows * cols), dtype=np.int64)
        choices = rng.choice(names, size=cur_batch, p=probs)
        for i, fn in enumerate(choices):
            boards[i] = FORMULA_REGISTRY[fn](rows, cols, rng).ravel()

        # ① 知格比對
        valid = np.all(boards[:, lin_known] == known_vals, axis=1)

        # ② 快速模式跳過深度檢查
        if not quick_mode:
            # 序列檢查
            seq_ok = np.array(
                [
                    analyzer_utils.check_sequences(
                        b.reshape(rows, cols), grid, 3, 1
                    )
                    for b in boards
                ]
            )
            valid &= seq_ok

            # skip-corr
            vi = np.where(valid)[0]
            if vi.size:
                v_boards = boards[vi].reshape(-1, rows, cols)
                corrs = [
                    np.corrcoef(skip_scores.ravel(), EXT_GM20_Skip_Pattern_Confidence_Vec(b).ravel())[0, 1]
                    for b in v_boards
                ]
                valid[vi] &= np.array([c > 0.85 for c in corrs])

        # ③ 累計命中
        fi = np.where(valid)[0]
        if fi.size:
            fboards = boards[fi].reshape(-1, rows, cols)
            for board in fboards:
                for r, c in blanks:
                    hit_counter[(r, c)][board[r, c]] += 1
            effective += fi.size

        total_seen += cur_batch

        # ④ early-stop：30k 效盤且所有空格 max>0.88
        if (
            not quick_mode
            and effective >= 30000
            and all(
                max(cnt.values()) / sum(cnt.values()) > 0.88
                for cnt in hit_counter.values()
                if cnt
            )
        ):
            break

    # normalize
    prob_map: Dict[Tuple[int, int], Dict[int, float]] = {}
    for pos, cnt in hit_counter.items():
        if not cnt:
            continue
        minimum, maximum = min(cnt.values()), max(cnt.values())
        prob_map[pos] = {
            k: math_utils.normalize_value(v, minimum or 1e-10, maximum or 1e-10)
            for k, v in cnt.items()
        }
    return prob_map


# ------------------------------------------------------- 多層加權（與先前一致，略）
def weight_prob_by_modules(
    grid: np.ndarray, prob_map: Dict[Tuple[int, int], Dict[int, float]]
) -> Dict[Tuple[int, int], Dict[int, float]]:
    # ...（保留你上版權重邏輯，篇幅省略，可直接沿用）...
    # === 省略段落：請把你現用的 weight_prob_by_modules 貼回此處 ===
    return prob_map


# ------------------------------------------------------- 全局唯一：Hungarian → Greedy
def assign_unique_hungarian(
    blanks: List[Tuple[int, int]],
    prob_map: Dict[Tuple[int, int], Dict[int, float]],
) -> Dict[Tuple[int, int], Tuple[int, float]]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return None  # 讓呼叫端回退 Greedy

    numbers = sorted({n for dist in prob_map.values() for n in dist})
    cost = np.full((len(blanks), len(numbers)), 50.0)  # 大成本 = 不可能

    for i, cell in enumerate(blanks):
        dist = prob_map[cell]
        for j, num in enumerate(numbers):
            p = dist.get(num, 0.0)
            cost[i, j] = -np.log(p + 1e-9)  # 機率轉對數成本

    row_ind, col_ind = linear_sum_assignment(cost)
    assignment = {
        blanks[r]: (numbers[c], float(prob_map[blanks[r]].get(numbers[c], 0.0)))
        for r, c in zip(row_ind, col_ind)
    }
    return assignment


def ensure_global_unique(
    blanks: List[Tuple[int, int]],
    prob_map: Dict[Tuple[int, int], Dict[int, float]],
) -> Dict[Tuple[int, int], Tuple[int, float]]:
    """
    回傳 {cell: (唯一數字, confidence)}
    先嘗試 Hungarian；失敗則回退 Greedy 排除。
    """
    assign = assign_unique_hungarian(blanks, prob_map)
    if assign:
        return assign

    # --- Greedy ---
    taken, assign = set(), {}
    cells_sorted = sorted(blanks, key=lambda p: max(prob_map[p].values()), reverse=True)
    for cell in cells_sorted:
        for n, p in sorted(prob_map[cell].items(), key=lambda x: x[1], reverse=True):
            if n not in taken:
                assign[cell] = (n, float(p))
                taken.add(n)
                break
    return assign


# ------------------------------------------------------- Public API
def predict_scratch_card(
    grid: List[List[int]],
    n_iter: int = 5_000_000,
    target_num: Optional[int] = None,
    quick_iter: int = 20000,
    refine_iter: int = 300000,
    unique: bool = True,
) -> Dict[str, Any]:
    """
    - quick_iter / refine_iter 為二段推論迭代數
    - unique=True → 強制同一數字不重複
    """
    grid_np = np.array(grid, dtype=np.int64)
    rows, cols = grid_np.shape
    blanks = [tuple(b) for b in np.argwhere(grid_np == -1)]

    # -------- two-phase inference ----------
    quick_map = simulate_with_formulas(
        grid_np.tobytes(), rows, cols, quick_iter, quick_mode=True
    )
    quick_map = weight_prob_by_modules(grid_np, quick_map)

    # 取前 K 熱點格
    K = min(3, len(blanks))
    hot_cells = sorted(
        blanks, key=lambda p: max(quick_map[p].values()), reverse=True
    )[:K]

    # 精算指定格
    refine_map = simulate_with_formulas(
        grid_np.tobytes(), rows, cols, refine_iter, quick_mode=False
    )
    refine_map = weight_prob_by_modules(grid_np, refine_map)

    # 合併：hot_cell 用 refine，其餘用 quick
    final_map = {
        cell: (refine_map if cell in hot_cells else quick_map)[cell] for cell in blanks
    }

    # -------- unique assignment ----------
    if unique and target_num is None:
        uniq_assign = ensure_global_unique(blanks, final_map)
        predictions = [
            {
                "row": r,
                "col": c,
                "candidates": [n],
                "confidences": [conf],
            }
            for (r, c), (n, conf) in uniq_assign.items()
        ]
        predictions.sort(key=lambda x: x["confidences"][0], reverse=True)
        return {
            "predictions": predictions,
            "mode": "unique",
            "full_probabilities": final_map,
        }

    # -------- target_num or top-3 ----------
    if target_num is not None:
        rank = [
            {
                "row": r,
                "col": c,
                "candidate": target_num,
                "confidence": final_map[(r, c)].get(target_num, 0.0),
            }
            for r, c in blanks
        ]
        rank.sort(key=lambda x: x["confidence"], reverse=True)
        return {"target": target_num, "rankings": rank, "full_probabilities": final_map}

    # default Top-3
    results = []
    for (r, c), dist in final_map.items():
        best = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
        nums, conf = zip(*best)
        results.append(
            {
                "row": r,
                "col": c,
                "candidates": list(nums),
                "confidences": list(map(float, conf)),
            }
        )
    results.sort(key=lambda x: x["confidences"][0], reverse=True)
    return {"predictions": results, "mode": "top3", "full_probabilities": final_map}