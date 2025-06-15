# analyzer.py
import os
from functools import lru_cache
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional, Union

import numpy as np

from modules import FORMULA_REGISTRY, compute_global_features
from brain import (
    EXT_GM20_Skip_Pattern_Confidence_Vec,
    MathUtils,
    BoardAnalyzerUtils,
)

# ---------------------------------------------------------------------
# Monte-Carlo + Heuristic 模擬
# ---------------------------------------------------------------------
@lru_cache(maxsize=128)
def simulate_with_formulas(
    grid_bytes: bytes,
    rows: int,
    cols: int,
    n_iter: int = 5_000_000,
    weights: Dict[str, float] | None = None,
) -> Dict[Tuple[int, int], Dict[int, float]]:
    """大量隨機生成完整盤面 → 篩掉不符規則 → 回傳每個空格的命中次數分布。"""
    grid = np.frombuffer(grid_bytes, dtype=np.int64).reshape(rows, cols)
    math_utils = MathUtils()
    analyzer = BoardAnalyzerUtils()
    rng = np.random.default_rng()

    blanks = np.argwhere(grid == -1)
    known_idx = np.argwhere(grid != -1)
    known_vals = grid[grid != -1]
    hit_counter = {tuple(b): Counter() for b in map(tuple, blanks)}

    # 公式權重
    w = weights or {"excel": 0.6, "shuffle": 0.4}
    names = list(w)

    lin_known = rows * known_idx[:, 0] + known_idx[:, 1]
    batch_size = 10_000 if rows * cols < 50 else 5_000 if rows * cols < 200 else 1_000

    # 讀取環境變數
    n_iter = (
        int(os.getenv("ITER", 5_000_000))
        if os.getenv("USE_FORMULA_ONLY") != "1"
        else 500_000
    )

    # baseline 特徵
    skip_scores = EXT_GM20_Skip_Pattern_Confidence_Vec(grid)
    mean_val, std_val = compute_global_features(grid.astype(np.float32))[:2]
    std_val = std_val or 1.0  # avoid div0

    # ---------------------------- main loop --------------------------
    for _ in range(n_iter // batch_size):
        # ① 批次生成候選盤
        boards = np.zeros((batch_size, rows * cols), dtype=np.int64)
        choices = rng.choice(names, size=batch_size, p=[w[n] for n in names])
        for i, fname in enumerate(choices):
            boards[i] = FORMULA_REGISTRY[fname](rows, cols, rng).ravel()

        # ② 已知數字比對
        valid = np.all(boards[:, lin_known] == known_vals, axis=1)

        # ③ 檢查算術 / 幾何序列（對齊 batch_size）
        seq_ok = np.array(
            [
                analyzer.check_sequences(
                    b.reshape(rows, cols), grid, min_len=3, allow_gaps=1
                )
                for b in boards
            ]
        )
        valid &= seq_ok

        # ④ Skip-pattern 相似度
        valid_idx = np.where(valid)[0]
        if valid_idx.size == 0:
            continue
        v_boards = boards[valid_idx].reshape(-1, rows, cols)
        v_scores = np.array(
            [EXT_GM20_Skip_Pattern_Confidence_Vec(b) for b in v_boards]
        )
        corr_ok = np.array(
            [np.corrcoef(skip_scores.ravel(), s.ravel())[0, 1] > 0.85 for s in v_scores]
        )
        valid[valid_idx] &= corr_ok

        # ⑤ 留下終版盤
        final_idx = np.where(valid)[0]
        if final_idx.size == 0:
            continue
        final_boards = boards[final_idx].reshape(-1, rows, cols)
        final_scores = v_scores[corr_ok]

        # ⑥ 累計各空格命中次數（resonance + global + skip 加權）
        for b_i, board in enumerate(final_boards):
            for r, c in blanks:
                window = board[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2]
                kn = window[window != -1]
                resonance = 1 / (1 + abs(board[r, c] - kn.mean()) * 0.5) if kn.size else 1
                g_w = np.exp(-((board[r, c] - mean_val) ** 2) / (2 * (std_val**2 + 1e-6)))
                hit_counter[(r, c)][board[r, c]] += (
                    final_scores[b_i, r, c] * resonance * g_w * 1.1
                )

            # 早停：全部格子 max prob > 0.95
            if all(max(c.values()) / sum(c.values()) > 0.95 for c in hit_counter.values()):
                break

    # -------------------------- normalize ----------------------------
    prob_map: Dict[Tuple[int, int], Dict[int, float]] = {}
    for pos, cnt in hit_counter.items():
        if not cnt:
            continue
        v_min, v_max = min(cnt.values()), max(cnt.values())
        prob_map[pos] = {
            k: math_utils.normalize_value(v, v_min or 1e-10, v_max or 1e-10)
            for k, v in cnt.items()
        }
    return prob_map


# ---------------------------------------------------------------------
# 匯總 / 多層加權
# ---------------------------------------------------------------------
def weight_prob_by_modules(
    grid: np.ndarray,
    prob_map: Dict[Tuple[int, int], Dict[int, float]],
) -> Dict[Tuple[int, int], Dict[int, float]]:
    math_utils = MathUtils()
    analyzer = BoardAnalyzerUtils()

    rows, cols = grid.shape
    blanks = np.argwhere(grid == -1)

    # 0️⃣ 缺席格子 → 均勻先驗
    legal_all = analyzer.get_legal_values_for_placement(grid) or {0}
    uniform = {n: 1.0 / len(legal_all) for n in legal_all}
    for r, c in blanks:
        if (r, c) not in prob_map or not prob_map[(r, c)]:
            prob_map[(r, c)] = dict(uniform)

    # 1️⃣ Local resonance
    for r, c in blanks:
        window = grid[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2]
        kn = window[window != -1]
        if kn.size:
            mean_kn = kn.mean()
            for n in prob_map[(r, c)]:
                prob_map[(r, c)][n] *= 1.2 / (1 + abs(n - mean_kn) * 0.5)

    # 2️⃣ Global value distribution
    mean_val, std_val = compute_global_features(grid.astype(np.float32))[:2]
    std_val = std_val or 1.0
    for r, c in blanks:
        for n in prob_map[(r, c)]:
            g_w = np.exp(-((n - mean_val) ** 2) / (2 * (std_val**2 + 1e-6)))
            prob_map[(r, c)][n] *= g_w * 1.15

    # 3️⃣ Skip-pattern confidence（加 0.05 底氣）
    skip_scores = EXT_GM20_Skip_Pattern_Confidence_Vec(grid)
    for r, c in blanks:
        skip_factor = max(skip_scores[r, c], 0.05) * 1.1
        for n in prob_map[(r, c)]:
            prob_map[(r, c)][n] *= skip_factor

    # 4️⃣ 既有 row/col 序列 boost
    for r, c in blanks:
        seq = analyzer.get_arithmetic_or_geometric_sequences
        has_seq = set().union(*seq(grid[r], 3, 1), *seq(grid[:, c], 3, 1))
        for n in prob_map[(r, c)]:
            if n in has_seq:
                prob_map[(r, c)][n] *= 1.7

    # 5️⃣ Normalize；若全 0 → 均勻
    for pos, dist in prob_map.items():
        tot = sum(dist.values())
        if tot == 0:
            n = len(dist) or 1
            prob_map[pos] = {k: 1.0 / n for k in dist}
        else:
            prob_map[pos] = {k: v / tot for k, v in dist.items()}

    return prob_map


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def predict_scratch_card(
    grid: List[List[int]],
    n_iter: int = 5_000_000,
    target_num: Optional[int] = None,
) -> Dict[str, Any]:
    """
    若 `target_num` 給定 → 只回傳該數字各空格的信心排序。
    否則回傳每格 top-3 候選。
    """
    grid_np = np.array(grid, dtype=np.int64)

    prob_map = simulate_with_formulas(
        grid_np.tobytes(), grid_np.shape[0], grid_np.shape[1], n_iter
    )
    prob_map = weight_prob_by_modules(grid_np, prob_map)

    rows, cols = grid_np.shape
    blanks = np.argwhere(grid_np == -1)

    results: List[Dict[str, Any]] = []
    if target_num is not None:
        # 只針對指定數字
        for r, c in blanks:
            conf = prob_map[(r, c)].get(target_num, 0.0)
            results.append(
                {"row": int(r), "col": int(c), "candidate": target_num, "confidence": conf}
            )
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return {"target": target_num, "rankings": results, "full_probabilities": prob_map}

    # 回傳各格 top-3
    for (r, c), dist in prob_map.items():
        best = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
        nums, conf = zip(*best)
        results.append(
            {
                "row": int(r),
                "col": int(c),
                "candidates": list(nums),
                "confidences": list(map(float, conf)),
            }
        )

    results.sort(key=lambda x: x["confidences"][0], reverse=True)
    full_probs = {f"{r},{c}": dist for (r, c), dist in prob_map.items()}
    return {"predictions": results, "full_probabilities": full_probs}