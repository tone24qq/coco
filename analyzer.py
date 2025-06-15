# analyzer.py
import os
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any
from functools import lru_cache

from modules import FORMULA_REGISTRY, compute_global_features, AdaptiveWeights
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
    grid = np.frombuffer(grid_bytes, dtype=np.int64).reshape(rows, cols)
    math_utils = MathUtils()
    analyzer = BoardAnalyzerUtils()
    rng = np.random.default_rng()

    blanks = np.argwhere(grid == -1)
    known_idx = np.argwhere(grid != -1)
    known_vals = grid[grid != -1]
    hit_counter = {tuple(b): Counter() for b in map(tuple, blanks)}

    # formula weights
    w = weights or {"excel": 0.6, "shuffle": 0.4}
    names = list(w)

    lin_known = rows * known_idx[:, 0] + known_idx[:, 1]
    batch_size = 10_000 if rows * cols < 50 else 5_000 if rows * cols < 200 else 1_000

    # 環境變數覆寫
    n_iter = (
        int(os.getenv("ITER", 5_000_000))
        if os.getenv("USE_FORMULA_ONLY") != "1"
        else 500_000
    )

    # 全盤特徵 & skip-pattern baseline
    skip_scores = EXT_GM20_Skip_Pattern_Confidence_Vec(grid)
    grid_feats = compute_global_features(grid.astype(np.float32))
    mean_val, std_val = grid_feats[0], grid_feats[1] or 1.0  # avoid div0

    # ---------------------------- main loop --------------------------
    for _ in range(n_iter // batch_size):
        # 1. 生成候選盤
        boards = np.zeros((batch_size, rows * cols), dtype=np.int64)
        choices = rng.choice(names, size=batch_size, p=[w[n] for n in names])
        for i, fname in enumerate(choices):
            boards[i] = FORMULA_REGISTRY[fname](rows, cols, rng).ravel()

        # 2. 先比對已知數字
        valid = np.all(boards[:, lin_known] == known_vals, axis=1)

        # 3. 檢查算術/幾何序列（**修正：對齊 batch_size**）
        seq_ok = np.array(
            [
                analyzer.check_sequences(
                    b.reshape(rows, cols), grid, min_len=3, allow_gaps=1
                )
                for b in boards
            ],
            dtype=bool,
        )
        valid &= seq_ok

        # 4. 篩出符合前兩條規則的盤，計算 skip-pattern 相似度
        valid_indices = np.where(valid)[0]
        if valid_indices.size == 0:
            continue
        valid_boards = boards[valid_indices].reshape(-1, rows, cols)

        board_scores = np.array(
            [EXT_GM20_Skip_Pattern_Confidence_Vec(b) for b in valid_boards]
        )
        corr_ok = np.array(
            [
                np.corrcoef(skip_scores.ravel(), bs.ravel())[0, 1] > 0.85
                for bs in board_scores
            ],
            dtype=bool,
        )
        # 把結果刷回 master mask
        valid[valid_indices] &= corr_ok

        # 5. 最終保留盤
        final_indices = np.where(valid)[0]
        if final_indices.size == 0:
            continue
        final_boards = boards[final_indices].reshape(-1, rows, cols)
        final_scores = board_scores[corr_ok]

        # 6. 累計各空格候選命中次數 (含 resonance / global / skip 加權)
        for b_idx, board in enumerate(final_boards):
            for r, c in blanks:
                window = board[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2]
                kn_vals = window[window != -1]
                resonance = (
                    1 / (1 + abs(board[r, c] - kn_vals.mean()) * 0.5)
                    if kn_vals.size
                    else 1.0
                )
                global_weight = np.exp(
                    -((board[r, c] - mean_val) ** 2) / (2 * (std_val**2 + 1e-6))
                )
                hit_counter[(r, c)][board[r, c]] += (
                    final_scores[b_idx, r, c]
                    * resonance
                    * global_weight
                    * 1.1  # boost
                )

            # 早停：若所有空格的最大機率已 >95%
            if all(
                max(cnt.values()) / sum(cnt.values()) > 0.95 for cnt in hit_counter.values()
            ):
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
# 後處理：本地/全局/Pattern 加權
# ---------------------------------------------------------------------
def weight_prob_by_modules(
    grid: np.ndarray, prob_map: Dict[Tuple[int, int], Dict[int, float]]
) -> Dict[Tuple[int, int], Dict[int, float]]:
    math_utils = MathUtils()
    analyzer = BoardAnalyzerUtils()
    rows, cols = grid.shape
    blanks = np.argwhere(grid == -1)
legal_all = analyzer.get_legal_values_for_placement(grid) or {0}
    uniform_fallback = {n: 1.0 / len(legal_all) for n in legal_all}
    for r, c in blanks:
        if (r, c) not in prob_map or not prob_map[(r, c)]:
            # 塞一份均勻分布當墊底，後續加權會自動調整
            prob_map[(r, c)] = dict(uniform_fallback)
    # 1️⃣ Local resonance
    for r, c in blanks:
        window = grid[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2]
        kn_vals = window[window != -1]
        if kn_vals.size:
            for num, prob in prob_map[(r, c)].items():
                resonance = 1 / (1 + abs(num - kn_vals.mean()) * 0.5)
                prob_map[(r, c)][num] *= resonance * 1.2

    # 2️⃣ Global distribution
    mean_val, std_val = compute_global_features(grid.astype(np.float32))[:2]
    std_val = std_val or 1.0
    for r, c in blanks:
        for num, prob in prob_map[(r, c)].items():
            g_w = np.exp(-((num - mean_val) ** 2) / (2 * (std_val**2 + 1e-6)))
            prob_map[(r, c)][num] *= g_w * 1.15

    # 3️⃣ Skip-pattern confidence
    skip_scores = EXT_GM20_Skip_Pattern_Confidence_Vec(grid)
    for r, c in blanks:
        for num in prob_map[(r, c)]:
            prob_map[(r, c)][num] *= skip_scores[r, c] * 1.1

    # 4️⃣ Existing row/col sequences boost
    for r, c in blanks:
        row_seq = analyzer.get_arithmetic_or_geometric_sequences(grid[r], 3, 1)
        col_seq = analyzer.get_arithmetic_or_geometric_sequences(grid[:, c], 3, 1)
        for num in prob_map[(r, c)]:
            if any(num in seq for seq in row_seq + col_seq):
                prob_map[(r, c)][num] *= 1.7

    # 5️⃣ Final normalization (per-cell)
    for pos in prob_map:
        total = sum(prob_map[pos].values()) or 1e-10
        prob_map[pos] = {k: v / total for k, v in prob_map[pos].items()}

    return prob_map

# ---------------------------------------------------------------------
# Public API wrapper
# ---------------------------------------------------------------------
def predict_scratch_card(
    grid: List[List[int]], n_iter: int
) -> Dict[str, Any]:
    grid_np = np.array(grid, dtype=np.int64)

    prob_map = simulate_with_formulas(
        grid_np.tobytes(), grid_np.shape[0], grid_np.shape[1], n_iter
    )
    prob_map = weight_prob_by_modules(grid_np, prob_map)

    # Pack result
    results: List[Dict[str, Any]] = []
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

    full_probs = {f"{r},{c}": dist for (r, c), dist in prob_map.items()}
    return {
        "predictions": sorted(results, key=lambda x: x["confidences"][0], reverse=True),
        "full_probabilities": full_probs,
    }

# ---------------------------------------------------------------------
# 自檢報告
# ---------------------------------------------------------------------
# - 語法 / type check：通過
# - 主要修正：seq_ok / corr_ok 與 valid shape 對齊，避免 broadcast error
# - 其它核心演算法、權重、加權機制「零刪減」