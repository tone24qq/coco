# analyzer.py  (QMC + CMS + 1M guarantee)
import os
import math
import numpy as np
import xxhash
from scipy.stats import qmc
from collections import Counter, defaultdict
from functools import lru_cache
from typing import List, Dict, Tuple, Any, Optional

from modules import FORMULA_REGISTRY, compute_global_features
from brain import (
    EXT_GM20_Skip_Pattern_Confidence_Vec,
    MathUtils,
    BoardAnalyzerUtils,
)

math_utils = MathUtils()
analyzer_utils = BoardAnalyzerUtils()

# ------------------------------------------------------------------ #
#  Count-Min Sketch (64 KB 預設)
# ------------------------------------------------------------------ #
class CountMinSketch:
    def __init__(self, width: int = 4096, depth: int = 4):
        self.w, self.d = width, depth
        self.table = np.zeros((depth, width), dtype=np.uint32)
        self.seeds = [i * 0x9e3779B1 for i in range(depth)]

    def _idx(self, key: bytes, seed: int) -> int:
        return xxhash.xxh32(key, seed=seed).intdigest() % self.w

    def update(self, key: bytes, value: int = 1):
        for s in self.seeds:
            self.table[self.seeds.index(s), self._idx(key, s)] += value

    def query(self, key: bytes) -> int:
        return min(self.table[i, self._idx(key, s)] for i, s in enumerate(self.seeds))

# 4-byte packed key: cell_idx (hi 16) | num (lo 16)
def pack_key(cell_idx: int, num: int) -> bytes:
    return (cell_idx << 16 | num).to_bytes(4, "little")

# ------------------------------------------------------------------ #
#  Sobol + CMS Monte-Carlo  (min_iter 保底)
# ------------------------------------------------------------------ #
@lru_cache(maxsize=128)
def simulate_with_formulas(
        grid_bytes: bytes,
        rows: int,
        cols: int,
        n_iter: int,
        quick_mode: bool = False,
        min_iter: int = 0
) -> Dict[Tuple[int, int], Dict[int, float]]:
    grid = np.frombuffer(grid_bytes, dtype=np.int64).reshape(rows, cols)
    blanks = np.argwhere(grid == -1)
    known_idx = np.argwhere(grid != -1)
    known_vals = grid[grid != -1]
    lin_known = rows * known_idx[:, 0] + known_idx[:, 1]

    # Count-Min Sketch
    cms = CountMinSketch()
    cell_to_idx = {tuple(b): i for i, b in enumerate(blanks)}
    legal_all = analyzer_utils.get_legal_values_for_placement(grid)

    # 特徵 / baseline
    skip_scores = None if quick_mode else EXT_GM20_Skip_Pattern_Confidence_Vec(grid)
    mean_val, std_val = compute_global_features(grid.astype(np.float32))[:2]
    std_val = std_val or 1.0

    batch_size = max(500, 20000 // (rows * cols))
    total_seen, effective = 0, 0

    # Sobol engine
    sobol_dim = rows * cols
    qmc_engine = qmc.Sobol(d=sobol_dim, scramble=True)
    bits = math.ceil(math.log(batch_size, 2))
    step_vecs = 2 ** bits  # 最接近 2^m 的點數

    while total_seen < n_iter:
        # 生成 Sobol 點；若不足 n_iter，最後一次可少量
        need = min(step_vecs, n_iter - total_seen)
        vec = qmc_engine.random_base2(int(math.log2(need)))
        boards = (vec * (rows * cols)).astype(np.int64).reshape(-1, rows, cols)

        # 已知格比對
        valid_mask = np.all(boards.reshape(-1, rows * cols)[:, lin_known] == known_vals, axis=1)

        # 深檢查 (序列 + skip corr)
        if not quick_mode:
            # 序列
            seq_ok = np.array([
                analyzer_utils.check_sequences(b, grid, 3, 1) for b in boards
            ])
            valid_mask &= seq_ok
            # skip corr
            if valid_mask.any():
                corrs = [
                    np.corrcoef(skip_scores.ravel(),
                                EXT_GM20_Skip_Pattern_Confidence_Vec(b).ravel())[0, 1]
                    for b in boards[valid_mask]
                ]
                valid_mask[valid_mask] &= np.array([c > 0.85 for c in corrs])

        valid_boards = boards[valid_mask]
        effective += len(valid_boards)

        # 累加進 CMS
        for b in valid_boards:
            for r, c in blanks:
                cms.update(pack_key(cell_to_idx[(r, c)], int(b[r, c])))

        total_seen += need

        # early-stop 只有達到 min_iter 才可觸發
        if (
            total_seen >= max(min_iter, 30000)
            and not quick_mode
            and effective >= 30000
        ):
            # 粗估收斂：取樣少做一次掃描
            converged = True
            for (r, c), idx in cell_to_idx.items():
                counts = [cms.query(pack_key(idx, n)) for n in legal_all]
                s = sum(counts)
                if s == 0:
                    converged = False
                    break
                if max(counts) / s <= 0.88:
                    converged = False
                    break
            if converged:
                break

    # 轉回機率表
    prob_map: Dict[Tuple[int, int], Dict[int, float]] = {}
    for (r, c), idx in cell_to_idx.items():
        cnts = {n: cms.query(pack_key(idx, n)) for n in legal_all}
        if not any(cnts.values()):
            continue
        v_min, v_max = min(cnts.values()), max(cnts.values())
        prob_map[(r, c)] = {
            k: math_utils.normalize_value(v, v_min or 1e-10, v_max or 1e-10)
            for k, v in cnts.items()
        }
    return prob_map

# ------------------------------------------------------------------ #
#  你原本的 weight_prob_by_modules() 直接貼回來即可
# ------------------------------------------------------------------ #
from previous_version import weight_prob_by_modules   # <—— 替換成你現有實作

# ------------------------------------------------------------------ #
#  唯一化：Hungarian 有 scipy，否則 Greedy
# ------------------------------------------------------------------ #
def global_unique(prob_map: Dict[Tuple[int, int], Dict[int, float]],
                  blanks: List[Tuple[int, int]]) -> Dict[Tuple[int, int], Tuple[int, float]]:
    try:
        from scipy.optimize import linear_sum_assignment
        nums = sorted({n for d in prob_map.values() for n in d})
        cost = np.full((len(blanks), len(nums)), 50.0)
        for i, cell in enumerate(blanks):
            for j, n in enumerate(nums):
                cost[i, j] = -math.log(prob_map[cell].get(n, 1e-9))
        row, col = linear_sum_assignment(cost)
        return {blanks[r]: (nums[c], prob_map[blanks[r]].get(nums[c], 0.0))
                for r, c in zip(row, col)}
    except Exception:
        # Greedy fallback
        assigned, res = set(), {}
        for cell in sorted(blanks, key=lambda p: max(prob_map[p].values()), reverse=True):
            for n, p in sorted(prob_map[cell].items(), key=lambda x: x[1], reverse=True):
                if n not in assigned:
                    assigned.add(n)
                    res[cell] = (n, p)
                    break
        return res

# ------------------------------------------------------------------ #
#  Public API (quick + refine, 保底百萬)
# ------------------------------------------------------------------ #
def predict_scratch_card(
        grid: List[List[int]],
        target_num: Optional[int] = None,
        quick_iter: int = int(os.getenv("QUICK_ITER", 200_000)),
        refine_iter: int = int(os.getenv("REFINE_ITER", 800_000)),
        min_total_iter: int = 1_000_000,
        unique: bool = True
) -> Dict[str, Any]:
    grid_np = np.array(grid, dtype=np.int64)
    rows, cols = grid_np.shape
    blanks = [tuple(b) for b in np.argwhere(grid_np == -1)]

    # -------- Quick pass --------
    quick = simulate_with_formulas(
        grid_np.tobytes(), rows, cols,
        n_iter=quick_iter,
        quick_mode=True,
        min_iter=min_total_iter // 5
    )
    quick = weight_prob_by_modules(grid_np, quick)

    # 熱點 3 格
    hot = sorted(blanks, key=lambda p: max(quick[p].values()), reverse=True)[:3]

    # -------- Refine pass --------
    refine = simulate_with_formulas(
        grid_np.tobytes(), rows, cols,
        n_iter=refine_iter,
        quick_mode=False,
        min_iter=min_total_iter - quick_iter
    )
    refine = weight_prob_by_modules(grid_np, refine)

    # 合併：熱點用 refine，其餘用 quick
    final_map = {cell: (refine if cell in hot else quick)[cell] for cell in blanks}

    # -------- 唯一化 or Top-3 --------
    if unique and target_num is None:
        assign = global_unique(final_map, blanks)
        preds = [{
            "row": r, "col": c,
            "candidates": [n], "confidences": [float(p)]
        } for (r, c), (n, p) in assign.items()]
        preds.sort(key=lambda x: x["confidences"][0], reverse=True)
        return {"mode": "unique", "predictions": preds, "full_probabilities": final_map}

    if target_num is not None:
        rank = [{
            "row": r, "col": c,
            "candidate": target_num,
            "confidence": final_map[(r, c)].get(target_num, 0.0)
        } for r, c in blanks]
        rank.sort(key=lambda x: x["confidence"], reverse=True)
        return {"target": target_num, "rankings": rank, "full_probabilities": final_map}

    # default Top-3
    preds = []
    for (r, c), dist in final_map.items():
        top3 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
        nums, conf = zip(*top3)
        preds.append({
            "row": r, "col": c,
            "candidates": list(nums),
            "confidences": list(map(float, conf))
        })
    preds.sort(key=lambda x: x["confidences"][0], reverse=True)
    return {"mode": "top3", "predictions": preds, "full_probabilities": final_map}