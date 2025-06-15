# ---------------------------- analyzer.py ---------------------------
import os, math, xxhash
from functools import lru_cache
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import warnings, numpy as np
warnings.filterwarnings("ignore",
                        message="invalid value encountered in divide",
                        category=RuntimeWarning)
from scipy.stats import qmc
from scipy.spatial.distance import cosine as _cosine

from modules import FORMULA_REGISTRY, compute_global_features
from brain   import (
    EXT_GM20_Skip_Pattern_Confidence_Vec,
    MathUtils,
    BoardAnalyzerUtils,
)

# ------------------------------------------------------------------ #
# 基本工具
# ------------------------------------------------------------------ #
math_utils    = MathUtils()
analyzer_util = BoardAnalyzerUtils()

def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """避免零向量產生 nan；結果 ∈[0,1]"""
    if np.allclose(a, 0) or np.allclose(b, 0):
        return 0.0
    return 1.0 - _cosine(a, b)

# -------- 全盤指紋（固定 10 維：μ, σ, diff_hist×8） ---------------
def global_fingerprint(arr: np.ndarray) -> np.ndarray:
    flat = arr[arr != -1]
    if flat.size == 0:
        return np.zeros(10, dtype=np.float32)
    mu, sigma = flat.mean(), flat.std() or 1.0
    diff_hist, _ = np.histogram(
        np.diff(np.sort(flat)),
        bins=8, range=(1, 64)
    )
    diff_hist = diff_hist.astype(np.float32)
    diff_hist /= (diff_hist.sum() or 1.0)
    return np.concatenate([[mu, sigma], diff_hist])

# -------- 9×9 Patch 分數 ------------------------------------------
def local_patch_score(p_true: np.ndarray, p_board: np.ndarray) -> float:
    mask = (p_true != -1)
    if not mask.any():
        return 0.5
    diff = np.abs(p_true[mask] - p_board[mask])
    gap  = diff.mean() / (p_true.max() + 1e-6)
    return max(0.0, 1.0 - gap)

def prebuild_patch_masks(r: int, c: int, rad: int = 4):
    masks = {}
    for i in range(r):
        for j in range(c):
            masks[(i, j)] = (
                slice(max(0, i - rad), min(r, i + rad + 1)),
                slice(max(0, j - rad), min(c, j + rad + 1)),
            )
    return masks

# -------- Float Count-Min Sketch -----------------------------------
class CountMinSketch:
    def __init__(self, width: int = 4096, depth: int = 4):
        self.w, self.d = width, depth
        self.tab   = np.zeros((depth, width), dtype=np.float32)
        self.seeds = [i * 0x9E3779B1 for i in range(depth)]

    def _idx(self, key: bytes, seed: int) -> int:
        return xxhash.xxh32(key, seed=seed).intdigest() % self.w

    def update(self, key: bytes, v: float = 1.0):
        for i, s in enumerate(self.seeds):
            self.tab[i, self._idx(key, s)] += v

    def query(self, key: bytes) -> float:
        return min(self.tab[i, self._idx(key, s)] for i, s in enumerate(self.seeds))

def pack_key(idx: int, num: int) -> bytes:
    return (idx << 16 | num).to_bytes(4, "little")

# ------------------------------------------------------------------ #
# Sobol + CMS Monte-Carlo
# ------------------------------------------------------------------ #
@lru_cache(maxsize=128)
def simulate_with_formulas(
    grid_bytes: bytes,
    rows: int,
    cols: int,
    n_iter: int,
    quick_mode: bool = False,
    min_iter: int = 0
) -> Dict[Tuple[int,int], Dict[int,float]]:
    grid   = np.frombuffer(grid_bytes, dtype=np.int64).reshape(rows, cols)
    blanks = np.argwhere(grid == -1)
    known  = np.argwhere(grid != -1)
    k_vals = grid[grid != -1]
    lin_k  = rows*known[:,0] + known[:,1]

    cms  = CountMinSketch()
    idx_map = {tuple(b): i for i, b in enumerate(blanks)}
    legal   = analyzer_util.get_legal_values_for_placement(grid)

    fp_true  = global_fingerprint(grid)
    masks    = prebuild_patch_masks(rows, cols)
    skip_ref = None if quick_mode else EXT_GM20_Skip_Pattern_Confidence_Vec(grid)

    batch = max(500, 20_000 // (rows*cols))
    sobol = qmc.Sobol(d=rows*cols, scramble=True)
    rand  = sobol.random                      # 無 2^k 限制

    total_seen, effective = 0, 0
    while total_seen < n_iter:
        need   = min(batch, n_iter - total_seen)
        boards = (rand(need) * (rows*cols)).astype(np.int64).reshape(-1, rows, cols)

        # 1. 已知格比對
        valid = np.all(boards.reshape(-1, rows*cols)[:, lin_k] == k_vals, axis=1)

        # 2. 序列 & skip 相似
        if not quick_mode:
            seq_ok = np.array([
                analyzer_util.check_sequences(b, grid, 3, 1)
                for b in boards
            ])
            valid &= seq_ok
            if valid.any():
                corrs = [
                    np.corrcoef(
                        skip_ref.ravel(),
                        EXT_GM20_Skip_Pattern_Confidence_Vec(b).ravel()
                    )[0, 1]
                    for b in boards[valid]
                ]
                valid[valid] &= np.array([c > 0.85 for c in corrs])

        finals = boards[valid]
        effective += len(finals)

        # 3. hit-counter 累加
        for b in finals:
            sim     = safe_cosine(fp_true, global_fingerprint(b))
            g_boost = 1.0 + max(sim - 0.90, 0.0) * 5.0
            for r, c in blanks:
                pr, pc  = masks[(r, c)]
                p_score = local_patch_score(grid[pr, pc], b[pr, pc])
                boost   = g_boost * (0.6 + 0.4 * p_score)
                cms.update(pack_key(idx_map[(r, c)], int(b[r, c])), boost)

        total_seen += need

        # 4. early-stop
        if (
            total_seen >= max(min_iter, 30_000)
            and not quick_mode
            and effective >= 30_000
        ):
            if all(
                (lambda lst: max(lst) / (sum(lst) or 1) > 0.88)(
                    [cms.query(pack_key(idx_map[p], n)) for n in legal]
                ) for p in idx_map
            ):
                break
                    # 5. 轉成機率表
    prob: Dict[Tuple[int,int], Dict[int,float]] = {}
    for pos, idx in idx_map.items():
        cnts = {n: cms.query(pack_key(idx, n)) for n in legal}
        if not any(cnts.values()):
            continue
        mn, mx = min(cnts.values()), max(cnts.values())
        prob[pos] = {k: math_utils.normalize_value(v, mn or 1e-10, mx or 1e-10)
                     for k, v in cnts.items()}
    return prob


# ------------------------------------------------------------------ #
# 後處理權重
# ------------------------------------------------------------------ #
def weight_prob_by_modules(
    grid: np.ndarray,
    prob: Dict[Tuple[int, int], Dict[int, float]]
) -> Dict[Tuple[int, int], Dict[int, float]]:
    rows, cols = grid.shape
    blanks = np.argwhere(grid == -1)
    legal  = analyzer_util.get_legal_values_for_placement(grid) or {0}
    uniform = {n: 1/len(legal) for n in legal}

    for r, c in blanks:
        if (r, c) not in prob or not prob[(r, c)]:
            prob[(r, c)] = dict(uniform)

    # 1. Local resonance
    for r, c in blanks:
        win = grid[max(0, r-1):r+2, max(0, c-1):c+2]
        kn  = win[win != -1]
        if kn.size:
            m = kn.mean()
            for n in prob[(r, c)]:
                prob[(r, c)][n] *= 1.2 / (1 + abs(n - m) * 0.5)

    # 2. Global distribution
    mean, std = compute_global_features(grid.astype(np.float32))[:2]
    std = std or 1.0
    for r, c in blanks:
        for n in prob[(r, c)]:
            g = math.exp(-((n - mean)**2) / (2*(std**2 + 1e-6)))
            prob[(r, c)][n] *= g * 1.15

    # 3. Skip-pattern boost
    skip = EXT_GM20_Skip_Pattern_Confidence_Vec(grid)
    for r, c in blanks:
        fac = max(skip[r, c], 0.05) * 1.1
        for n in prob[(r, c)]:
            prob[(r, c)][n] *= fac

    # 4. Row/Col sequence boost
    seq_fn = analyzer_util.get_arithmetic_or_geometric_sequences
    for r, c in blanks:
        has = set().union(*seq_fn(grid[r], 3, 1),
                          *seq_fn(grid[:, c], 3, 1))
        for n in prob[(r, c)]:
            if n in has:
                prob[(r, c)][n] *= 1.7

    # 5. Normalize
    for pos, dist in prob.items():
        s = sum(dist.values()) or 1e-10
        prob[pos] = {k: v / s for k, v in dist.items()}
    return prob


# ------------------------------------------------------------------ #
# 全局唯一化
# ------------------------------------------------------------------ #
def global_unique(prob, blanks):
    try:
        from scipy.optimize import linear_sum_assignment
        nums = sorted({n for d in prob.values() for n in d})
        cost = np.full((len(blanks), len(nums)), 50.0)
        for i, cell in enumerate(blanks):
            for j, n in enumerate(nums):
                cost[i, j] = -math.log(prob[cell].get(n, 1e-9))
        r, c = linear_sum_assignment(cost)
        return {blanks[i]: (nums[j],
                            prob[blanks[i]].get(nums[j], 0.0))
                for i, j in zip(r, c)}
    except Exception:
        taken, out = set(), {}
        for cell in sorted(
            blanks,
            key=lambda p: max(prob[p].values()),
            reverse=True
        ):
            for n, pscore in sorted(
                prob[cell].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                if n not in taken:
                    taken.add(n)
                    out[cell] = (n, pscore)
                    break
        return out


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #
def predict_scratch_card(
    grid: List[List[int]],
    target_num: Optional[int] = None,
    quick_iter:  int = int(os.getenv("QUICK_ITER", 200_000)),
    refine_iter: int = int(os.getenv("REFINE_ITER", 800_000)),
    min_total_iter: int = int(os.getenv("MIN_TOTAL_ITER", 1_000_000)),
    unique: bool = True
) -> Dict[str, Any]:
    gp = np.array(grid, dtype=np.int64)
    rows, cols = gp.shape
    blanks = [tuple(b) for b in np.argwhere(gp == -1)]

    # -------- Quick pass --------
    quick_map = simulate_with_formulas(
        gp.tobytes(), rows, cols,
        n_iter=quick_iter, quick_mode=True,
        min_iter=min_total_iter // 5
    )
    quick_map = weight_prob_by_modules(gp, quick_map)

    # 取熱度最高 3 格再細算
    hot_cells = sorted(
        blanks,
        key=lambda p: max(quick_map[p].values()),
        reverse=True
    )[:3]

    # -------- Refine pass --------
    refine_map = simulate_with_formulas(
        gp.tobytes(), rows, cols,
        n_iter=refine_iter, quick_mode=False,
        min_iter=min_total_iter - quick_iter
    )
    refine_map = weight_prob_by_modules(gp, refine_map)

    # 合併：熱點→refine，其餘→quick
    prob_map = {cell: (refine_map if cell in hot_cells else quick_map)[cell]
                for cell in blanks}
                    # ---- 轉成 Python scalar 並把 tuple key → 字串 ----
    full_probs = {
        f"{int(r)},{int(c)}": {int(k): float(v) for k, v in dist.items()}
        for (r, c), dist in prob_map.items()
    }

    # ---- Specific number (target_num) 模式 → predictions ----
    if target_num is not None:
        preds = [{
            "row": int(r),
            "col": int(c),
            "candidates": [int(target_num)],
            "confidences": [float(prob_map[(r, c)].get(target_num, 0.0))]
        } for r, c in blanks]

        preds.sort(key=lambda x: x["confidences"][0], reverse=True)
        return {
            "predictions": preds,
            "full_probabilities": full_probs
        }
            # ---- 全局唯一化 (Hungarian / Greedy) ----
    if unique:
        assign = global_unique(prob_map, blanks)
        preds = [{
            "row": int(r),
            "col": int(c),
            "candidates": [int(n)],
            "confidences": [float(p)]
        } for (r, c), (n, p) in assign.items()]
        preds.sort(key=lambda x: x["confidences"][0], reverse=True)
        return {
            "predictions": preds,
            "full_probabilities": full_probs
        }

    # ---- Default：每格 Top-3 ----
    preds = []
    for (r, c), dist in prob_map.items():
        best = sorted(dist.items(),
                      key=lambda x: x[1],
                      reverse=True)[:3]
        nums, conf = zip(*best)
        preds.append({
            "row": int(r),
            "col": int(c),
            "candidates": [int(n) for n in nums],
            "confidences": [float(x) for x in conf]
        })

    preds.sort(key=lambda x: x["confidences"][0], reverse=True)
    return {
        "predictions": preds,
        "full_probabilities": full_probs
    }
# ----------------------------- END ----------------------------------