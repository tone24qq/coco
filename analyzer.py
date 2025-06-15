# --------------------------  analyzer.py  ---------------------------
import os, math, xxhash, pathlib, csv
from functools import lru_cache
from typing    import List, Dict, Tuple, Any, Optional

import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import cosine as _cosine

from modules import FORMULA_REGISTRY, compute_global_features
from brain   import (
    EXT_GM20_Skip_Pattern_Confidence_Vec,
    MathUtils,
    BoardAnalyzerUtils,
)

# -------- utils -----------------------------------------------------
math_utils    = MathUtils()
analyzer_util = BoardAnalyzerUtils()

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - _cosine(a, b)         # 0~1，相似越高越接近 1

# -------- 1. 全盤指紋 -----------------------------------------------
def global_fingerprint(arr: np.ndarray) -> np.ndarray:
    flat = arr[arr != -1]
    if flat.size == 0: return np.zeros(12)
    mu, sigma = flat.mean(), flat.std() or 1
    diff_hist = np.histogram(np.diff(np.sort(flat)), bins=8,
                             range=(1, flat.max()+1))[0]
    diff_hist = diff_hist / (diff_hist.sum() or 1)
    return np.concatenate([[mu, sigma], diff_hist])

# -------- 2. 9×9 Patch Score ---------------------------------------
def local_patch_score(p_true: np.ndarray, p_board: np.ndarray) -> float:
    mask = (p_true != -1)
    if not mask.any():               # 全空 patch
        return 0.5
    diff = np.abs(p_true[mask] - p_board[mask])
    mean_gap = diff.mean()
    norm_gap = mean_gap / (p_true.max() + 1e-6)
    return max(0.0, 1.0 - norm_gap)  # 0~1，越小差距越大

def prebuild_patch_mask(rows: int, cols: int, radius: int = 4):
    m = {}
    for r in range(rows):
        for c in range(cols):
            m[(r, c)] = (
                slice(max(0, r - radius), min(rows, r + radius + 1)),
                slice(max(0, c - radius), min(cols, c + radius + 1)),
            )
    return m

# -------- 3. Count-Min Sketch --------------------------------------
class CountMinSketch:
    def __init__(self, width: int = 4096, depth: int = 4):
        self.w, self.d = width, depth
        self.tab = np.zeros((depth, width), dtype=np.uint32)
        self.seeds = [i * 0x9E3779B1 for i in range(depth)]

    def _h(self, key: bytes, seed: int) -> int:
        return xxhash.xxh32(key, seed=seed).intdigest() % self.w

    def update(self, key: bytes, v: float = 1):
        for i, s in enumerate(self.seeds):
            self.tab[i, self._h(key, s)] += v

    def query(self, key: bytes) -> int:
        return min(self.tab[i, self._h(key, s)] for i, s in enumerate(self.seeds))

def pack_key(idx: int, num: int) -> bytes:
    return (idx << 16 | num).to_bytes(4, "little")

# ------------------- 4. Monte-Carlo with boosts --------------------
@lru_cache(maxsize=128)
def simulate_with_formulas(
    grid_bytes: bytes,
    rows: int,
    cols: int,
    n_iter: int,
    quick_mode: bool = False,
    min_iter: int = 0,
) -> Dict[Tuple[int, int], Dict[int, float]]:
    grid = np.frombuffer(grid_bytes, dtype=np.int64).reshape(rows, cols)
    blanks  = np.argwhere(grid == -1)
    known   = np.argwhere(grid != -1)
    k_vals  = grid[grid != -1]
    lin_k   = rows * known[:, 0] + known[:, 1]

    cms  = CountMinSketch()
    idx_map = {tuple(b): i for i, b in enumerate(blanks)}
    legal   = analyzer_util.get_legal_values_for_placement(grid)

    # baseline
    fp_true = global_fingerprint(grid)
    patch_mask = prebuild_patch_mask(rows, cols)
    skip_base  = None if quick_mode else EXT_GM20_Skip_Pattern_Confidence_Vec(grid)
    mean, std  = compute_global_features(grid.astype(np.float32))[:2]
    std = std or 1.0

    # Sobol engine
    batch = max(500, 20_000 // (rows * cols))
    sob_dim = rows * cols
    engine  = qmc.Sobol(d=sob_dim, scramble=True)
    step    = 2 ** math.ceil(math.log(batch, 2))

    total, eff = 0, 0
    while total < n_iter:
        need = min(step, n_iter - total)
        vec  = engine.random_base2(int(math.log2(need)))
        boards = (vec * (rows * cols)).astype(np.int64).reshape(-1, rows, cols)

        # 已知格一致
        valid = np.all(boards.reshape(-1, rows * cols)[:, lin_k] == k_vals, axis=1)

        # 序列 & skip corr
        if not quick_mode:
            seq_ok = np.array([analyzer_util.check_sequences(
                                b, grid, 3, 1) for b in boards])
            valid &= seq_ok
            if valid.any():
                corrs = [np.corrcoef(skip_base.ravel(),
                                     EXT_GM20_Skip_Pattern_Confidence_Vec(b).ravel())[0,1]
                         for b in boards[valid]]
                valid[valid] &= np.array([c > 0.85 for c in corrs])

        finals = boards[valid]
        eff += len(finals)

        # ------- 累加（boost = 全盤 × Patch） -------
        for b in finals:
            g_sim   = cosine(fp_true, global_fingerprint(b))
            g_boost = 1.0 + max(g_sim - 0.90, 0) * 5  # 0.90↑ → ×1~1.5

            for r, c in blanks:
                pr, pc = patch_mask[(r, c)]
                p_true, p_b = grid[pr, pc], b[pr, pc]
                p_score = local_patch_score(p_true, p_b)     # 0~1
                boost   = g_boost * (0.6 + 0.4 * p_score)   # 0.6~1.5

                cms.update(pack_key(idx_map[(r, c)], int(b[r, c])), boost)

        total += need

        # early-stop（達 min_iter 才檢查）
        if total >= max(min_iter, 30_000) and not quick_mode and eff >= 30_000:
            conv = all(
                (lambda lst: max(lst) / (sum(lst) or 1) > 0.88)(
                    [cms.query(pack_key(idx_map[p], n)) for n in legal]
                )
                for p in idx_map
            )
            if conv: break

    # ---- normalize to prob_map ----
    prob_map: Dict[Tuple[int, int], Dict[int, float]] = {}
    for (r, c), i in idx_map.items():
        cnt = {n: cms.query(pack_key(i, n)) for n in legal}
        if not any(cnt.values()): continue
        mn, mx = min(cnt.values()), max(cnt.values())
        prob_map[(r, c)] = {k: math_utils.normalize_value(v, mn or 1e-10, mx or 1e-10)
                            for k, v in cnt.items()}
    return prob_map

# ------------------- 5. weight_prob_by_modules (同前) ----------------
def weight_prob_by_modules(
    grid: np.ndarray,
    prob: Dict[Tuple[int, int], Dict[int, float]]
) -> Dict[Tuple[int, int], Dict[int, float]]:
    rows, cols = grid.shape
    blanks = np.argwhere(grid == -1)
    legal  = analyzer_util.get_legal_values_for_placement(grid) or {0}
    uni    = {n: 1/len(legal) for n in legal}

    for r, c in blanks:
        if (r,c) not in prob or not prob[(r,c)]:
            prob[(r,c)] = dict(uni)

    # Local resonance
    for r, c in blanks:
        win = grid[max(0,r-1):r+2, max(0,c-1):c+2]
        kn  = win[win != -1]
        if kn.size:
            m = kn.mean()
            for n in prob[(r,c)]:
                prob[(r,c)][n] *= 1.2 / (1+abs(n-m)*0.5)

    # Global dist
    mean, std = compute_global_features(grid.astype(np.float32))[:2]
    std = std or 1.0
    for r, c in blanks:
        for n in prob[(r,c)]:
            g = math.exp(-((n-mean)**2)/(2*(std**2+1e-6)))
            prob[(r,c)][n] *= g*1.15

    # Skip boost
    skip = EXT_GM20_Skip_Pattern_Confidence_Vec(grid)
    for r, c in blanks:
        fac = max(skip[r,c], 0.05)*1.1
        for n in prob[(r,c)]:
            prob[(r,c)][n] *= fac

    # Row/col seq
    for r, c in blanks:
        seq = analyzer_util.get_arithmetic_or_geometric_sequences
        has = set().union(*seq(grid[r], 3, 1), *seq(grid[:, c], 3, 1))
        for n in prob[(r,c)]:
            if n in has:
                prob[(r,c)][n] *= 1.7

    # normalize
    for pos, dist in prob.items():
        s = sum(dist.values()) or 1e-10
        prob[pos] = {k: v/s for k,v in dist.items()}
    return prob

# ------------------- 6. 唯一化 (Hungarian→Greedy) ------------------
def global_unique(prob, blanks):
    try:
        from scipy.optimize import linear_sum_assignment
        nums  = sorted({n for d in prob.values() for n in d})
        cost  = np.full((len(blanks), len(nums)), 50.0)

        for i, cell in enumerate(blanks):
            for j, n in enumerate(nums):
                # 這行補齊 ↓↓↓
                cost[i, j] = -math.log(prob[cell].get(n, 1e-9))  # 取對數成本

        row, col = linear_sum_assignment(cost)
        return {blanks[r]: (nums[c], prob[blanks[r]].get(nums[c], 0.0))
                for r, c in zip(row, col)}

    except Exception:
        # ---- Greedy 後備 ----
        taken, assign = set(), {}
        for cell in sorted(blanks, key=lambda p: max(prob[p].values()), reverse=True):
            for n, p_score in sorted(prob[cell].items(), key=lambda x: x[1], reverse=True):
                if n not in taken:
                    taken.add(n)
                    assign[cell] = (n, p_score)
                    break
        return assign
        # -------------------- 7. Public API --------------------
def predict_scratch_card(
    grid: List[List[int]],
    target_num: Optional[int] = None,
    quick_iter:  int = int(os.getenv("QUICK_ITER", 200_000)),
    refine_iter: int = int(os.getenv("REFINE_ITER", 800_000)),
    min_total_iter: int = int(os.getenv("MIN_TOTAL_ITER", 1_000_000)),
    unique: bool = True
) -> Dict[str, Any]:
    """
    - 若 `target_num` 指定：只回傳該數字在各空格的信心排序。
    - `unique=True`：全局唯一化（Hungarian→Greedy）。
    - 不指定 `target_num` → 回傳每格 Top-3 候選。
    """
    grid_np = np.array(grid, dtype=np.int64)
    rows, cols = grid_np.shape
    blanks = [tuple(b) for b in np.argwhere(grid_np == -1)]

    # -------- Quick pass --------
    quick_map = simulate_with_formulas(
        grid_np.tobytes(), rows, cols,
        n_iter=quick_iter, quick_mode=True,
        min_iter=min_total_iter // 5
    )
    quick_map = weight_prob_by_modules(grid_np, quick_map)

    # 熱度最高的 3 格
    hot_cells = sorted(blanks, key=lambda p: max(quick_map[p].values()), reverse=True)[:3]

    # -------- Refine pass --------
    refine_map = simulate_with_formulas(
        grid_np.tobytes(), rows, cols,
        n_iter=refine_iter, quick_mode=False,
        min_iter=min_total_iter - quick_iter
    )
    refine_map = weight_prob_by_modules(grid_np, refine_map)

    # 合併：熱點用 refine，其餘用 quick
    prob_map = {cell: (refine_map if cell in hot_cells else quick_map)[cell]
                for cell in blanks}

    # -------- unique assignment --------
    if unique and target_num is None:
        assign = global_unique(prob_map, blanks)
        preds = [{
            "row": r, "col": c,
            "candidates": [n],
            "confidences": [float(p)]
        } for (r, c), (n, p) in assign.items()]
        preds.sort(key=lambda x: x["confidences"][0], reverse=True)
        return {"mode": "unique",
                "predictions": preds,
                "full_probabilities": prob_map}

    # -------- target_num only --------
    if target_num is not None:
        rank = [{
            "row": r, "col": c,
            "candidate": target_num,
            "confidence": prob_map[(r, c)].get(target_num, 0.0)
        } for r, c in blanks]
        rank.sort(key=lambda x: x["confidence"], reverse=True)
        return {"target": target_num,
                "rankings": rank,
                "full_probabilities": prob_map}

    # -------- default Top-3 --------
    preds = []
    for (r, c), dist in prob_map.items():
        best = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
        nums, conf = zip(*best)
        preds.append({
            "row": r, "col": c,
            "candidates": list(nums),
            "confidences": list(map(float, conf))
        })
    preds.sort(key=lambda x: x["confidences"][0], reverse=True)
    return {"mode": "top3",
            "predictions": preds,
            "full_probabilities": prob_map}
# ------------------  END  analyzer.py ------------------