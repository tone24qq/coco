from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter

from modules import register_strategy


# ----------------------------------------------------------------------
# 1. focus — 3×3 已知密度卷算
# ----------------------------------------------------------------------
@register_strategy("focus", weight=0.20)
def compute_focus_score(grid: NDArray) -> NDArray:
    mask = (grid != -1).astype(float)
    raw = uniform_filter(mask, size=3, mode="constant") * 9  # 0–9
    out = np.where(grid == -1, raw, 0.0)
    mn, mx = out.min(), out.max()
    return (out - mn) / (mx - mn + 1e-9)


# ----------------------------------------------------------------------
# 2. skip — 行/列等差跳格
# ----------------------------------------------------------------------
@register_strategy("skip", weight=0.15)
def detect_skip_patterns(grid: NDArray) -> NDArray:
    arr = grid.copy()
    rows, cols = arr.shape
    score = np.zeros_like(arr, dtype=float)

    # 行向量化
    idx = np.arange(cols)
    for r in range(rows):
        row_vals = arr[r]
        pos = np.flatnonzero(row_vals != -1)
        if pos.size >= 3:
            diffs = np.diff(row_vals[pos])
            if np.all(diffs == diffs[0]):
                k = diffs[0]
                base = row_vals[pos[0]]
                expected = base + (idx - pos[0]) * k
                mask = (row_vals == -1) & (expected >= 1) & (expected <= rows * cols)
                score[r, mask] = 1.0

    # 列向量化（轉箭重用）
    arrT = arr.T
    idx = np.arange(rows)
    for c in range(cols):
        col_vals = arrT[c]
        pos = np.flatnonzero(col_vals != -1)
        if pos.size >= 3:
            diffs = np.diff(col_vals[pos])
            if np.all(diffs == diffs[0]):
                k = diffs[0]
                base = col_vals[pos[0]]
                expected = base + (idx - pos[0]) * k
                mask = (col_vals == -1) & (expected >= 1) & (expected <= rows * cols)
                score[mask, c] = 1.0

    return score


# ----------------------------------------------------------------------
# 3. diff — 連續差值趨勢（±1…5）
# ----------------------------------------------------------------------
@register_strategy("diff", weight=0.15)
def compute_difference_trend(grid: NDArray) -> NDArray:
    arr = grid.copy()
    score = np.zeros_like(arr, dtype=float)

    # 行
    diff_row = np.abs(np.diff(arr, axis=1))
    trend_row = (diff_row[:, :-1] == diff_row[:, 1:]) & (diff_row[:, :-1] <= 5)
    mask_row = trend_row & (arr[:, 1:-1] == -1)
    score[:, 1:-1] = np.maximum(score[:, 1:-1], mask_row.astype(float))

    # 列
    diff_col = np.abs(np.diff(arr, axis=0))
    trend_col = (diff_col[:-1, :] == diff_col[1:, :]) & (diff_col[:-1, :] <= 5)
    mask_col = trend_col & (arr[1:-1, :] == -1)
    score[1:-1, :] = np.maximum(score[1:-1, :], mask_col.astype(float))

    return score.astype(float)


# ----------------------------------------------------------------------
# 4. mirror — 左右/上下鏡像
# ----------------------------------------------------------------------
@register_strategy("mirror", weight=0.20)
def detect_mirror_sequences(grid: NDArray) -> NDArray:
    arr = grid
    rows, cols = arr.shape
    score = np.zeros_like(arr, dtype=float)

    # 水平鏡像
    left = arr[:, :-2]
    right = arr[:, 2:]
    mid_h = (arr[:, 1:-1] == -1) & (left == right) & (left != -1)
    score[:, 1:-1] += mid_h.astype(float)

    # 垂直鏡像
    top = arr[:-2, :]
    bottom = arr[2:, :]
    mid_v = (arr[1:-1, :] == -1) & (top == bottom) & (top != -1)
    score[1:-1, :] += mid_v.astype(float)

    return score


# ----------------------------------------------------------------------
# 5. conn — 1/距離 連通性熱圖
# ----------------------------------------------------------------------
@register_strategy("conn", weight=0.15)
def connectivity_heatmap(grid: NDArray) -> NDArray:
    arr = grid
    ys, xs = np.nonzero(arr != -1)
    if ys.size == 0:
        return np.zeros_like(arr, dtype=float)

    Y, X = np.indices(arr.shape)
    d = np.min(np.abs(Y[..., None] - ys) + np.abs(X[..., None] - xs), axis=-1)
    score = np.where(arr == -1, 1.0 / (d + 1e-3), 0.0)
    mn, mx = score.min(), score.max()
    return (score - mn) / (mx - mn + 1e-9)


# ----------------------------------------------------------------------
# 6. tail — 尾數 (mod 10) 熱區
# ----------------------------------------------------------------------
@register_strategy("tail", weight=0.15)
def sequence_tail_analyzer(grid: NDArray) -> NDArray:
    arr = grid
    tails = arr % 10
    tails[arr == -1] = -1
    score = np.zeros_like(arr, dtype=float)
    for t in range(10):
        mask = tails == t
        if not mask.any():
            continue
        count = np.sum(mask)
        bonus = uniform_filter(mask.astype(float), size=3, mode="constant") * 9 / count
        score[arr == -1] += bonus[arr == -1]
    mn, mx = score.min(), score.max()
    return (score - mn) / (mx - mn + 1e-9)


# ----------------------------------------------------------------------
# 7. diag — 對角對符
# ----------------------------------------------------------------------
@register_strategy("diag", weight=0.10)
def diagonal_consistency_score(grid: NDArray) -> NDArray:
    arr = grid
    score = np.zeros_like(arr, dtype=float)
    ul = arr[:-2, :-2]
    br = arr[2:, 2:]
    ur = arr[:-2, 2:]
    bl = arr[2:, :-2]
    mid = arr[1:-1, 1:-1] == -1
    diag1 = (ul == br) & (ul != -1) & mid
    diag2 = (ur == bl) & (ur != -1) & mid
    score[1:-1, 1:-1] = diag1 | diag2
    return score


# ----------------------------------------------------------------------
# 8. row_col_bias（原本已有，但這裡給向量化參考）
# ----------------------------------------------------------------------
@register_strategy("row_col_bias", weight=0.08)
def row_col_bias(grid: NDArray) -> NDArray:
    rows, cols = grid.shape
    r_idx = np.linspace(-1, 1, rows)[:, None]
    c_idx = np.linspace(-1, 1, cols)[None, :]
    base = 1 - (r_idx**2 + c_idx**2)  # 中央高、邊角低
    base = np.where(grid == -1, base, 0.0)
    mn, mx = base.min(), base.max()
    return (base - mn) / (mx - mn + 1e-9)


# ----------------------------------------------------------------------
# 融合器 —— fuse_scores_vect
# ----------------------------------------------------------------------
def fuse_scores_vect(
    score_map: dict[str, NDArray], weights: dict[str, float], grid: NDArray
) -> NDArray:
    """Broadcast 權量後 tensordot 合併，再做 0-1 正規化。"""
    keys, mats = zip(*score_map.items())
    w = np.array([weights.get(k, 0.0) for k in keys], dtype=float)
    stack = np.stack(mats, axis=0)  # shape (K, H, W)
    combined = np.tensordot(w, stack, axes=1)  # (H, W)
    combined[grid != -1] = 0.0
    mn, mx = combined.min(), combined.max()
    return (combined - mn) / (mx - mn + 1e-9)
