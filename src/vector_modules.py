from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

Board = List[List[int]]
Cell = Tuple[int, int]


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def _unopened_indices(unopened_cells: List[Cell]) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray([r for r, _ in unopened_cells], dtype=np.int32)
    cols = np.asarray([c for _, c in unopened_cells], dtype=np.int32)
    return rows, cols


def focus_score_vectorized(board: Board, unopened_cells: List[Cell], window_size: int = 3) -> Dict[Cell, float]:
    if not unopened_cells:
        return {}
    board_arr = np.asarray(board, dtype=np.int32)
    known = (board_arr != -1).astype(np.float64)
    k = max(1, int(window_size))
    if k % 2 == 0:
        k += 1
    rad = k // 2
    padded = np.pad(known, ((rad, rad), (rad, rad)), mode="constant", constant_values=0.0)
    prefix = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    h, w = known.shape
    window_sum = np.zeros_like(known)
    for r in range(h):
        r0 = r
        r1 = r + k
        for c in range(w):
            c0 = c
            c1 = c + k
            window_sum[r, c] = prefix[r1, c1] - prefix[r0, c1] - prefix[r1, c0] + prefix[r0, c0]
    density = _clip01(window_sum / float(k * k))
    rr, cc = _unopened_indices(unopened_cells)
    values = density[rr, cc]
    return {cell: float(values[i]) for i, cell in enumerate(unopened_cells)}


def connectivity_heatmap_vectorized(
    board: Board,
    unopened_cells: List[Cell],
    decay: str = "inverse_distance",
    decay_gamma: float = 0.35,
) -> Dict[Cell, float]:
    if not unopened_cells:
        return {}
    board_arr = np.asarray(board, dtype=np.int32)
    known_pos = np.argwhere(board_arr != -1)
    rr, cc = _unopened_indices(unopened_cells)
    q = np.stack([rr, cc], axis=1).astype(np.float64)
    if known_pos.size == 0:
        return {cell: 0.5 for cell in unopened_cells}
    kp = known_pos.astype(np.float64)
    dr = np.abs(q[:, None, 0] - kp[None, :, 0])
    dc = np.abs(q[:, None, 1] - kp[None, :, 1])
    manhattan = dr + dc
    if decay == "manhattan":
        contrib = 1.0 / (1.0 + manhattan)
    elif decay == "exp_decay":
        contrib = np.exp(-float(decay_gamma) * manhattan)
    else:
        contrib = 1.0 / np.maximum(1.0, manhattan)
    score = _clip01(contrib.mean(axis=1) * 2.0)
    return {cell: float(score[i]) for i, cell in enumerate(unopened_cells)}


def difference_trend_vectorized(board: Board, unopened_cells: List[Cell], target_number: int) -> Dict[Cell, float]:
    if not unopened_cells:
        return {}
    board_arr = np.asarray(board, dtype=np.int32)
    known = board_arr != -1
    rows, cols = board_arr.shape

    left_val = np.full_like(board_arr, -1)
    right_val = np.full_like(board_arr, -1)
    up_val = np.full_like(board_arr, -1)
    down_val = np.full_like(board_arr, -1)

    for r in range(rows):
        last = -1
        for c in range(cols):
            left_val[r, c] = last
            if known[r, c]:
                last = board_arr[r, c]
        last = -1
        for c in range(cols - 1, -1, -1):
            right_val[r, c] = last
            if known[r, c]:
                last = board_arr[r, c]

    for c in range(cols):
        last = -1
        for r in range(rows):
            up_val[r, c] = last
            if known[r, c]:
                last = board_arr[r, c]
        last = -1
        for r in range(rows - 1, -1, -1):
            down_val[r, c] = last
            if known[r, c]:
                last = board_arr[r, c]

    rr, cc = _unopened_indices(unopened_cells)
    left_n = left_val[rr, cc]
    right_n = right_val[rr, cc]
    up_n = up_val[rr, cc]
    down_n = down_val[rr, cc]

    row_mid = np.where(
        (left_n != -1) & (right_n != -1),
        np.abs(target_number - (left_n + right_n) / 2.0),
        np.nan,
    )
    col_mid = np.where(
        (up_n != -1) & (down_n != -1),
        np.abs(target_number - (up_n + down_n) / 2.0),
        np.nan,
    )
    stacked = np.vstack([row_mid, col_mid])
    valid = np.isfinite(stacked)
    count = valid.sum(axis=0)
    safe = np.where(valid, stacked, 0.0)
    trend_delta = np.where(count > 0, safe.sum(axis=0) / np.maximum(count, 1), np.nan)
    fallback = np.abs(np.nanmean(np.vstack([left_n, right_n, up_n, down_n]), axis=0) - target_number)
    delta = np.where(np.isnan(trend_delta), np.nan_to_num(fallback, nan=float(target_number)), trend_delta)
    score = _clip01(1.0 / (1.0 + delta / max(rows * cols / 6.0, 1.0)))
    return {cell: float(score[i]) for i, cell in enumerate(unopened_cells)}


def skip_patterns_vectorized(board: Board, unopened_cells: List[Cell], target_number: int) -> Dict[Cell, float]:
    if not unopened_cells:
        return {}
    b = np.asarray(board, dtype=np.int32)
    rows, cols = b.shape
    rr, cc = _unopened_indices(unopened_cells)

    score = np.full(rr.shape[0], 0.5, dtype=np.float64)
    for i, (r, c) in enumerate(zip(rr.tolist(), cc.tolist())):
        supports = []
        if c - 2 >= 0 and c + 2 < cols:
            a, m, z = b[r, c - 2], b[r, c - 1], b[r, c + 1]
            if a != -1 and m != -1 and z != -1:
                step = m - a
                pred = z + step
                supports.append(1.0 / (1.0 + abs(pred - target_number)))
        if r - 2 >= 0 and r + 2 < rows:
            a, m, z = b[r - 2, c], b[r - 1, c], b[r + 1, c]
            if a != -1 and m != -1 and z != -1:
                step = m - a
                pred = z + step
                supports.append(1.0 / (1.0 + abs(pred - target_number)))
        if supports:
            score[i] = float(np.mean(supports))
    score = _clip01(score)
    return {cell: float(score[i]) for i, cell in enumerate(unopened_cells)}


def mirror_sequences_vectorized(board: Board, unopened_cells: List[Cell], target_number: int) -> Dict[Cell, float]:
    if not unopened_cells:
        return {}
    b = np.asarray(board, dtype=np.int32)
    rows, cols = b.shape
    rr, cc = _unopened_indices(unopened_cells)

    out = np.full(rr.shape[0], 0.4, dtype=np.float64)
    for i, (r, c) in enumerate(zip(rr.tolist(), cc.tolist())):
        vals = []
        if cols > 1:
            mc = cols - 1 - c
            v = b[r, mc]
            if v != -1:
                vals.append(1.0 / (1.0 + abs(v - target_number)))
        if rows > 1:
            mr = rows - 1 - r
            v = b[mr, c]
            if v != -1:
                vals.append(1.0 / (1.0 + abs(v - target_number)))
        if rows > 1 and cols > 1:
            mr = rows - 1 - r
            mc = cols - 1 - c
            v = b[mr, mc]
            if v != -1:
                vals.append(1.0 / (1.0 + abs(v - target_number)))
        if vals:
            out[i] = float(np.mean(vals))
    clipped = _clip01(out)
    return {cell: float(clipped[i]) for i, cell in enumerate(unopened_cells)}


def tail_analyzer_vectorized(
    board: Board,
    unopened_cells: List[Cell],
    target_number: int,
    window_size: int = 3,
) -> Dict[Cell, float]:
    if not unopened_cells:
        return {}
    board_arr = np.asarray(board, dtype=np.int32)
    target_tail = int(target_number) % 10
    known = board_arr != -1
    tail_match = ((board_arr % 10) == target_tail) & known

    k = max(1, int(window_size))
    if k % 2 == 0:
        k += 1
    rad = k // 2

    padded_known = np.pad(known.astype(np.float64), ((rad, rad), (rad, rad)), mode="constant")
    padded_match = np.pad(tail_match.astype(np.float64), ((rad, rad), (rad, rad)), mode="constant")

    def _box_sum(arr: np.ndarray) -> np.ndarray:
        prefix = np.pad(arr, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
        h, w = known.shape
        out = np.zeros((h, w), dtype=np.float64)
        for r in range(h):
            for c in range(w):
                r0, r1 = r, r + k
                c0, c1 = c, c + k
                out[r, c] = prefix[r1, c1] - prefix[r0, c1] - prefix[r1, c0] + prefix[r0, c0]
        return out

    known_cnt = _box_sum(padded_known)
    match_cnt = _box_sum(padded_match)
    ratio = (match_cnt + 1.0) / (known_cnt + 2.0)
    rr, cc = _unopened_indices(unopened_cells)
    vals = _clip01(ratio[rr, cc])
    return {cell: float(vals[i]) for i, cell in enumerate(unopened_cells)}
