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


def zone_type_for_cell(rows: int, cols: int, cell: Cell) -> str:
    r, c = cell
    if (r in (0, rows - 1)) and (c in (0, cols - 1)):
        return "corner"
    if r in (0, rows - 1) or c in (0, cols - 1):
        return "edge"
    return "center"


def support_context(board: Board, cell: Cell, local_radius: int = 1) -> Dict[str, float]:
    rows = len(board)
    cols = len(board[0])
    r, c = cell
    known = np.asarray(board, dtype=np.int32) != -1
    row_known = float(np.sum(known[r, :]))
    col_known = float(np.sum(known[:, c]))
    global_known = float(np.sum(known))
    row_avail = float(cols)
    col_avail = float(rows)
    global_avail = float(rows * cols)
    lr = max(1, int(local_radius))
    rr0, rr1 = max(0, r - lr), min(rows, r + lr + 1)
    cc0, cc1 = max(0, c - lr), min(cols, c + lr + 1)
    local_block = known[rr0:rr1, cc0:cc1]
    local_avail = float(local_block.size - 1)
    local_known = float(np.sum(local_block) - (1.0 if known[r, c] else 0.0))
    available_support = row_avail + col_avail + global_avail + max(local_avail, 0.0)
    observed_support = row_known + col_known + global_known + max(local_known, 0.0)
    coverage = observed_support / max(available_support, 1.0)
    return {
        "available_support_count": available_support,
        "normalized_support": coverage,
        "local_support": local_known / max(local_avail, 1.0),
        "row_support": row_known / max(row_avail, 1.0),
        "col_support": col_known / max(col_avail, 1.0),
        "global_support": global_known / max(global_avail, 1.0),
        "coverage_ratio": coverage,
        "zone_type": zone_type_for_cell(rows, cols, cell),
    }


def _compute_local_tail_evidence(
    board: Board,
    candidate: Cell,
    target_number: int,
    window_size: int = 3,
) -> Dict[str, float]:
    rows = len(board)
    cols = len(board[0])
    r, c = candidate
    k = max(1, int(window_size))
    if k % 2 == 0:
        k += 1
    radius = k // 2
    target_tail = int(target_number) % 10

    known_neighbors = 0
    same_tail_neighbors = 0
    near_value_neighbors = 0
    same_decade_neighbors = 0
    row_same_tail_count = 0
    col_same_tail_count = 0

    for rr in range(max(0, r - radius), min(rows, r + radius + 1)):
        for cc in range(max(0, c - radius), min(cols, c + radius + 1)):
            if rr == r and cc == c:
                continue
            value = board[rr][cc]
            if value == -1:
                continue
            known_neighbors += 1
            if value % 10 == target_tail:
                same_tail_neighbors += 1
                if rr == r:
                    row_same_tail_count += 1
                if cc == c:
                    col_same_tail_count += 1
            if abs(value - target_number) in {1, 2}:
                near_value_neighbors += 1
            if (value - 1) // 10 == (target_number - 1) // 10:
                same_decade_neighbors += 1

    local_tail_ratio = float(same_tail_neighbors) / max(float(known_neighbors), 1.0)
    has_structural_anchor = float((row_same_tail_count + col_same_tail_count) > 0 or near_value_neighbors > 0)
    strong_tail_signal = (
        (same_tail_neighbors >= 2 and known_neighbors >= 4 and (row_same_tail_count >= 1 or col_same_tail_count >= 1))
        or (near_value_neighbors >= 1 and same_tail_neighbors >= 1 and known_neighbors >= 3)
        or (local_tail_ratio >= 0.40 and (row_same_tail_count + col_same_tail_count) >= 2)
    )
    return {
        "known_neighbors": float(known_neighbors),
        "same_tail_neighbors": float(same_tail_neighbors),
        "row_same_tail_count": float(row_same_tail_count),
        "col_same_tail_count": float(col_same_tail_count),
        "near_value_neighbors": float(near_value_neighbors),
        "same_decade_neighbors": float(same_decade_neighbors),
        "local_tail_ratio": float(local_tail_ratio),
        "has_structural_anchor": float(has_structural_anchor),
        "strong_tail_signal": float(strong_tail_signal),
    }


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
    coverage = np.zeros_like(known)
    for r in range(h):
        r0 = max(0, r - rad)
        r1 = min(h, r + rad + 1)
        for c in range(w):
            c0 = max(0, c - rad)
            c1 = min(w, c + rad + 1)
            coverage[r, c] = float((r1 - r0) * (c1 - c0))
    density = window_sum / np.maximum(coverage, 1.0)
    board_baseline = float(np.mean(known))
    corrected = 0.5 + (density - board_baseline)
    density = _clip01(corrected)
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
    observed_density = float(np.mean(board_arr != -1))
    raw = contrib.mean(axis=1)
    baseline = np.median(raw) if raw.size else 0.0
    coverage_corrected = raw - baseline
    score = _clip01(0.5 + coverage_corrected / max(observed_density + 1e-6, 1e-3))
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
    row_single = np.where(
        (left_n != -1) ^ (right_n != -1),
        np.minimum(np.abs(target_number - left_n), np.abs(target_number - right_n)),
        np.nan,
    )
    col_single = np.where(
        (up_n != -1) ^ (down_n != -1),
        np.minimum(np.abs(target_number - up_n), np.abs(target_number - down_n)),
        np.nan,
    )
    stacked = np.vstack([row_mid, col_mid, row_single, col_single])
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
        observed = 0
        if c - 2 >= 0 and c + 2 < cols:
            a, m, z = b[r, c - 2], b[r, c - 1], b[r, c + 1]
            if a != -1 and m != -1 and z != -1:
                step = m - a
                pred = z + step
                supports.append(1.0 / (1.0 + abs(pred - target_number)))
                observed += 1
        if r - 2 >= 0 and r + 2 < rows:
            a, m, z = b[r - 2, c], b[r - 1, c], b[r + 1, c]
            if a != -1 and m != -1 and z != -1:
                step = m - a
                pred = z + step
                supports.append(1.0 / (1.0 + abs(pred - target_number)))
                observed += 1
        if supports:
            score[i] = float(np.mean(supports))
        elif observed == 0:
            score[i] = 0.5
    score = _clip01(score)
    return {cell: float(score[i]) for i, cell in enumerate(unopened_cells)}


def mirror_sequences_vectorized(board: Board, unopened_cells: List[Cell], target_number: int) -> Dict[Cell, float]:
    if not unopened_cells:
        return {}
    b = np.asarray(board, dtype=np.int32)
    rows, cols = b.shape
    rr, cc = _unopened_indices(unopened_cells)

    out = np.full(rr.shape[0], 0.5, dtype=np.float64)
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
    out: Dict[Cell, float] = {}
    for i, cell in enumerate(unopened_cells):
        evidence = _compute_local_tail_evidence(board, cell, target_number, window_size=window_size)
        if evidence["strong_tail_signal"] < 0.5:
            out[cell] = 0.5
            continue
        normalized_tail_ratio = max(0.0, min(1.0, (float(vals[i]) - 0.5) / 0.5))
        out[cell] = float(max(0.5, min(0.72, 0.50 + 0.22 * normalized_tail_ratio)))
    return out
