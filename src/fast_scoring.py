from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


def _id(x):
    return x


def _njit(**kwargs):
    if njit is None:
        return _id
    return njit(**kwargs)


def prepare_fast_inputs(
    board: list[list[int]],
    unopened_cells: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    board_arr = np.asarray(board, dtype=np.int32)
    rows = np.asarray([r for r, _ in unopened_cells], dtype=np.int32)
    cols = np.asarray([c for _, c in unopened_cells], dtype=np.int32)
    known_mask = board_arr != -1
    return board_arr, rows, cols, known_mask


@_njit(cache=True)
def logic_rule_numba(
    board_arr: np.ndarray,
    unopened_rows: np.ndarray,
    unopened_cols: np.ndarray,
    target_number: int,
) -> np.ndarray:
    out = np.empty(unopened_rows.shape[0], dtype=np.float64)
    rows, cols = board_arr.shape
    for i in range(unopened_rows.shape[0]):
        r = unopened_rows[i]
        c = unopened_cols[i]
        neigh_count = 0
        contradiction_votes = 0
        abs_delta_sum = 0.0
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                continue
            v = board_arr[rr, cc]
            if v == -1:
                continue
            neigh_count += 1
            abs_delta_sum += abs(v - target_number)
            if rr == r and cc < c and v > target_number:
                contradiction_votes += 1
            if rr == r and cc > c and v < target_number:
                contradiction_votes += 1
            if cc == c and rr < r and v > target_number:
                contradiction_votes += 1
            if cc == c and rr > r and v < target_number:
                contradiction_votes += 1
        if neigh_count == 0:
            out[i] = 0.5
        else:
            local_support = 1.0 / (1.0 + (abs_delta_sum / neigh_count))
            contradiction_penalty = contradiction_votes / neigh_count
            score = local_support - 0.7 * contradiction_penalty
            if score < 0.0:
                score = 0.0
            if score > 1.0:
                score = 1.0
            out[i] = score
    return out


def prior_model_fast(board_arr: np.ndarray, unopened_rows: np.ndarray, unopened_cols: np.ndarray) -> np.ndarray:
    rows, cols = board_arr.shape
    center_r = (rows - 1) / 2.0
    center_c = (cols - 1) / 2.0
    max_dist = max(center_r + center_c, 1.0)
    dist = np.abs(unopened_rows - center_r) + np.abs(unopened_cols - center_c)
    return 1.0 - (dist / max_dist)


@_njit(cache=True)
def directional_consistency_numba(
    board_arr: np.ndarray,
    unopened_rows: np.ndarray,
    unopened_cols: np.ndarray,
    target_number: int,
) -> np.ndarray:
    out = np.empty(unopened_rows.shape[0], dtype=np.float64)
    rows, cols = board_arr.shape
    board_size = rows * cols
    scale = max(board_size / 3.0, 1.0)
    for i in range(unopened_rows.shape[0]):
        r = unopened_rows[i]
        c = unopened_cols[i]
        left_n = right_n = up_n = down_n = 0
        left_sat = right_sat = up_sat = down_sat = 0
        left_min = right_min = up_min = down_min = 1e9
        left_sum = right_sum = up_sum = down_sum = 0.0
        row_violation_count = 0
        col_violation_count = 0

        row_vals_count = 1
        row_vals_sum = float(target_number)
        col_vals_count = 1
        col_vals_sum = float(target_number)

        for cc in range(cols):
            v = board_arr[r, cc]
            if v == -1:
                continue
            row_vals_count += 1
            row_vals_sum += v
            if cc < c:
                left_n += 1
                if v < target_number:
                    left_sat += 1
                if v > target_number:
                    row_violation_count += 1
                d = abs(v - target_number)
                left_sum += d
                if d < left_min:
                    left_min = d
            elif cc > c:
                right_n += 1
                if v > target_number:
                    right_sat += 1
                if v < target_number:
                    row_violation_count += 1
                d = abs(v - target_number)
                right_sum += d
                if d < right_min:
                    right_min = d
        for rr in range(rows):
            v = board_arr[rr, c]
            if v == -1:
                continue
            col_vals_count += 1
            col_vals_sum += v
            if rr < r:
                up_n += 1
                if v < target_number:
                    up_sat += 1
                if v > target_number:
                    col_violation_count += 1
                d = abs(v - target_number)
                up_sum += d
                if d < up_min:
                    up_min = d
            elif rr > r:
                down_n += 1
                if v > target_number:
                    down_sat += 1
                if v < target_number:
                    col_violation_count += 1
                d = abs(v - target_number)
                down_sum += d
                if d < down_min:
                    down_min = d

        def order_score(n, sat):
            return 0.5 if n == 0 else sat / n

        def distance_score(n, s, m):
            if n == 0:
                return 0.5
            nearest = m if m < 1e8 else 0.0
            avg = s / n
            v = 1.0 - ((0.6 * nearest + 0.4 * avg) / scale)
            if v < 0.0:
                v = 0.0
            if v > 1.0:
                v = 1.0
            return v

        support = (
            order_score(left_n, left_sat)
            + order_score(right_n, right_sat)
            + order_score(up_n, up_sat)
            + order_score(down_n, down_sat)
            + distance_score(left_n, left_sum, left_min)
            + distance_score(right_n, right_sum, right_min)
            + distance_score(up_n, up_sum, up_min)
            + distance_score(down_n, down_sum, down_min)
            + 0.5
            + 0.5
        ) / 10.0
        violation_penalty = (row_violation_count + col_violation_count) / 4.0
        if violation_penalty > 1.0:
            violation_penalty = 1.0
        score = support - 0.8 * violation_penalty
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        out[i] = score
    return out


@_njit(cache=True)
def line_consistency_numba(
    board_arr: np.ndarray,
    unopened_rows: np.ndarray,
    unopened_cols: np.ndarray,
    target_number: int,
) -> np.ndarray:
    out = np.empty(unopened_rows.shape[0], dtype=np.float64)
    rows, cols = board_arr.shape
    for i in range(unopened_rows.shape[0]):
        r = unopened_rows[i]
        c = unopened_cols[i]
        row_known = 0
        col_known = 0
        for cc in range(cols):
            if board_arr[r, cc] != -1:
                row_known += 1
        for rr in range(rows):
            if board_arr[rr, c] != -1:
                col_known += 1
        info = (row_known + col_known) / max(rows + cols, 1)
        score = 0.45 + 0.55 * info
        if score > 1.0:
            score = 1.0
        out[i] = score
    return out


@_njit(cache=True)
def evaluate_pairwise_gain_numba(
    board_arr: np.ndarray,
    known_mask: np.ndarray,
    candidate_row: int,
    candidate_col: int,
    anchor_rows: np.ndarray,
    anchor_cols: np.ndarray,
    anchor_values: np.ndarray,
    target_number: int,
    max_trials: int,
) -> Tuple[float, int, int, int]:
    best_gain = 0.0
    best_anchor_idx = -1
    best_anchor_value = -1
    trials = 0
    rows, cols = board_arr.shape
    for ai in range(anchor_rows.shape[0]):
        ar = anchor_rows[ai]
        ac = anchor_cols[ai]
        if ar == candidate_row and ac == candidate_col:
            continue
        for vi in range(anchor_values.shape[0]):
            if trials >= max_trials:
                return best_gain, best_anchor_idx, best_anchor_value, trials
            av = anchor_values[vi]
            trials += 1
            if av == target_number:
                continue
            conflict = False
            for rr in range(rows):
                for cc in range(cols):
                    if known_mask[rr, cc] and board_arr[rr, cc] == av:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                continue
            dist = abs(ar - candidate_row) + abs(ac - candidate_col)
            gain = 1.0 / (1.0 + dist + abs(av - target_number))
            if gain > best_gain:
                best_gain = gain
                best_anchor_idx = ai
                best_anchor_value = av
    return best_gain, best_anchor_idx, best_anchor_value, trials
