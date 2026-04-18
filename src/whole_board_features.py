from __future__ import annotations

from collections import Counter
from math import log2, sqrt
from typing import Dict, Iterable, List, Tuple


Cell = Tuple[int, int]
PRIMARY_FEATURE_PREFIXES = (
    "residue_",
    "multiple10_",
    "residue_interaction_",
    "local5x5_",
    "row_residue_",
    "col_residue_",
    "row_decade_",
    "col_decade_",
    "neighbor_residue_",
    "neighbor_multiple10_",
)

DEPRECATED_FEATURE_PREFIXES = (
    "known_ratio",
    "unknown_ratio",
    "row_known_entropy",
    "col_known_entropy",
    "tail_entropy",
    "tail_hist_max_ratio",
    "same_tail_adjacency_rate",
    "same_decade_proximity_rate",
    "consecutive_neighbor_rate",
    "edge_center_balance",
)


def _neighbors(rows: int, cols: int, r: int, c: int) -> Iterable[Cell]:
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            yield rr, cc


def _tail(v: int) -> int:
    return int(v % 10)


def _decade(v: int) -> int:
    return int((v - 1) // 10)


def _entropy(values: List[int]) -> float:
    if not values:
        return 0.0
    total = len(values)
    counts = Counter(values)
    return -sum((cnt / total) * log2(cnt / total) for cnt in counts.values() if cnt > 0)


def _relative_center_distance(rows: int, cols: int, r: int, c: int) -> float:
    center_r = (rows - 1) / 2.0
    center_c = (cols - 1) / 2.0
    dist = abs(r - center_r) + abs(c - center_c)
    max_dist = max(center_r + center_c, 1.0)
    return float(dist / max_dist)


def _relative_edge_distance(rows: int, cols: int, r: int, c: int) -> float:
    min_dist = min(r, rows - 1 - r, c, cols - 1 - c)
    max_possible = max(min((rows - 1) / 2.0, (cols - 1) / 2.0), 1.0)
    return float(min_dist / max_possible)


def _window_cells(rows: int, cols: int, r: int, c: int, radius: int = 2) -> List[Cell]:
    out: List[Cell] = []
    for rr in range(max(0, r - radius), min(rows, r + radius + 1)):
        for cc in range(max(0, c - radius), min(cols, c + radius + 1)):
            out.append((rr, cc))
    return out


def _known_values(board: List[List[int]], cells: Iterable[Cell]) -> List[int]:
    values: List[int] = []
    for r, c in cells:
        v = int(board[r][c])
        if v != -1:
            values.append(v)
    return values


def is_primary_feature_column(name: str) -> bool:
    normalized = name
    if normalized.startswith("board_state_"):
        normalized = normalized[len("board_state_") :]
    if normalized.startswith("candidate_delta_"):
        normalized = normalized[len("candidate_delta_") :]
    if normalized.startswith("delta_"):
        normalized = normalized[len("delta_") :]
    return any(normalized.startswith(prefix) for prefix in PRIMARY_FEATURE_PREFIXES)


def compute_board_state_features(board: List[List[int]], target_number: int) -> Dict[str, float]:
    rows = len(board)
    cols = len(board[0]) if rows else 0
    known_cells: List[Cell] = []
    known_values: List[int] = []

    for r in range(rows):
        for c in range(cols):
            v = int(board[r][c])
            if v != -1:
                known_cells.append((r, c))
                known_values.append(v)

    tail_values = [_tail(v) for v in known_values]
    decade_values = [_decade(v) for v in known_values]
    residue_hist = Counter(tail_values)
    decade_hist = Counter(decade_values)

    out: Dict[str, float] = {
        "board_rows": float(rows),
        "board_cols": float(cols),
        "board_size": float(rows * cols),
        "target_number_norm": float(target_number / max(rows * cols, 1)),
        "residue_target_tail": float(_tail(target_number)),
        "multiple10_target_is_multiple_of_10": float(int(target_number % 10 == 0)),
        "multiple10_target_dist_to_nearest_multiple_of_10": float(
            min(target_number % 10, (10 - target_number % 10) % 10)
        ),
        "multiple10_target_signed_delta_to_nearest_multiple_of_10": float(
            (target_number // 10 * 10) - target_number
            if target_number % 10 <= 5
            else ((target_number // 10 + 1) * 10) - target_number
        ),
        "residue_global_entropy": float(_entropy(tail_values)),
        "multiple10_global_decade_entropy": float(_entropy(decade_values)),
    }
    for d in range(10):
        out[f"residue_global_tail_hist_{d}"] = float(residue_hist.get(d, 0) / max(len(known_values), 1))
        out[f"multiple10_global_decade_hist_{d}"] = float(decade_hist.get(d, 0) / max(len(known_values), 1))
    return out


def compute_candidate_delta_features(
    board: List[List[int]],
    target_number: int,
    cand_row: int,
    cand_col: int,
    board_state_features: Dict[str, float],
) -> Dict[str, float]:
    rows = len(board)
    cols = len(board[0]) if rows else 0
    prev_value = int(board[cand_row][cand_col])

    window_cells = _window_cells(rows, cols, cand_row, cand_col, radius=2)
    window_values = _known_values(board, window_cells)
    row_cells = [(cand_row, cc) for cc in range(cols)]
    col_cells = [(rr, cand_col) for rr in range(rows)]
    row_values = _known_values(board, row_cells)
    col_values = _known_values(board, col_cells)
    neighbor_cells = list(_neighbors(rows, cols, cand_row, cand_col))
    neighbor_values = _known_values(board, neighbor_cells)

    target_tail = _tail(target_number)
    target_decade = _decade(target_number)

    row_tail_values = [_tail(v) for v in row_values]
    col_tail_values = [_tail(v) for v in col_values]
    win_tail_values = [_tail(v) for v in window_values]
    row_dec_values = [_decade(v) for v in row_values]
    col_dec_values = [_decade(v) for v in col_values]
    win_dec_values = [_decade(v) for v in window_values]

    row_tail_hist = Counter(row_tail_values)
    col_tail_hist = Counter(col_tail_values)
    win_tail_hist = Counter(win_tail_values)
    row_dec_hist = Counter(row_dec_values)
    col_dec_hist = Counter(col_dec_values)
    win_dec_hist = Counter(win_dec_values)

    same_tail_local = sum(1 for v in window_values if _tail(v) == target_tail)
    same_decade_local = sum(1 for v in window_values if _decade(v) == target_decade)
    neighbors_same_tail = sum(1 for v in neighbor_values if _tail(v) == target_tail)
    neighbors_multiple10 = sum(1 for v in neighbor_values if v % 10 == 0)
    neighbors_near_multiple10 = sum(1 for v in neighbor_values if min(v % 10, (10 - v % 10) % 10) <= 1)

    diff_local = [abs(v - target_number) for v in window_values]
    diff_bins = {1, 2, 5, 10, 20}

    row_band = min((cand_row * 3) // max(rows, 1), 2)
    col_band = min((cand_col * 3) // max(cols, 1), 2)
    local_density = len(window_values) / max(len(window_cells), 1)

    nearest_floor = (target_number // 10) * 10
    nearest_ceil = ((target_number + 9) // 10) * 10
    if abs(target_number - nearest_floor) <= abs(nearest_ceil - target_number):
        nearest = nearest_floor
    else:
        nearest = nearest_ceil

    # local adjacency patterns
    local_cell_set = set(window_cells)
    same_tail_adjacent = 0
    same_decade_adjacent = 0
    same_tail_same_decade = 0
    consecutive_pairs = 0
    for rr, cc in window_cells:
        v = int(board[rr][cc])
        if v == -1:
            continue
        for nr, nc in _neighbors(rows, cols, rr, cc):
            if (nr, nc) <= (rr, cc) or (nr, nc) not in local_cell_set:
                continue
            u = int(board[nr][nc])
            if u == -1:
                continue
            if _tail(v) == _tail(u):
                same_tail_adjacent += 1
            if _decade(v) == _decade(u):
                same_decade_adjacent += 1
            if _tail(v) == _tail(u) and _decade(v) == _decade(u):
                same_tail_same_decade += 1
            if abs(v - u) == 1:
                consecutive_pairs += 1

    target_tail_mode = Counter(win_tail_values).most_common(1)[0][0] if win_tail_values else -1
    target_dec_mode = Counter(win_dec_values).most_common(1)[0][0] if win_dec_values else -1

    residue_tail_mode_match = 1.0 if target_tail_mode == target_tail else 0.0
    residue_dec_mode_match = 1.0 if target_dec_mode == target_decade else 0.0

    target_is_multiple10 = int(target_number % 10 == 0)
    out = {
        "cand_row": float(cand_row + 1),
        "cand_col": float(cand_col + 1),
        "relative_row": float((cand_row + 1) / max(rows, 1)),
        "relative_col": float((cand_col + 1) / max(cols, 1)),
        "relative_center_distance": _relative_center_distance(rows, cols, cand_row, cand_col),
        "relative_edge_distance": _relative_edge_distance(rows, cols, cand_row, cand_col),
        "is_feasible": float(int(prev_value == -1)),
        # A. 尾數訊號族
        **{f"residue_tail_{d}": float(int(target_tail == d)) for d in range(10)},
        "residue_same_tail_count_local5x5": float(same_tail_local),
        "residue_same_tail_ratio_local5x5": float(same_tail_local / max(len(window_values), 1)),
        "residue_same_tail_adjacent_count": float(same_tail_adjacent),
        "residue_target_tail_match_count_local5x5": float(same_tail_local),
        "neighbor_residue_same_tail_count": float(neighbors_same_tail),
        "neighbor_residue_same_tail_ratio": float(neighbors_same_tail / max(len(neighbor_values), 1)),
        "row_residue_entropy": float(_entropy(row_tail_values)),
        "col_residue_entropy": float(_entropy(col_tail_values)),
        "local5x5_residue_entropy": float(_entropy(win_tail_values)),
        # B. 十倍數訊號族
        "multiple10_is_multiple_of_10": float(target_is_multiple10),
        "multiple10_dist_to_nearest_multiple_of_10": float(abs(target_number - nearest)),
        "multiple10_signed_delta_to_nearest_multiple_of_10": float(nearest - target_number),
        "local5x5_multiple10_count": float(sum(1 for v in window_values if v % 10 == 0)),
        "local5x5_multiple10_ratio": float(sum(1 for v in window_values if v % 10 == 0) / max(len(window_values), 1)),
        "row_multiple10_count": float(sum(1 for v in row_values if v % 10 == 0)),
        "col_multiple10_count": float(sum(1 for v in col_values if v % 10 == 0)),
        "neighbor_multiple10_count": float(neighbors_multiple10),
        "neighbor_multiple10_near_multiple10_count": float(neighbors_near_multiple10),
        "local5x5_same_decade_count": float(same_decade_local),
        "row_decade_entropy": float(_entropy(row_dec_values)),
        "col_decade_entropy": float(_entropy(col_dec_values)),
        "local5x5_decade_entropy": float(_entropy(win_dec_values)),
        # C. 交互
        "residue_interaction_tail_x_decade": float(target_tail * 10 + target_decade),
        "residue_interaction_tail_x_row_band": float(target_tail * 10 + row_band),
        "residue_interaction_tail_x_col_band": float(target_tail * 10 + col_band),
        "residue_interaction_tail_x_local_density": float(target_tail * local_density),
        "residue_interaction_multiple10_x_neighbor_abs_delta_bin": float(
            target_is_multiple10 * sum(1 for d in diff_local if d <= 2)
        ),
        "residue_interaction_multiple10_x_same_tail_ratio": float(
            target_is_multiple10 * (same_tail_local / max(len(window_values), 1))
        ),
        "residue_interaction_target_tail_x_window_tail_mode": float(target_tail * residue_tail_mode_match),
        "residue_interaction_target_decade_x_window_decade_mode": float(target_decade * residue_dec_mode_match),
        "local5x5_count_abs_delta_eq_10": float(sum(1 for d in diff_local if d == 10)),
        "local5x5_consecutive_pair_count": float(consecutive_pairs),
        "local5x5_same_decade_adjacent_count": float(same_decade_adjacent),
        "local5x5_same_tail_and_same_decade_count": float(same_tail_same_decade),
    }

    for d in range(10):
        out[f"row_residue_hist_{d}"] = float(row_tail_hist.get(d, 0) / max(len(row_tail_values), 1))
        out[f"col_residue_hist_{d}"] = float(col_tail_hist.get(d, 0) / max(len(col_tail_values), 1))
        out[f"local5x5_residue_hist_{d}"] = float(win_tail_hist.get(d, 0) / max(len(win_tail_values), 1))
        out[f"row_decade_hist_{d}"] = float(row_dec_hist.get(d, 0) / max(len(row_dec_values), 1))
        out[f"col_decade_hist_{d}"] = float(col_dec_hist.get(d, 0) / max(len(col_dec_values), 1))
        out[f"local5x5_decade_hist_{d}"] = float(win_dec_hist.get(d, 0) / max(len(win_dec_values), 1))

    for b in sorted(diff_bins):
        out[f"local5x5_count_abs_delta_in_{b}"] = float(sum(1 for d in diff_local if d == b))

    return out


def merge_feature_layers(board_state: Dict[str, float], candidate_delta: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out.update({f"board_state_{k}": float(v) for k, v in board_state.items()})
    out.update({f"candidate_delta_{k}": float(v) for k, v in candidate_delta.items()})
    return out


def euclidean(values: List[float]) -> float:
    return float(sqrt(sum(v * v for v in values)))
