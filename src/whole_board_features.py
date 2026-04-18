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
    "global_residue_",
    "global_decade_",
)

FEATURE_SCHEMA_VERSION = "whole_board_features_v3_refactored"
NEAR_CONSTANT_STD_EPS = 1e-8
NEAR_CONSTANT_DOMINANT_RATIO = 0.999

FEATURE_RENAME_MAP = {
    "local5x5_count_abs_delta_in_1": "local5x5_count_abs_delta_eq_1",
    "local5x5_count_abs_delta_in_2": "local5x5_count_abs_delta_eq_2",
    "local5x5_count_abs_delta_in_5": "local5x5_count_abs_delta_eq_5",
    "local5x5_count_abs_delta_in_10": "local5x5_count_abs_delta_eq_10",
    "local5x5_count_abs_delta_in_20": "local5x5_count_abs_delta_eq_20",
}

FEATURE_MERGE_MAP = {
    "candidate_delta_row_multiple10_count": "merged into candidate_delta_row_multiple10_ratio",
    "candidate_delta_col_multiple10_count": "merged into candidate_delta_col_multiple10_ratio",
    "candidate_delta_local5x5_multiple10_count": "merged into candidate_delta_local5x5_multiple10_ratio",
    "candidate_delta_residue_target_tail_match_count_local5x5": (
        "merged into candidate_delta_residue_same_tail_count_local5x5"
    ),
}

DEPRECATED_FEATURE_PREFIXES = (
    "board_rows",
    "board_cols",
    "board_size",
    "target_number_norm",
    "cand_row",
    "cand_col",
    "relative_row",
    "relative_col",
    "relative_center_distance",
    "relative_edge_distance",
    "is_feasible",
)

_DYNAMIC_OPTIONAL_MARKERS = (
    "_decade_hist_bin_",
    "_decade_mode_bin_",
)


def _neighbors(rows: int, cols: int, r: int, c: int) -> Iterable[Cell]:
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            yield rr, cc


def _directional_neighbor(rows: int, cols: int, r: int, c: int, dr: int, dc: int) -> Cell | None:
    rr, cc = r + dr, c + dc
    if 0 <= rr < rows and 0 <= cc < cols:
        return rr, cc
    return None


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


def _gini_from_counts(counts: Counter[int], total: int) -> float:
    if total <= 0:
        return 0.0
    probs = [cnt / total for cnt in counts.values() if cnt > 0]
    return float(1.0 - sum(p * p for p in probs))


def _hist_summary(prefix: str, values: List[int]) -> Dict[str, float]:
    if not values:
        return {
            f"{prefix}_entropy": 0.0,
            f"{prefix}_mode_bin": -1.0,
            f"{prefix}_top1_ratio": 0.0,
            f"{prefix}_top1_top2_gap": 0.0,
            f"{prefix}_gini": 0.0,
        }

    counts = Counter(values)
    total = len(values)
    ranked = counts.most_common(2)
    top1 = ranked[0][1] / total
    top2 = ranked[1][1] / total if len(ranked) > 1 else 0.0
    return {
        f"{prefix}_entropy": float(_entropy(values)),
        f"{prefix}_mode_bin": float(ranked[0][0]),
        f"{prefix}_top1_ratio": float(top1),
        f"{prefix}_top1_top2_gap": float(top1 - top2),
        f"{prefix}_gini": _gini_from_counts(counts, total),
    }


def _dynamic_hist(prefix: str, values: List[int], max_bin: int) -> Dict[str, float]:
    counts = Counter(values)
    total = max(len(values), 1)
    out: Dict[str, float] = {}
    for b in range(max_bin + 1):
        out[f"{prefix}_hist_bin_{b}"] = float(counts.get(b, 0) / total)
    return out


def _nearest_multiple_of_10(target_number: int) -> int:
    floor_v = (target_number // 10) * 10
    ceil_v = ((target_number + 9) // 10) * 10
    return floor_v if abs(target_number - floor_v) <= abs(ceil_v - target_number) else ceil_v


def _max_decade(rows: int, cols: int) -> int:
    board_size = max(rows * cols, 1)
    return int((board_size - 1) // 10)


def is_primary_feature_column(name: str) -> bool:
    normalized = name
    if normalized.startswith("board_state_"):
        normalized = normalized[len("board_state_") :]
    if normalized.startswith("candidate_delta_"):
        normalized = normalized[len("candidate_delta_") :]
    if normalized.startswith("delta_"):
        normalized = normalized[len("delta_") :]
    return any(normalized.startswith(prefix) for prefix in PRIMARY_FEATURE_PREFIXES)


def is_dynamic_optional_feature_column(name: str) -> bool:
    return any(marker in name for marker in _DYNAMIC_OPTIONAL_MARKERS)


def compute_board_state_features(board: List[List[int]], target_number: int) -> Dict[str, float]:
    rows = len(board)
    cols = len(board[0]) if rows else 0
    known_values = [int(board[r][c]) for r in range(rows) for c in range(cols) if int(board[r][c]) != -1]

    target_tail = _tail(target_number)
    target_decade = _decade(target_number)
    target_is_multiple10 = int(target_number % 10 == 0)
    nearest = _nearest_multiple_of_10(target_number)

    tail_values = [_tail(v) for v in known_values]
    decade_values = [_decade(v) for v in known_values]
    tail_counts = Counter(tail_values)

    out: Dict[str, float] = {
        "residue_target_tail_index": float(target_tail),
        "multiple10_target_is_multiple_of_10": float(target_is_multiple10),
        "multiple10_target_dist_to_nearest_multiple_of_10": float(abs(target_number - nearest)),
        "multiple10_target_signed_delta_to_nearest_multiple_of_10": float(nearest - target_number),
    }
    out.update(_hist_summary("global_residue", tail_values))
    out.update(_hist_summary("global_decade", decade_values))
    # backward-compatible alias for tests/runtime that still reference old key naming
    out["residue_global_entropy"] = out["global_residue_entropy"]

    for t in range(10):
        out[f"global_residue_tail_hist_{t}"] = float(tail_counts.get(t, 0) / max(len(tail_values), 1))
        out[f"residue_target_tail_{t}"] = float(int(target_tail == t))

    max_dec = _max_decade(rows, cols)
    out.update(_dynamic_hist("global_decade", decade_values, max_dec))
    out["global_decade_target_decade_ratio"] = float(
        Counter(decade_values).get(target_decade, 0) / max(len(decade_values), 1)
    )

    return out


def compute_candidate_delta_features(
    board: List[List[int]],
    target_number: int,
    cand_row: int,
    cand_col: int,
    board_state_features: Dict[str, float],
) -> Dict[str, float]:
    _ = board_state_features
    rows = len(board)
    cols = len(board[0]) if rows else 0
    prev_value = int(board[cand_row][cand_col])

    target_tail = _tail(target_number)
    target_decade = _decade(target_number)
    target_is_multiple10 = int(target_number % 10 == 0)
    max_dec = _max_decade(rows, cols)

    window_cells = _window_cells(rows, cols, cand_row, cand_col, radius=2)
    window_values = _known_values(board, window_cells)
    row_values = _known_values(board, [(cand_row, cc) for cc in range(cols)])
    col_values = _known_values(board, [(rr, cand_col) for rr in range(rows)])
    neighbor_cells = list(_neighbors(rows, cols, cand_row, cand_col))
    neighbor_values = _known_values(board, neighbor_cells)

    row_tail = [_tail(v) for v in row_values]
    col_tail = [_tail(v) for v in col_values]
    win_tail = [_tail(v) for v in window_values]

    row_decade = [_decade(v) for v in row_values]
    col_decade = [_decade(v) for v in col_values]
    win_decade = [_decade(v) for v in window_values]

    row_tail_counts = Counter(row_tail)
    col_tail_counts = Counter(col_tail)
    win_tail_counts = Counter(win_tail)

    same_tail_local = sum(1 for v in window_values if _tail(v) == target_tail)
    same_decade_local = sum(1 for v in window_values if _decade(v) == target_decade)

    local_tail_ratio = same_tail_local / max(len(window_values), 1)
    global_tail_ratio = row_tail_counts.get(target_tail, 0) / max(len(row_tail), 1)

    diffs = [abs(v - target_number) for v in window_values]
    diff_bins = (1, 2, 5, 10, 20)

    directional = {
        "up": _directional_neighbor(rows, cols, cand_row, cand_col, -1, 0),
        "down": _directional_neighbor(rows, cols, cand_row, cand_col, 1, 0),
        "left": _directional_neighbor(rows, cols, cand_row, cand_col, 0, -1),
        "right": _directional_neighbor(rows, cols, cand_row, cand_col, 0, 1),
    }

    neighbor_same_tail = 0
    neighbor_multiple10 = 0
    near_multiple10 = 0
    direction_out: Dict[str, float] = {}
    for name, cell in directional.items():
        if cell is None:
            direction_out[f"neighbor_residue_{name}_same_tail"] = 0.0
            direction_out[f"neighbor_multiple10_{name}_is_multiple10"] = 0.0
            direction_out[f"neighbor_multiple10_{name}_abs_delta"] = 0.0
            continue
        v = int(board[cell[0]][cell[1]])
        if v == -1:
            direction_out[f"neighbor_residue_{name}_same_tail"] = 0.0
            direction_out[f"neighbor_multiple10_{name}_is_multiple10"] = 0.0
            direction_out[f"neighbor_multiple10_{name}_abs_delta"] = 0.0
            continue
        is_same_tail = int(_tail(v) == target_tail)
        is_mul10 = int(v % 10 == 0)
        neighbor_same_tail += is_same_tail
        neighbor_multiple10 += is_mul10
        near_multiple10 += int(min(v % 10, (10 - v % 10) % 10) <= 1)
        direction_out[f"neighbor_residue_{name}_same_tail"] = float(is_same_tail)
        direction_out[f"neighbor_multiple10_{name}_is_multiple10"] = float(is_mul10)
        direction_out[f"neighbor_multiple10_{name}_abs_delta"] = float(abs(v - target_number))

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
            same_tail_adjacent += int(_tail(v) == _tail(u))
            same_decade_adjacent += int(_decade(v) == _decade(u))
            same_tail_same_decade += int(_tail(v) == _tail(u) and _decade(v) == _decade(u))
            consecutive_pairs += int(abs(v - u) == 1)

    row_band = min((cand_row * 3) // max(rows, 1), 2)
    col_band = min((cand_col * 3) // max(cols, 1), 2)
    local_density = len(window_values) / max(len(window_cells), 1)

    out: Dict[str, float] = {
        "cand_row": float(cand_row + 1),
        "cand_col": float(cand_col + 1),
        "relative_row": float((cand_row + 1) / max(rows, 1)),
        "relative_col": float((cand_col + 1) / max(cols, 1)),
        "relative_center_distance": _relative_center_distance(rows, cols, cand_row, cand_col),
        "relative_edge_distance": _relative_edge_distance(rows, cols, cand_row, cand_col),
        "is_feasible": float(int(prev_value == -1)),
        "residue_same_tail_count_local5x5": float(same_tail_local),
        "residue_same_tail_ratio_local5x5": float(local_tail_ratio),
        "residue_same_tail_adjacent_count": float(same_tail_adjacent),
        "neighbor_residue_same_tail_count": float(neighbor_same_tail),
        "neighbor_residue_same_tail_ratio": float(neighbor_same_tail / max(len(neighbor_values), 1)),
        "neighbor_multiple10_count": float(neighbor_multiple10),
        "neighbor_multiple10_near_multiple10_count": float(near_multiple10),
        "multiple10_is_multiple_of_10": float(target_is_multiple10),
        "multiple10_dist_to_nearest_multiple_of_10": float(
            abs(target_number - _nearest_multiple_of_10(target_number))
        ),
        "multiple10_signed_delta_to_nearest_multiple_of_10": float(
            _nearest_multiple_of_10(target_number) - target_number
        ),
        "local5x5_multiple10_ratio": float(sum(1 for v in window_values if v % 10 == 0) / max(len(window_values), 1)),
        "row_multiple10_ratio": float(sum(1 for v in row_values if v % 10 == 0) / max(len(row_values), 1)),
        "col_multiple10_ratio": float(sum(1 for v in col_values if v % 10 == 0) / max(len(col_values), 1)),
        "local5x5_same_decade_count": float(same_decade_local),
        "local5x5_consecutive_pair_count": float(consecutive_pairs),
        "local5x5_same_decade_adjacent_count": float(same_decade_adjacent),
        "local5x5_same_tail_and_same_decade_count": float(same_tail_same_decade),
        "residue_interaction_tail_x_decade": float(target_tail * 10 + target_decade),
        "residue_interaction_tail_x_row_band": float(target_tail * 10 + row_band),
        "residue_interaction_tail_x_col_band": float(target_tail * 10 + col_band),
        "residue_interaction_tail_x_local_density": float(target_tail * local_density),
        "residue_interaction_multiple10_x_same_tail_ratio": float(target_is_multiple10 * local_tail_ratio),
        "residue_interaction_target_tail_x_window_tail_mode": float(
            target_tail * float(Counter(win_tail).most_common(1)[0][0] if win_tail else -1)
        ),
        "residue_interaction_target_decade_x_window_decade_mode": float(
            target_decade * float(Counter(win_decade).most_common(1)[0][0] if win_decade else -1)
        ),
        "residue_interaction_local_tail_ratio_lift": float(local_tail_ratio - global_tail_ratio),
        "residue_interaction_local_same_decade_ratio_lift": float(
            same_decade_local / max(len(window_values), 1)
            - Counter(row_decade).get(target_decade, 0) / max(len(row_decade), 1)
        ),
    }
    out.update(direction_out)

    out.update(_hist_summary("row_residue", row_tail))
    out.update(_hist_summary("col_residue", col_tail))
    out.update(_hist_summary("local5x5_residue", win_tail))
    out.update(_hist_summary("row_decade", row_decade))
    out.update(_hist_summary("col_decade", col_decade))
    out.update(_hist_summary("local5x5_decade", win_decade))

    out.update(_dynamic_hist("row_decade", row_decade, max_dec))
    out.update(_dynamic_hist("col_decade", col_decade, max_dec))
    out.update(_dynamic_hist("local5x5_decade", win_decade, max_dec))

    for t in range(10):
        out[f"row_residue_hist_{t}"] = float(row_tail_counts.get(t, 0) / max(len(row_tail), 1))
        out[f"col_residue_hist_{t}"] = float(col_tail_counts.get(t, 0) / max(len(col_tail), 1))
        out[f"local5x5_residue_hist_{t}"] = float(win_tail_counts.get(t, 0) / max(len(win_tail), 1))

    for b in diff_bins:
        out[f"local5x5_count_abs_delta_eq_{b}"] = float(sum(1 for d in diffs if d == b))
        out[f"local5x5_count_abs_delta_le_{b}"] = float(sum(1 for d in diffs if d <= b))

    return out


def merge_feature_layers(board_state: Dict[str, float], candidate_delta: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out.update({f"board_state_{k}": float(v) for k, v in board_state.items()})
    out.update({f"candidate_delta_{k}": float(v) for k, v in candidate_delta.items()})
    return out


def euclidean(values: List[float]) -> float:
    return float(sqrt(sum(v * v for v in values)))
