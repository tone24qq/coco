from __future__ import annotations

from collections import Counter
from math import log2, sqrt
from typing import Dict, Iterable, List, Tuple


Cell = Tuple[int, int]


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


def compute_board_state_features(board: List[List[int]], target_number: int) -> Dict[str, float]:
    rows = len(board)
    cols = len(board[0]) if rows else 0
    known_cells: List[Cell] = []
    unknown_cells: List[Cell] = []
    known_values: List[int] = []

    for r in range(rows):
        for c in range(cols):
            v = int(board[r][c])
            if v == -1:
                unknown_cells.append((r, c))
            else:
                known_cells.append((r, c))
                known_values.append(v)

    known_ratio = len(known_cells) / max(rows * cols, 1)
    row_known = [sum(1 for v in row if v != -1) / max(cols, 1) for row in board]
    col_known = [sum(1 for r in range(rows) if board[r][c] != -1) / max(rows, 1) for c in range(cols)]

    tail_counts = Counter(_tail(v) for v in known_values)
    tail_entropy = _entropy([_tail(v) for v in known_values])

    same_tail_edges = 0
    same_decade_edges = 0
    consecutive_edges = 0
    edge_count = 0
    for r in range(rows):
        for c in range(cols):
            v = board[r][c]
            if v == -1:
                continue
            for rr, cc in _neighbors(rows, cols, r, c):
                if (rr, cc) < (r, c):
                    continue
                u = board[rr][cc]
                if u == -1:
                    continue
                edge_count += 1
                same_tail_edges += int(_tail(v) == _tail(u))
                same_decade_edges += int(_decade(v) == _decade(u))
                consecutive_edges += int(abs(v - u) == 1)

    edge_cells = [(r, c) for r in range(rows) for c in range(cols) if r in (0, rows - 1) or c in (0, cols - 1)]
    center_cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in set(edge_cells)]
    edge_known = sum(1 for r, c in edge_cells if board[r][c] != -1)
    center_known = sum(1 for r, c in center_cells if board[r][c] != -1)

    return {
        "board_rows": float(rows),
        "board_cols": float(cols),
        "board_size": float(rows * cols),
        "target_number_norm": float(target_number / max(rows * cols, 1)),
        "known_ratio": float(known_ratio),
        "unknown_ratio": float(1.0 - known_ratio),
        "row_known_entropy": float(_entropy([int(v * 1000) for v in row_known])),
        "col_known_entropy": float(_entropy([int(v * 1000) for v in col_known])),
        "tail_entropy": float(tail_entropy),
        "tail_hist_max_ratio": float(max(tail_counts.values(), default=0) / max(len(known_values), 1)),
        "same_tail_adjacency_rate": float(same_tail_edges / max(edge_count, 1)),
        "same_decade_proximity_rate": float(same_decade_edges / max(edge_count, 1)),
        "consecutive_neighbor_rate": float(consecutive_edges / max(edge_count, 1)),
        "edge_center_balance": float(edge_known / max(edge_known + center_known, 1)),
    }


def compute_candidate_delta_features(
    board: List[List[int]],
    target_number: int,
    cand_row: int,
    cand_col: int,
    board_state_features: Dict[str, float],
) -> Dict[str, float]:
    rows = len(board)
    cols = len(board[0]) if rows else 0
    cloned = [row[:] for row in board]
    prev_value = cloned[cand_row][cand_col]
    cloned[cand_row][cand_col] = target_number
    placed = compute_board_state_features(cloned, target_number=target_number)

    delta: Dict[str, float] = {}
    for key, value in placed.items():
        if key in ("board_rows", "board_cols", "board_size", "target_number_norm"):
            continue
        delta[f"delta_{key}"] = float(value - board_state_features.get(key, 0.0))

    is_feasible = int(prev_value == -1)
    rel_row = (cand_row + 1) / max(rows, 1)
    rel_col = (cand_col + 1) / max(cols, 1)

    return {
        "cand_row": float(cand_row + 1),
        "cand_col": float(cand_col + 1),
        "relative_row": float(rel_row),
        "relative_col": float(rel_col),
        "relative_center_distance": _relative_center_distance(rows, cols, cand_row, cand_col),
        "relative_edge_distance": _relative_edge_distance(rows, cols, cand_row, cand_col),
        "is_feasible": float(is_feasible),
        **delta,
    }


def merge_feature_layers(board_state: Dict[str, float], candidate_delta: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out.update({f"board_state_{k}": float(v) for k, v in board_state.items()})
    out.update({f"candidate_delta_{k}": float(v) for k, v in candidate_delta.items()})
    return out


def euclidean(values: List[float]) -> float:
    return float(sqrt(sum(v * v for v in values)))
