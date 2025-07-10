"""Sample-based neighbor and line statistics."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

BLANK_VAL = -1


def _matches_neighbor(grid: np.ndarray, board: np.ndarray, r: int, c: int) -> bool:
    rows, cols = grid.shape
    r0, r1 = max(0, r - 1), min(rows, r + 2)
    c0, c1 = max(0, c - 1), min(cols, c + 2)
    sub_mask = grid[r0:r1, c0:c1] != BLANK_VAL
    return np.array_equal(board[r0:r1, c0:c1][sub_mask], grid[r0:r1, c0:c1][sub_mask])


def _matches_line(grid: np.ndarray, board: np.ndarray, r: int, c: int) -> bool:
    """Return ``True`` if row or column known values match."""

    row_mask = grid[r] != BLANK_VAL
    col_mask = grid[:, c] != BLANK_VAL

    row_ok = not row_mask.any() or np.array_equal(board[r][row_mask], grid[r][row_mask])
    col_ok = not col_mask.any() or np.array_equal(
        board[:, c][col_mask], grid[:, c][col_mask]
    )

    if row_mask.any() and col_mask.any():
        return row_ok or col_ok
    return row_ok and col_ok


def compute_neighbor_line_stats(
    grid: np.ndarray,
    target_num: int,
    *,
    samples_dir: str = "samples",
    enable_neighbor_match: bool = True,
    enable_line_match: bool = True,
    weight_neighbor: float = 0.5,
    weight_line: float = 0.5,
) -> np.ndarray:
    """Return score map using sample-based neighbor and line statistics."""

    if not enable_neighbor_match and not enable_line_match:
        raise ValueError("At least one matching mode must be enabled")

    arr = np.asarray(grid, dtype=int)
    rows, cols = arr.shape
    blanks = np.argwhere(arr == BLANK_VAL)
    if blanks.size == 0:
        return np.zeros((rows, cols), dtype=float)

    from analyzer import _load_samples_for_shape

    boards: List[Tuple[np.ndarray, str]] = _load_samples_for_shape(
        samples_dir, rows, cols
    )
    if not boards:
        return np.zeros((rows, cols), dtype=float)

    neigh_counts = np.zeros((rows, cols), dtype=float)
    line_counts = np.zeros((rows, cols), dtype=float)
    match_counts = np.zeros((rows, cols), dtype=int)

    for board, _ in boards:
        for br, bc in blanks:
            if enable_neighbor_match and not _matches_neighbor(arr, board, br, bc):
                continue
            if enable_line_match and not _matches_line(arr, board, br, bc):
                continue
            r0, r1 = max(0, br - 1), min(rows, br + 2)
            c0, c1 = max(0, bc - 1), min(cols, bc + 2)
            neigh_counts[br, bc] += np.count_nonzero(board[r0:r1, c0:c1] == target_num)
            row_count = np.count_nonzero(board[br] == target_num)
            col_count = np.count_nonzero(board[:, bc] == target_num)
            if board[br, bc] == target_num:
                col_count -= 1
            line_counts[br, bc] += row_count + col_count
            match_counts[br, bc] += 1

    for r, c in blanks:
        if match_counts[r, c] > 0:
            neigh_counts[r, c] /= float(match_counts[r, c])
            line_counts[r, c] /= float(match_counts[r, c])

    if neigh_counts.max() > 0:
        neigh_counts /= float(neigh_counts.max())
    if line_counts.max() > 0:
        line_counts /= float(line_counts.max())

    score = weight_neighbor * neigh_counts + weight_line * line_counts
    mx = score.max(initial=0.0)
    return score / mx if mx > 0 else score
