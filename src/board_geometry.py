from __future__ import annotations

from typing import List, Optional, Tuple

Cell = Tuple[int, int]


def main_diagonal_cells(rows: int, cols: int) -> List[Cell]:
    limit = min(rows, cols)
    return [(i, i) for i in range(limit)]


def anti_diagonal_cells(rows: int, cols: int) -> List[Cell]:
    limit = min(rows, cols)
    return [(i, cols - 1 - i) for i in range(limit)]


def cell_on_main_diagonal(cell: Cell, rows: int, cols: int) -> bool:
    r, c = cell
    return 0 <= r < rows and 0 <= c < cols and r == c and r < min(rows, cols)


def cell_on_anti_diagonal(cell: Cell, rows: int, cols: int) -> bool:
    r, c = cell
    return 0 <= r < rows and 0 <= c < cols and (r + c == cols - 1) and r < min(rows, cols)


def relative_rank_in_line(cells: List[Cell], cell: Cell) -> Optional[int]:
    try:
        return cells.index(cell)
    except ValueError:
        return None
