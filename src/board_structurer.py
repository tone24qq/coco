from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cell_classifier import classify_cell_empty_or_filled
from .cell_digit_reader import read_cell_digit
from .grid_detector import GridDetection


@dataclass
class BoardParseResult:
    sample_id: str
    shape: str
    row_count: int
    col_count: int
    grid: List[List[Optional[int]]]
    cell_confidence: List[List[float]]
    low_confidence_cells: List[Dict[str, object]]
    parse_confidence: float
    image_path: str
    ticket_type: str | None


def _cell_crop(board: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    pad_y = max(1, (y1 - y0) // 10)
    pad_x = max(1, (x1 - x0) // 10)
    return board[y0 + pad_y : y1 - pad_y, x0 + pad_x : x1 - pad_x]


def structure_board(
    sample_id: str,
    image_path: str,
    detection: GridDetection,
    ticket_type: str | None = None,
) -> BoardParseResult:
    rows, cols = detection.row_count, detection.col_count
    max_value = rows * cols
    grid: List[List[Optional[int]]] = [[None for _ in range(cols)] for _ in range(rows)]
    conf: List[List[float]] = [[0.0 for _ in range(cols)] for _ in range(rows)]
    low: List[Dict[str, object]] = []

    for r in range(rows):
        for c in range(cols):
            y0, y1 = detection.row_lines[r], detection.row_lines[r + 1]
            x0, x1 = detection.col_lines[c], detection.col_lines[c + 1]
            cell = _cell_crop(detection.board_image, y0, y1, x0, x1)
            filled = classify_cell_empty_or_filled(cell)
            if not filled.is_filled:
                conf[r][c] = filled.confidence
                continue
            digit = read_cell_digit(cell, max_value=max_value)
            comb_conf = 0.5 * filled.confidence + 0.5 * digit.confidence
            conf[r][c] = comb_conf
            if digit.value is None:
                low.append({"row": r, "col": c, "reason": "digit_low_confidence"})
                continue
            grid[r][c] = int(digit.value)

    # structural correction: duplicates beyond one occurrence become low confidence nulls
    seen: Dict[int, Tuple[int, int, float]] = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v is None:
                continue
            if v < 1 or v > max_value:
                grid[r][c] = None
                low.append({"row": r, "col": c, "reason": "out_of_range"})
                continue
            if v in seen:
                pr, pc, pconf = seen[v]
                if conf[r][c] > pconf:
                    grid[pr][pc] = None
                    low.append({"row": pr, "col": pc, "reason": "duplicate_value"})
                    seen[v] = (r, c, conf[r][c])
                else:
                    grid[r][c] = None
                    low.append({"row": r, "col": c, "reason": "duplicate_value"})
            else:
                seen[v] = (r, c, conf[r][c])

    filled_ratio = float(np.mean([[1.0 if x is not None else 0.0 for x in row] for row in grid]))
    mean_conf = float(np.mean(np.array(conf))) if rows and cols else 0.0
    parse_conf = max(0.0, min(1.0, 0.5 * detection.confidence + 0.3 * mean_conf + 0.2 * filled_ratio))

    return BoardParseResult(
        sample_id=sample_id,
        shape=f"{rows}x{cols}",
        row_count=rows,
        col_count=cols,
        grid=grid,
        cell_confidence=conf,
        low_confidence_cells=low,
        parse_confidence=parse_conf,
        image_path=image_path,
        ticket_type=ticket_type,
    )
