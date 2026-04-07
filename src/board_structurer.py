from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .board_solver import solve_board
from .cell_classifier import classify_cell
from .cell_digit_reader import read_cell_digit
from .grid_detector import GridDetection
from .ticket_specs import TicketSpec


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
    board_confidence: float
    warp_confidence: float
    shape_confidence: float
    cell_class_confidence_mean: float
    digit_confidence_mean: float
    global_consistency_confidence: float
    final_parse_confidence: float
    numbers_all: List[int]
    value_to_position: Dict[str, List[Dict[str, int]]]
    missing_values: List[int]
    black_cells: List[Dict[str, int]]
    pending_cells: List[Dict[str, object]]
    parse_diagnostics: Dict[str, object]


class BoardStructureError(ValueError):
    pass


def _cell_crop(board: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    pad_y = max(1, (y1 - y0) // 10)
    pad_x = max(1, (x1 - x0) // 10)
    return board[y0 + pad_y : y1 - pad_y, x0 + pad_x : x1 - pad_x]


def find_number_positions(
    grid: List[List[Optional[int]]], value: int
) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v == value:
                out.append({"row": r + 1, "col": c + 1, "row0": r, "col0": c})
    return out


def structure_board(
    sample_id: str,
    image_path: str,
    detection: GridDetection,
    spec: TicketSpec,
    ticket_type: str | None = None,
) -> BoardParseResult:
    rows, cols = detection.row_count, detection.col_count
    if (rows, cols) != spec.expected_shape:
        raise BoardStructureError("shape_mismatch")
    max_value = rows * cols
    grid: List[List[Optional[int]]] = [[None for _ in range(cols)] for _ in range(rows)]
    conf: List[List[float]] = [[0.0 for _ in range(cols)] for _ in range(rows)]
    low: List[Dict[str, object]] = []
    class_conf: List[float] = []
    digit_conf: List[float] = []
    black_cells: List[Dict[str, int]] = []
    cell_candidates: Dict[tuple[int, int], List[int]] = {}
    cell_labels: Dict[tuple[int, int], str] = {}
    ocr_backend = "fallback_template"

    for r in range(rows):
        for c in range(cols):
            y0, y1 = detection.row_lines[r], detection.row_lines[r + 1]
            x0, x1 = detection.col_lines[c], detection.col_lines[c + 1]
            cell = _cell_crop(detection.board_image, y0, y1, x0, x1)
            cls = classify_cell(cell)
            class_conf.append(cls.confidence)
            cell_labels[(r, c)] = cls.label
            conf[r][c] = cls.confidence
            if cls.label == "blank":
                continue
            if cls.label == "solid_black":
                black_cells.append({"row": r + 1, "col": c + 1})
                continue
            digit = read_cell_digit(cell, max_value=max_value)
            ocr_backend = digit.ocr_backend
            digit_conf.append(digit.confidence)
            cell_candidates[(r, c)] = [int(x["value"]) for x in digit.top_candidates]
            conf[r][c] = 0.4 * cls.confidence + 0.6 * digit.confidence
            if digit.value is None:
                low.append(
                    {
                        "row": r,
                        "col": c,
                        "reason": "digit_low_confidence",
                        "needs_review": True,
                    }
                )
                continue
            grid[r][c] = int(digit.value)
            if cls.needs_review:
                low.append(
                    {
                        "row": r,
                        "col": c,
                        "reason": "classification_low_confidence",
                        "needs_review": True,
                    }
                )

    solved = solve_board(
        grid=grid, cell_candidates=cell_candidates, cell_labels=cell_labels, spec=spec
    )
    final_grid = solved.grid
    low.extend(solved.pending_cells)

    numbers_all = sorted([int(v) for row in final_grid for v in row if v is not None])
    value_to_position: Dict[str, List[Dict[str, int]]] = {}
    for v in numbers_all:
        key = str(v)
        if key not in value_to_position:
            value_to_position[key] = find_number_positions(final_grid, v)

    c_class = float(np.mean(class_conf)) if class_conf else 0.0
    d_conf = float(np.mean(digit_conf)) if digit_conf else 0.0
    mean_conf = float(np.mean(np.array(conf))) if rows and cols else 0.0
    final_parse_conf = max(
        0.0,
        min(
            1.0,
            0.16 * detection.board_confidence
            + 0.16 * detection.warp_confidence
            + 0.16 * detection.shape_confidence
            + 0.16 * c_class
            + 0.16 * d_conf
            + 0.20 * solved.consistency_confidence,
        ),
    )

    return BoardParseResult(
        sample_id=sample_id,
        shape=f"{rows}x{cols}",
        row_count=rows,
        col_count=cols,
        grid=final_grid,
        cell_confidence=conf,
        low_confidence_cells=low,
        parse_confidence=mean_conf,
        image_path=image_path,
        ticket_type=ticket_type,
        board_confidence=detection.board_confidence,
        warp_confidence=detection.warp_confidence,
        shape_confidence=detection.shape_confidence,
        cell_class_confidence_mean=c_class,
        digit_confidence_mean=d_conf,
        global_consistency_confidence=solved.consistency_confidence,
        final_parse_confidence=final_parse_conf,
        numbers_all=numbers_all,
        value_to_position=value_to_position,
        missing_values=solved.missing_values,
        black_cells=black_cells,
        pending_cells=solved.pending_cells,
        parse_diagnostics={"ocr_backend": ocr_backend, "duplicates": solved.duplicates},
    )
