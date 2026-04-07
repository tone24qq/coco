from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .board_query import build_value_to_position
from .ticket_specs import TicketSpec


@dataclass
class ContractResult:
    status: str
    contract_passed: bool
    needs_manual_review: bool
    shape_ok: bool
    confidence_ok: bool
    low_conf_ok: bool
    complete_grid: bool
    missing_values: List[int]
    duplicate_values: List[int]
    rows: int
    cols: int


def evaluate_board_contract(
    grid: List[List[Optional[int]]],
    spec: TicketSpec,
    parse_confidence: float,
    low_confidence_cells: List[Dict[str, object]],
    min_confidence: float = 0.55,
    max_low_conf_cells: int = 8,
    strict: bool = False,
) -> ContractResult:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    shape_ok = (rows, cols) == spec.expected_shape
    numbers = [int(v) for row in grid for v in row if v is not None]
    legal = spec.legal_values
    illegal = sorted({v for v in numbers if v not in legal})
    dupes = sorted({v for v in numbers if numbers.count(v) > 1})
    missing = sorted(list(legal - set(numbers)))
    complete = len(missing) == 0
    confidence_ok = parse_confidence >= min_confidence
    low_conf_ok = len(low_confidence_cells) <= max_low_conf_cells

    if not shape_ok:
        status = "shape_mismatch"
    elif not confidence_ok:
        status = "low_confidence_parse"
    elif strict and (dupes or illegal):
        status = "contract_violation"
    elif strict and not complete:
        status = "incomplete_grid"
    elif not low_conf_ok:
        status = "needs_manual_review"
    else:
        status = "ok"

    needs_review = status in (
        "shape_mismatch",
        "low_confidence_parse",
        "incomplete_grid",
        "needs_manual_review",
        "contract_violation",
    )
    return ContractResult(
        status=status,
        contract_passed=(status == "ok"),
        needs_manual_review=needs_review,
        shape_ok=shape_ok,
        confidence_ok=confidence_ok,
        low_conf_ok=low_conf_ok,
        complete_grid=complete,
        missing_values=missing,
        duplicate_values=dupes + illegal,
        rows=rows,
        cols=cols,
    )


def build_output_schema(
    *,
    status: str,
    source_mode: str,
    shape: str,
    grid: List[List[Optional[int]]],
    black_cells: List[Dict[str, int]],
    low_confidence_cells: List[Dict[str, object]],
    parse_confidence: float,
    contract: ContractResult,
    parse_diagnostics: Dict[str, object],
) -> Dict[str, object]:
    value_to_position = build_value_to_position(grid)
    numbers_all = sorted([int(v) for row in grid for v in row if v is not None])
    return {
        "status": status,
        "source_mode": source_mode,
        "shape": shape,
        "rows": contract.rows,
        "cols": contract.cols,
        "grid": grid,
        "numbers_all": numbers_all,
        "value_to_position": value_to_position,
        "black_cells": black_cells,
        "missing_values": contract.missing_values,
        "legal_value_min": 1,
        "legal_value_max": contract.rows * contract.cols,
        "low_confidence_cells": low_confidence_cells,
        "parse_confidence": parse_confidence,
        "contract_passed": contract.contract_passed,
        "needs_manual_review": contract.needs_manual_review,
        "parse_diagnostics": parse_diagnostics,
    }
