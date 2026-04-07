from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .ticket_specs import TicketSpec


@dataclass
class SolverResult:
    grid: List[List[Optional[int]]]
    pending_cells: List[Dict[str, object]]
    consistency_confidence: float
    duplicates: List[int]
    missing_values: List[int]


def solve_board(
    grid: List[List[Optional[int]]],
    cell_candidates: Dict[tuple[int, int], List[int]],
    cell_labels: Dict[tuple[int, int], str],
    spec: TicketSpec,
) -> SolverResult:
    legal = spec.legal_values
    out = [row[:] for row in grid]
    pending: List[Dict[str, object]] = []

    seen: Dict[int, tuple[int, int]] = {}
    duplicates: List[int] = []
    for r, row in enumerate(out):
        for c, v in enumerate(row):
            if v is None:
                continue
            if v not in legal:
                out[r][c] = None
                pending.append({"row": r, "col": c, "reason": "illegal_value"})
                continue
            if v in seen:
                duplicates.append(v)
                out[r][c] = None
                pending.append({"row": r, "col": c, "reason": "duplicate_value"})
            else:
                seen[v] = (r, c)

    present = {v for row in out for v in row if v is not None}
    missing = sorted(list(legal - present))

    for r, row in enumerate(out):
        for c, v in enumerate(row):
            label = cell_labels.get((r, c), "printed_number")
            if label == "solid_black":
                out[r][c] = None
                continue
            if v is not None:
                continue
            cands = [x for x in cell_candidates.get((r, c), []) if x in missing]
            if len(cands) == 1:
                out[r][c] = cands[0]
                missing.remove(cands[0])
            else:
                pending.append({"row": r, "col": c, "reason": "pending_review", "candidates": cands[:3]})

    final_present = {v for row in out for v in row if v is not None}
    final_missing = sorted(list(legal - final_present))
    conf = max(0.0, min(1.0, 1.0 - len(final_missing) / max(len(legal), 1)))
    return SolverResult(
        grid=out,
        pending_cells=pending,
        consistency_confidence=conf,
        duplicates=sorted(set(duplicates)),
        missing_values=final_missing,
    )
