from __future__ import annotations

from typing import Dict, List, Optional, Set

from .board_query import find_number_positions


def predict_number_positions(
    grid: List[List[Optional[int]]],
    query_number: int,
    missing_values: List[int],
    low_confidence_cells: List[Dict[str, object]],
    black_cells: List[Dict[str, int]],
    manual_override_cells: Set[tuple[int, int]] | None = None,
) -> Dict[str, object]:
    exact = find_number_positions(grid, query_number)
    if exact["found"] and not exact["contract_violation"]:
        return {
            "query_number": query_number,
            "query_status": "exact_found",
            "exact_positions": exact["positions"],
            "top5_position_candidates": [
                {
                    **exact["positions"][0],
                    "score": 1.0,
                    "reason": "exact_match",
                }
            ],
        }

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    black = {(x["row"], x["col"]) for x in black_cells}
    low = {
        (int(x.get("row", -1)) + 1, int(x.get("col", -1)) + 1)
        for x in low_confidence_cells
    }
    manual = manual_override_cells or set()

    candidates = []
    if query_number not in missing_values and not exact["found"]:
        return {
            "query_number": query_number,
            "query_status": "not_possible",
            "exact_positions": [],
            "top5_position_candidates": [],
        }

    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            if (r, c) in black:
                continue
            v = grid[r - 1][c - 1]
            if v is not None and v != query_number:
                continue
            score = 0.2
            reason = ["candidate_cell"]
            if (r, c) in low:
                score += 0.5
                reason.append("low_confidence_replaceable")
            if v is None:
                score += 0.2
                reason.append("empty_cell")
            if (r, c) in manual:
                score -= 0.3
                reason.append("manual_override_penalty")
            candidates.append(
                {
                    "row_1based": r,
                    "col_1based": c,
                    "row_0based": r - 1,
                    "col_0based": c - 1,
                    "score": max(0.0, min(1.0, score)),
                    "reason": ",".join(reason),
                }
            )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return {
        "query_number": query_number,
        "query_status": "predicted" if candidates else "not_possible",
        "exact_positions": exact["positions"],
        "top5_position_candidates": candidates[:5],
    }
