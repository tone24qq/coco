from __future__ import annotations

from typing import Dict, List, Optional


def build_value_to_position(
    grid: List[List[Optional[int]]],
) -> Dict[str, List[Dict[str, int]]]:
    out: Dict[str, List[Dict[str, int]]] = {}
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v is None:
                continue
            key = str(int(v))
            out.setdefault(key, []).append(
                {
                    "row_1based": r + 1,
                    "col_1based": c + 1,
                    "row_0based": r,
                    "col_0based": c,
                }
            )
    return out


def find_number_positions(
    grid: List[List[Optional[int]]], value: int
) -> Dict[str, object]:
    value_to_position = build_value_to_position(grid)
    positions = value_to_position.get(str(value), [])
    return {
        "value": value,
        "positions": positions,
        "found": len(positions) > 0,
        "contract_violation": len(positions) > 1,
        "status": "found" if positions else "not_found",
    }
