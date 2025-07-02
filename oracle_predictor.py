"""Oracle predictor leveraging known board numbering pattern."""

from __future__ import annotations

from typing import Dict, List


def predict_target_location(grid: List[List[int]], target: int) -> Dict[str, int]:
    """Return the exact cell location for ``target``.

    This assumes the board numbers are arranged sequentially from 1 to N in
    row-major order as produced by :func:`modules.generate_unique_grid`.
    """
    if not grid or not grid[0]:
        raise ValueError("Grid must be non-empty")

    rows, cols = len(grid), len(grid[0])
    r = (target - 1) // cols
    c = (target - 1) % cols
    if r < 0 or r >= rows or c < 0 or c >= cols:
        raise ValueError("target out of range for given grid size")
    return {"row": r, "col": c}
