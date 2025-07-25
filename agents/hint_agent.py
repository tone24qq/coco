"""Hint agent for Sudoku-like puzzles."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from coco_common.csp_utils import get_subgrid_indices


def predict(board: np.ndarray, target: int, **kwargs: Any) -> List[Dict[str, Any]]:
    """Return legal positions for ``target`` without solving the puzzle."""
    n = board.shape[0]
    sub = int(np.sqrt(n))
    digits = set(range(1, n + 1))

    predictions: List[Dict[str, Any]] = []
    for r in range(n):
        for c in range(n):
            if board[r, c] != -1:
                continue
            if target not in digits:
                continue
            # row and column check
            if target in board[r] or target in board[:, c]:
                continue
            sg_r, sg_c = get_subgrid_indices(r, c, sub)
            if target in board[sg_r : sg_r + sub, sg_c : sg_c + sub]:
                continue
            predictions.append({"row": r, "col": c, "score": 1.0})
    return predictions
