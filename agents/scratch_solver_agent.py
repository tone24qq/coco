"""Scratch-card solver agent with simple heuristics.

This agent assumes numbers on the board are unique within the range
1..N*M (where ``N`` and ``M`` are board dimensions). Blank cells are
represented by ``-1``. The agent predicts positions of a target number
by assigning equal score to all blank cells when the target is hidden
or returning the revealed location when the target is already visible.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def predict(board: np.ndarray, target: int, **_: Any) -> List[Dict[str, Any]]:
    """Predict candidate positions for ``target``.

    Parameters
    ----------
    board:
        2D numpy array with ``-1`` for hidden cells. Values must be unique
        and in the range ``1`` .. ``rows*cols``.
    target:
        The number to locate.

    Returns
    -------
    list of dict
        Each dict contains ``row``, ``col`` and ``score`` keys.
    """
    if board.ndim != 2:
        raise ValueError("board must be 2D")

    rows, cols = board.shape
    max_val = rows * cols

    numbers = board[board != -1]
    if numbers.size != len(np.unique(numbers)):
        raise ValueError("board numbers must be unique")
    if numbers.size and (numbers.min() < 1 or numbers.max() > max_val):
        raise ValueError("board numbers out of range")

    # target already revealed
    locations = np.argwhere(board == target)
    if locations.size:
        return [{"row": int(r), "col": int(c), "score": 1.0} for r, c in locations]

    blanks = np.argwhere(board == -1)
    if blanks.size == 0:
        return []

    score = 1.0 / len(blanks)
    return [{"row": int(r), "col": int(c), "score": float(score)} for r, c in blanks]
