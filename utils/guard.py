"""Guard helpers for scratch card predictions."""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np

from dataset import BLANK_VALUE


def ensure_only_blank(
    board: np.ndarray, preds: Iterable[Any], blank_value: int = BLANK_VALUE
) -> List[Any]:
    """Filter predictions to ensure returned positions are blank."""
    rows, cols = board.shape
    flat = board.ravel()
    out: List[Any] = []
    bad: List[Sequence[int]] = []
    for p in preds:
        r = p["row"] if isinstance(p, dict) else getattr(p, "row")
        c = p["col"] if isinstance(p, dict) else getattr(p, "col")
        idx = r * cols + c
        if 0 <= idx < flat.size and flat[idx] == blank_value:
            out.append(p)
        else:
            bad.append((r, c))
    if bad:
        logging.getLogger(__name__).error(
            "[GUARD] filtered non-blank cells: %s values=%s",
            bad,
            [int(flat[r * cols + c]) for (r, c) in bad],
        )
    return out


def index_to_coord(idx: int, shape: Tuple[int, int]) -> Tuple[int, int]:
    """Return ``(row, col)`` coordinate for a flattened ``idx``.

    Parameters
    ----------
    idx:
        Flat index in ``0``..``rows*cols-1``.
    shape:
        ``(rows, cols)`` shape of the board.

    Returns
    -------
    tuple[int, int]
        Row and column index corresponding to ``idx``.
    """

    r, c = np.unravel_index(idx, shape)
    return int(r), int(c)
