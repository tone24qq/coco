"""Guard helpers for scratch card predictions."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np


def ensure_only_blank(board: np.ndarray, preds: Iterable[Any], blank_value: int = -1):
    """Filter predictions to ensure returned positions are blank."""
    rows, cols = board.shape
    flat = board.ravel()
    out = []
    bad = []
    for p in preds:
        r = p["row"] if isinstance(p, dict) else getattr(p, "row")
        c = p["col"] if isinstance(p, dict) else getattr(p, "col")
        idx = r * cols + c
        if 0 <= idx < flat.size and flat[idx] == blank_value:
            out.append(p)
        else:
            bad.append((r, c))
    if bad:
        logging.getLogger(__name__).error("[GUARD] filtered non-blank cells: %s", bad)
    return out
