"""Utilities for loading board datasets from NPZ archives."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def load_boards_from_npz(npz_path: str | Path) -> List[List[List[int]]]:
    """Load boards from an ``.npz`` file.

    The archive must contain an array named ``boards`` with shape
    ``(n, rows, cols)``. The function will return a list of boards as nested
    Python lists.
    """

    path = Path(npz_path)
    with np.load(path) as data:
        if "boards" not in data:
            raise KeyError(f"'boards' not found in {path}")
        boards = data["boards"]
    if boards.ndim == 2:
        boards = boards[None, ...]
    return boards.astype(int).tolist()
