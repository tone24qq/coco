# Utilities for loading board datasets from compressed archives

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import List


def load_boards_from_zip(
    zip_path: str | Path, rows: int, cols: int
) -> List[List[List[int]]]:
    """Load boards from a ZIP archive.

    Parameters
    ----------
    zip_path : str | Path
        Path to the ZIP archive containing board JSON files.
    rows : int
        Number of rows of the desired boards.
    cols : int
        Number of columns of the desired boards.

    Returns
    -------
    list[list[list[int]]]
        List of boards.
    """
    filename = f"boards_{rows}x{cols}_50000.json"
    zpath = Path(zip_path)
    with zipfile.ZipFile(zpath, "r") as zf:
        if filename not in zf.namelist():
            raise FileNotFoundError(f"{filename} not found in {zpath}")
        with zf.open(filename) as f:
            boards: List[List[List[int]]] = json.load(f)
    return boards
