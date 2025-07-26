import json
import os
import zipfile
from typing import Iterable, List

import numpy as np


def _extract_boards(obj: object) -> Iterable[np.ndarray]:
    """Yield ``numpy`` boards from ``obj``.

    Supported structures:
    - ``{"board": [...]}``
    - ``{"boards": [...]}``
    - ``[[...], [...], ...]`` (list of boards)
    """

    if isinstance(obj, dict):
        if "board" in obj:
            yield np.array(obj["board"], dtype=int)
        elif "boards" in obj:
            for board in obj["boards"]:
                yield np.array(board, dtype=int)
    elif isinstance(obj, list):
        if obj and isinstance(obj[0], list) and obj[0] and isinstance(obj[0][0], list):
            for board in obj:
                yield np.array(board, dtype=int)


def load_boards_from_archives(data_dir: str) -> List[np.ndarray]:
    """Recursively load boards from ``data_dir``.

    All ``.json`` files and JSON files inside ``.zip`` archives are read and
    converted to ``numpy`` arrays. Files may contain a single object with a
    ``board`` field or a list of boards.
    """

    boards: List[np.ndarray] = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            path = os.path.join(root, fname)
            if fname.endswith(".zip"):
                with zipfile.ZipFile(path) as zf:
                    for inner in zf.namelist():
                        if inner.endswith(".json"):
                            with zf.open(inner) as f:
                                obj = json.load(f)
                                boards.extend(list(_extract_boards(obj)))
            elif fname.endswith(".json"):
                with open(path) as f:
                    obj = json.load(f)
                    boards.extend(list(_extract_boards(obj)))
    return boards
