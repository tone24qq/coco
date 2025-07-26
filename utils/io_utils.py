import json
import os
import zipfile
from typing import List

import numpy as np


def load_boards_from_archives(data_dir: str) -> List[np.ndarray]:
    """Recursively load boards from ``data_dir``.

    All ``.json`` files and JSON files inside ``.zip`` archives are read and
    converted to ``numpy`` arrays.
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
                                boards.append(np.array(obj["board"], dtype=int))
            elif fname.endswith(".json"):
                with open(path) as f:
                    obj = json.load(f)
                    boards.append(np.array(obj["board"], dtype=int))
    return boards
