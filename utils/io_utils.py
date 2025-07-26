import json
import os
import zipfile
from typing import List

import numpy as np


def load_boards_from_archives(data_dir: str) -> List[np.ndarray]:
    """Load scratch card boards from JSON files or ZIP archives.

    The function walks through ``data_dir`` and reads either ``.json`` files or
    JSON files inside ``.zip`` archives. Each JSON file is expected to have a
    ``{"board": [[...], ...]}`` structure.

    Parameters
    ----------
    data_dir : str
        Directory containing ``.json`` or ``.zip`` files.

    Returns
    -------
    List[np.ndarray]
        List of numpy arrays representing boards.
    """

    boards: List[np.ndarray] = []
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        if fname.endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                for inner in zf.namelist():
                    if inner.endswith(".json"):
                        with zf.open(inner) as f:
                            obj = json.load(f)
                            arr = np.array(obj["board"], dtype=int)
                            boards.append(arr)
        elif fname.endswith(".json"):
            with open(path) as f:
                obj = json.load(f)
                arr = np.array(obj["board"], dtype=int)
                boards.append(arr)
    return boards
