import json
import os
import zipfile
from typing import Iterable, List, Tuple

import numpy as np


def _extract_boards(
    obj: object, mask_target: bool = False
) -> Iterable[Tuple[np.ndarray, int]]:
    """Yield ``(board, target)`` pairs from ``obj``.

    Supported structures:
    - ``{"board": [...]}``
    - ``{"boards": [...]}``
    - ``[[...], [...], ...]`` (list of boards)
    """

    if isinstance(obj, dict):
        if "board" in obj and "target" in obj:
            board = np.array(obj["board"], dtype=int)
            target = int(obj["target"])
            if mask_target:
                pos = np.argwhere(board == target)
                if pos.shape[0] != 1:
                    raise ValueError(f"target {target} not found exactly once in board")
                r, c = pos[0]
                board = board.copy()
                board[r, c] = -1
            yield (board, target)
        elif "boards" in obj:
            for item in obj["boards"]:
                if isinstance(item, dict) and "board" in item and "target" in item:
                    board = np.array(item["board"], dtype=int)
                    target = int(item["target"])
                    if mask_target:
                        pos = np.argwhere(board == target)
                        if pos.shape[0] != 1:
                            raise ValueError(
                                f"target {target} not found exactly once in board"
                            )
                        r, c = pos[0]
                        board = board.copy()
                        board[r, c] = -1
                    yield (board, target)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and "board" in item and "target" in item:
                board = np.array(item["board"], dtype=int)
                target = int(item["target"])
                if mask_target:
                    pos = np.argwhere(board == target)
                    if pos.shape[0] != 1:
                        raise ValueError(
                            f"target {target} not found exactly once in board"
                        )
                    r, c = pos[0]
                    board = board.copy()
                    board[r, c] = -1
                yield (board, target)


def load_boards_from_archives(
    data_dir: str, *, mask_target: bool = False
) -> List[Tuple[np.ndarray, int]]:
    """Recursively load boards and targets from ``data_dir``.

    All ``.json`` files and JSON files inside ``.zip`` archives are read and
    converted to ``numpy`` arrays. Files may contain a single object with a
    ``board`` field or a list of boards. When ``mask_target`` is ``True``, the
    cell containing ``target`` is replaced by ``-1`` so models can learn to
    infer the missing number.
    """

    boards: List[Tuple[np.ndarray, int]] = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            path = os.path.join(root, fname)
            if fname.endswith(".zip"):
                with zipfile.ZipFile(path) as zf:
                    for inner in zf.namelist():
                        if inner.endswith(".json"):
                            with zf.open(inner) as f:
                                obj = json.load(f)
                                boards.extend(list(_extract_boards(obj, mask_target)))
            elif fname.endswith(".json"):
                with open(path) as f:
                    obj = json.load(f)
                    boards.extend(list(_extract_boards(obj, mask_target)))
    return boards
