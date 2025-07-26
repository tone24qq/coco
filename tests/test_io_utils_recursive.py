import json
import zipfile
from pathlib import Path

import numpy as np

from utils.io_utils import load_boards_from_archives


def test_recursive_loader(tmp_path: Path) -> None:
    sub = tmp_path / "sub" / "deep"
    sub.mkdir(parents=True)
    board = np.arange(4).reshape(2, 2).tolist()
    (tmp_path / "a.json").write_text(json.dumps({"board": board, "target": 1}))
    (sub / "b.json").write_text(json.dumps({"board": board, "target": 1}))
    zpath = tmp_path / "c.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("inner.json", json.dumps({"board": board, "target": 1}))
    boards = load_boards_from_archives(str(tmp_path))
    assert len(boards) == 3
    assert all(isinstance(b[0], np.ndarray) and b[1] == 1 for b in boards)
