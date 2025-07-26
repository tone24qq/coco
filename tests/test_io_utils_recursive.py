import json
import zipfile
from pathlib import Path

import numpy as np

from utils.io_utils import load_boards_from_archives


def test_recursive_loader(tmp_path: Path) -> None:
    sub = tmp_path / "sub" / "deep"
    sub.mkdir(parents=True)
    board = np.arange(4).reshape(2, 2).tolist()
    (tmp_path / "a.json").write_text(json.dumps({"board": board}))
    (sub / "b.json").write_text(json.dumps({"board": board}))
    zpath = tmp_path / "c.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("inner.json", json.dumps({"board": board}))
    boards = load_boards_from_archives(str(tmp_path))
    assert len(boards) == 3
    assert all(isinstance(b, np.ndarray) for b in boards)
