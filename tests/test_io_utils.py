import json
import zipfile
from pathlib import Path

import numpy as np

from utils.io_utils import load_boards_from_archives


def test_load_boards_from_archives(tmp_path: Path) -> None:
    board = np.arange(20).reshape(4, 5).tolist()
    json_path = tmp_path / "board.json"
    json_path.write_text(json.dumps({"board": board, "target": 3}))

    zip_path = tmp_path / "boards.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a.json", json.dumps({"board": board, "target": 3}))

    boards = load_boards_from_archives(str(tmp_path))
    assert len(boards) == 2
    assert all(isinstance(b[0], np.ndarray) and isinstance(b[1], int) for b in boards)
