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


def test_load_boards_mask_target(tmp_path: Path) -> None:
    board = np.arange(1, 5).reshape(2, 2)
    (tmp_path / "a.json").write_text(json.dumps({"board": board.tolist(), "target": 3}))
    boards = load_boards_from_archives(str(tmp_path), mask_target=True)
    assert len(boards) == 1
    out_board, target = boards[0]
    assert target == 3
    expected = board.copy()
    expected[expected == 3] = -1
    assert np.array_equal(out_board, expected)
