import json
from pathlib import Path

import numpy as np

from utils.io_utils import load_boards_from_archives


def test_load_boards_list_format(tmp_path: Path) -> None:
    data = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ]
    (tmp_path / "boards.json").write_text(json.dumps(data))
    boards = load_boards_from_archives(str(tmp_path))
    assert len(boards) == 2
    assert all(isinstance(b, np.ndarray) for b in boards)
    assert (boards[0] == np.array([[1, 2], [3, 4]])).all()


def test_load_boards_dict_list_format(tmp_path: Path) -> None:
    data = {
        "boards": [
            [[9, 10], [11, 12]],
            [[13, 14], [15, 16]],
        ]
    }
    (tmp_path / "boards2.json").write_text(json.dumps(data))
    boards = load_boards_from_archives(str(tmp_path))
    assert len(boards) == 2
    assert (boards[1] == np.array([[13, 14], [15, 16]])).all()
