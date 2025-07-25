import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

PUZZLE_DIR = pathlib.Path(__file__).parent / "puzzles"


def _load(path: pathlib.Path) -> np.ndarray:
    data = path.read_text().split()
    n = int(len(data) ** 0.5)
    values = [int(x) for x in data]
    return np.array(values).reshape(n, n)


@pytest.fixture(params=list(PUZZLE_DIR.glob("**/*_board.txt")))
def board_and_solution(request):
    board_path = request.param
    sol_path = board_path.with_name(board_path.name.replace("_board", "_solution"))
    return _load(board_path), _load(sol_path)
