import json
import zipfile
from pathlib import Path

import numpy as np

import analyzer


def _create_zip(samples_dir: Path, boards):
    zpath = samples_dir / "s.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for i, board in enumerate(boards):
            rows = len(board)
            cols = len(board[0]) if rows else 0
            data = {"rows": rows, "cols": cols, "grid": board}
            zf.writestr(f"b{i}.json", json.dumps(data))


def test_match_samples_partial(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    base_board = [[1, 2], [3, 4]]
    _create_zip(samples, [base_board])
    grid = np.array([[1, -1], [3, 5]])
    matches = analyzer.match_samples(grid, 2, str(samples))
    assert len(matches) == 1
    assert np.array_equal(matches[0], np.array(base_board))


def test_match_samples_nearest(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    base_board = [[1, 2], [3, 4]]
    _create_zip(samples, [base_board])
    grid = np.array([[6, -1], [8, 9]])
    matches = analyzer.match_samples(grid, 2, str(samples))
    assert matches


def test_match_samples_dedup_and_ratio_sort(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    b1 = [[1, 2, 6], [3, 4, 5]]
    b2 = [[1, 2, 6], [7, 4, 5]]  # partial mism 1
    b3 = [[9, 2, 6], [8, 4, 5]]  # approx ratio 2/3
    b4 = [[9, 2, 6], [8, 7, 5]]  # approx ratio 1.0
    _create_zip(samples, [b1, b1, b2, b3, b4])
    grid = np.array([[1, -1, -1], [3, 4, -1]])
    matches = analyzer.match_samples(grid, 2, str(samples), n1=2, n2=4, top_k=10)
    assert len(matches) == 4
    assert np.array_equal(matches[0], np.array(b1))
    assert np.array_equal(matches[1], np.array(b2))
    assert np.array_equal(matches[2], np.array(b3))
    assert np.array_equal(matches[3], np.array(b4))
