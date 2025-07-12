import json
import zipfile
from pathlib import Path

import numpy as np

import analyzer


def _create_zip(samples_dir: Path, boards):
    zpath = samples_dir / "s.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for i, board in enumerate(boards):
            data = {"rows": 2, "cols": 2, "grid": board}
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


def test_match_samples_dedup(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    base_board = [[1, 2], [3, 4]]
    _create_zip(samples, [base_board, base_board, base_board])
    grid = np.array([[1, 2], [3, 4]])
    matches = analyzer.match_samples(grid, 2, str(samples))
    assert len(matches) == 1


def test_match_samples_legacy(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    base_board = [[1, 2], [3, 4]]
    _create_zip(samples, [base_board])
    grid = np.array([[1, -1], [3, 5]])
    matches = analyzer.match_samples(grid, 2, str(samples), legacy=True)
    assert not matches
