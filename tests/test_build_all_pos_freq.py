# isort: skip_file
import json
import zipfile

import numpy as np

from build_all_pos_freq import build_and_save_all_pos_freq, iter_all_json_from_zip


def _create_sample_zip(path):
    board1 = [[-1, 1], [2, -1]]
    board2 = [[3, 4], [5, 6]]
    board3 = [[7, -1], [8, 9]]
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("f1.json", json.dumps({"rows": 2, "cols": 2, "grid": board1}))
        zf.writestr("f2.json", json.dumps(board2))
        zf.writestr("f3.json", json.dumps({"2x2": [board3]}))
    return [board1, board2, board3]


def test_iter_all_json_from_zip(tmp_path):
    zip_path = tmp_path / "s.zip"
    boards = _create_sample_zip(zip_path)
    items = list(iter_all_json_from_zip(zip_path))
    assert len(items) == 3
    for (rows, cols, grid), expected in zip(items, boards):
        assert (rows, cols) == (2, 2)
        assert grid == expected


def test_build_and_save_all_pos_freq(tmp_path):
    samples_dir = tmp_path / "samples"
    output_dir = tmp_path / "priors"
    samples_dir.mkdir()
    _create_sample_zip(samples_dir / "s.zip")
    build_and_save_all_pos_freq(samples_dir, output_dir)
    out = output_dir / "pos_freq_2x2.npz"
    assert out.exists()
    data = np.load(out)
    freq = data["freq"]
    counts = np.array([[2, 2], [3, 2]])
    expected = counts / counts.sum()
    assert np.allclose(freq, expected)
