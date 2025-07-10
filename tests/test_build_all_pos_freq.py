# isort: skip_file
import numpy as np

from build_all_pos_freq import build_and_save_all_pos_freq, iter_all_json_from_zip


def _create_sample_npz(path):
    boards = np.array(
        [
            [[-1, 1], [2, -1]],
            [[3, 4], [5, 6]],
            [[7, -1], [8, 9]],
        ],
        dtype=np.int8,
    )
    np.savez(path, boards=boards)
    return boards.tolist()


def test_iter_all_json_from_zip(tmp_path):
    zip_path = tmp_path / "s.npz"
    boards = _create_sample_npz(zip_path)
    items = list(iter_all_json_from_zip(zip_path))
    assert len(items) == 3
    for (rows, cols, grid), expected in zip(items, boards):
        assert (rows, cols) == (2, 2)
        assert grid == expected


def test_build_and_save_all_pos_freq(tmp_path):
    samples_dir = tmp_path / "samples"
    output_dir = tmp_path / "priors"
    samples_dir.mkdir()
    _create_sample_npz(samples_dir / "s.npz")
    build_and_save_all_pos_freq(samples_dir, output_dir)
    out = output_dir / "pos_freq_2x2.npz"
    assert out.exists()
    data = np.load(out)
    freq = data["freq"]
    counts = np.array([[2, 2], [3, 2]])
    expected = counts / counts.sum()
    assert np.allclose(freq, expected)
