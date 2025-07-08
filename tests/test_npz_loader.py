import numpy as np
import pytest

import analyzer


def test_load_global_pos_freq_npz(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    arr = np.ones((4, 5, 21))
    np.savez(d / "global_pos_freq_4x5.npz", freq=arr)
    loaded = analyzer.load_global_pos_freq_npz((4, 5), d)
    assert np.array_equal(loaded, arr)


def test_load_global_pos_freq_npz_missing(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        analyzer.load_global_pos_freq_npz((2, 2), d)
