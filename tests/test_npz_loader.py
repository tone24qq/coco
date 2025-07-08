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


def test_npz_usage_stats(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    arr_a = np.ones((2, 2, 5))
    arr_b = np.ones((3, 3, 10))
    np.savez(d / "global_pos_freq_2x2.npz", freq=arr_a)
    np.savez(d / "global_pos_freq_3x3.npz", freq=arr_b)

    analyzer._NPZ_USAGE_STATS.clear()
    analyzer._load_global_pos_freq_npz_cached.cache_clear()

    analyzer.load_global_pos_freq_npz((2, 2), d)
    analyzer.load_global_pos_freq_npz((3, 3), d)
    analyzer.load_global_pos_freq_npz((2, 2), d)

    assert analyzer._NPZ_USAGE_STATS["2x2"] == 2
    assert analyzer._NPZ_USAGE_STATS["3x3"] == 1
