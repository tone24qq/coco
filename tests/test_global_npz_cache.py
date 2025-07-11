import numpy as np

import analyzer


def test_load_global_pos_freq_npz_and_get(tmp_path):
    d = tmp_path / "npz"
    d.mkdir()
    arr = np.ones((2, 2, 5))
    np.savez(d / "global_pos_freq_2x2.npz", freq=arr)
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    freq = analyzer.load_global_pos_freq_npz((2, 2), d)
    assert freq is not None
    assert np.array_equal(freq, arr)
    assert analyzer.get_global_pos_freq((3, 3)) is None
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
