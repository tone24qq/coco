import numpy as np

import analyzer


def test_load_sample_stats_npz(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    freq = np.ones((3, 4, 5))
    np.savez(samples / "sample_stats_4x5.npz", freq=freq)
    analyzer._SAMPLE_STATS_LOADED.clear()
    analyzer._SAMPLE_CACHE.clear()
    analyzer.load_all_sample_stats(str(samples))
    assert (4, 5, "npz") in analyzer._SAMPLE_STATS_LOADED
    cached = analyzer._SAMPLE_CACHE.get((4, 5, "npz"))
    assert cached and np.array_equal(cached[0][0], freq)
