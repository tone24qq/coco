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


def test_load_sample_stats_from_parts_via_loader(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    board = np.array([[1, 2], [3, 4]], dtype=np.int8)
    np.savez(samples / "boards_2x2_part0.npz", boards=board[None])
    analyzer._SAMPLE_STATS_LOADED.clear()
    analyzer.get_sample_stats_cached.cache_clear()
    analyzer.load_all_sample_stats(str(samples))
    assert (2, 2, str(samples)) in analyzer._SAMPLE_STATS_LOADED
    assert analyzer.get_sample_stats_cached(2, 2, str(samples)) is not None
