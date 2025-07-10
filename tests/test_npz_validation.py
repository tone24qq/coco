import numpy as np

import analyzer


def test_invalid_npz_counter(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    np.savez(samples / "sample_stats_2x2.npz", freq=np.zeros((2, 2, 5), int))
    np.savez(samples / "sample_stats_3x3.npz", freq=np.zeros((3, 3, 10), int))
    with open(samples / "boards_3x3_part0.npz", "wb") as f:
        f.write(b"corrupt")

    analyzer.get_sample_stats_cached.cache_clear()
    before = analyzer.INVALID_NPZ_COUNTER._value.get()
    analyzer.get_sample_stats_cached(2, 2, str(samples))
    analyzer.get_sample_stats_cached(3, 3, str(samples))
    after = analyzer.INVALID_NPZ_COUNTER._value.get()
    assert after - before == 1
    assert analyzer.get_sample_stats_cached.cache_info().currsize <= 6
