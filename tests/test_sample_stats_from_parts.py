import numpy as np

import analyzer


def test_load_sample_stats_from_board_parts(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    b1 = np.array([[1, 2], [3, 4]], dtype=np.int8)
    b2 = np.array([[2, 1], [4, 3]], dtype=np.int8)
    np.savez(samples / "boards_2x2_part0.npz", boards=b1[None])
    np.savez(samples / "boards_2x2_part1.npz", boards=b2[None])
    analyzer.get_sample_stats_cached.cache_clear()
    freq = analyzer.get_sample_stats_cached(2, 2, str(samples))
    rr, cc = np.indices(b1.shape)
    expected = np.zeros((2, 2, 5), dtype=int)
    for b in (b1, b2):
        np.add.at(expected, (rr, cc, b), 1)
    assert freq is not None
    assert np.array_equal(freq, expected)
