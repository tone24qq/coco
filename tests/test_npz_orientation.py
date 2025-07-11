import numpy as np

import analyzer


def test_compute_position_probabilities_board_oriented(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    freq = np.zeros((2, 2, 5), dtype=float)
    freq[0, 0, 1] = 1.0
    freq[0, 1, 2] = 1.0
    freq[1, 0, 3] = 1.0
    freq[1, 1, 4] = 1.0
    np.savez(d / "global_pos_freq_2x2.npz", freq=freq)

    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    analyzer._load_global_pos_freq_npz_cached.cache_clear()
    analyzer.compute_position_probabilities.cache_clear()
    analyzer.load_global_pos_freq_npz((2, 2), d)
    probs = analyzer.compute_position_probabilities(str(tmp_path / "samples"), 2, 2)
    assert probs[(0, 0)][1] == 1.0
    assert probs[(0, 1)][2] == 1.0
    assert probs[(1, 0)][3] == 1.0
    assert probs[(1, 1)][4] == 1.0


def test_compute_position_probabilities_target_first(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    freq = np.zeros((5, 2, 2), dtype=float)
    freq[1, 0, 0] = 1.0
    freq[2, 0, 1] = 1.0
    freq[3, 1, 0] = 1.0
    freq[4, 1, 1] = 1.0
    np.savez(d / "global_pos_freq_2x2.npz", freq=freq)

    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    analyzer._load_global_pos_freq_npz_cached.cache_clear()
    analyzer.compute_position_probabilities.cache_clear()
    analyzer.load_global_pos_freq_npz((2, 2), d)
    probs = analyzer.compute_position_probabilities(str(tmp_path / "samples"), 2, 2)
    assert probs[(0, 0)][1] == 1.0
    assert probs[(0, 1)][2] == 1.0
    assert probs[(1, 0)][3] == 1.0
    assert probs[(1, 1)][4] == 1.0


def test_compute_position_probabilities_swapped_axes(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    freq = np.zeros((2, 2, 5), dtype=float)
    freq[0, 0, 1] = 1.0
    freq[1, 0, 2] = 1.0
    freq[0, 1, 3] = 1.0
    freq[1, 1, 4] = 1.0
    freq_swapped = freq.transpose(1, 0, 2)
    np.savez(d / "global_pos_freq_2x2.npz", freq=freq_swapped)

    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    analyzer._load_global_pos_freq_npz_cached.cache_clear()
    analyzer.compute_position_probabilities.cache_clear()
    analyzer.load_global_pos_freq_npz((2, 2), d)
    probs = analyzer.compute_position_probabilities(str(tmp_path / "samples"), 2, 2)
    assert probs[(0, 0)][1] == 1.0
    assert probs[(0, 1)][2] == 1.0
    assert probs[(1, 0)][3] == 1.0
    assert probs[(1, 1)][4] == 1.0
