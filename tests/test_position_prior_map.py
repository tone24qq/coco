import numpy as np

import analyzer
from position_prior import build_position_prior_map


def test_build_position_prior_map(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    np.savez(samples / "2x2.npz", boards=np.array([[[1, 2], [3, 4]]], dtype=np.int8))

    out_npz = tmp_path / "out_npz"
    out_npz.mkdir()
    from position_prior import build_position_prior

    build_position_prior(str(samples), str(out_npz / "global_pos_freq_2x2.npz"))
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    analyzer._load_global_pos_freq_npz_cached.cache_clear()
    analyzer.compute_position_probabilities.cache_clear()
    analyzer.load_global_pos_freq_npz((2, 2), out_npz)

    priors = build_position_prior_map(str(samples))
    assert (2, 2) in priors
    assert priors[(2, 2)][(0, 0)][1] == 1.0
