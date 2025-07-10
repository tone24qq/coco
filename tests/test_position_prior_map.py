import numpy as np

from position_prior import build_position_prior_map


def test_build_position_prior_map(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    np.savez(samples / "2x2.npz", boards=np.array([[[1, 2], [3, 4]]], dtype=np.int8))

    priors = build_position_prior_map(str(samples))
    assert (2, 2) in priors
    assert priors[(2, 2)][(0, 0)][1] == 1.0
