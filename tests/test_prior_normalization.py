import logging
import os

import numpy as np

import analyzer


def test_prior_mix_is_normalized(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    arr = np.array([[[1, 2], [3, 4]]])
    np.savez(samples / "s.npz", boards=arr)
    counts = np.zeros((2, 2, 5), dtype=float)
    for board in arr:
        for r in range(2):
            for c in range(2):
                counts[r, c, board[r, c]] += 1
    totals = counts.sum(axis=2, keepdims=True)
    totals[totals == 0] = 1
    freq = counts / totals
    np.savez(samples / "pos_freq.npz", freq=freq)

    logging.disable(logging.CRITICAL)
    os.environ["FAST_TEST"] = "1"
    res = analyzer.predict_scratch_card(
        grid=[[-1, -1], [-1, -1]],
        iterations=1,
        global_iter=1,
        focus_iter=0,
        top_n=1,
        epsilon=0.0,
        unique=False,
        sample_gamma=1.0,
        history_dir=str(samples),
    )
    for dist in res["full_probabilities"].values():
        assert abs(sum(dist.values()) - 1.0) < 1e-9
