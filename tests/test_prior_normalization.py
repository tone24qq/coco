import logging
import os

import analyzer


def test_prior_mix_is_normalized():

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
        history_dir="",
    )
    for dist in res["full_probabilities"].values():
        assert abs(sum(dist.values()) - 1.0) < 1e-9
