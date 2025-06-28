import json
import logging
import os
import zipfile

import analyzer


def test_prior_mix_is_normalized(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    data = {"rows": 2, "cols": 2, "grid": [[1, 2], [3, 4]]}
    zpath = samples / "s.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("a.json", json.dumps(data))

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
