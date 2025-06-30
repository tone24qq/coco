import json
import zipfile

import analyzer
from position_prior import build_position_prior


def test_global_position_prior(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    data = {"rows": 2, "cols": 2, "grid": [[1, 2], [3, 4]]}
    with zipfile.ZipFile(samples / "s.zip", "w") as zf:
        zf.writestr("a.json", json.dumps(data))

    out = samples / "pos_freq.npz"
    build_position_prior(str(samples), str(out), buckets=2)

    analyzer.load_global_pos_freq(str(samples))
    probs = analyzer.compute_position_probabilities(str(samples), 2, 2)
    assert probs[(0, 0)][1] == 1.0
    assert probs[(0, 1)][2] == 1.0
