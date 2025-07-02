import json
import zipfile

from position_prior import build_position_prior_map


def test_build_position_prior_map(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    data = {"rows": 2, "cols": 2, "grid": [[1, 2], [3, 4]]}
    with zipfile.ZipFile(samples / "s.zip", "w") as zf:
        zf.writestr("a.json", json.dumps(data))

    priors = build_position_prior_map(str(samples))
    assert (2, 2) in priors
    assert priors[(2, 2)][(0, 0)][1] == 1.0
