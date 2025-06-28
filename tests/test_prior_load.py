import json
import zipfile

import numpy as np

import analyzer


def test_prior_load(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    grid = [[1, 2], [3, 4]]
    data = {"rows": 2, "cols": 2, "grid": grid}
    with zipfile.ZipFile(samples / "s.zip", "w") as zf:
        zf.writestr("d.json", json.dumps(data))
    cube = np.zeros((2, 2, 5), dtype=np.int64)
    arr = np.array(grid)
    rr, cc = np.indices(arr.shape)
    mask = arr >= 1
    np.add.at(cube, (rr[mask], cc[mask], arr[mask]), 1)
    np.save(samples / "prior.npy", cube)
    probs = analyzer.compute_position_probabilities(str(samples), 2, 2)
    assert probs[(0, 0)][1] == 1.0
