import json
import zipfile

import numpy as np

import analyzer


def test_load_and_filter_samples(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    board = [[1, 2], [3, 4]]
    with zipfile.ZipFile(samples / "z.zip", "w") as zf:
        zf.writestr("b.json", json.dumps({"rows": 2, "cols": 2, "grid": board}))

    analyzer._SAMPLE_CACHE.clear()
    loaded = analyzer._load_samples_for_shape(str(samples), 2, 2)
    assert loaded == [(np.array(board, dtype=int), "z.zip")]

    grid = np.array([[-1, -1], [-1, -1]])
    filtered = analyzer.filter_matching_samples(grid, loaded)
    assert filtered == loaded
