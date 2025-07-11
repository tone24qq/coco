import json
import zipfile

import numpy as np

import analyzer


def test_predict_target_cell_fusion(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    board = {"rows": 2, "cols": 2, "grid": [[1, 2], [3, 4]]}
    zpath = samples / "s.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for i in range(5):
            zf.writestr(f"b{i}.json", json.dumps(board))

    grid = np.array([[1, -1], [3, 4]])
    pos = analyzer.predict_target_cell(grid, 2, history_dir=str(samples))
    assert pos == (0, 1)
