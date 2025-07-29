import json
import zipfile
from pathlib import Path

import numpy as np

from precompute_heatmap import collect_statistics


def test_collect_statistics(tmp_path: Path) -> None:
    board1 = np.array([[1, 2], [3, 4]])
    board2 = np.array([[4, 3], [2, 1]])
    (tmp_path / "a.json").write_text(
        json.dumps({"board": board1.tolist(), "target": 2})
    )
    with zipfile.ZipFile(tmp_path / "b.zip", "w") as zf:
        zf.writestr("b.json", json.dumps({"board": board2.tolist(), "target": 4}))
    heatmaps, counts = collect_statistics(str(tmp_path))
    shape = (2, 2)
    assert shape in heatmaps and shape in counts
    hm = heatmaps[shape]
    cnt = counts[shape]
    hm /= hm.sum()
    cnt /= cnt.sum()
    assert np.isclose(hm[0, 0], 2 / 6)
    assert np.isclose(hm[0, 1], 2 / 6)
    assert np.allclose(cnt, 0.25)
