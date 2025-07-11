import json
import zipfile

import numpy as np

import analyzer


def test_get_global_heatmap_basic(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    freq = np.zeros((5, 2, 2), dtype=float)
    freq[1, 0, 0] = 1.0
    np.savez(samples / "pos_freq.npz", freq=freq)
    heat = analyzer.get_global_heatmap(2, 2, 1, str(samples))
    assert heat.shape == (2, 2)
    assert heat[0, 0] == 1.0
    assert heat[0, 1] == 0.0


def test_get_global_heatmap_fallback(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    data = {"rows": 2, "cols": 2, "grid": [[1, 2], [3, 4]]}
    with zipfile.ZipFile(samples / "s.zip", "w") as zf:
        zf.writestr("a.json", json.dumps(data))
    heat = analyzer.get_global_heatmap(2, 2, 1, str(samples))
    hist = analyzer.compute_history_frequency(str(samples), 1, 2, 2)
    assert np.allclose(heat, hist)


def test_match_samples_limit(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    board = {"rows": 2, "cols": 2, "grid": [[1, 2], [3, 4]]}
    with zipfile.ZipFile(samples / "s.zip", "w") as zf:
        for i in range(5):
            zf.writestr(f"b{i}.json", json.dumps(board))
    grid = np.array([[1, 2], [3, -1]])
    matches = analyzer.match_samples(grid, 4, str(samples), limit=2)
    assert len(matches) == 2
