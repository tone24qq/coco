import json
import zipfile

import numpy as np

import analyzer


def test_compute_history_frequency(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    data1 = {"rows": 2, "cols": 2, "grid": [[1, 2], [3, 4]]}
    data2 = {"rows": 2, "cols": 2, "grid": [[2, 2], [1, 2]]}
    zpath = samples / "s.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("a.json", json.dumps(data1))
        zf.writestr("b.json", json.dumps(data2))

    freq = analyzer.compute_history_frequency(str(samples), 2, 2, 2)
    assert freq.shape == (2, 2)
    assert abs(freq[0, 0] - 0.25) < 1e-6
    assert abs(freq[0, 1] - 0.5) < 1e-6
    assert abs(freq[1, 1] - 0.25) < 1e-6


def test_compute_history_frequency_precomputed(tmp_path, monkeypatch):
    prior_dir = tmp_path / "priors"
    prior_dir.mkdir()
    arr = np.array([[0.1, 0.9], [0.0, 0.0]])
    np.save(prior_dir / "2x2.npy", arr)
    monkeypatch.setattr(analyzer, "PRIORS_DIR", prior_dir)
    analyzer._PRIOR_CACHE.clear()
    freq = analyzer.compute_history_frequency(str(tmp_path / "samples"), 2, 2, 2)
    assert np.allclose(freq, arr)


def test_predict_with_history(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    data = {"rows": 2, "cols": 2, "grid": [[2, 1], [3, 2]]}
    zpath = samples / "s.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("c.json", json.dumps(data))

    grid = [[-1, -1], [-1, -1]]
    result = analyzer.predict_scratch_card(
        grid,
        target_num=2,
        iterations=4,
        global_iter=2,
        focus_iter=2,
        history_dir=str(samples),
        gamma_history=1.0,
    )
    assert "predictions" in result
