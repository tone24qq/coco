import json
import zipfile

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
    assert freq[0, 0] == 1
    assert freq[0, 1] == 2
    assert freq[1, 1] == 1


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
