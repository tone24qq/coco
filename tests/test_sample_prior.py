import json
import zipfile

import analyzer


def test_compute_position_probabilities(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    data1 = {"rows": 2, "cols": 2, "grid": [[1, 2], [3, 4]]}
    data2 = {"rows": 2, "cols": 2, "grid": [[2, 1], [4, 3]]}
    zpath = samples / "s.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("a.json", json.dumps(data1))
        zf.writestr("b.json", json.dumps(data2))

    probs = analyzer.compute_position_probabilities(str(samples), 2, 2)
    assert (0, 0) in probs
    cell = probs[(0, 0)]
    assert abs(sum(cell.values()) - 1.0) < 1e-6
    assert cell.get(1, 0) == cell.get(2, 0)


def test_predict_with_sample_prior(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    data = {"rows": 2, "cols": 2, "grid": [[1, 2], [3, 4]]}
    zpath = samples / "s.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("c.json", json.dumps(data))

    grid = [[-1, 2], [3, 4]]
    res = analyzer.predict_scratch_card(
        grid,
        target_num=1,
        iterations=4,
        global_iter=2,
        focus_iter=2,
        history_dir=str(samples),
        sample_gamma=1.0,
    )
    assert "predictions" in res
