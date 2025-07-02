import json
import zipfile

import analyzer


def _make_samples(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    data1 = {"rows": 2, "cols": 2, "grid": [[1, 2], [3, 4]], "mode": "excel"}
    data2 = {"rows": 2, "cols": 2, "grid": [[2, 1], [3, 4]], "mode": "shuffle"}
    with zipfile.ZipFile(samples / "s.zip", "w") as zf:
        zf.writestr("a.json", json.dumps(data1))
        zf.writestr("b.json", json.dumps(data2))
    return samples


def test_compute_position_distribution(tmp_path):
    samples = _make_samples(tmp_path)
    stats = analyzer.compute_position_distribution(str(samples), 2, 2)
    assert stats[(0, 0)][1] == 1
    assert stats[(0, 0)][2] == 1
    stats_excel = analyzer.compute_position_distribution(
        str(samples), 2, 2, mode="excel"
    )
    assert stats_excel[(0, 0)][1] == 1
    assert 2 not in stats_excel[(0, 0)]


def test_compute_number_distribution(tmp_path):
    samples = _make_samples(tmp_path)
    dist = analyzer.compute_number_distribution(str(samples), 2, 2)
    assert dist[1][(0, 0)] == 1
    assert dist[1][(0, 1)] == 1
    excel_only = analyzer.compute_number_distribution(str(samples), 2, 2, mode="excel")
    assert excel_only[1][(0, 0)] == 1
    assert (0, 1) not in excel_only[1]


def test_predict_number(tmp_path):
    samples = _make_samples(tmp_path)
    stats = analyzer.compute_position_distribution(str(samples), 2, 2)
    grid = [[-1, 2], [3, 4]]
    preds = analyzer.predict_number(grid, stats)
    assert preds
    cell, num, score = preds[0]
    assert cell == (0, 0)
    assert num == 1
    assert abs(score - 1.0) < 1e-6
