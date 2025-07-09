import numpy as np

import analyzer


def _make_samples(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    freq = np.zeros((2, 2, 5), dtype=int)
    boards = [np.array([[1, 2], [3, 4]]), np.array([[2, 1], [3, 4]])]
    for b in boards:
        rr, cc = np.indices(b.shape)
        np.add.at(freq, (rr, cc, b), 1)
    np.savez(
        samples / "2x2.npz",
        freq=freq,
        meta={"samples": len(boards), "schema_version": 1, "generated_at": "now"},
    )
    return samples


def test_compute_position_distribution(tmp_path):
    samples = _make_samples(tmp_path)
    stats = analyzer.compute_position_distribution(str(samples), 2, 2)
    assert stats[(0, 0)][1] == 1
    assert stats[(0, 0)][2] == 1
    stats_excel = analyzer.compute_position_distribution(
        str(samples), 2, 2, mode="excel"
    )
    assert stats_excel == stats


def test_compute_number_distribution(tmp_path):
    samples = _make_samples(tmp_path)
    dist = analyzer.compute_number_distribution(str(samples), 2, 2)
    assert dist[1][(0, 0)] == 1
    assert dist[1][(0, 1)] == 1
    excel_only = analyzer.compute_number_distribution(str(samples), 2, 2, mode="excel")
    assert excel_only == dist


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
