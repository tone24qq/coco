import analyzer


def test_compute_position_distribution():
    stats = analyzer.compute_position_distribution("", 2, 2, n_synth=10, seed=0)
    assert sum(stats[(0, 0)].values()) == 10
    assert all(sum(v.values()) == 10 for v in stats.values())


def test_predict_number():
    stats = analyzer.compute_position_distribution("", 2, 2, n_synth=10, seed=0)
    grid = [[-1, 2], [3, 4]]
    preds = analyzer.predict_number(grid, stats)
    assert preds
    cell, num, score = preds[0]
    assert cell == (0, 0)
    assert num == 1
    assert abs(score - 1.0) < 1e-6
