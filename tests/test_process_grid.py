import numpy as np

import analyzer


def test_process_grid_weighted(monkeypatch):
    grid = np.array([[1, -1], [-1, 4]])

    def fake_weight_prob(_grid, prob_map, target_num=None):
        res = {}
        for cell, dist in prob_map.items():
            val = next(iter(dist))
            res[cell] = {val: 0.8 if cell == (0, 1) else 0.2}
        return res

    monkeypatch.setattr(analyzer, "weight_prob_by_modules", fake_weight_prob)

    preds = analyzer.process_grid(grid)
    assert len(preds) == 2
    assert preds[0]["probability"] != preds[1]["probability"]
    assert abs(sum(p["probability"] for p in preds) - 100.0) < 1e-6
