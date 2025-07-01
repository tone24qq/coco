import numpy as np

import modules


def _make_grid(r: int, c: int) -> np.ndarray:
    grid = np.arange(1, r * c + 1, dtype=int).reshape(r, c)
    grid[r // 2, c // 2] = -1
    return grid


def test_focus_score_normalized():
    g = _make_grid(5, 5)
    s = modules.compute_focus_score(g)
    assert s.shape == g.shape
    assert 0.0 <= float(s.max()) <= 1.0


def test_skip_pattern_output():
    g = _make_grid(4, 4)
    s = modules.detect_skip_patterns(g)
    assert s.shape == g.shape
    assert np.all(s >= 0)
