import warnings

import numpy as np

import analyzer
import brain

warnings.filterwarnings("ignore", category=DeprecationWarning)
np.seterr(all="ignore")


def random_board(rng, r, c):
    board = np.arange(1, r * c + 1, dtype=int).reshape(r, c)
    blanks = rng.choice(r * c, max(1, (r * c) // 4), replace=False)
    board.ravel()[blanks] = -1
    return board


def test_q13_basic():
    rng = np.random.default_rng(0)
    for _ in range(3):
        r = int(rng.integers(4, 8))
        c = int(rng.integers(4, 8))
        grid = random_board(rng, r, c)
        s = brain.EXT_Q13_GlobalConsistencySpectrum_Vec(grid)
        assert s.shape == grid.shape
        assert np.all(s[grid != -1] == 0)
        assert 0.0 <= float(s.max()) <= 1.0


def test_select_modules_includes_q13(monkeypatch):
    grid = np.array([[1, -1], [2, 3]])
    monkeypatch.setenv("ENABLE_SPECTRUM", "1")
    mods = analyzer.select_modules(grid, target=None)
    assert "EXT_Q13_GlobalConsistencySpectrum_Vec" in mods
    monkeypatch.delenv("ENABLE_SPECTRUM", raising=False)
