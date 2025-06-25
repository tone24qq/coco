import numpy as np

import analyzer
import brain


def random_board(rng, r, c):
    board = np.arange(1, r * c + 1, dtype=int).reshape(r, c)
    blanks = rng.choice(r * c, max(1, (r * c) // 5), replace=False)
    board.ravel()[blanks] = -1
    return board


def test_q11_q12_basic():
    rng = np.random.default_rng(0)
    for _ in range(5):
        r = int(rng.integers(4, 8))
        c = int(rng.integers(4, 8))
        grid = random_board(rng, r, c)
        target = int(rng.integers(1, r * c + 1))
        s11 = brain.EXT_Q11_GlobalDigitAffinity_Vec(grid, target=target)
        s12 = brain.EXT_Q12_ArithmeticProgression_Vec(grid)
        assert s11.shape == grid.shape and s12.shape == grid.shape
        assert np.all(s11[grid != -1] == 0)
        assert np.all(s12[grid != -1] == 0)
        assert np.any(s11[grid == -1] >= 0)
        assert 0.0 <= float(s11.max()) <= 1.0
        assert 0.0 <= float(s12.max()) <= 1.0


def test_select_modules_includes_new():
    grid = np.array([[1, -1], [2, 3]])
    mods = analyzer.select_modules(grid, target=5)
    assert "EXT_Q11_GlobalDigitAffinity_Vec" in mods
    assert "EXT_Q12_ArithmeticProgression_Vec" in mods
    mods2 = analyzer.select_modules(grid, target=None)
    assert "EXT_Q12_ArithmeticProgression_Vec" in mods2
