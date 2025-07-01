import numpy as np

import analyzer
import modules


def random_board(rng, r, c):
    board = np.arange(1, r * c + 1, dtype=int).reshape(r, c)
    blanks = rng.choice(r * c, max(1, (r * c) // 5), replace=False)
    board.ravel()[blanks] = -1
    return board


def test_tail_and_skip_basic():
    rng = np.random.default_rng(0)
    for _ in range(5):
        r = int(rng.integers(4, 8))
        c = int(rng.integers(4, 8))
        grid = random_board(rng, r, c)
        s1 = modules.sequence_tail_analyzer(grid)
        s2 = modules.detect_skip_patterns(grid)
        assert s1.shape == grid.shape and s2.shape == grid.shape
        assert np.all(s1[grid != -1] >= 0)
        assert np.all(s2[grid != -1] >= 0)


def test_select_modules_includes_new():
    grid = np.array([[1, -1], [2, 3]])
    mods = analyzer.select_modules(grid, target=5)
    for name in ["focus", "skip", "diff", "mirror", "conn", "tail"]:
        assert name in mods
    mods2 = analyzer.select_modules(grid, target=None)
    for name in ["focus", "skip", "diff", "mirror", "conn", "tail"]:
        assert name in mods2
