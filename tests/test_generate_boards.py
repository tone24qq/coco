import numpy as np

import analyzer
import modules


def test_generate_full_boards_uses_formulas(monkeypatch):
    calls = []

    def dummy(rows, cols, rng):
        calls.append(True)
        return np.arange(1, rows * cols + 1).reshape(rows, cols)

    modules.FORMULA_REGISTRY["dummy"] = dummy
    rng = np.random.default_rng(0)
    grid = np.array([[1, -1], [-1, 4]], dtype=int)
    boards = analyzer.generate_full_boards(
        2, 2, 3, rng, ("dummy",), np.array([1.0]), grid
    )
    assert boards.shape == (3, 2, 2)
    assert calls and all(boards[:, 0, 0] == 1) and all(boards[:, 1, 1] == 4)
    del modules.FORMULA_REGISTRY["dummy"]
