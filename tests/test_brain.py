# tests/test_brain.py
import inspect

import numpy as np

import brain

MODULE_FNS = [
    m for n, m in inspect.getmembers(brain, inspect.isfunction) if n.startswith("EXT_")
]


def test_module_shapes(make_grid):
    grid = np.array(make_grid(8, 10))
    for fn in MODULE_FNS:
        out = fn(grid, target=42) if "target" in fn.__code__.co_varnames else fn(grid)
        assert out.shape == grid.shape
        assert np.isfinite(out).all(), f"{fn.__name__} 出現 NaN/Inf"
