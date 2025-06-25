import base64

import numpy as np

from analyzer import monte_carlo_prob_map, prob_map_to_png


def test_monte_carlo_prob_map_basic():
    grid = np.array([[1, -1], [2, -1]])
    prob = monte_carlo_prob_map(grid, 3, n_iter=32, seed=42)
    assert prob.shape == grid.shape
    assert 0.0 <= prob[0, 1] <= 1.0


def test_prob_map_to_png_roundtrip():
    mat = np.array([[0.0, 1.0], [0.5, 0.2]])
    data = prob_map_to_png(mat)
    assert data.startswith(b"\x89PNG")
    b64 = base64.b64encode(data).decode("ascii")
    assert isinstance(b64, str)
