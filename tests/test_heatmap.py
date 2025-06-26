import base64

import numpy as np

from analyzer import (
    heatmap_to_base64,
    monte_carlo_prob_map,
    prob_map_to_png,
    probability_heatmap,
    render_heatmap,
)


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


def test_heatmap_to_base64():
    mat = np.array([[0.1, 0.2], [0.3, 0.4]])
    b64 = heatmap_to_base64(mat)
    assert isinstance(b64, str)
    assert b64.startswith("iVBOR")


def test_probability_heatmap_base64():
    grid = np.array([[1, -1], [2, -1]])
    prob = probability_heatmap(grid, 3, n_iter=16, seed=1)
    assert prob.shape == grid.shape
    img = render_heatmap(prob, "base64")
    assert isinstance(img, str)
    assert img.startswith("iVBOR")


def test_probability_heatmap_raw():
    grid = np.array([[1, -1], [2, -1]])
    prob = probability_heatmap(grid, 3, n_iter=16, seed=1)
    arr = render_heatmap(prob, "raw")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == grid.shape
