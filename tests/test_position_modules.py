import numpy as np

from src.position_modules import MODULE_REGISTRY


def _assert_module_output(module_name: str, grid: np.ndarray) -> None:
    fn = MODULE_REGISTRY[module_name]
    out = fn(grid)
    assert out.shape == grid.shape
    assert np.all(out[grid != -1] == 0)


def test_modules_shape_and_non_missing_zero() -> None:
    grid = np.array(
        [
            [1, -1, 3],
            [-1, 5, -1],
            [7, 8, 9],
        ]
    )
    for module_name in MODULE_REGISTRY:
        _assert_module_output(module_name, grid)


def test_all_empty_grid() -> None:
    grid = np.full((4, 4), -1)
    for module_name in MODULE_REGISTRY:
        out = MODULE_REGISTRY[module_name](grid)
        assert out.shape == grid.shape


def test_all_full_grid() -> None:
    grid = np.arange(16).reshape(4, 4)
    for module_name in MODULE_REGISTRY:
        out = MODULE_REGISTRY[module_name](grid)
        assert out.shape == grid.shape
        assert np.allclose(out, 0)


def test_single_missing() -> None:
    grid = np.array([[1, 2], [3, -1]])
    for module_name in MODULE_REGISTRY:
        out = MODULE_REGISTRY[module_name](grid)
        assert out.shape == grid.shape
        assert out[1, 1] >= 0
        assert np.all(out[grid != -1] == 0)


def test_multiple_shapes() -> None:
    shapes = [(2, 5), (5, 2), (6, 6)]
    for shape in shapes:
        grid = np.ones(shape, dtype=int)
        grid.flat[0] = -1
        grid.flat[-1] = -1
        for module_name in MODULE_REGISTRY:
            out = MODULE_REGISTRY[module_name](grid)
            assert out.shape == shape
