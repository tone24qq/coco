# brain3.py

import numpy as np

def ex19(grid: np.ndarray) -> np.ndarray:
    """
    水平二项和递推模式（与 ex5/ex19_b1 相同逻辑）。
    """
    # 复用 brain1 中的 ex5
    from brain1 import ex5 as _ex5
    return _ex5(grid)

def ex20(grid: np.ndarray) -> np.ndarray:
    """
    垂直二项和递推模式（与 ex6 相同逻辑）。
    """
    # 复用 brain1 中的 ex6
    from brain1 import ex6 as _ex6
    return _ex6(grid)

def ex21(grid: np.ndarray) -> np.ndarray:
    """
    移动平均模式。如果某中心格缺失，且上下相加等于左右相加，则标记中心格。
    """
    output = np.zeros(grid.shape, dtype=float)
    if grid.shape[0] < 3 or grid.shape[1] < 3:
        return output
    center = grid[1:-1, 1:-1]
    up    = grid[:-2, 1:-1]
    down  = grid[2:, 1:-1]
    left  = grid[1:-1, :-2]
    right = grid[1:-1, 2:]
    mask = (
        (center == -1) &
        (up != -1) & (down != -1) &
        (left != -1) & (right != -1) &
        ((up + down) == (left + right)) &
        (((up + down) % 2) == 0)
    )
    output[1:-1, 1:-1][mask] = 1.0
    return output

def ex22(grid: np.ndarray) -> np.ndarray:
    """
    垂直对称模式。如果盘面上下对称且某位置一侧缺失，则标记该缺失位置。
    """
    output = np.zeros(grid.shape, dtype=float)
    n_rows, n_cols = grid.shape
    top = grid[:n_rows//2, :]
    bottom = grid[-(n_rows//2):, :][::-1, :]
    mask_top_missing = (top == -1) & (bottom != -1)
    output[:n_rows//2, :][mask_top_missing] = 1.0
    mask_bottom_missing = (top != -1) & (bottom == -1)
    output[-(n_rows//2):, :][::-1, :][mask_bottom_missing] = 1.0
    return output

def ex23(grid: np.ndarray) -> np.ndarray:
    """
    水平对称模式。如果盘面左右对称且某位置一侧缺失，则标记该缺失位置。
    """
    output = np.zeros(grid.shape, dtype=float)
    n_rows, n_cols = grid.shape
    left = grid[:, :n_cols//2]
    right = grid[:, -(n_cols//2):][:, ::-1]
    mask_left_missing = (left == -1) & (right != -1)
    output[:, :n_cols//2][mask_left_missing] = 1.0
    mask_right_missing = (left != -1) & (right == -1)
    output[:, -(n_cols//2):][:, ::-1][mask_right_missing] = 1.0
    return output

def ex24(grid: np.ndarray) -> np.ndarray:
    """
    未实现模式：暂时返回全零阵列。
    """
    return np.zeros(grid.shape, dtype=float)

def ex25(grid: np.ndarray) -> np.ndarray:
    """
    未实现模式：暂时返回全零阵列。
    """
    return np.zeros(grid.shape, dtype=float)

def ex26(grid: np.ndarray) -> np.ndarray:
    """
    未实现模式：暂时返回全零阵列。
    """
    return np.zeros(grid.shape, dtype=float)