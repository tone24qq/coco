# brain2.py

import numpy as np

def ex10(grid: np.ndarray) -> np.ndarray:
    """
    检测反对角线（NE-SW）等差序列模式。
    """
    output = np.zeros(grid.shape, dtype=float)
    a = grid[:-2, 2:]
    b = grid[1:-1, 1:-1]
    c = grid[2:, :-2]
    mask1 = (a == -1) & (b != -1) & (c != -1)
    output[:-2, 2:][mask1] = 1.0
    mask2 = (b == -1) & (a != -1) & (c != -1) & (((c + a) % 2) == 0)
    output[1:-1, 1:-1][mask2] = 1.0
    mask3 = (c == -1) & (a != -1) & (b != -1)
    output[2:, :-2][mask3] = 1.0
    return output

def ex11(grid: np.ndarray) -> np.ndarray:
    """
    检测主对角线（NW-SE）三连相同数字模式（与 ex7 同逻辑）。
    """
    output = np.zeros(grid.shape, dtype=float)
    a = grid[:-2, :-2]
    b = grid[1:-1, 1:-1]
    c = grid[2:, 2:]
    mask1 = (a == -1) & (b != -1) & (c != -1) & (b == c)
    output[:-2, :-2][mask1] = 1.0
    mask2 = (b == -1) & (a != -1) & (c != -1) & (a == c)
    output[1:-1, 1:-1][mask2] = 1.0
    mask3 = (c == -1) & (a != -1) & (b != -1) & (a == b)
    output[2:, 2:][mask3] = 1.0
    return output

def ex12(grid: np.ndarray) -> np.ndarray:
    """
    检测反对角线（NE-SW）三连相同数字模式（与 ex8 同逻辑）。
    """
    output = np.zeros(grid.shape, dtype=float)
    a = grid[:-2, 2:]
    b = grid[1:-1, 1:-1]
    c = grid[2:, :-2]
    mask1 = (a == -1) & (b != -1) & (c != -1) & (b == c)
    output[:-2, 2:][mask1] = 1.0
    mask2 = (b == -1) & (a != -1) & (c != -1) & (a == c)
    output[1:-1, 1:-1][mask2] = 1.0
    mask3 = (c == -1) & (a != -1) & (b != -1) & (a == b)
    output[2:, :-2][mask3] = 1.0
    return output

def ex13(grid: np.ndarray) -> np.ndarray:
    """
    同一行所有已知值相同时，标记该行隐藏格。
    """
    output = np.zeros(grid.shape, dtype=float)
    n_rows, n_cols = grid.shape
    for i in range(n_rows):
        row = grid[i, :]
        known = row[row != -1]
        if known.size > 0 and np.nanmin(known) == np.nanmax(known):
            output[i, row == -1] = 1.0
    return output

def ex14(grid: np.ndarray) -> np.ndarray:
    """
    同一列所有已知值相同时，标记该列隐藏格。
    """
    output = np.zeros(grid.shape, dtype=float)
    n_rows, n_cols = grid.shape
    for j in range(n_cols):
        col = grid[:, j]
        known = col[col != -1]
        if known.size > 0 and np.nanmin(known) == np.nanmax(known):
            output[col == -1, j] = 1.0
    return output

def ex15(grid: np.ndarray) -> np.ndarray:
    """
    2×2 方块中三格相同数字模式。
    """
    output = np.zeros(grid.shape, dtype=float)
    tl = grid[:-1, :-1]
    tr = grid[:-1, 1:]
    bl = grid[1:, :-1]
    br = grid[1:, 1:]
    mask_br = (br == -1) & (tl != -1) & (tr != -1) & (bl != -1) & (tl == tr) & (tl == bl)
    output[1:, 1:][mask_br] = 1.0
    mask_bl = (bl == -1) & (tl != -1) & (tr != -1) & (br != -1) & (tl == tr) & (tl == br)
    output[1:, :-1][mask_bl] = 1.0
    mask_tr = (tr == -1) & (tl != -1) & (bl != -1) & (br != -1) & (tl == bl) & (tl == br)
    output[:-1, 1:][mask_tr] = 1.0
    mask_tl = (tl == -1) & (tr != -1) & (bl != -1) & (br != -1) & (tr == bl) & (tr == br)
    output[:-1, :-1][mask_tl] = 1.0
    return output