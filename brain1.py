# brain1.py

import numpy as np

def ex1(grid: np.ndarray) -> np.ndarray:
    """
    检测水平连续递增序列模式（差值为1）。如果有任何三连续横列格子中，
    两端已知且为连续数字，中间遮蔽，则推测中间位置应为连续数字。
    """
    output = np.zeros(grid.shape, dtype=float)
    a = grid[:, :-2]    # 三连窗的第一格
    b = grid[:, 1:-1]   # 三连窗的第二格
    c = grid[:, 2:]     # 三连窗的第三格
    # 情况1：第一格缺失，中间和最后一格已知且连续
    mask1 = (a == -1) & (b != -1) & (c != -1) & ((c - b) == 1)
    output[:, :-2][mask1] = 1.0
    # 情况2：中间格缺失，已知两端且连续
    mask2 = (b == -1) & (a != -1) & (c != -1) & ((c - a) == 2)
    output[:, 1:-1][mask2] = 1.0
    # 情况3：最后格缺失，已知前两格且连续
    mask3 = (c == -1) & (a != -1) & (b != -1) & ((b - a) == 1)
    output[:, 2:][mask3] = 1.0
    return output

def ex3(grid: np.ndarray) -> np.ndarray:
    """
    如果某行中有两个已知格数值相等，且该行上有隐藏格，则该隐藏格很可能与它们相同。
    """
    output = np.zeros(grid.shape, dtype=float)
    rows, cols = grid.shape
    for i in range(rows):
        row = grid[i, :]
        known = row[row != -1]
        if known.size > 1 and np.all(known == known[0]):
            # 该行所有已知格数值相等
            output[i, row == -1] = 1.0
    return output

def ex4(grid: np.ndarray) -> np.ndarray:
    """
    如果某列中有两个已知格数值相等，且该列上有隐藏格，则该隐藏格很可能与它们相同。
    """
    output = np.zeros(grid.shape, dtype=float)
    rows, cols = grid.shape
    for j in range(cols):
        col = grid[:, j]
        known = col[col != -1]
        if known.size > 1 and np.all(known == known[0]):
            output[col == -1, j] = 1.0
    return output

def ex5(grid: np.ndarray) -> np.ndarray:
    """
    水平二项和递推（类 Fibonacci）。若三连横窗口中的一格缺失，且已知两格满足 a+b=c，
    则该缺失格很可能是 a+b 或 b−a 等可推导数值。
    """
    output = np.zeros(grid.shape, dtype=float)
    a = grid[:, :-2]
    b = grid[:, 1:-1]
    c = grid[:, 2:]
    # 三种情况都标记 1.0，实际此处仅表示“可能”
    mask_last = (c == -1) & (a != -1) & (b != -1)
    output[:, 2:][mask_last] = 1.0
    mask_first = (a == -1) & (b != -1) & (c != -1)
    output[:, :-2][mask_first] = 1.0
    mask_mid = (b == -1) & (a != -1) & (c != -1)
    output[:, 1:-1][mask_mid] = 1.0
    return output

def ex6(grid: np.ndarray) -> np.ndarray:
    """
    垂直二项和递推（类 Fibonacci）。纵向版本。
    """
    output = np.zeros(grid.shape, dtype=float)
    a = grid[:-2, :]
    b = grid[1:-1, :]
    c = grid[2:, :]
    mask_last = (c == -1) & (a != -1) & (b != -1)
    output[2:, :][mask_last] = 1.0
    mask_first = (a == -1) & (b != -1) & (c != -1)
    output[:-2, :][mask_first] = 1.0
    mask_mid = (b == -1) & (a != -1) & (c != -1)
    output[1:-1, :][mask_mid] = 1.0
    return output

def ex7(grid: np.ndarray) -> np.ndarray:
    """
    主对角线（NW-SE）三连相同数字模式。若三连对角线上有两格相同且另一缺失，则该缺失格可能相同。
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

def ex8(grid: np.ndarray) -> np.ndarray:
    """
    反对角线（NE-SW）三连相同数字模式。纵向与水平对称的 ex7。
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

def ex9(grid: np.ndarray) -> np.ndarray:
    """
    2×2 方块中三格相同数字模式。如果 2×2 块中三个已知相同，第四缺失，则该缺失格可能相同。
    """
    output = np.zeros(grid.shape, dtype=float)
    tl = grid[:-1, :-1]   # top-left
    tr = grid[:-1, 1:]    # top-right
    bl = grid[1:, :-1]    # bottom-left
    br = grid[1:, 1:]     # bottom-right

    mask_br = (br == -1) & (tl != -1) & (tr != -1) & (bl != -1) & (tl == tr) & (tl == bl)
    output[1:, 1:][mask_br] = 1.0

    mask_bl = (bl == -1) & (tl != -1) & (tr != -1) & (br != -1) & (tl == tr) & (tl == br)
    output[1:, :-1][mask_bl] = 1.0

    mask_tr = (tr == -1) & (tl != -1) & (bl != -1) & (br != -1) & (tl == bl) & (tl == br)
    output[:-1, 1:][mask_tr] = 1.0

    mask_tl = (tl == -1) & (tr != -1) & (bl != -1) & (br != -1) & (tr == bl) & (tr == br)
    output[:-1, :-1][mask_tl] = 1.0
    return output

def ex16(grid: np.ndarray) -> np.ndarray:
    """
    水平对称模式。如果某行左右对称且一个格缺失，则该缺失格可能与对称位置相同。
    """
    output = np.zeros(grid.shape, dtype=float)
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols // 2):
            left = grid[i, j]
            right = grid[i, cols - 1 - j]
            if left == -1 and right != -1:
                output[i, j] = 1.0
            if right == -1 and left != -1:
                output[i, cols - 1 - j] = 1.0
    return output

def ex17(grid: np.ndarray) -> np.ndarray:
    """
    垂直对称模式。如果某列上下对称且一个格缺失，则该缺失格可能与对称位置相同。
    """
    output = np.zeros(grid.shape, dtype=float)
    rows, cols = grid.shape
    for j in range(cols):
        for i in range(rows // 2):
            top = grid[i, j]
            bottom = grid[rows - 1 - i, j]
            if top == -1 and bottom != -1:
                output[i, j] = 1.0
            if bottom == -1 and top != -1:
                output[rows - 1 - i, j] = 1.0
    return output

def ex18(grid: np.ndarray) -> np.ndarray:
    """
    移动平均模式。如果某中心格缺失，且上下相加等于左右相加（中心为四邻平均），则标记中心格。
    """
    output = np.zeros(grid.shape, dtype=float)
    rows, cols = grid.shape
    if rows < 3 or cols < 3:
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

def ex19(grid: np.ndarray) -> np.ndarray:
    """
    水平二项和递推模式（与 ex5 相同逻辑，但放在 brain1）。
    """
    # 本处直接调用 ex5 实现
    from brain1 import ex5 as _ex5
    return _ex5(grid)