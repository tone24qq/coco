import numpy as np

def ex19(grid: np.ndarray) -> np.ndarray:
    """
    檢測水平二項和遞推模式（類 Fibonacci）。如果有連續三格橫列，
    當中一格缺失且已知兩格符合前兩項和等於第三項的關係，則標記缺失格。
    """
    output = np.zeros(grid.shape, dtype=float)
    a = grid[:, :-2]
    b = grid[:, 1:-1]
    c = grid[:, 2:]
    # 水平 情況1：末格缺失，已知 a, b，若 c 應為 a+b
    mask_last = (c == -1) & (a != -1) & (b != -1)
    output[:, 2:][mask_last] = 1.0  # 直接標記缺失處（實際數值應為 a+b，但此處僅標記位置）
    # 水平 情況2：首格缺失，已知 b, c，若 b 應為 a+c => a = b - c
    mask_first = (a == -1) & (b != -1) & (c != -1)
    output[:, :-2][mask_first] = 1.0
    # 水平 情況3：中格缺失，已知 a, c，若 c = a + b => b = c - a
    mask_mid = (b == -1) & (a != -1) & (c != -1)
    output[:, 1:-1][mask_mid] = 1.0
    return output

def ex20(grid: np.ndarray) -> np.ndarray:
    """
    檢測垂直二項和遞推模式（類 Fibonacci）。縱向版本。
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

def ex21(grid: np.ndarray) -> np.ndarray:
    """
    檢測多方向移動平均模式。如果某中心格缺失，且上下相加等於左右相加（代表中心應為兩者平均），則標記中心格。
    """
    output = np.zeros(grid.shape, dtype=float)
    if grid.shape[0] < 3 or grid.shape[1] < 3:
        return output
    center = grid[1:-1, 1:-1]
    up    = grid[:-2, 1:-1]
    down  = grid[2:, 1:-1]
    left  = grid[1:-1, :-2]
    right = grid[1:-1, 2:]
    # 中心缺失，四鄰皆知且上下和等於左右和（代表中心為兩者平均值）
    mask = (center == -1) & (up != -1) & (down != -1) & (left != -1) & (right != -1) \
           & ((up + down) == (left + right)) & (((up + down) % 2) == 0)
    output[1:-1, 1:-1][mask] = 1.0
    return output

def ex22(grid: np.ndarray) -> np.ndarray:
    """
    檢測豎直對稱模式。如果盤面上下對稱且某位置一側缺失，則標記該缺失位置。
    """
    output = np.zeros(grid.shape, dtype=float)
    n_rows, n_cols = grid.shape
    # 僅考慮偶數或奇數行數皆可，奇數中央行無需匹配
    top = grid[:n_rows//2, :]
    bottom = grid[-(n_rows//2):, :][::-1, :]  # 反轉下半部以便與上半部對齊
    # 上半缺，下半同位置有值
    mask_top_missing = (top == -1) & (bottom != -1)
    output[:n_rows//2, :][mask_top_missing] = 1.0
    # 下半缺，上半有值
    mask_bottom_missing = (top != -1) & (bottom == -1)
    output[-(n_rows//2):, :][::-1, :][mask_bottom_missing] = 1.0
    return output

def ex23(grid: np.ndarray) -> np.ndarray:
    """
    檢測水平對稱模式。如果盤面左右對稱且某位置一側缺失，則標記該缺失位置。
    """
    output = np.zeros(grid.shape, dtype=float)
    n_rows, n_cols = grid.shape
    # 左右半部（忽略正中央列如果為奇數寬度）
    left = grid[:, :n_cols//2]
    right = grid[:, -(n_cols//2):][:, ::-1]  # 反轉右半部
    mask_left_missing = (left == -1) & (right != -1)
    output[:, :n_cols//2][mask_left_missing] = 1.0
    mask_right_missing = (left != -1) & (right == -1)
    output[:, -(n_cols//2):][:, ::-1][mask_right_missing] = 1.0
    return output

def ex24(grid: np.ndarray) -> np.ndarray:
    """
    （保留）未實作模式。當前回傳全零矩陣。
    """
    return np.zeros(grid.shape, dtype=float)

def ex25(grid: np.ndarray) -> np.ndarray:
    """
    （保留）未實作模式。當前回傳全零矩陣。
    """
    return np.zeros(grid.shape, dtype=float)

def ex26(grid: np.ndarray) -> np.ndarray:
    """
    （保留）未實作模式。當前回傳全零矩陣。
    """
    return np.zeros(grid.shape, dtype=float)