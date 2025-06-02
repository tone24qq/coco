import numpy as np

def ex1(grid: np.ndarray) -> np.ndarray:
    """
    檢測水平連續遞增序列模式（差值為1）。如果有任何三連續橫列格子中，
    兩端已知且為連續數字，中間遮蔽，則推測中間位置應為連續數字。
    """
    # 建立輸出矩陣
    output = np.zeros(grid.shape, dtype=float)
    # 提取橫向連續三格的視窗
    a = grid[:, :-2]    # 三連窗的第一格
    b = grid[:, 1:-1]   # 三連窗的第二格
    c = grid[:, 2:]     # 三連窗的第三格
    # 情況1：第一格缺失，中間和最後一格已知且連號
    mask1 = (a == -1) & (b != -1) & (c != -1) & ((c - b) == 1)
    output[:, :-2][mask1] = 1.0
    # 情況2：中間缺失，兩端已知且相差2（連號序列中間缺一）
    mask2 = (b == -1) & (a != -1) & (c != -1) & ((c - a) == 2)
    # 需要確認a與c連號（差為2）且中間應插入 a+1
    output[:, 1:-1][mask2] = 1.0
    # 情況3：最後一格缺失，前兩格已知且連號
    mask3 = (c == -1) & (a != -1) & (b != -1) & ((b - a) == 1)
    output[:, 2:][mask3] = 1.0
    return output

def ex2(grid: np.ndarray) -> np.ndarray:
    """
    檢測垂直連續遞增序列模式（差值為1）。原理同 ex1，但沿縱向掃描。
    """
    output = np.zeros(grid.shape, dtype=float)
    # 提取縱向連續三格的視窗
    a = grid[:-2, :]   # 第一格
    b = grid[1:-1, :]  # 第二格
    c = grid[2:, :]    # 第三格
    # 垂直 情況1：首格缺失且下兩格連續
    mask1 = (a == -1) & (b != -1) & (c != -1) & ((c - b) == 1)
    output[:-2, :][mask1] = 1.0
    # 垂直 情況2：中格缺失，首末差為2
    mask2 = (b == -1) & (a != -1) & (c != -1) & ((c - a) == 2)
    output[1:-1, :][mask2] = 1.0
    # 垂直 情況3：末格缺失，前兩格連續
    mask3 = (c == -1) & (a != -1) & (b != -1) & ((b - a) == 1)
    output[2:, :][mask3] = 1.0
    return output

def ex3(grid: np.ndarray) -> np.ndarray:
    """
    檢測水平等差數列模式（任意差值）。如任何三連續橫列格子中，
    存在一格缺失且其餘兩格已知，且這兩格符合等差數列條件，
    則該缺失格視為可能的補完位置。
    """
    output = np.zeros(grid.shape, dtype=float)
    a = grid[:, :-2]
    b = grid[:, 1:-1]
    c = grid[:, 2:]
    # 水平 情況1：第一格缺失，則差值 = c - b，可推第一格 = b - 差值
    mask1 = (a == -1) & (b != -1) & (c != -1)
    # 任意兩數皆可形成等差序列，因此無需其他條件
    output[:, :-2][mask1] = 1.0
    # 水平 情況2：中間缺失，需確認兩端差值為偶數（等差中點可整數）
    mask2 = (b == -1) & (a != -1) & (c != -1) & (((c + a) % 2) == 0)
    output[:, 1:-1][mask2] = 1.0
    # 水平 情況3：最後一格缺失
    mask3 = (c == -1) & (a != -1) & (b != -1)
    output[:, 2:][mask3] = 1.0
    return output

def ex4(grid: np.ndarray) -> np.ndarray:
    """
    檢測垂直等差數列模式（任意差值）。與 ex3 類似，但沿縱向。
    """
    output = np.zeros(grid.shape, dtype=float)
    a = grid[:-2, :]
    b = grid[1:-1, :]
    c = grid[2:, :]
    # 垂直 情況1：首格缺失
    mask1 = (a == -1) & (b != -1) & (c != -1)
    output[:-2, :][mask1] = 1.0
    # 垂直 情況2：中格缺失（兩端差值為偶數）
    mask2 = (b == -1) & (a != -1) & (c != -1) & (((c + a) % 2) == 0)
    output[1:-1, :][mask2] = 1.0
    # 垂直 情況3：末格缺失
    mask3 = (c == -1) & (a != -1) & (b != -1)
    output[2:, :][mask3] = 1.0
    return output

def ex5(grid: np.ndarray) -> np.ndarray:
    """
    檢測水平等比數列模式。如任何三連續橫列格子中存在一格缺失，
    其餘兩格已知且可構成等比數列，則該缺失格為可能的補完位置。
    """
    output = np.zeros(grid.shape, dtype=float)
    a = grid[:, :-2]
    b = grid[:, 1:-1]
    c = grid[:, 2:]
    # 水平 等比 情況1：第一格缺失（已知 b, c 構成等比）
    mask1_base = (a == -1) & (b != -1) & (c != -1) & (c != 0)
    # 檢查 b^2 是否可被 c 整除（確保 ratio 為有理數且 a = b^2/c 整數）
    mask1 = mask1_base & ((b * b) % c == 0)
    output[:, :-2][mask1] = 1.0
    # 水平 等比 情況2：中間缺失（已知 a, c）
    mask2_base = (b == -1) & (a != -1) & (c != -1) & (a >= 0) & (c >= 0)
    # a*c 必須是完全平方數，b = sqrt(a*c)
    prod = a * c
    # 避免浮點誤差，使用整數運算檢查完全平方
    int_sqrt = ((prod >= 0) * np.rint(np.sqrt(prod)).astype(int))
    mask2 = mask2_base & (int_sqrt * int_sqrt == prod) & ((a == 0) == (c == 0))
    output[:, 1:-1][mask2] = 1.0
    # 水平 等比 情況3：最後一格缺失
    mask3_base = (c == -1) & (a != -1) & (b != -1) & (a != 0)
    mask3 = mask3_base & ((b * b) % a == 0)
    output[:, 2:][mask3] = 1.0
    return output

def ex6(grid: np.ndarray) -> np.ndarray:
    """
    檢測垂直等比數列模式（類似 ex5 垂直方向）。
    """
    output = np.zeros(grid.shape, dtype=float)
    a = grid[:-2, :]
    b = grid[1:-1, :]
    c = grid[2:, :]
    # 垂直 等比 情況1：首格缺失
    mask1_base = (a == -1) & (b != -1) & (c != -1) & (c != 0)
    mask1 = mask1_base & ((b * b) % c == 0)
    output[:-2, :][mask1] = 1.0
    # 垂直 等比 情況2：中格缺失
    mask2_base = (b == -1) & (a != -1) & (c != -1) & (a >= 0) & (c >= 0)
    prod = a * c
    int_sqrt = ((prod >= 0) * np.rint(np.sqrt(prod)).astype(int))
    mask2 = mask2_base & (int_sqrt * int_sqrt == prod) & ((a == 0) == (c == 0))
    output[1:-1, :][mask2] = 1.0
    # 垂直 等比 情況3：末格缺失
    mask3_base = (c == -1) & (a != -1) & (b != -1) & (a != 0)
    mask3 = mask3_base & ((b * b) % a == 0)
    output[2:, :][mask3] = 1.0
    return output

def ex7(grid: np.ndarray) -> np.ndarray:
    """
    檢測水平三連相同數字模式。如同一橫列的連續三格中有一格缺失，
    另外兩格已知且數值相同，則視為潛在圖案，標記該缺失格。
    """
    output = np.zeros(grid.shape, dtype=float)
    a = grid[:, :-2]
    b = grid[:, 1:-1]
    c = grid[:, 2:]
    # 水平 情況1：第一格缺失，後兩格相同
    mask1 = (a == -1) & (b != -1) & (c != -1) & (b == c)
    output[:, :-2][mask1] = 1.0
    # 水平 情況2：中間缺失，首末相同
    mask2 = (b == -1) & (a != -1) & (c != -1) & (a == c)
    output[:, 1:-1][mask2] = 1.0
    # 水平 情況3：最後一格缺失，前兩格相同
    mask3 = (c == -1) & (a != -1) & (b != -1) & (a == b)
    output[:, 2:][mask3] = 1.0
    return output

def ex8(grid: np.ndarray) -> np.ndarray:
    """
    檢測垂直三連相同數字模式。垂直方向的版本。
    """
    output = np.zeros(grid.shape, dtype=float)
    a = grid[:-2, :]
    b = grid[1:-1, :]
    c = grid[2:, :]
    mask1 = (a == -1) & (b != -1) & (c != -1) & (b == c)
    output[:-2, :][mask1] = 1.0
    mask2 = (b == -1) & (a != -1) & (c != -1) & (a == c)
    output[1:-1, :][mask2] = 1.0
    mask3 = (c == -1) & (a != -1) & (b != -1) & (a == b)
    output[2:, :][mask3] = 1.0
    return output

def ex9(grid: np.ndarray) -> np.ndarray:
    """
    檢測主對角線方向（NW-SE）等差序列模式。與 ex3 類似，但沿對角方向檢查。
    """
    output = np.zeros(grid.shape, dtype=float)
    # 主對角線三格窗口
    a = grid[:-2, :-2]
    b = grid[1:-1, 1:-1]
    c = grid[2:, 2:]
    # 對角 情況1：第一格缺失
    mask1 = (a == -1) & (b != -1) & (c != -1)
    output[:-2, :-2][mask1] = 1.0
    # 對角 情況2：中間缺失（兩端差值偶數）
    mask2 = (b == -1) & (a != -1) & (c != -1) & (((c + a) % 2) == 0)
    output[1:-1, 1:-1][mask2] = 1.0
    # 對角 情況3：最後一格缺失
    mask3 = (c == -1) & (a != -1) & (b != -1)
    output[2:, 2:][mask3] = 1.0
    return output