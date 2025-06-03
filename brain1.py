# brain1.py
"""
模組組合1：數列推理模組，如等差數列、等比數列模式偵測。
"""
import numpy as np

def arithmetic_progression_row_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    等差數列 (橫列) 模組：
    偵測每一列中是否可透過在遮蔽格放入 target 形成等差數列。
    給每個遮蔽格一個分數 (0 或 1) 表示符合等差條件。
    """
    rows, cols = grid.shape
    score = np.zeros((rows, cols), dtype=float)
    # 中間位置：檢查左右鄰居存在且 target 恰為兩者平均數
    if cols >= 3:
        left = grid[:, :-2]
        center_blank = (grid[:, 1:-1] == -1)
        right = grid[:, 2:]
        valid = center_blank & (left != -1) & (right != -1)
        # 兩側值和為 2*target 表示 target 為平均數
        match = (left + right == 2 * target)
        score[:, 1:-1][valid & match] = 1.0
    # 左端位置：若該列第一格是遮蔽且後面至少兩格有值，檢查 target 是否延續等差序列
    if cols >= 2:
        for r in range(rows):
            if grid[r, 0] == -1:
                # 找出此列第一個和第二個非遮蔽值
                known_indices = np.where(grid[r] != -1)[0]
                if known_indices.size >= 2:
                    i1, i2 = known_indices[0], known_indices[1]
                    val1, val2 = grid[r, i1], grid[r, i2]
                    # 以前兩個已知值差值作等差公差
                    d = val2 - val1
                    expected = val1 - d
                    if expected == target:
                        score[r, 0] = 1.0
    # 右端位置：若該列最後一格是遮蔽且該列有至少兩個已知值，檢查 target 是否延續等差序列
    if cols >= 2:
        for r in range(rows):
            if grid[r, cols-1] == -1:
                known_indices = np.where(grid[r] != -1)[0]
                if known_indices.size >= 2:
                    second_last_idx = known_indices[-2]
                    last_idx = known_indices[-1]
                    val_second_last = grid[r, second_last_idx]
                    val_last = grid[r, last_idx]
                    d = val_last - val_second_last
                    expected = val_last + d
                    if expected == target:
                        score[r, cols-1] = 1.0
    return score

def arithmetic_progression_col_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    等差數列 (直行) 模組：
    偵測每一直行中是否可透過在遮蔽格放入 target 形成縱向等差數列。
    （透過轉置呼叫橫列等差模組實現）
    """
    # 轉置 grid 後套用橫向等差推理，再轉置回原形
    return arithmetic_progression_row_score(grid.T, target).T

def geometric_progression_row_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    等比數列 (橫列) 模組：
    偵測每一列中是否可透過在遮蔽格放入 target 形成等比數列。
    給每個遮蔽格一個分數 (0 或 1) 表示符合等比條件。
    """
    rows, cols = grid.shape
    score = np.zeros((rows, cols), dtype=float)
    # 中間位置：檢查左右鄰居存在且 target^2 = left * right（等比中項）
    if cols >= 3:
        left = grid[:, :-2]
        center_blank = (grid[:, 1:-1] == -1)
        right = grid[:, 2:]
        valid = center_blank & (left != -1) & (right != -1)
        match = (left * right == target * target)
        score[:, 1:-1][valid & match] = 1.0
    # 左端位置：若首格遮蔽且至少兩個後續有值，檢查 target 是否延續等比序列
    if cols >= 2:
        for r in range(rows):
            if grid[r, 0] == -1:
                known_indices = np.where(grid[r] != -1)[0]
                if known_indices.size >= 2:
                    i1, i2 = known_indices[0], known_indices[1]
                    a, b = grid[r, i1], grid[r, i2]
                    # 檢查 b 是否為 a 的整數倍（公比）
                    if a != 0 and b % a == 0:
                        ratio = b // a
                        expected = a // ratio
                        if a % ratio == 0 and expected == target:
                            score[r, 0] = 1.0
    # 右端位置：若末格遮蔽且至少兩個前面有值，檢查 target 是否延續等比序列
    if cols >= 2:
        for r in range(rows):
            if grid[r, cols-1] == -1:
                known_indices = np.where(grid[r] != -1)[0]
                if known_indices.size >= 2:
                    second_last_idx = known_indices[-2]
                    last_idx = known_indices[-1]
                    a, b = grid[r, second_last_idx], grid[r, last_idx]
                    if a != 0 and b % a == 0:
                        ratio = b // a
                        expected = b * ratio
                        if expected == target:
                            score[r, cols-1] = 1.0
    return score

def geometric_progression_col_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    等比數列 (直行) 模組：
    偵測每一直行中是否可透過在遮蔽格放入 target 形成縱向等比數列。
    （透過轉置呼叫橫列等比模組實現）
    """
    return geometric_progression_row_score(grid.T, target).T