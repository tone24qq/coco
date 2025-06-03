# brain3.py
"""
模組組合3：乘法關係、質數鄰近及總和值模式模組。
"""
import numpy as np

def neighbor_product_horizontal_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    鄰積模式 (水平方向) 模組：
    若遮蔽格左右鄰居存在且相乘結果等於 target，則視為滿足條件。
    """
    rows, cols = grid.shape
    score = np.zeros((rows, cols), dtype=float)
    if cols >= 3:
        left = grid[:, :-2]
        center_blank = (grid[:, 1:-1] == -1)
        right = grid[:, 2:]
        valid = center_blank & (left != -1) & (right != -1)
        match = (left * right == target)
        score[:, 1:-1][valid & match] = 1.0
    return score

def neighbor_product_vertical_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    鄰積模式 (垂直方向) 模組：
    若遮蔽格上下鄰居存在且相乘結果等於 target，則視為滿足條件。
    """
    return neighbor_product_horizontal_score(grid.T, target).T

def neighbor_factor_horizontal_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    鄰乘因子模式 (水平方向) 模組：
    若遮蔽格左右鄰居存在，且其中一側數字乘以 target 等於另一側數字，則視為滿足條件。
    """
    rows, cols = grid.shape
    score = np.zeros((rows, cols), dtype=float)
    if cols >= 3:
        left = grid[:, :-2]
        center_blank = (grid[:, 1:-1] == -1)
        right = grid[:, 2:]
        valid = center_blank & (left != -1) & (right != -1)
        match = ((left * target == right) | (right * target == left))
        score[:, 1:-1][valid & match] = 1.0
    return score

def neighbor_factor_vertical_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    鄰乘因子模式 (垂直方向) 模組：
    若遮蔽格上下鄰居存在，且其中一側數字乘以 target 等於另一側數字，則視為滿足條件。
    """
    return neighbor_factor_horizontal_score(grid.T, target).T

def prime_cluster_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    質數簇集模式模組：
    若 target 為質數，則鄰近質數越多的位置分數越高；若 target 非質數，則鄰近合數（非質數且 >1）越多的位置分數越高。
    """
    rows, cols = grid.shape
    score = np.zeros((rows, cols), dtype=float)
    # 建立質數查核表（埃拉托斯特尼篩法）
    try:
        max_val = grid[grid != -1].max()
    except ValueError:
        max_val = 0
    if target > max_val:
        max_val = target
    if max_val < 2:
        is_prime = np.zeros(max_val + 1, dtype=bool)
    else:
        is_prime = np.ones(max_val + 1, dtype=bool)
        is_prime[:2] = False  # 0和1不是質數
        for p in range(2, int(max_val**0.5) + 1):
            if is_prime[p]:
                is_prime[p*p : max_val+1 : p] = False
    # 建立與 grid 同形狀的質數與合數掩碼
    prime_mask = (grid > 1) & (is_prime[grid])
    composite_mask = (grid > 1) & (~is_prime[grid])
    # 利用零邊界填充計算每格上下左右鄰居的質數/合數數量
    padded_prime = np.pad(prime_mask.astype(int), ((1,1),(1,1)), constant_values=0)
    padded_composite = np.pad(composite_mask.astype(int), ((1,1),(1,1)), constant_values=0)
    up_prime = padded_prime[0:rows, 1:cols+1]
    down_prime = padded_prime[2:rows+2, 1:cols+1]
    left_prime = padded_prime[1:rows+1, 0:cols]
    right_prime = padded_prime[1:rows+1, 2:cols+2]
    prime_neighbor_count = up_prime + down_prime + left_prime + right_prime
    up_composite = padded_composite[0:rows, 1:cols+1]
    down_composite = padded_composite[2:rows+2, 1:cols+1]
    left_composite = padded_composite[1:rows+1, 0:cols]
    right_composite = padded_composite[1:rows+1, 2:cols+2]
    composite_neighbor_count = up_composite + down_composite + left_composite + right_composite
    blank_mask = (grid == -1)
    # 根據 target 是否質數，選擇質數鄰居數或合數鄰居數作為分數
    if target > 1 and is_prime[target]:
        score = prime_neighbor_count * blank_mask
    else:
        score = composite_neighbor_count * blank_mask
    return score.astype(float)

def constant_row_sum_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    列恆和模式模組：
    假設每列總和相同，若插入 target 能使該列總和達成與其他列相同，則給予分數。
    僅考慮每列至多一個遮蔽格的情況。
    """
    rows, cols = grid.shape
    score = np.zeros((rows, cols), dtype=float)
    # 計算各列已知值之和，並找出完整列的共同總和
    row_sums = np.sum(np.where(grid == -1, 0, grid), axis=1)
    complete_rows = (np.sum(grid == -1, axis=1) == 0)
    common_sum = None
    if np.any(complete_rows):
        full_sums = row_sums[complete_rows]
        if full_sums.size > 1:
            if np.all(full_sums == full_sums[0]):
                common_sum = full_sums[0]
        else:
            common_sum = full_sums[0]
    if common_sum is None:
        return score  # 無共同總和，不計分
    # 檢查每列若只有一個遮蔽格，放入 target 是否使該列和達到 common_sum
    for r in range(rows):
        if np.sum(grid[r] == -1) == 1:
            c = np.argmax(grid[r] == -1)
            if row_sums[r] + target == common_sum:
                score[r, c] = 1.0
    return score

def constant_col_sum_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    行恆和模式模組：
    假設每行總和相同，若插入 target 能使該行總和達成與其他行相同，則給予分數。
    僅考慮每行至多一個遮蔽格的情況。
    """
    return constant_row_sum_score(grid.T, target).T