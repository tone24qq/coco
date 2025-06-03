# brain2.py
"""
模組組合2：奇偶模式與鄰近基本算術關係模組（和、差模式）。
"""
import numpy as np

def uniform_parity_row_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    單一奇偶模式 (橫列) 模組：
    若某一列所有已知數字同為奇數或偶數，則該列遮蔽格若 target 奇偶一致則給予高分。
    """
    rows, cols = grid.shape
    score = np.zeros((rows, cols), dtype=float)
    mask = (grid != -1)
    known_count = mask.sum(axis=1)
    # odd_count 計算每列奇數的個數
    odd_count = ((grid % 2 != 0) & mask).sum(axis=1)
    all_even = (odd_count == 0)  # 該列所有已知值為偶數
    all_odd = (odd_count == known_count) & (known_count > 0)  # 該列所有已知值為奇數
    # 需至少2個已知值才判定存在奇偶模式
    uniform_pattern = (known_count >= 2) & (all_even | all_odd)
    # 根據 target 奇偶判斷符合模式的列
    if target % 2 == 0:
        rows_match = uniform_pattern & all_even
    else:
        rows_match = uniform_pattern & all_odd
    # 將符合條件列中的所有遮蔽格設為1分
    if rows_match.any():
        blank_mask = (grid == -1)
        score[rows_match, :] = blank_mask[rows_match, :].astype(float)
    return score

def uniform_parity_col_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    單一奇偶模式 (直行) 模組：
    與 uniform_parity_row_score 類似，但檢查直行。
    """
    return uniform_parity_row_score(grid.T, target).T

def alternate_parity_row_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    交錯奇偶模式 (橫列) 模組：
    偵測每個遮蔽格是否需要與相鄰數字奇偶相反來滿足交錯模式。
    若左右鄰居同為偶數或奇數，則該遮蔽格應為相反奇偶；若鄰居奇偶相反則無法交錯匹配。
    """
    rows, cols = grid.shape
    score = np.zeros((rows, cols), dtype=float)
    # 左鄰居與右鄰居的值（無鄰居或遮蔽則設為 -1）
    left_vals = np.pad(grid, ((0,0),(1,0)), constant_values=-1)[:, :-1]
    right_vals = np.pad(grid, ((0,0),(0,1)), constant_values=-1)[:, 1:]
    left_exist = (left_vals != -1)
    right_exist = (right_vals != -1)
    # 左右鄰居奇偶（0: 偶, 1: 奇），不存在鄰居時值無效
    left_par = left_vals % 2
    right_par = right_vals % 2
    # Case 1: 左右鄰居皆存在且奇偶相同 => 遮蔽格需為相反奇偶
    both_exist = left_exist & right_exist
    same_parity_neighbors = both_exist & (left_par == right_par)
    target_par = 0 if target % 2 == 0 else 1
    # 需要 target 奇偶與鄰居相反
    needed_parity_mask = same_parity_neighbors & (target_par != left_par)
    # Case 2: 僅單側鄰居存在 => 遮蔽格需與該鄰居奇偶相反
    left_only = left_exist & ~right_exist
    right_only = ~left_exist & right_exist
    satisfied_left = left_only & (target_par != left_par)
    satisfied_right = right_only & (target_par != right_par)
    # 綜合滿足交錯條件的遮蔽格
    satisfied_mask = needed_parity_mask | satisfied_left | satisfied_right
    blank_mask = (grid == -1)
    score[satisfied_mask & blank_mask] = 1.0
    return score

def alternate_parity_col_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    交錯奇偶模式 (直行) 模組：
    與 alternate_parity_row_score 類似，但檢查直行上下鄰居。
    """
    return alternate_parity_row_score(grid.T, target).T

def neighbor_sum_horizontal_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    鄰和模式 (水平方向) 模組：
    若遮蔽格左右兩側的數字存在且相加等於 target，則視為滿足條件。
    """
    rows, cols = grid.shape
    score = np.zeros((rows, cols), dtype=float)
    if cols >= 3:
        left = grid[:, :-2]
        center_blank = (grid[:, 1:-1] == -1)
        right = grid[:, 2:]
        valid = center_blank & (left != -1) & (right != -1)
        match = (left + right == target)
        score[:, 1:-1][valid & match] = 1.0
    return score

def neighbor_sum_vertical_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    鄰和模式 (垂直方向) 模組：
    若遮蔽格上下兩側的數字存在且相加等於 target，則視為滿足條件。
    """
    return neighbor_sum_horizontal_score(grid.T, target).T

def neighbor_diff_horizontal_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    鄰差模式 (水平方向) 模組：
    若遮蔽格左右兩側的數字存在且差的絕對值等於 target，則視為滿足條件。
    """
    rows, cols = grid.shape
    score = np.zeros((rows, cols), dtype=float)
    if cols >= 3:
        left = grid[:, :-2]
        center_blank = (grid[:, 1:-1] == -1)
        right = grid[:, 2:]
        valid = center_blank & (left != -1) & (right != -1)
        match = (np.abs(left - right) == target)
        score[:, 1:-1][valid & match] = 1.0
    return score

def neighbor_diff_vertical_score(grid: np.ndarray, target: int) -> np.ndarray:
    """
    鄰差模式 (垂直方向) 模組：
    若遮蔽格上下兩側的數字存在且差的絕對值等於 target，則視為滿足條件。
    """
    return neighbor_diff_horizontal_score(grid.T, target).T