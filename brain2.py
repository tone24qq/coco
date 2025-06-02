import numpy as np

def ex10(grid: np.ndarray) -> np.ndarray:
    """
    檢測反對角線方向（NE-SW）等差序列模式。
    """
    output = np.zeros(grid.shape, dtype=float)
    # 反對角線（三格從右上到左下）
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
    檢測主對角線方向（NW-SE）三連相同數字模式。
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
    檢測反對角線方向（NE-SW）三連相同數字模式。
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
    檢測整列（橫列）相同數字模式。如果某一橫列已知數字全都相同且至少一格缺失，
    則將該列所有缺失格標記為潛在相同數字。
    """
    output = np.zeros(grid.shape, dtype=float)
    n_rows, n_cols = grid.shape
    for i in range(n_rows):
        row = grid[i, :]
        if np.any(row == -1) and np.nanmin(row[row != -1]) == np.nanmax(row[row != -1]):
            # 該列的已知數值都相等
            output[i, row == -1] = 1.0
    return output

def ex14(grid: np.ndarray) -> np.ndarray:
    """
    檢測整行（直行）相同數字模式。縱向版本的 ex13。
    """
    output = np.zeros(grid.shape, dtype=float)
    n_rows, n_cols = grid.shape
    for j in range(n_cols):
        col = grid[:, j]
        if np.any(col == -1) and np.nanmin(col[col != -1]) == np.nanmax(col[col != -1]):
            output[col == -1, j] = 1.0
    return output

def ex15(grid: np.ndarray) -> np.ndarray:
    """
    檢測 2x2 方塊內相同數字模式。如果一個2x2區塊有三個格子相同數字，
    一個缺失，則缺失格也很可能是相同數字。
    """
    output = np.zeros(grid.shape, dtype=float)
    # 定義2x2區塊四個元素
    tl = grid[:-1, :-1]   # top-left
    tr = grid[:-1, 1:]    # top-right
    bl = grid[1:, :-1]    # bottom-left
    br = grid[1:, 1:]     # bottom-right
    # 情況：右下角缺失，其餘三格已知且相等
    mask_br = (br == -1) & (tl != -1) & (tr != -1) & (bl != -1) & (tl == tr) & (tl == bl)
    output[1:, 1:][mask_br] = 1.0
    # 情況：左下角缺失
    mask_bl = (bl == -1) & (tl != -1) & (tr != -1) & (br != -1) & (tl == tr) & (tl == br)
    output[1:, :-1][mask_bl] = 1.0
    # 情況：右上角缺失
    mask_tr = (tr == -1) & (tl != -1) & (bl != -1) & (br != -1) & (tl == bl) & (tl == br)
    output[:-1, 1:][mask_tr] = 1.0
    # 情況：左上角缺失
    mask_tl = (tl == -1) & (tr != -1) & (bl != -1) & (br != -1) & (tr == bl) & (tr == br)
    output[:-1, :-1][mask_tl] = 1.0
    return output

def ex16(grid: np.ndarray) -> np.ndarray:
    """
    檢測四角相同數字模式。如果整個盤面的四個角落中，
    有三個角的數字已知且相同，最後一個角落缺失，則標記該缺失角落。
    """
    output = np.zeros(grid.shape, dtype=float)
    # 盤面四個角座標
    corners = [(0, 0), (0, grid.shape[1]-1), (grid.shape[0]-1, 0), (grid.shape[0]-1, grid.shape[1]-1)]
    corner_vals = [grid[i, j] for i, j in corners]
    known_vals = [v for v in corner_vals if v != -1]
    # 若有且只有一個角缺失，且其餘角已知值都相等
    if corner_vals.count(-1) == 1 and len(known_vals) > 0 and min(known_vals) == max(known_vals):
        missing_index = corner_vals.index(-1)
        i, j = corners[missing_index]
        output[i, j] = 1.0
    return output

def ex17(grid: np.ndarray) -> np.ndarray:
    """
    檢測十字形（+形狀）相同數字模式。如果中心或臂端有缺失，且十字形其他格皆已知且相同，
    則缺失格可能也是該相同數字。
    """
    output = np.zeros(grid.shape, dtype=float)
    n_rows, n_cols = grid.shape
    if n_rows < 3 or n_cols < 3:
        return output  # 太小的盤面無法形成十字形
    # 取出內部區域 (排除邊界) 作為中心可能位置
    center = grid[1:-1, 1:-1]
    up    = grid[:-2, 1:-1]
    down  = grid[2:, 1:-1]
    left  = grid[1:-1, :-2]
    right = grid[1:-1, 2:]
    # Case 1: 中心缺失，四個方向皆已知且相等
    mask_center = (center == -1) & (up != -1) & (down != -1) & (left != -1) & (right != -1) \
                  & (up == down) & (up == left) & (up == right)
    output[1:-1, 1:-1][mask_center] = 1.0
    # Case 2: 上方缺失（中心與其餘方向已知且相等）
    mask_up = (up == -1) & (center != -1) & (down != -1) & (left != -1) & (right != -1) \
              & (center == down) & (center == left) & (center == right)
    output[:-2, 1:-1][mask_up] = 1.0
    # Case 3: 下方缺失
    mask_down = (down == -1) & (center != -1) & (up != -1) & (left != -1) & (right != -1) \
                & (center == up) & (center == left) & (center == right)
    output[2:, 1:-1][mask_down] = 1.0
    # Case 4: 左方缺失
    mask_left = (left == -1) & (center != -1) & (up != -1) & (down != -1) & (right != -1) \
                & (center == up) & (center == down) & (center == right)
    output[1:-1, :-2][mask_left] = 1.0
    # Case 5: 右方缺失
    mask_right = (right == -1) & (center != -1) & (up != -1) & (down != -1) & (left != -1) \
                 & (center == up) & (center == down) & (center == left)
    output[1:-1, 2:][mask_right] = 1.0
    return output

def ex18(grid: np.ndarray) -> np.ndarray:
    """
    （保留）其他模式偵測槽位。當前未實作特殊邏輯，回傳全零矩陣。
    """
    # 尚未定義的新模式，可在此擴充
    return np.zeros(grid.shape, dtype=float)