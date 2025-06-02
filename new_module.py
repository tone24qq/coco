import numpy as np

def ex27(grid: np.ndarray) -> np.ndarray:
    """
    新增模式：跨行連續數列檢測。
    將整個盤面視為按行連續展開的一維序列，若整體為等差序列且僅缺少一個數字，則標記該缺失位置。
    """
    output = np.zeros(grid.shape, dtype=float)
    # 將盤面按行優先展平為一維
    rows, cols = grid.shape
    flat = grid.flatten()
    # 若整體僅有一處缺失
    missing_indices = np.where(flat == -1)[0]
    if flat.size == 0 or missing_indices.size != 1:
        return output  # 僅處理單一缺失的情況
    miss_idx = missing_indices[0]
    known = np.delete(flat, miss_idx)
    # 判斷已知部分是否等差序列（允許序列首尾缺一值）
    if known.size < 2:
        return output
    diffs = np.diff(known)
    # 以已知部分最常見的差值作為序列公差
    # （這裡假設除缺失處外序列公差一致）
    diff_values, diff_counts = np.unique(diffs, return_counts=True)
    if diff_values.size == 0:
        return output
    common_diff = diff_values[np.argmax(diff_counts)]
    # 檢查是否除一處外所有差值相等且那一處差值正好是 common_diff 的兩倍
    double_gap_indices = np.where(diffs == 2 * common_diff)[0]
    if diffs.size > 0 and double_gap_indices.size == 1:
        # 存在一個雙倍差，認定那裡是缺失點
        pass
    # 確認所有已知差值等於 common_diff
    if np.all((diffs == common_diff) | (diffs == 2 * common_diff)):
        # 預測缺失值可能位置合理，標記輸出
        # 換算缺失索引回 2D 座標
        r, c = divmod(miss_idx, cols)
        output[r, c] = 1.0
    return output

def ex28(grid: np.ndarray) -> np.ndarray:
    """
    新增模式：跨列連續數列檢測。
    將盤面視為按列連續展開的一維序列進行等差序列檢測，原理同 ex27。
    """
    output = np.zeros(grid.shape, dtype=float)
    # 將盤面按列優先展平為一維
    rows, cols = grid.shape
    flat = grid.T.flatten()  # 轉置後展平，相當於列序展開
    missing_indices = np.where(flat == -1)[0]
    if flat.size == 0 or missing_indices.size != 1:
        return output
    miss_idx = missing_indices[0]
    known = np.delete(flat, miss_idx)
    if known.size < 2:
        return output
    diffs = np.diff(known)
    diff_values, diff_counts = np.unique(diffs, return_counts=True)
    if diff_values.size == 0:
        return output
    common_diff = diff_values[np.argmax(diff_counts)]
    if np.all((diffs == common_diff) | (diffs == 2 * common_diff)):
        # 換算缺失索引回原始 2D 座標（列序展開轉回矩陣坐標）
        c, r = divmod(miss_idx, rows)
        output[r, c] = 1.0
    return output