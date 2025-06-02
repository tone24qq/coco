import numpy as np
import math
import brain1, brain2, brain3, new_module

# 模組權重設定：使用者可根據需要調整各 ex 模組的權重
MODULE_WEIGHTS = {i: 1.0 for i in range(1, 29)}  # 預設 ex1～ex28 權重皆為 1.0

# 正規化方法設定：可選 "minmax" 或 "zscore"
NORMALIZE_METHOD = "minmax"

def analyze_full_board(grid: np.ndarray) -> np.ndarray:
    """
    統一分析入口函式。對給定的數字盤面 (grid) 執行所有模組分析，
    將各模組回傳的分數矩陣正規化並依權重加權合併，回傳最終的分數矩陣。
    """
    # 基本防呆：確認輸入為二維陣列
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)
    if grid.ndim != 2:
        raise ValueError("Input grid must be a 2D numpy array.")
    # 將輸入盤面資料轉為 float 型別以利後續計算（保留 -1 等特殊值）
    grid = grid.astype(float)
    n_rows, n_cols = grid.shape
    # 動態收集所有 exXX 函式
    modules = []
    for module in (brain1, brain2, brain3, new_module):
        for name in dir(module):
            if name.startswith("ex") and name[2:].isdigit():
                func = getattr(module, name)
                if callable(func):
                    modules.append(func)
    # 按編號順序排序函式列表
    modules.sort(key=lambda f: int(f.__name__[2:]))
    # 執行所有模組分析並收集結果矩陣
    score_matrices = []
    for func in modules:
        mod_num = int(func.__name__[2:])
        weight = MODULE_WEIGHTS.get(mod_num, 1.0)
        try:
            result = func(grid)
        except Exception as e:
            # 模組內部錯誤被捕捉，使用全零矩陣代替，並記錄錯誤訊息
            result = np.zeros((n_rows, n_cols))
            print(f"Module ex{mod_num} raised an error: {e}")
        # 確保輸出為 numpy 陣列且形狀與輸入相同
        result = np.array(result, dtype=float)
        if result.shape != grid.shape:
            # 若形狀不符，輸出空矩陣代替
            result = np.zeros((n_rows, n_cols), dtype=float)
        # 模組輸出正規化（Min-Max 或 Z-Score）
        if NORMALIZE_METHOD.lower() == "minmax":
            min_val = result.min()
            max_val = result.max()
            if max_val > min_val:
                normed = (result - min_val) / (max_val - min_val)
            else:
                # 若全零或常數矩陣，正規化後用0矩陣代表無顯著信號
                normed = np.zeros_like(result)
        elif NORMALIZE_METHOD.lower() == "zscore":
            mean = result.mean()
            std = result.std()
            if std > 0:
                normed = (result - mean) / std
            else:
                # 同樣處理常數矩陣的情況
                normed = np.zeros_like(result)
        else:
            # 未知的正規化方法，直接使用原值（不建議）
            normed = result.copy()
        # 將正規化矩陣乘以該模組權重
        weighted = normed * weight
        score_matrices.append(weighted)
    # 將所有模組的加權分數矩陣累加
    if score_matrices:
        total_score = np.sum(score_matrices, axis=0)
    else:
        total_score = np.zeros((n_rows, n_cols))
    return total_score

# 測試與驗證（單元測試範例）
if __name__ == "__main__":
    # 1. 極端值測試盤面：所有值相同或極端，確保正規化處理不出錯
    grid1 = np.full((3, 3), 9999)  # 3x3 全部是極端高值
    print("Test1 (Extreme values uniform grid):")
    print(analyze_full_board(grid1))  # 預期全零矩陣（無任何特殊模式）

    # 2. 空白盤測試：全部為 -1（未知），無已知值可分析
    grid2 = np.full((4, 4), -1)  # 4x4 全部遮蔽
    print("\nTest2 (Empty/masked grid):")
    print(analyze_full_board(grid2))  # 預期全零矩陣

    # 3. 遮蔽盤測試：包含部分已知且存在明顯模式可推論遮蔽值
    # 例如：中間遮蔽，周圍數字形成十字對稱且相等，預期中間為相同數字
    grid3 = np.array([
        [4, 6, 8],
        [6, -1, 6],
        [8, 6, 4]
    ])
    print("\nTest3 (Patterned grid with a mask):")
    result3 = analyze_full_board(grid3)
    print(result3)
    # 預期輸出: 中心點得高分 (多個模組推斷該遮蔽位應為6)，其他位置為0。
    # （由於有多個模組支持，此處中心格的分數可能累加成 4.0）

    # 4. 規律盤完整測試：盤面數字完整呈現某種規律但無遮蔽值
    grid4 = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7]
    ])
    print("\nTest4 (Fully revealed pattern grid):")
    print(analyze_full_board(grid4))
    # 預期輸出: 全零矩陣（雖有遞增規律，但無缺失值需要推理）

    # 5. 錯誤輸入測試：非2D或無效輸入，應當被攔截處理不致當機
    bad_input = [1, 2, 3]  # 非二維陣列
    print("\nTest5 (Invalid input):")
    try:
        _ = analyze_full_board(np.array(bad_input))
        print("Error: invalid input was not caught as expected.")
    except Exception as e:
        print(f"Caught expected exception for invalid input: {e}")