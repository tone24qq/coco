import numpy as np
import math
import brain1, brain2, brain3, new_module

# 模組權重設定：使用者可自行調整 ex1～ex28 的權重
MODULE_WEIGHTS = {i: 1.0 for i in range(1, 29)}

# 正規化方式："minmax" 或 "zscore"
NORMALIZE_METHOD = "minmax"

def analyze_full_board(grid: np.ndarray) -> np.ndarray:
    """
    統一分析入口。對傳入的數字盤面 (grid) 執行所有模組分析，
    再將各模組回傳的分數矩陣做正規化並加權合併，最終回傳整張分數矩陣。
    如果所有分數都為 0，代表無任何模組有信心，則對每個遮蔽格給一個小基礎分 (0.1) 作為 fallback。
    """
    # 防呆：若非 NumPy 陣列，自動轉換
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)
    if grid.ndim != 2:
        raise ValueError("Input grid 必須為 2D numpy array。")
    grid = grid.astype(float)  # 轉為浮點型，保留 -1 等特殊數值
    n_rows, n_cols = grid.shape

    # 動態蒐集所有 exXX 模組函式
    modules = []
    for module in (brain1, brain2, brain3, new_module):
        for name in dir(module):
            if name.startswith("ex") and name[2:].isdigit():
                func = getattr(module, name)
                if callable(func):
                    modules.append(func)
    # 按函式名稱中的數字排序 (ex1, ex2, ...)
    modules.sort(key=lambda f: int(f.__name__[2:]))

    # 執行所有模組並收集分數矩陣
    score_matrices = []
    for func in modules:
        mod_num = int(func.__name__[2:])
        weight = MODULE_WEIGHTS.get(mod_num, 1.0)
        try:
            result = func(grid)
        except Exception as e:
            # 模組執行失敗時，用全零矩陣取代，並列印錯誤訊息
            result = np.zeros((n_rows, n_cols))
            print(f"Module ex{mod_num} 執行錯誤：{e}")
        result = np.array(result, dtype=float)
        if result.shape != grid.shape:
            # 若輸出形狀不符，用全零矩陣取代
            result = np.zeros((n_rows, n_cols), dtype=float)

        # 正規化：Min-Max 或 Z-Score
        if NORMALIZE_METHOD.lower() == "minmax":
            min_val = result.min()
            max_val = result.max()
            if max_val > min_val:
                normed = (result - min_val) / (max_val - min_val)
            else:
                normed = np.zeros_like(result)
        elif NORMALIZE_METHOD.lower() == "zscore":
            mean = result.mean()
            std = result.std()
            if std > 0:
                normed = (result - mean) / std
            else:
                normed = np.zeros_like(result)
        else:
            # 未知正規化方式，直接複製原始結果
            normed = result.copy()

        # 乘上對應權重
        weighted = normed * weight
        score_matrices.append(weighted)

    # 將所有加權後的分數累加
    if score_matrices:
        total_score = np.sum(score_matrices, axis=0)
    else:
        total_score = np.zeros((n_rows, n_cols))

    # 如果所有分數都為 0，對每個遮蔽格給一個很小的基礎分 (0.1) 作為 fallback
    if total_score.max() == 0:
        mask = (grid == -1)
        total_score[mask] = 0.1
        # 已知格保持 0
        total_score[~mask] = 0.0

    return total_score

# 測試與驗證部分保留（如原本有 __main__ 那段可不動）
if __name__ == "__main__":
    # （此處可保留你原本測試程式碼）
    import numpy as np

    grid1 = np.full((3, 3), 9999)
    print("測試1 (極端值均一格子)：")
    print(analyze_full_board(grid1))

    grid2 = np.full((4, 4), -1)
    print("\n測試2 (全遮蔽盤)：")
    print(analyze_full_board(grid2))

    grid3 = np.array([
        [4, 6, 8],
        [6, -1, 6],
        [8, 6, 4]
    ])
    print("\n測試3 (有模式且中間遮蔽)：")
    print(analyze_full_board(grid3))

    grid4 = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7]
    ])
    print("\n測試4 (完整遞增模式盤)：")
    print(analyze_full_board(grid4))

    bad_input = [1, 2, 3]
    print("\n測試5 (錯誤輸入)：")
    try:
        _ = analyze_full_board(np.array(bad_input))
        print("錯誤：沒有捕捉到非法輸入！")
    except Exception as e:
        print(f"成功捕捉錯誤：{e}")