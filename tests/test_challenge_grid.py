import pytest
import inspect
from your_module import recover_masked_grid, locate_target_by_partial_grid

# 1. 定義 Challenge / Answer grids
CHALLENGE_GRID = [
    [56, -1, 82, -1, 70, -1, -1, 47, -1, 19, 24, -1],
    [-1, 2, -1, 94, -1, 5, -1, -1, 6, -1, 85, 92],
    [60, -1, -1, 7, 84, -1, 21, 15, -1, -1, 66, 40],
    [-1, 48, 99, 10, -1, 37, -1, 38, 75, 55, 97, -1],
    [53, -1, -1, 41, 23, -1, 20, -1, 34, -1, 69, 67],
    [-1, 91, 26, 17, -1, -1, 93, 32, 9, -1, 87, -1],
    [90, 3, -1, 80, -1, 95, 45, -1, 35, 81, -1, 1],
    [74, -1, 28, 4, 49, -1, -1, 65, 8, 100, -1, 51],
    [-1, 79, 14, 46, -1, 102, -1, 105, -1, 106, 108, -1],
    [107, -1, -1, 112, 113, 114, 115, -1, 117, 118, -1, 120],
]

ANSWER_GRID = [
    [56, 88, 82, 39, 70, 89, 12, 47, 44, 19, 24, 52],
    [42, 2, 98, 94, 54, 5, 18, 58, 6, 27, 85, 92],
    [60, 13, 73, 7, 84, 77, 21, 15, 31, 71, 66, 40],
    [62, 48, 99, 10, 59, 37, 16, 38, 75, 55, 97, 29],
    [53, 72, 36, 41, 23, 76, 20, 83, 34, 86, 69, 67],
    [30, 91, 26, 17, 63, 61, 93, 32, 9, 57, 87, 50],
    [90, 3, 33, 80, 96, 95, 45, 25, 35, 81, 11, 1],
    [74, 64, 28, 4, 49, 78, 22, 65, 8, 100, 43, 51],
    [68, 79, 14, 46, 100, 102, 109, 105, 104, 106, 108, 101],
    [107, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
]

# 2. 不報錯測試
def test_recover_masked_grid_no_error():
    # 確保函式可順利執行
    recovered = recover_masked_grid([row[:] for row in CHALLENGE_GRID])
    assert recovered is not None

# 3. 形狀與原值保留
def test_recover_masked_grid_shape_and_preserve():
    out = recover_masked_grid([row[:] for row in CHALLENGE_GRID])
    assert len(out) == len(CHALLENGE_GRID)
    for r, row in enumerate(CHALLENGE_GRID):
        assert len(out[r]) == len(row)
        for c, val in enumerate(row):
            if val != -1:
                assert out[r][c] == val, f"原值被改寫 r={r},c={c}"

# 4. 合法性與不重複
def test_recover_masked_grid_valid_values_no_duplicates():
    out = recover_masked_grid([row[:] for row in CHALLENGE_GRID])
    flat = [v for row in out for v in row]
    total = len(flat)
    assert set(flat) == set(range(1, total+1)), "必須包含 1…rows×cols，且不重複"

# 5. 完整正確性
def test_recover_masked_grid_exact_answer():
    out = recover_masked_grid([row[:] for row in CHALLENGE_GRID])
    assert out == ANSWER_GRID

# 6. Top-3 命中率測試（如有提供 locate_target_by_partial_grid）
def test_locate_target_top3_includes_answer():
    for r in range(len(CHALLENGE_GRID)):
        for c in range(len(CHALLENGE_GRID[0])):
            if CHALLENGE_GRID[r][c] == -1:
                target = ANSWER_GRID[r][c]
                top3 = locate_target_by_partial_grid(CHALLENGE_GRID, target)
                assert (r, c) in top3, f"target={target} 應包含位置 {(r,c)}"

# 7. 防作弊（簡易檢測：確保模組內沒硬編碼 ANSWER_GRID）
def test_no_hardcoded_answer_in_module():
    import your_module
    src = inspect.getsource(your_module)
    assert "ANSWER_GRID" not in src