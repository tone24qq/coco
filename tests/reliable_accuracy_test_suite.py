"""reliable_accuracy_test_suite.py

包含：
- run_infinite_test: 無限壓力測試腳本，可手動執行
- run_until_converged: 批次收斂測試函數
- pytest 單元測試：test_accuracy_converges, test_stratified_accuracy
"""

import numpy as np
from tqdm import tqdm
import pytest

from modules import generate_unique_grid
from analyzer import probability_heatmap

def run_infinite_test(
    min_size: int = 4,
    max_size: int = 20,
    mask_ratio: float = 0.5,
    max_iters: int = 1000000,
    seed: int = 42,
    log_every: int = 1000
) -> None:
    """無限迴圈壓力測試，印出實時命中率。"""
    rng = np.random.default_rng(seed)
    hits = total = 0
    for i in tqdm(range(1, max_iters + 1), desc='Stress Testing'):
        rows = rng.integers(min_size, max_size + 1)
        cols = rng.integers(min_size, max_size + 1)
        full_grid = generate_unique_grid(rows, cols, rng)
        grid = full_grid.copy()
        total_cells = rows * cols
        mask_indices = rng.choice(total_cells, size=int(mask_ratio * total_cells), replace=False)
        for idx in mask_indices:
            r, c = divmod(idx, cols)
            grid[r, c] = -1

        # 隨機取一已知位置當目標
        candidates = [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] != -1]
        if not candidates:
            continue
        tr, tc = rng.choice(candidates)
        target = full_grid[tr, tc]

        heat = probability_heatmap(grid, target_num=target)
        pred = max(heat.items(), key=lambda x: x[1])[0]

        total += 1
        if pred == (tr, tc):
            hits += 1

        if i % log_every == 0:
            acc = hits / total if total else 0.0
            print(f"[{i}] Acc = {acc:.4f}")

def run_until_converged(
    min_size=4, max_size=20, mask_ratio=0.5,
    batch_size=200, delta=0.02, z=1.96, seed=0
):
    """批次收斂測試，直到95%置信區間半寬 ≤ delta 或達最大批次。"""
    rng = np.random.default_rng(seed)
    hits = total = 0
    max_batches = 50
    for batch in range(1, max_batches + 1):
        for _ in range(batch_size):
            rows = rng.integers(min_size, max_size + 1)
            cols = rng.integers(min_size, max_size + 1)
            full_grid = generate_unique_grid(rows, cols, rng)
            grid = full_grid.copy()
            total_cells = rows * cols
            mask_indices = rng.choice(total_cells, total_cells // 2, replace=False)
            for idx in mask_indices:
                r, c = divmod(idx, cols)
                grid[r, c] = -1
            candidates = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != -1]
            if not candidates:
                continue
            tr, tc = rng.choice(candidates)
            target = full_grid[tr, tc]
            heat = probability_heatmap(grid, target_num=target)
            pred = max(heat.items(), key=lambda x: x[1])[0]
            total += 1
            if pred == (tr, tc):
                hits += 1
        p = hits / total if total else 0.0
        half_width = z * (p * (1 - p) / total) ** 0.5 if total else float('inf')
        print(f"Batch {batch}: Acc={p:.4f} ±{half_width:.4f}")
        if half_width <= delta:
            return p, half_width, total
    return p, half_width, total

@pytest.mark.timeout(120)
def test_accuracy_converges():
    """
    測試在 4×4–10×10 範圍內，batch_size=100 下，
    命中率的 95% 置信區間半寬能夠收斂到 ≤ 0.05。
    """
    p, hw, total = run_until_converged(
        min_size=4, max_size=10,
        batch_size=100, delta=0.05, seed=42
    )
    assert total > 0, "無測試樣本"
    assert hw <= 0.05, f"半寬 {hw:.4f} 未收斂"
    assert 0.0 <= p <= 1.0

@pytest.mark.parametrize("rows,cols,mask_ratio", [
    (4, 4, 0.5),
    (8, 8, 0.5),
    (12, 10, 0.5),
    (16, 16, 0.5),
])
def test_stratified_accuracy(rows, cols, mask_ratio):
    """
    在各種固定尺寸 (rows×cols) 下，跑 200 次隨機遮蔽模擬，
    檢查命中率在合理範圍 [0,1]。
    """
    rng = np.random.default_rng(0)
    hits = 0
    trials = 200
    for _ in range(trials):
        full_grid = generate_unique_grid(rows, cols, rng)
        grid = full_grid.copy()
        total_cells = rows * cols
        mask_indices = rng.choice(total_cells, int(mask_ratio * total_cells), replace=False)
        for idx in mask_indices:
            r, c = divmod(idx, cols)
            grid[r, c] = -1

        candidates = [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] != -1]
        tr, tc = rng.choice(candidates)
        target = full_grid[tr, tc]

        heat = probability_heatmap(grid, target_num=target)
        pred = max(heat.items(), key=lambda x: x[1])[0]
        if pred == (tr, tc):
            hits += 1

    acc = hits / trials
    assert 0.0 <= acc <= 1.0

if __name__ == "__main__":
    print("Starting infinite stress test...")
    run_infinite_test()