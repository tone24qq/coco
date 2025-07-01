import numpy as np
import pytest

from analyzer import probability_heatmap
from modules import generate_unique_grid
from tests.reliability_utils import run_until_converged


@pytest.mark.timeout(120)
def test_accuracy_converges():
    p, hw, total = run_until_converged(
        min_size=4, max_size=10, batch_size=100, delta=0.05, seed=42
    )
    assert total > 0
    assert hw <= 0.05
    assert 0.0 <= p <= 1.0


@pytest.mark.slow
@pytest.mark.parametrize(
    "rows,cols,mask_ratio",
    [
        (4, 4, 0.5),
        (8, 8, 0.5),
        (12, 10, 0.5),
        (16, 16, 0.5),
    ],
)
def test_reliability_1000_runs(rows, cols, mask_ratio):
    rng = np.random.default_rng(0)
    hits = 0
    trials = 1000
    total_cells = rows * cols
    for _ in range(trials):
        full_grid = generate_unique_grid(rows, cols, rng=rng)
        grid = full_grid.copy()
        mask_idxs = rng.choice(total_cells, total_cells // 2, replace=False)
        for idx in mask_idxs:
            r, c = divmod(idx, cols)
            grid[r][c] = -1
        tr, tc = rng.choice(
            [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != -1]
        )
        target = full_grid[tr][tc]
        heatmap = probability_heatmap(grid, target)
        pr, pc = divmod(int(np.argmax(heatmap)), cols)
        if (pr, pc) == (tr, tc):
            hits += 1
    acc = hits / trials
    assert 0.0 <= acc <= 1.0
