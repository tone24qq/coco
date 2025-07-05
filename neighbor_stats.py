import functools
from collections import Counter
from typing import Dict

import numpy as np

__all__ = ["compute_neighbor_distribution", "neighbor_compatibility_score"]


@functools.lru_cache(maxsize=128)
def compute_neighbor_distribution(
    rows: int, cols: int, target: int | None, n_sims: int = 10000
) -> Dict[int, float]:
    """Monte Carlo estimate of neighbor value distribution for ``target``."""
    if target is None:
        return {}

    cnt: Counter[int] = Counter()
    deltas = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    rng = np.random.default_rng()
    for _ in range(n_sims):
        arr = np.arange(1, rows * cols + 1)
        rng.shuffle(arr)
        grid = arr.reshape(rows, cols)
        r, c = np.argwhere(grid == target)[0]
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cnt[grid[nr, nc]] += 1
    total = sum(cnt.values()) or 1
    return {v: cnt[v] / total for v in cnt}


def neighbor_compatibility_score(
    grid: np.ndarray, dist: Dict[int, float]
) -> np.ndarray:
    """Score each blank cell by compatibility with neighbor distribution."""
    rows, cols = grid.shape
    score = np.zeros((rows, cols), dtype=float)
    deltas = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    ranked_values = [
        v for v, _ in sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    ]
    max_level = len(ranked_values) + 1

    levels = np.full((rows, cols), max_level, dtype=float)

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            prod = 1.0
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                    prod *= dist.get(grid[nr, nc], 1e-6)
            score[r, c] = prod

            for idx, val in enumerate(ranked_values):
                if any(
                    0 <= r + dr < rows
                    and 0 <= c + dc < cols
                    and grid[r + dr, c + dc] == val
                    for dr, dc in deltas
                ):
                    levels[r, c] = idx + 1
                    break

    levels_inv = 1.0 / levels
    score *= levels_inv
    mx = score.max(initial=0.0)
    return score / mx if mx > 0 else score
