import numpy as np

import brain


def ref_q2(
    grid: np.ndarray, *, w_adj: float = 1.0, threshold: float = 0.0
) -> np.ndarray:
    rows, cols = grid.shape
    valid = grid != -1
    counts = np.zeros((rows, cols), dtype=float)
    known = np.zeros((rows, cols), dtype=int)
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dr, dc in dirs:
        r1 = slice(max(0, -dr), rows - max(0, dr))
        c1 = slice(max(0, -dc), cols - max(0, dc))
        r2 = slice(max(0, dr), rows - max(0, -dr))
        c2 = slice(max(0, dc), cols - max(0, -dc))
        a = grid[r1, c1]
        b = grid[r2, c2]
        mask = valid[r1, c1] & valid[r2, c2] & (np.abs(a - b) == 1)
        counts[r1, c1] += mask
        counts[r2, c2] += mask
        known[r1, c1] += valid[r2, c2]
        known[r2, c2] += valid[r1, c1]
    active = (counts >= 2) | (known == 0)
    score = np.where(active, counts, 0.0)
    if threshold > 0:
        return np.zeros_like(score)
    mx = score.max(initial=0.0)
    if mx > 0:
        score = (score / mx) * w_adj
    else:
        score.fill(0.0)
    return score


def ref_q3(grid: np.ndarray) -> np.ndarray:
    rows, cols = grid.shape
    score = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        for c in range(cols):
            r2, c2 = rows - 1 - r, cols - 1 - c
            if r > r2 or (r == r2 and c >= c2):
                continue
            if grid[r, c] != -1 and grid[r2, c2] != -1 and grid[r, c] == grid[r2, c2]:
                score[r, c] += 1.0
                score[r2, c2] += 1.0
    mx = score.max(initial=0.0)
    if mx > 0:
        score /= mx
    return score


def ref_q6(grid: np.ndarray) -> np.ndarray:
    rows, cols = grid.shape
    score = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols and grid[r, c] == grid[r, c + 1]:
                score[r, c] += 1.0
                score[r, c + 1] += 1.0
            if r + 1 < rows and grid[r, c] == grid[r + 1, c]:
                score[r, c] += 1.0
                score[r + 1, c] += 1.0
            val = grid[r, c]
            matches = 0.0
            if r > 0 and grid[r - 1, c] == val:
                matches += 1.0
            if r + 1 < rows and grid[r + 1, c] == val:
                matches += 1.0
            if c > 0 and grid[r, c - 1] == val:
                matches += 1.0
            if c + 1 < cols and grid[r, c + 1] == val:
                matches += 1.0
            score[r, c] += matches / 4.0
    mx = score.max(initial=1.0)
    score /= mx
    return score


def _make_grid(r: int, c: int) -> np.ndarray:
    grid = np.arange(1, r * c + 1, dtype=int).reshape(r, c)
    grid[r // 2, c // 2] = -1
    return grid


def test_q2_vectorized_matches_reference():
    g = _make_grid(5, 6)
    assert np.allclose(
        brain.EXT_Q2_PotentialPath_Vec(g, w_adj=1.0, threshold=0.0),
        ref_q2(g, w_adj=1.0, threshold=0.0),
    )


def test_q3_vectorized_matches_reference():
    g = _make_grid(6, 5)
    assert np.allclose(brain.EXT_Q3_DiscontinuitySym_Vec(g), ref_q3(g))


def test_q6_vectorized_matches_reference():
    g = _make_grid(5, 5)
    assert np.allclose(brain.EXT_Q6_LineBridge_Vec(g), ref_q6(g))


def test_q2_threshold_blocks_score():
    g = np.array([[1, 2], [3, -1]])
    priors = {
        (0, 0): {1: 0.1},
        (0, 1): {2: 0.1},
        (1, 0): {3: 0.1},
        (1, 1): {},
    }
    out = brain.EXT_Q2_PotentialPath_Vec(g, target=1, priors=priors, threshold=0.5)
    assert np.all(out == 0)
