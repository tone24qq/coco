from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GlobalDecodeResult:
    grid: list[list[int | None]]
    low_confidence_cells: list[dict[str, object]]
    decode_backend: str


def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    n = cost.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    out: list[tuple[int, int]] = []
    for j in range(1, n + 1):
        if p[j] > 0:
            out.append((p[j] - 1, j - 1))
    return out


def decode_with_constraints(
    rows: int,
    cols: int,
    candidates: dict[tuple[int, int], list[dict[str, float | int]]],
    min_assign_score: float = 0.05,
) -> GlobalDecodeResult:
    total = rows * cols
    mat = np.full((total, total), 9.0, dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            for rank, item in enumerate(candidates.get((r, c), [])[:6]):
                v = int(item.get("value", 0))
                if v < 1 or v > total:
                    continue
                score = float(item.get("score", 0.0))
                score = max(0.0, min(1.0, score)) * (1.0 - rank * 0.06)
                mat[idx, v - 1] = min(mat[idx, v - 1], 1.0 - score)

    assignment = _hungarian(mat)
    grid: list[list[int | None]] = [[None for _ in range(cols)] for _ in range(rows)]
    low: list[dict[str, object]] = []
    for ci, vi in assignment:
        r, c = divmod(int(ci), cols)
        r = int(r)
        c = int(c)
        score = float(1.0 - mat[ci, vi])
        if score >= min_assign_score:
            grid[r][c] = int(vi + 1)
        else:
            low.append(
                {
                    "row": int(r),
                    "col": int(c),
                    "reason": "global_decode_low_score",
                    "score": float(score),
                    "needs_review": True,
                }
            )
    return GlobalDecodeResult(grid=grid, low_confidence_cells=low, decode_backend="hungarian_unique_assignment")
