from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


Array2D = np.ndarray


def _normalize_masked(scores: Array2D, missing_mask: Array2D) -> Array2D:
    out = np.zeros_like(scores, dtype=float)
    vals = scores[missing_mask]
    if vals.size == 0:
        return out
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if np.isclose(vmax, vmin):
        out[missing_mask] = 1.0
    else:
        out[missing_mask] = (vals - vmin) / (vmax - vmin)
    return out


def compute_focus_score(grid: Array2D) -> Array2D:
    missing_mask = grid == -1
    known_mask = ~missing_mask
    r, c = grid.shape
    scores = np.zeros((r, c), dtype=float)
    for i in range(r):
        for j in range(c):
            if not missing_mask[i, j]:
                continue
            r0, r1 = max(0, i - 1), min(r, i + 2)
            c0, c1 = max(0, j - 1), min(c, j + 2)
            scores[i, j] = float(np.sum(known_mask[r0:r1, c0:c1]))
    return _normalize_masked(scores, missing_mask)


def _line_skip_score(line: Array2D) -> Array2D:
    n = line.shape[0]
    out = np.zeros(n, dtype=float)
    known_idx = np.where(line != -1)[0]
    if known_idx.size < 3:
        return out
    for i in range(known_idx.size - 2):
        a, b, c = known_idx[i : i + 3]
        if (b - a) == (c - b):
            step = b - a
            dv1 = line[b] - line[a]
            dv2 = line[c] - line[b]
            if dv1 == dv2:
                nxt = c + step
                if 0 <= nxt < n and line[nxt] == -1:
                    out[nxt] += 1.0
                prev = a - step
                if 0 <= prev < n and line[prev] == -1:
                    out[prev] += 1.0
                mid = a + step
                if line[mid] == -1:
                    out[mid] += 0.5
    return out


def detect_skip_patterns(grid: Array2D) -> Array2D:
    missing_mask = grid == -1
    r, c = grid.shape
    scores = np.zeros((r, c), dtype=float)
    for i in range(r):
        scores[i, :] += _line_skip_score(grid[i, :])
    for j in range(c):
        scores[:, j] += _line_skip_score(grid[:, j])
    scores[~missing_mask] = 0.0
    return _normalize_masked(scores, missing_mask)


def _line_trend_score(line: Array2D) -> Array2D:
    n = line.shape[0]
    out = np.zeros(n, dtype=float)
    for i in range(2, n):
        if line[i - 2] != -1 and line[i - 1] != -1 and line[i] == -1:
            out[i] += 1.0
    return out


def compute_difference_trend(grid: Array2D) -> Array2D:
    missing_mask = grid == -1
    r, c = grid.shape
    scores = np.zeros((r, c), dtype=float)
    for i in range(r):
        scores[i, :] += _line_trend_score(grid[i, :])
    for j in range(c):
        scores[:, j] += _line_trend_score(grid[:, j])
    scores[~missing_mask] = 0.0
    return _normalize_masked(scores, missing_mask)


def detect_mirror_sequences(grid: Array2D) -> Array2D:
    missing_mask = grid == -1
    r, c = grid.shape
    scores = np.zeros((r, c), dtype=float)
    for i in range(r):
        for j in range(c):
            if not missing_mask[i, j]:
                continue
            max_k = min(j + 1, c - j)
            for k in range(1, max_k):
                l, rr = j - k, j + k
                if grid[i, l] != -1 and grid[i, rr] != -1 and grid[i, l] == grid[i, rr]:
                    scores[i, j] += 1.0
            max_kv = min(i + 1, r - i)
            for k in range(1, max_kv):
                u, d = i - k, i + k
                if grid[u, j] != -1 and grid[d, j] != -1 and grid[u, j] == grid[d, j]:
                    scores[i, j] += 1.0
    return _normalize_masked(scores, missing_mask)


def connectivity_heatmap(grid: Array2D) -> Array2D:
    missing_mask = grid == -1
    known_positions = np.argwhere(grid != -1)
    r, c = grid.shape
    scores = np.zeros((r, c), dtype=float)
    if known_positions.size == 0:
        return scores
    for i in range(r):
        for j in range(c):
            if not missing_mask[i, j]:
                continue
            dist = np.abs(known_positions[:, 0] - i) + np.abs(known_positions[:, 1] - j)
            dist = np.maximum(dist, 1)
            scores[i, j] = float(np.sum(1.0 / dist))
    return _normalize_masked(scores, missing_mask)


def sequence_tail_analyzer(grid: Array2D) -> Array2D:
    missing_mask = grid == -1
    r, c = grid.shape
    scores = np.zeros((r, c), dtype=float)
    known_positions = np.argwhere(grid != -1)
    if known_positions.size == 0:
        return scores
    known_values = grid[grid != -1]
    tails = known_values % 10
    counts = np.bincount(tails, minlength=10).astype(float)
    freq = counts / max(float(np.sum(counts)), 1.0)
    for i in range(r):
        for j in range(c):
            if not missing_mask[i, j]:
                continue
            v = 0.0
            for (x, y) in known_positions:
                dist = abs(x - i) + abs(y - j)
                if dist == 0:
                    continue
                tail = int(grid[x, y] % 10)
                v += freq[tail] / dist
            scores[i, j] = v
    return _normalize_masked(scores, missing_mask)


def ext_a2_weighted_proximity_vec(
    grid: Array2D,
    radius: int = 2,
    value_weight_factor: float = 1.0,
    decay: float = 1.0,
) -> Array2D:
    missing_mask = grid == -1
    r, c = grid.shape
    scores = np.zeros((r, c), dtype=float)
    known_positions = np.argwhere(grid != -1)
    for i in range(r):
        for j in range(c):
            if not missing_mask[i, j]:
                continue
            s = 0.0
            for x, y in known_positions:
                dist = abs(x - i) + abs(y - j)
                if dist == 0 or dist > radius:
                    continue
                s += (grid[x, y] * value_weight_factor) / (dist**decay)
            scores[i, j] = s
    return _normalize_masked(scores, missing_mask)


def ext_d3_potential_field_vec(grid: Array2D, decay_exponent: float = 1.0) -> Array2D:
    missing_mask = grid == -1
    r, c = grid.shape
    scores = np.zeros((r, c), dtype=float)
    known_positions = np.argwhere(grid != -1)
    for i in range(r):
        for j in range(c):
            if not missing_mask[i, j]:
                continue
            s = 0.0
            for x, y in known_positions:
                dist = abs(x - i) + abs(y - j)
                dist = max(dist, 1)
                s += grid[x, y] / (dist**decay_exponent)
            scores[i, j] = s
    return _normalize_masked(scores, missing_mask)


def ext_f10_discontinuity_vec(grid: Array2D) -> Array2D:
    missing_mask = grid == -1
    r, c = grid.shape
    scores = np.zeros((r, c), dtype=float)
    for i in range(r):
        for j in range(c):
            if grid[i, j] != -1:
                continue
            if 0 < j < c - 1 and grid[i, j - 1] != -1 and grid[i, j + 1] != -1:
                scores[i, j] += 1.0
            if 0 < i < r - 1 and grid[i - 1, j] != -1 and grid[i + 1, j] != -1:
                scores[i, j] += 1.0
            if 0 < i < r - 1 and 0 < j < c - 1:
                if grid[i - 1, j - 1] != -1 and grid[i + 1, j + 1] != -1:
                    scores[i, j] += 0.5
                if grid[i - 1, j + 1] != -1 and grid[i + 1, j - 1] != -1:
                    scores[i, j] += 0.5
    return _normalize_masked(scores, missing_mask)


def center_baseline_score(grid: Array2D) -> Array2D:
    missing_mask = grid == -1
    r, c = grid.shape
    rr, cc = np.indices((r, c))
    cr = (r - 1) / 2.0
    cc0 = (c - 1) / 2.0
    dist = np.sqrt((rr - cr) ** 2 + (cc - cc0) ** 2)
    score = np.max(dist) - dist
    score[~missing_mask] = 0.0
    return _normalize_masked(score, missing_mask)


MODULE_REGISTRY = {
    "focus": compute_focus_score,
    "skip": detect_skip_patterns,
    "diff": compute_difference_trend,
    "mirror": detect_mirror_sequences,
    "connectivity": connectivity_heatmap,
    "tail": sequence_tail_analyzer,
    "a2": ext_a2_weighted_proximity_vec,
    "d3": ext_d3_potential_field_vec,
    "f10": ext_f10_discontinuity_vec,
}


@dataclass
class ModuleOutput:
    scores: Dict[str, Array2D]
    missing_mask: Array2D


def compute_all_module_scores(grid: Array2D, enabled_modules: List[str]) -> ModuleOutput:
    missing_mask = grid == -1
    outputs: Dict[str, Array2D] = {}
    for module in enabled_modules:
        fn = MODULE_REGISTRY[module]
        scores = fn(grid)
        scores[~missing_mask] = 0.0
        outputs[module] = scores
    return ModuleOutput(scores=outputs, missing_mask=missing_mask)


def rank_missing_cells(score_grid: Array2D, missing_mask: Array2D) -> List[Tuple[Tuple[int, int], float]]:
    cells = np.argwhere(missing_mask)
    ranked = [((int(i), int(j)), float(score_grid[i, j])) for i, j in cells]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
