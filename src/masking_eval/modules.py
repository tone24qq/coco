from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


Array2D = np.ndarray
Cell = Tuple[int, int]


def _get_lines(grid: Array2D, cell: Cell) -> Dict[str, Array2D]:
    i, j = cell
    row = grid[i, :]
    col = grid[:, j]
    diag = np.array([grid[i + k, j + k] for k in range(-min(i, j), min(grid.shape[0] - i, grid.shape[1] - j))])
    anti = np.array(
        [
            grid[i + k, j - k]
            for k in range(
                -min(i, grid.shape[1] - 1 - j),
                min(grid.shape[0] - i, j + 1),
            )
        ]
    )
    return {"row": row, "col": col, "diag": diag, "anti": anti}


def compute_focus_score(masked_grid: Array2D, target_cell: Cell) -> float:
    i, j = target_cell
    r, c = masked_grid.shape
    r0, r1 = max(0, i - 1), min(r, i + 2)
    c0, c1 = max(0, j - 1), min(c, j + 2)
    known = masked_grid[r0:r1, c0:c1] != -1
    return float(np.sum(known) / known.size)


def connectivity_heatmap(masked_grid: Array2D, target_cell: Cell) -> float:
    known = np.argwhere(masked_grid != -1)
    if known.size == 0:
        return 0.0
    i, j = target_cell
    dist = np.abs(known[:, 0] - i) + np.abs(known[:, 1] - j)
    dist = np.maximum(dist, 1)
    return float(np.sum(1.0 / dist))


def ext_heatmap_prior(target_cell: Cell, heatmap_prior: Array2D | None) -> float:
    if heatmap_prior is None:
        return 0.0
    i, j = target_cell
    return float(heatmap_prior[i, j])


def ext_a2_proximity(masked_grid: Array2D, target_cell: Cell, radius: int = 2, decay: float = 1.0) -> float:
    i, j = target_cell
    r, c = masked_grid.shape
    score = 0.0
    for x in range(max(0, i - radius), min(r, i + radius + 1)):
        for y in range(max(0, j - radius), min(c, j + radius + 1)):
            if masked_grid[x, y] == -1 or (x == i and y == j):
                continue
            d = max(abs(x - i) + abs(y - j), 1)
            score += float(masked_grid[x, y]) / (d**decay)
    return score


def ext_d3_potential(masked_grid: Array2D, target_cell: Cell, decay_exponent: float = 1.25) -> float:
    return connectivity_heatmap(masked_grid, target_cell) / max(decay_exponent, 1e-6)


def _count_local_arith(line: Array2D) -> float:
    score = 0.0
    for k in range(len(line) - 2):
        a, b, c = line[k], line[k + 1], line[k + 2]
        if -1 in (a, b, c):
            continue
        if (b - a) == (c - b):
            score += 1.0
    return score


def detect_skip_patterns(candidate_grid: Array2D, target_cell: Cell) -> float:
    lines = _get_lines(candidate_grid, target_cell)
    return _count_local_arith(lines["row"]) + _count_local_arith(lines["col"])


def compute_difference_trend(candidate_grid: Array2D, target_cell: Cell) -> float:
    lines = _get_lines(candidate_grid, target_cell)
    score = 0.0
    for line in (lines["row"], lines["col"]):
        idx = np.where(line != -1)[0]
        if idx.size >= 3:
            diffs = np.diff(line[idx])
            score += float(1.0 / (1.0 + np.var(diffs)))
    return score


def detect_mirror_sequences(candidate_grid: Array2D, target_cell: Cell) -> float:
    i, j = target_cell
    r, c = candidate_grid.shape
    score = 0.0
    for k in range(1, min(j + 1, c - j)):
        if candidate_grid[i, j - k] != -1 and candidate_grid[i, j + k] != -1:
            score += float(candidate_grid[i, j - k] == candidate_grid[i, j + k])
    for k in range(1, min(i + 1, r - i)):
        if candidate_grid[i - k, j] != -1 and candidate_grid[i + k, j] != -1:
            score += float(candidate_grid[i - k, j] == candidate_grid[i + k, j])
    return score


def sequence_tail_analyzer(masked_grid: Array2D, target_cell: Cell, candidate_value: int) -> float:
    i, j = target_cell
    known = np.argwhere(masked_grid != -1)
    if known.size == 0:
        return 0.0
    values = masked_grid[masked_grid != -1]
    freq = np.bincount(values % 10, minlength=10).astype(float)
    freq /= max(np.sum(freq), 1.0)
    cand_tail = candidate_value % 10
    return float(np.sum([freq[cand_tail] / max(abs(x - i) + abs(y - j), 1) for x, y in known]))


def ext_f10_discontinuity(candidate_grid: Array2D, target_cell: Cell) -> float:
    i, j = target_cell
    r, c = candidate_grid.shape
    score = 0.0
    if 0 < j < c - 1 and candidate_grid[i, j - 1] != -1 and candidate_grid[i, j + 1] != -1:
        score += 1.0
    if 0 < i < r - 1 and candidate_grid[i - 1, j] != -1 and candidate_grid[i + 1, j] != -1:
        score += 1.0
    return score


# discovery candidate-specific modules

def local_arithmetic_completion_score(candidate_grid: Array2D, target_cell: Cell) -> float:
    lines = _get_lines(candidate_grid, target_cell)
    score = 0.0
    for line in lines.values():
        score += _count_local_arith(line)
        idx = np.where(line != -1)[0]
        if idx.size >= 3:
            score += 1.0 / (1.0 + np.var(np.diff(line[idx])))
    return float(score)


def local_delta_consistency_score(candidate_grid: Array2D, target_cell: Cell) -> float:
    lines = _get_lines(candidate_grid, target_cell)
    score = 0.0
    for line in lines.values():
        clean = line[line != -1]
        if clean.size >= 4:
            d1 = np.diff(clean)
            d2 = np.diff(d1)
            score += float(1.0 / (1.0 + np.var(d1))) + float(1.0 / (1.0 + np.var(d2)))
    return score


def mirror_pair_agreement_score(candidate_grid: Array2D, target_cell: Cell) -> float:
    return detect_mirror_sequences(candidate_grid, target_cell)


def rank_gap_repair_score(candidate_grid: Array2D, target_cell: Cell) -> float:
    lines = _get_lines(candidate_grid, target_cell)
    score = 0.0
    for line in (lines["row"], lines["col"]):
        clean = line[line != -1]
        if clean.size >= 3:
            ord_idx = np.argsort(clean)
            score += float(1.0 / (1.0 + np.var(np.diff(clean[ord_idx]))))
    return score


def modulo_family_score(masked_grid: Array2D, target_cell: Cell, candidate_value: int) -> float:
    mods = [2, 4, 5, 8, 10, 16, 20]
    i, j = target_cell
    known = np.argwhere(masked_grid != -1)
    score = 0.0
    for mod in mods:
        vals = masked_grid[masked_grid != -1]
        freq = np.bincount(vals % mod, minlength=mod).astype(float)
        freq /= max(np.sum(freq), 1.0)
        r = candidate_value % mod
        score += float(np.sum([freq[r] / max(abs(x - i) + abs(y - j), 1) for x, y in known]))
    return score


def neighborhood_uniqueness_pressure(candidate_grid: Array2D, target_cell: Cell) -> float:
    i, j = target_cell
    r, c = candidate_grid.shape
    r0, r1 = max(0, i - 1), min(r, i + 2)
    c0, c1 = max(0, j - 1), min(c, j + 2)
    patch = candidate_grid[r0:r1, c0:c1]
    vals = patch[patch != -1]
    if vals.size == 0:
        return 0.0
    return float(len(np.unique(vals)) / vals.size)


def directional_pattern_agreement(candidate_grid: Array2D, target_cell: Cell) -> float:
    lines = _get_lines(candidate_grid, target_cell)
    score = 0.0
    for line in lines.values():
        clean = line[line != -1]
        if clean.size >= 3:
            score += float(1.0 / (1.0 + np.var(np.diff(clean))))
    return score


def a2_delta_score(masked_grid: Array2D, target_cell: Cell, candidate_grid: Array2D) -> float:
    return ext_a2_proximity(candidate_grid, target_cell) - ext_a2_proximity(masked_grid, target_cell)


def d3_delta_score(masked_grid: Array2D, target_cell: Cell, candidate_grid: Array2D) -> float:
    return ext_d3_potential(candidate_grid, target_cell) - ext_d3_potential(masked_grid, target_cell)


def f10_repair_delta(masked_grid: Array2D, target_cell: Cell, candidate_grid: Array2D) -> float:
    return ext_f10_discontinuity(candidate_grid, target_cell) - ext_f10_discontinuity(masked_grid, target_cell)


BASE_MODULES = ["focus", "connectivity", "heatmap", "a2", "d3", "skip", "diff", "mirror", "tail", "f10"]
DISCOVERY_MODULES = [
    "local_arith_completion",
    "local_delta_consistency",
    "mirror_agreement",
    "rank_gap_repair",
    "modulo_family",
    "neighborhood_uniqueness_pressure",
    "directional_pattern_agreement",
    "a2_delta",
    "d3_delta",
    "f10_repair_delta",
]


def compute_module_score(
    name: str,
    masked_grid: Array2D,
    target_cell: Cell,
    candidate_value: int,
    candidate_grid: Array2D,
    heatmap_prior: Array2D | None,
) -> float:
    if name == "focus":
        return compute_focus_score(masked_grid, target_cell)
    if name == "connectivity":
        return connectivity_heatmap(masked_grid, target_cell)
    if name == "heatmap":
        return ext_heatmap_prior(target_cell, heatmap_prior)
    if name == "a2":
        return ext_a2_proximity(masked_grid, target_cell)
    if name == "d3":
        return ext_d3_potential(masked_grid, target_cell)
    if name == "skip":
        return detect_skip_patterns(candidate_grid, target_cell)
    if name == "diff":
        return compute_difference_trend(candidate_grid, target_cell)
    if name == "mirror":
        return detect_mirror_sequences(candidate_grid, target_cell)
    if name == "tail":
        return sequence_tail_analyzer(masked_grid, target_cell, candidate_value)
    if name == "f10":
        return ext_f10_discontinuity(candidate_grid, target_cell)
    if name == "local_arith_completion":
        return local_arithmetic_completion_score(candidate_grid, target_cell)
    if name == "local_delta_consistency":
        return local_delta_consistency_score(candidate_grid, target_cell)
    if name in {"mirror_pair_agreement", "mirror_agreement"}:
        return mirror_pair_agreement_score(candidate_grid, target_cell)
    if name == "rank_gap_repair":
        return rank_gap_repair_score(candidate_grid, target_cell)
    if name == "modulo_family":
        return modulo_family_score(masked_grid, target_cell, candidate_value)
    if name in {"neighborhood_uniqueness", "neighborhood_uniqueness_pressure"}:
        return neighborhood_uniqueness_pressure(candidate_grid, target_cell)
    if name == "directional_pattern_agreement":
        return directional_pattern_agreement(candidate_grid, target_cell)
    if name == "a2_delta":
        return a2_delta_score(masked_grid, target_cell, candidate_grid)
    if name == "d3_delta":
        return d3_delta_score(masked_grid, target_cell, candidate_grid)
    if name == "f10_repair_delta":
        return f10_repair_delta(masked_grid, target_cell, candidate_grid)
    raise KeyError(f"Unknown module: {name}")
