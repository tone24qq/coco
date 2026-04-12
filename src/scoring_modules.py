from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Optional, Protocol, Tuple
import numpy as np

from src.board_geometry import (
    anti_diagonal_cells,
    cell_on_anti_diagonal,
    cell_on_main_diagonal,
    main_diagonal_cells,
    relative_rank_in_line,
)
from src.fast_scoring import (
    directional_consistency_numba,
    evaluate_pairwise_gain_numba,
    line_consistency_numba,
    logic_rule_numba,
    prepare_fast_inputs,
    prior_model_fast,
)
from src.neighborhood_association import NeighborhoodAssociationModule
from src.vector_modules import (
    connectivity_heatmap_vectorized,
    difference_trend_vectorized,
    focus_score_vectorized,
    mirror_sequences_vectorized,
    skip_patterns_vectorized,
    tail_analyzer_vectorized,
)

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback exercised in tests via monkeypatch
    linear_sum_assignment = None

Board = List[List[int]]
Cell = Tuple[int, int]
NEUTRAL_SCORE = 0.5


@dataclass
class ModuleScoreResult:
    scores: Dict[Cell, float]
    explanation: str
    details: Dict[Cell, Dict[str, float]] = field(default_factory=dict)


class ScoringModule(Protocol):
    name: str

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        ...


class LogicRuleModule:
    name = "logic_rule"

    def __init__(self, fast_enabled: bool = True) -> None:
        self.fast_enabled = fast_enabled

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        fast_scores: Dict[Cell, float] = {}
        if self.fast_enabled and unopened_cells:
            try:
                board_arr, rows_arr, cols_arr, _ = prepare_fast_inputs(board, unopened_cells)
                arr = logic_rule_numba(board_arr, rows_arr, cols_arr, int(target_number))
                fast_scores = {cell: float(arr[i]) for i, cell in enumerate(unopened_cells)}
            except Exception:
                fast_scores = {}
        rows, cols = len(board), len(board[0])
        result: Dict[Cell, float] = {}
        details: Dict[Cell, Dict[str, float]] = {}
        for r, c in unopened_cells:
            neighbors = []
            contradiction_votes = 0
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] != -1:
                    v = board[rr][cc]
                    neighbors.append(v)
                    if rr == r and cc < c and v > target_number:
                        contradiction_votes += 1
                    if rr == r and cc > c and v < target_number:
                        contradiction_votes += 1
                    if cc == c and rr < r and v > target_number:
                        contradiction_votes += 1
                    if cc == c and rr > r and v < target_number:
                        contradiction_votes += 1
            if not neighbors:
                result[(r, c)] = fast_scores.get((r, c), 0.5)
                details[(r, c)] = {
                    "local_support_score": 0.5,
                    "local_contradiction_penalty": 0.0,
                    "neighbor_count": 0.0,
                    "contradiction_votes": 0.0,
                }
                continue
            mean_abs_delta = sum(abs(v - target_number) for v in neighbors) / len(neighbors)
            local_support = 1.0 / (1.0 + mean_abs_delta)
            contradiction_penalty = _clip(contradiction_votes / max(len(neighbors), 1))
            score = _clip(local_support - 0.7 * contradiction_penalty)
            result[(r, c)] = fast_scores.get((r, c), score)
            details[(r, c)] = {
                "local_support_score": local_support,
                "local_contradiction_penalty": contradiction_penalty,
                "neighbor_count": float(len(neighbors)),
                "contradiction_votes": float(contradiction_votes),
            }
        return ModuleScoreResult(result, "logic_rule: 以局部 support 與局部矛盾懲罰共同評分", details=details)


class PatternModelModule:
    name = "pattern_model"

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        rows, cols = len(board), len(board[0])
        result: Dict[Cell, float] = {}
        target_tail = target_number % 10
        for r, c in unopened_cells:
            known_neighbors = 0
            tail_matches = 0
            for rr in range(max(0, r - 1), min(rows, r + 2)):
                for cc in range(max(0, c - 1), min(cols, c + 2)):
                    if rr == r and cc == c:
                        continue
                    val = board[rr][cc]
                    if val == -1:
                        continue
                    known_neighbors += 1
                    if (val % 10) == target_tail:
                        tail_matches += 1
            if known_neighbors == 0:
                result[(r, c)] = 0.3
            else:
                result[(r, c)] = (tail_matches + 1) / (known_neighbors + 1)
        return ModuleScoreResult(result, "pattern_model: 使用局部尾數分佈與已開密度作啟發式評分")


class PriorModelModule:
    name = "prior_model"

    def __init__(self, fast_enabled: bool = True) -> None:
        self.fast_enabled = fast_enabled

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        del target_number
        if self.fast_enabled and unopened_cells:
            try:
                board_arr, rows_arr, cols_arr, _ = prepare_fast_inputs(board, unopened_cells)
                scores_arr = prior_model_fast(board_arr, rows_arr, cols_arr)
                return ModuleScoreResult(
                    {cell: float(scores_arr[i]) for i, cell in enumerate(unopened_cells)},
                    "prior_model: 使用位置先驗（中心偏好）評分",
                )
            except Exception:
                pass
        rows, cols = len(board), len(board[0])
        center_r = (rows - 1) / 2
        center_c = (cols - 1) / 2
        max_dist = max(center_r + center_c, 1.0)
        result: Dict[Cell, float] = {}
        for r, c in unopened_cells:
            dist = abs(r - center_r) + abs(c - center_c)
            result[(r, c)] = 1.0 - (dist / max_dist)
        return ModuleScoreResult(result, "prior_model: 使用位置先驗（中心偏好）評分")


class FocusScoreModule:
    name = "focus_score"

    def __init__(self, window_size: int = 3) -> None:
        self.window_size = max(1, int(window_size))

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        del target_number
        scores = focus_score_vectorized(board, unopened_cells, window_size=self.window_size)
        return ModuleScoreResult(scores, "focus_score: 局部視窗已知格密度")


class ConnectivityHeatmapModule:
    name = "connectivity_heatmap"

    def __init__(self, decay: str = "inverse_distance", decay_gamma: float = 0.35) -> None:
        self.decay = str(decay)
        self.decay_gamma = float(decay_gamma)

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        del target_number
        scores = connectivity_heatmap_vectorized(
            board, unopened_cells, decay=self.decay, decay_gamma=self.decay_gamma
        )
        return ModuleScoreResult(scores, "connectivity_heatmap: 與已知格連通熱度")


class DifferenceTrendModule:
    name = "difference_trend"

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        scores = difference_trend_vectorized(board, unopened_cells, target_number)
        return ModuleScoreResult(scores, "difference_trend: 行列差值趨勢一致性")


class SkipPatternsModule:
    name = "skip_patterns"

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        scores = skip_patterns_vectorized(board, unopened_cells, target_number)
        return ModuleScoreResult(scores, "skip_patterns: 行列跳格規律支持度")


class MirrorSequencesModule:
    name = "mirror_sequences"

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        scores = mirror_sequences_vectorized(board, unopened_cells, target_number)
        return ModuleScoreResult(scores, "mirror_sequences: 水平/垂直/對角鏡像支持度")


class TailAnalyzerModule:
    name = "tail_analyzer"

    def __init__(self, window_size: int = 3) -> None:
        self.window_size = max(1, int(window_size))

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        scores = tail_analyzer_vectorized(board, unopened_cells, target_number, window_size=self.window_size)
        return ModuleScoreResult(scores, "tail_analyzer: 尾數分布局部相容性")


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _known_values(cells: List[Tuple[int, int, int]]) -> List[int]:
    return [v for _, _, v in cells if v != -1]


def _order_score(values: List[int], target_number: int, expect_less: bool) -> float:
    if not values:
        return NEUTRAL_SCORE
    if expect_less:
        satisfied = sum(1 for v in values if v < target_number)
    else:
        satisfied = sum(1 for v in values if v > target_number)
    return _clip(satisfied / len(values))


def _distance_score(values: List[int], target_number: int, board_size: int) -> float:
    if not values:
        return NEUTRAL_SCORE
    diffs = sorted(abs(v - target_number) for v in values)
    nearest = diffs[0]
    avg = sum(diffs) / len(diffs)
    scale = max(board_size / 3.0, 1.0)
    return _clip(1.0 - ((0.6 * nearest + 0.4 * avg) / scale))


def _smoothness_score(values: List[int]) -> float:
    if len(values) <= 2:
        return NEUTRAL_SCORE
    ordered = sorted(values)
    gaps = [ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1)]
    if not gaps:
        return NEUTRAL_SCORE
    mu = sum(gaps) / len(gaps)
    if mu <= 0:
        return NEUTRAL_SCORE
    variance = sum((g - mu) ** 2 for g in gaps) / len(gaps)
    cv = math.sqrt(variance) / mu
    return _clip(1.0 / (1.0 + cv))


def _monotonicity_score(positioned_values: List[Tuple[int, int]]) -> float:
    if len(positioned_values) <= 2:
        return NEUTRAL_SCORE
    positioned_values = sorted(positioned_values)
    diffs = [positioned_values[i + 1][1] - positioned_values[i][1] for i in range(len(positioned_values) - 1)]
    non_decreasing = sum(1 for d in diffs if d >= 0) / len(diffs)
    non_increasing = sum(1 for d in diffs if d <= 0) / len(diffs)
    return max(non_decreasing, non_increasing)


def _residual_score(values: List[int], board_size: int) -> float:
    if len(values) <= 2:
        return NEUTRAL_SCORE
    ordered = sorted(values)
    gaps = [ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1)]
    med = median(gaps)
    if med <= 0:
        med = max(board_size / max(len(values), 1), 1.0)
    residual = sum(abs(g - med) for g in gaps) / len(gaps)
    return _clip(1.0 - residual / max(board_size / 2.0, 1.0))


def _percentile_fit(known_values: List[int], target_number: int, line_length: int, idx: int, board_size: int) -> float:
    if len(known_values) < 2:
        return NEUTRAL_SCORE
    sorted_vals = sorted(known_values)
    low = sorted_vals[0]
    high = sorted_vals[-1]
    if high == low:
        return NEUTRAL_SCORE
    expected = idx / max(line_length - 1, 1)
    actual = (target_number - low) / (high - low)
    return _clip(1.0 - abs(actual - expected))


def _line_values_with_positions(
    board: Board,
    cells: List[Cell],
    candidate: Cell,
    target_number: int,
) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for i, (r, c) in enumerate(cells):
        v = target_number if (r, c) == candidate else board[r][c]
        if v != -1:
            out.append((i, v))
    return out


def _directional_components(board: Board, candidate: Cell, target_number: int) -> Dict[str, float]:
    rows, cols = len(board), len(board[0])
    r, c = candidate
    left = [board[r][cc] for cc in range(0, c) if board[r][cc] != -1]
    right = [board[r][cc] for cc in range(c + 1, cols) if board[r][cc] != -1]
    up = [board[rr][c] for rr in range(0, r) if board[rr][c] != -1]
    down = [board[rr][c] for rr in range(r + 1, rows) if board[rr][c] != -1]
    row_vals = [board[r][cc] for cc in range(cols) if board[r][cc] != -1] + [target_number]
    col_vals = [board[rr][c] for rr in range(rows) if board[rr][c] != -1] + [target_number]
    board_size = rows * cols
    row_violation_count = sum(1 for v in left if v > target_number) + sum(1 for v in right if v < target_number)
    col_violation_count = sum(1 for v in up if v > target_number) + sum(1 for v in down if v < target_number)
    components = {
        "left_order_score": _order_score(left, target_number, expect_less=True),
        "right_order_score": _order_score(right, target_number, expect_less=False),
        "up_order_score": _order_score(up, target_number, expect_less=True),
        "down_order_score": _order_score(down, target_number, expect_less=False),
        "left_distance_score": _distance_score(left, target_number, board_size),
        "right_distance_score": _distance_score(right, target_number, board_size),
        "up_distance_score": _distance_score(up, target_number, board_size),
        "down_distance_score": _distance_score(down, target_number, board_size),
        "row_balance_score": _smoothness_score(row_vals),
        "col_balance_score": _smoothness_score(col_vals),
        "row_violation_count": float(row_violation_count),
        "col_violation_count": float(col_violation_count),
    }
    support_keys = [k for k in components.keys() if k.endswith("_score")]
    support = sum(components[k] for k in support_keys) / max(len(support_keys), 1)
    violation_penalty = _clip((row_violation_count + col_violation_count) / 4.0)
    components["directional_support_score"] = support
    components["directional_contradiction_penalty"] = violation_penalty
    components["directional_consistency"] = _clip(support - 0.8 * violation_penalty)
    return components


def _line_components(board: Board, candidate: Cell, target_number: int) -> Dict[str, float]:
    rows, cols = len(board), len(board[0])
    r, c = candidate
    board_size = rows * cols
    row_cells = [(r, cc) for cc in range(cols)]
    col_cells = [(rr, c) for rr in range(rows)]
    row_pos = _line_values_with_positions(board, row_cells, candidate, target_number)
    col_pos = _line_values_with_positions(board, col_cells, candidate, target_number)

    row_vals = [v for _, v in row_pos]
    col_vals = [v for _, v in col_pos]
    row_residual = _residual_score(row_vals, board_size)
    col_residual = _residual_score(col_vals, board_size)
    row_mono = _monotonicity_score(row_pos)
    col_mono = _monotonicity_score(col_pos)
    row_pct = _percentile_fit(row_vals, target_number, cols, c, board_size)
    col_pct = _percentile_fit(col_vals, target_number, rows, r, board_size)

    main_diag = main_diagonal_cells(rows, cols) if cell_on_main_diagonal(candidate, rows, cols) else []
    anti_diag = anti_diagonal_cells(rows, cols) if cell_on_anti_diagonal(candidate, rows, cols) else []

    diag_scores = []
    diag_monotonicity = []
    diag_percentiles = []
    for diag_cells in (main_diag, anti_diag):
        if not diag_cells:
            diag_scores.append(NEUTRAL_SCORE)
            diag_monotonicity.append(NEUTRAL_SCORE)
            diag_percentiles.append(NEUTRAL_SCORE)
            continue
        pos_vals = _line_values_with_positions(board, diag_cells, candidate, target_number)
        vals = [v for _, v in pos_vals]
        diag_scores.append(_residual_score(vals, board_size))
        diag_monotonicity.append(_monotonicity_score(pos_vals))
        diag_idx = relative_rank_in_line(diag_cells, candidate)
        if diag_idx is None:
            diag_percentiles.append(NEUTRAL_SCORE)
            continue
        diag_percentiles.append(_percentile_fit(vals, target_number, len(diag_cells), diag_idx, board_size))

    diag_violation_count = 0
    if main_diag:
        for rr, cc in main_diag:
            v = board[rr][cc]
            if v == -1 or (rr, cc) == candidate:
                continue
            if (rr < r and cc < c and v > target_number) or (rr > r and cc > c and v < target_number):
                diag_violation_count += 1
    if anti_diag:
        for rr, cc in anti_diag:
            v = board[rr][cc]
            if v == -1 or (rr, cc) == candidate:
                continue
            if (rr < r and cc > c and v > target_number) or (rr > r and cc < c and v < target_number):
                diag_violation_count += 1

    monotonic_break_flag = float(
        (row_mono < 0.5) or (col_mono < 0.5) or (sum(diag_monotonicity) / len(diag_monotonicity) < 0.5)
    )
    percentile_outlier_flag = float(
        (row_pct < 0.25) or (col_pct < 0.25) or (sum(diag_percentiles) / len(diag_percentiles) < 0.25)
    )
    gap_outlier_flag = float(
        (row_residual < 0.25) or (col_residual < 0.25) or (sum(diag_scores) / len(diag_scores) < 0.25)
    )

    components = {
        "row_residual_score": row_residual,
        "col_residual_score": col_residual,
        "main_diag_score": diag_scores[0],
        "anti_diag_score": diag_scores[1],
        "row_monotonicity_score": row_mono,
        "col_monotonicity_score": col_mono,
        "diag_monotonicity_score": sum(diag_monotonicity) / len(diag_monotonicity),
        "row_percentile_fit": row_pct,
        "col_percentile_fit": col_pct,
        "diag_percentile_fit": sum(diag_percentiles) / len(diag_percentiles),
        "diag_violation_count": float(diag_violation_count),
        "monotonic_break_flag": monotonic_break_flag,
        "percentile_outlier_flag": percentile_outlier_flag,
        "gap_outlier_flag": gap_outlier_flag,
    }
    support_keys = [k for k in components.keys() if k.endswith("_score") or k.endswith("_fit")]
    support = sum(components[k] for k in support_keys) / max(len(support_keys), 1)
    penalty = _clip(
        (diag_violation_count / 3.0)
        + 0.35 * monotonic_break_flag
        + 0.25 * percentile_outlier_flag
        + 0.25 * gap_outlier_flag
    )
    components["line_support_score"] = support
    components["line_contradiction_penalty"] = _clip(penalty)
    components["line_consistency"] = _clip(support - 0.75 * components["line_contradiction_penalty"])
    return components


def _cell_number_compatibility(board: Board, cell: Cell, number: int) -> float:
    directional = _directional_components(board, cell, number)["directional_consistency"]
    line = _line_components(board, cell, number)["line_consistency"]
    rows, cols = len(board), len(board[0])
    center_r = (rows - 1) / 2.0
    center_c = (cols - 1) / 2.0
    max_dist = max(center_r + center_c, 1.0)
    prior = 1.0 - ((abs(cell[0] - center_r) + abs(cell[1] - center_c)) / max_dist)
    return _clip(0.4 * directional + 0.4 * line + 0.2 * prior)


def _cell_number_compatibility_cached(
    board: Board,
    cell: Cell,
    number: int,
    cache: Dict[Tuple[Cell, int], float],
) -> float:
    key = (cell, int(number))
    if key not in cache:
        cache[key] = _cell_number_compatibility(board, cell, number)
    return cache[key]


class DirectionalConsistencyModule:
    name = "directional_consistency"

    def __init__(self, fast_enabled: bool = True) -> None:
        self.fast_enabled = fast_enabled

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        py_scores: Dict[Cell, float] = {}
        details: Dict[Cell, Dict[str, float]] = {}
        fast_scores: Dict[Cell, float] = {}
        if self.fast_enabled and unopened_cells:
            try:
                board_arr, rows_arr, cols_arr, _ = prepare_fast_inputs(board, unopened_cells)
                arr = directional_consistency_numba(board_arr, rows_arr, cols_arr, int(target_number))
                fast_scores = {cell: float(arr[i]) for i, cell in enumerate(unopened_cells)}
            except Exception:
                fast_scores = {}
        for cell in unopened_cells:
            components = _directional_components(board, cell, target_number)
            py_scores[cell] = components["directional_consistency"]
            details[cell] = components
        scores = dict(py_scores)
        if fast_scores and unopened_cells:
            py_top1 = max(unopened_cells, key=lambda c: py_scores.get(c, 0.0))
            fast_top1 = max(unopened_cells, key=lambda c: fast_scores.get(c, 0.0))
            if py_top1 == fast_top1:
                scores = {cell: fast_scores.get(cell, py_scores[cell]) for cell in unopened_cells}
        return ModuleScoreResult(
            scores,
            "directional_consistency: 以左右上下順序、距離與列欄平滑性評估",
            details=details,
        )


class LineConsistencyModule:
    name = "line_consistency"

    def __init__(self, fast_enabled: bool = True) -> None:
        self.fast_enabled = fast_enabled

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        py_scores: Dict[Cell, float] = {}
        details: Dict[Cell, Dict[str, float]] = {}
        fast_scores: Dict[Cell, float] = {}
        if self.fast_enabled and unopened_cells:
            try:
                board_arr, rows_arr, cols_arr, _ = prepare_fast_inputs(board, unopened_cells)
                arr = line_consistency_numba(board_arr, rows_arr, cols_arr, int(target_number))
                fast_scores = {cell: float(arr[i]) for i, cell in enumerate(unopened_cells)}
            except Exception:
                fast_scores = {}
        for cell in unopened_cells:
            components = _line_components(board, cell, target_number)
            py_scores[cell] = components["line_consistency"]
            details[cell] = components
        scores = dict(py_scores)
        if fast_scores and unopened_cells:
            py_top1 = max(unopened_cells, key=lambda c: py_scores.get(c, 0.0))
            fast_top1 = max(unopened_cells, key=lambda c: fast_scores.get(c, 0.0))
            if py_top1 == fast_top1:
                scores = {cell: fast_scores.get(cell, py_scores[cell]) for cell in unopened_cells}
        return ModuleScoreResult(
            scores,
            "line_consistency: 以整行整列與對角殘差/單調性/分位吻合度評估",
            details=details,
        )


class GlobalAssignmentPriorModule:
    name = "global_assignment_prior"

    def __init__(
        self,
        assignment_mode: str = "exact",
        top_m_candidates: int = 8,
        exact_max_candidates: int = 20,
    ) -> None:
        self.assignment_mode = assignment_mode
        self.top_m_candidates = max(1, int(top_m_candidates))
        self.exact_max_candidates = max(2, int(exact_max_candidates))

    @staticmethod
    def _greedy_assignment_cost(
        board: Board,
        cells: List[Cell],
        numbers: List[int],
        compatibility_cache: Optional[Dict[Tuple[Cell, int], float]] = None,
    ) -> Optional[float]:
        if not cells or not numbers:
            return 0.0
        if len(cells) != len(numbers):
            return None
        ranked_pairs = []
        compat_cache = compatibility_cache if compatibility_cache is not None else {}
        for number in numbers:
            for cell in cells:
                compat = _cell_number_compatibility_cached(board, cell, number, compat_cache)
                ranked_pairs.append((1.0 - compat, number, cell))
        ranked_pairs.sort(key=lambda x: x[0])
        used_cells: set[Cell] = set()
        used_numbers: set[int] = set()
        total = 0.0
        count = 0
        for cost, number, cell in ranked_pairs:
            if number in used_numbers or cell in used_cells:
                continue
            used_numbers.add(number)
            used_cells.add(cell)
            total += cost
            count += 1
            if len(used_cells) == len(cells):
                break
        if count == 0 or count != len(cells):
            return None
        return total / count

    @staticmethod
    def _exact_assignment_cost(
        board: Board,
        cells: List[Cell],
        numbers: List[int],
        compatibility_cache: Optional[Dict[Tuple[Cell, int], float]] = None,
    ) -> Optional[float]:
        if linear_sum_assignment is None or not cells or not numbers or len(cells) != len(numbers):
            return None
        matrix = []
        compat_cache = compatibility_cache if compatibility_cache is not None else {}
        for cell in cells:
            row_costs = []
            for number in numbers:
                compatibility = _cell_number_compatibility_cached(board, cell, number, compat_cache)
                row_costs.append(1.0 - compatibility)
            matrix.append(row_costs)
        row_idx, col_idx = linear_sum_assignment(matrix)
        if len(row_idx) == 0:
            return None
        cost_total = 0.0
        for r_idx, c_idx in zip(row_idx.tolist(), col_idx.tolist()):
            cost_total += matrix[r_idx][c_idx]
        return cost_total / len(row_idx)

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        if len(unopened_cells) <= 1:
            return ModuleScoreResult(
                {cell: NEUTRAL_SCORE for cell in unopened_cells},
                "global_assignment_prior: 資訊不足，返回中性分數",
            )
        rows, cols = len(board), len(board[0])
        n_total = rows * cols
        anchor_costs: Dict[Cell, float] = {}
        anchor_details: Dict[Cell, Dict[str, float]] = {}
        directional_scores = DirectionalConsistencyModule().score(board, unopened_cells, target_number).scores
        line_scores = LineConsistencyModule().score(board, unopened_cells, target_number).scores
        prior_scores = PriorModelModule().score(board, unopened_cells, target_number).scores
        pre_ranked = sorted(
            unopened_cells,
            key=lambda cell: (
                directional_scores.get(cell, 0.5) + line_scores.get(cell, 0.5) + prior_scores.get(cell, 0.5)
            ),
            reverse=True,
        )
        active_anchors = pre_ranked[: self.top_m_candidates]
        compatibility_cache: Dict[Tuple[Cell, int], float] = {}
        for anchor in active_anchors:
            board_with_anchor = [list(row) for row in board]
            board_with_anchor[anchor[0]][anchor[1]] = target_number
            others = [cell for cell in unopened_cells if cell != anchor]
            opened_with_anchor = {
                board_with_anchor[r][c]
                for r in range(rows)
                for c in range(cols)
                if board_with_anchor[r][c] != -1
            }
            pool = [x for x in range(1, n_total + 1) if x not in opened_with_anchor]
            used_exact = 0
            used_greedy = 0
            assignment_cost = None
            exact_allowed = self.assignment_mode == "exact" and len(unopened_cells) <= self.exact_max_candidates
            if exact_allowed:
                assignment_cost = self._exact_assignment_cost(
                    board_with_anchor, others, pool, compatibility_cache=compatibility_cache
                )
                used_exact = int(assignment_cost is not None)
            if assignment_cost is None:
                assignment_cost = self._greedy_assignment_cost(
                    board_with_anchor, others, pool, compatibility_cache=compatibility_cache
                )
                used_greedy = 1
            infeasible = assignment_cost is None
            if infeasible:
                assignment_cost = 1.0
            anchor_costs[anchor] = float(assignment_cost)
            anchor_details[anchor] = {
                "global_assignment_mode": 1.0 if exact_allowed else 0.0,
                "used_exact_assignment": float(used_exact),
                "used_greedy_fallback": float(used_greedy),
                "exact_forced_off_by_candidate_count": float(self.assignment_mode == "exact" and not exact_allowed),
                "used_compatibility_fallback": 0.0,
                "forced_anchor_total_assignment_cost": float(assignment_cost * max(len(others), 1)),
                "forced_anchor_avg_assignment_cost": float(assignment_cost),
                "infeasible_or_high_cost_flag": float(infeasible),
                "reduced_assignment_path": float(len(active_anchors) < len(unopened_cells)),
                "reduced_assignment_skipped": 0.0,
            }

        best_cost = min(anchor_costs.values()) if anchor_costs else 1.0
        scores: Dict[Cell, float] = {}
        details: Dict[Cell, Dict[str, float]] = {}
        for anchor, avg_cost in anchor_costs.items():
            delta = max(0.0, avg_cost - best_cost)
            high_cost_flag = float(avg_cost >= 0.75 or delta >= 0.2)
            score = _clip(1.0 - (avg_cost + 1.2 * delta + 0.6 * high_cost_flag))
            scores[anchor] = score
            details[anchor] = {
                **anchor_details[anchor],
                "anchor_cost_delta_vs_best": float(delta),
                "infeasible_or_high_cost_flag": max(
                    anchor_details[anchor]["infeasible_or_high_cost_flag"],
                    high_cost_flag,
                ),
                "global_assignment_score": score,
            }

        for anchor in unopened_cells:
            if anchor in scores:
                continue
            cheap_score = _cell_number_compatibility_cached(board, anchor, target_number, compatibility_cache)
            scores[anchor] = _clip(0.85 * cheap_score + 0.15 * NEUTRAL_SCORE)
            details[anchor] = {
                "global_assignment_mode": 1.0 if self.assignment_mode == "exact" else 0.0,
                "used_exact_assignment": 0.0,
                "used_greedy_fallback": 0.0,
                "exact_forced_off_by_candidate_count": 0.0,
                "used_compatibility_fallback": 1.0,
                "forced_anchor_total_assignment_cost": 0.0,
                "forced_anchor_avg_assignment_cost": 0.0,
                "infeasible_or_high_cost_flag": 0.0,
                "anchor_cost_delta_vs_best": 0.0,
                "global_assignment_score": scores[anchor],
                "reduced_assignment_path": 1.0,
                "reduced_assignment_skipped": 1.0,
            }
        return ModuleScoreResult(
            scores,
            "global_assignment_prior: 固定 target 後估計剩餘數字全局唯一分配一致性",
            details=details,
        )


class PairwiseConditionalConsistencyModule:
    name = "pairwise_conditional_consistency"

    def __init__(
        self,
        anchor_top_k_cells: int = 5,
        anchor_top_k_values: int = 5,
        max_pair_trials_per_candidate: int = 20,
        gating_enabled: bool = True,
        contradiction_penalty_weight: float = 1.0,
        hard_violation_threshold: float = 2.0,
        hard_gate_multiplier: float = 0.05,
        soft_gate_floor: float = 0.25,
        submodule_weights: Optional[Dict[str, float]] = None,
        runtime_mode: str = "full",
        candidate_top_n: int = 8,
        global_assignment_mode: str = "greedy",
        global_assignment_top_m_candidates: int = 4,
    ) -> None:
        self.anchor_top_k_cells = max(1, anchor_top_k_cells)
        self.anchor_top_k_values = max(1, anchor_top_k_values)
        self.max_pair_trials_per_candidate = max(1, max_pair_trials_per_candidate)
        self.gating_enabled = gating_enabled
        self.contradiction_penalty_weight = contradiction_penalty_weight
        self.hard_violation_threshold = hard_violation_threshold
        self.hard_gate_multiplier = hard_gate_multiplier
        self.soft_gate_floor = soft_gate_floor
        self.runtime_mode = str(runtime_mode)
        self.candidate_top_n = max(1, int(candidate_top_n))
        raw_weights = submodule_weights or {}
        total = sum(max(0.0, float(v)) for v in raw_weights.values())
        if total <= 0:
            self.submodule_weights = {
                "logic_rule": 0.35,
                "directional_consistency": 0.25,
                "line_consistency": 0.25,
                "global_assignment_prior": 0.15,
            }
        else:
            self.submodule_weights = {k: max(0.0, float(v)) / total for k, v in raw_weights.items()}
        self.logic_module = LogicRuleModule()
        self.directional_module = DirectionalConsistencyModule()
        self.line_module = LineConsistencyModule()
        self.global_module = GlobalAssignmentPriorModule(
            assignment_mode=global_assignment_mode,
            top_m_candidates=global_assignment_top_m_candidates,
        )
        self.seed_ranked_candidates: List[Cell] = []

    def set_seed_ranked_candidates(self, ranked: List[Cell]) -> None:
        self.seed_ranked_candidates = list(ranked)

    def _candidate_composite_from_results(
        self,
        module_results: Dict[str, ModuleScoreResult],
        cell: Cell,
    ) -> float:
        support = 0.0
        contradiction = 0.0
        weight_total = 0.0
        details_by_module: Dict[str, Dict[str, float]] = {}
        for name, result in module_results.items():
            w = float(self.submodule_weights.get(name, 0.0))
            if w <= 0:
                continue
            score = float(result.scores.get(cell, NEUTRAL_SCORE))
            details = result.details.get(cell, {}) if result.details else {}
            support += score * w
            contradiction += _extract_pairwise_contradiction(name, score, details) * w
            weight_total += w
            details_by_module[name] = details
        if weight_total <= 0:
            return 0.0
        contradiction_penalty = contradiction / weight_total
        gate_multiplier = 1.0
        row_v = float(details_by_module.get("directional_consistency", {}).get("row_violation_count", 0.0))
        col_v = float(details_by_module.get("directional_consistency", {}).get("col_violation_count", 0.0))
        diag_v = float(details_by_module.get("line_consistency", {}).get("diag_violation_count", 0.0))
        line_flags = (
            float(details_by_module.get("line_consistency", {}).get("monotonic_break_flag", 0.0))
            + float(details_by_module.get("line_consistency", {}).get("percentile_outlier_flag", 0.0))
            + float(details_by_module.get("line_consistency", {}).get("gap_outlier_flag", 0.0))
        )
        violation_score = row_v + col_v + diag_v + line_flags
        if self.gating_enabled:
            if violation_score >= self.hard_violation_threshold:
                gate_multiplier = self.hard_gate_multiplier
            else:
                gate_multiplier = max(self.soft_gate_floor, 1.0 - 0.25 * contradiction_penalty)
        gated = gate_multiplier * support
        return gated - self.contradiction_penalty_weight * contradiction_penalty

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        if not unopened_cells:
            return ModuleScoreResult({}, "pairwise_conditional_consistency: no unopened cells")
        rows, cols = len(board), len(board[0])
        n_total = rows * cols
        opened_numbers = {board[r][c] for r in range(rows) for c in range(cols) if board[r][c] != -1}
        remaining_numbers = [x for x in range(1, n_total + 1) if x not in opened_numbers and x != target_number]
        module_results = {
            "logic_rule": self.logic_module.score(board, unopened_cells, target_number),
            "directional_consistency": self.directional_module.score(board, unopened_cells, target_number),
            "line_consistency": self.line_module.score(board, unopened_cells, target_number),
            "global_assignment_prior": self.global_module.score(board, unopened_cells, target_number),
        }
        target_support_scores = module_results["directional_consistency"].scores
        ranked_anchor_cells = sorted(unopened_cells, key=lambda c: target_support_scores.get(c, 0.0), reverse=True)
        anchor_cells = ranked_anchor_cells[: self.anchor_top_k_cells]
        ranked_values = sorted(remaining_numbers, key=lambda x: abs(x - target_number))
        anchor_values = ranked_values[: self.anchor_top_k_values]

        board_arr, _, _, known_mask = prepare_fast_inputs(board, unopened_cells)
        base_composite_cache: Dict[Cell, float] = {
            c: self._candidate_composite_from_results(module_results, c) for c in unopened_cells
        }
        candidate_subset = set(unopened_cells)
        if self.runtime_mode == "fast":
            baseline_ranked = (
                [c for c in self.seed_ranked_candidates if c in base_composite_cache]
                if self.seed_ranked_candidates
                else sorted(unopened_cells, key=lambda c: base_composite_cache[c], reverse=True)
            )
            candidate_subset = set(baseline_ranked[: self.candidate_top_n])
        scores: Dict[Cell, float] = {}
        details: Dict[Cell, Dict[str, float]] = {}
        for candidate in unopened_cells:
            base_score = base_composite_cache[candidate]
            if candidate not in candidate_subset:
                scores[candidate] = 0.0
                details[candidate] = {
                    "best_anchor_row": -1.0,
                    "best_anchor_col": -1.0,
                    "best_anchor_value": -1.0,
                    "conditional_gain": 0.0,
                    "pair_trials_used": 0.0,
                    "base_cache_hits": 1.0,
                    "conditioned_cache_size": 0.0,
                    "runtime_reduced_path": 1.0,
                    "runtime_mode_fast": 1.0,
                }
                continue
            anchor_rows = [a[0] for a in anchor_cells if a != candidate]
            anchor_cols = [a[1] for a in anchor_cells if a != candidate]
            gains = evaluate_pairwise_gain_numba(
                board_arr,
                known_mask,
                int(candidate[0]),
                int(candidate[1]),
                np.asarray(anchor_rows, dtype=np.int32),
                np.asarray(anchor_cols, dtype=np.int32),
                np.asarray(anchor_values, dtype=np.int32),
                int(target_number),
                int(self.max_pair_trials_per_candidate),
            )
            heuristic_gain, best_anchor_idx, best_anchor_value, pair_trials = gains
            best_gain = max(0.0, min(1.0, float(heuristic_gain)))
            best_anchor = None
            screened: List[Tuple[float, Cell, int]] = []
            if best_anchor_idx >= 0 and best_anchor_idx < len(anchor_rows):
                screened.append(
                    (
                        float(best_gain),
                        (int(anchor_rows[best_anchor_idx]), int(anchor_cols[best_anchor_idx])),
                        int(best_anchor_value),
                    )
                )
            for anchor in anchor_cells:
                if anchor == candidate:
                    continue
                for v in anchor_values[:2]:
                    screened.append((0.0, anchor, int(v)))
            screened = screened[: min(3, len(screened))]
            for heuristic_gain, anchor, anchor_val in screened:
                locality = 1.0 / (1.0 + abs(anchor[0] - candidate[0]) + abs(anchor[1] - candidate[1]))
                value_closeness = 1.0 / (1.0 + abs(anchor_val - target_number))
                gain = max(0.0, min(1.0, 0.65 * float(heuristic_gain) + 0.25 * locality + 0.10 * value_closeness))
                gain = max(0.0, gain - 0.2 * max(0.0, 0.5 - base_score))
                if gain > best_gain:
                    best_gain = gain
                    best_anchor = anchor
                    best_anchor_value = anchor_val
            scores[candidate] = _clip(best_gain)
            details[candidate] = {
                "best_anchor_row": float((best_anchor[0] + 1) if best_anchor else -1),
                "best_anchor_col": float((best_anchor[1] + 1) if best_anchor else -1),
                "best_anchor_value": float(best_anchor_value if best_anchor_value is not None else -1),
                "conditional_gain": float(0.0 if math.isnan(best_gain) else best_gain),
                "pair_trials_used": float(pair_trials),
                "base_cache_hits": float(int(candidate in base_composite_cache)),
                "conditioned_cache_size": 0.0,
                "runtime_reduced_path": float(self.runtime_mode == "fast"),
                "runtime_mode_fast": float(self.runtime_mode == "fast"),
            }

        return ModuleScoreResult(
            scores,
            "pairwise_conditional_consistency: 估計在有限條件假設下 target 候選分數的最大增益",
            details=details,
        )


def _extract_pairwise_contradiction(module_name: str, module_score: float, details: Dict[str, float]) -> float:
    if module_name == "logic_rule":
        return float(details.get("local_contradiction_penalty", 0.0))
    if module_name == "directional_consistency":
        return float(details.get("directional_contradiction_penalty", 0.0))
    if module_name == "line_consistency":
        return float(details.get("line_contradiction_penalty", 0.0))
    if module_name == "global_assignment_prior":
        return float(details.get("anchor_cost_delta_vs_best", 0.0)) + 0.5 * float(
            details.get("infeasible_or_high_cost_flag", 0.0)
        )
    return _clip(1.0 - module_score)


MODULE_FACTORIES = {
    "logic_rule": lambda cfg: LogicRuleModule(fast_enabled=bool(cfg.get("fast_enabled", True))),
    "pattern_model": lambda _cfg: PatternModelModule(),
    "prior_model": lambda cfg: PriorModelModule(fast_enabled=bool(cfg.get("fast_enabled", True))),
    "directional_consistency": lambda cfg: DirectionalConsistencyModule(
        fast_enabled=bool(cfg.get("fast_enabled", True))
    ),
    "line_consistency": lambda cfg: LineConsistencyModule(fast_enabled=bool(cfg.get("fast_enabled", True))),
    "global_assignment_prior": lambda cfg: GlobalAssignmentPriorModule(
        assignment_mode=str(cfg.get("assignment_mode", "exact")),
        top_m_candidates=int(cfg.get("top_m_candidates", 8)),
        exact_max_candidates=int(cfg.get("exact_max_candidates", 20)),
    ),
    "pairwise_conditional_consistency": lambda cfg: PairwiseConditionalConsistencyModule(
        anchor_top_k_cells=int(cfg.get("anchor_top_k_cells", 5)),
        anchor_top_k_values=int(cfg.get("anchor_top_k_values", 5)),
        max_pair_trials_per_candidate=int(cfg.get("max_pair_trials_per_candidate", 20)),
        gating_enabled=bool(cfg.get("gating_enabled", True)),
        contradiction_penalty_weight=float(cfg.get("contradiction_penalty_weight", 1.0)),
        hard_violation_threshold=float(cfg.get("hard_violation_threshold", 2.0)),
        hard_gate_multiplier=float(cfg.get("hard_gate_multiplier", 0.05)),
        soft_gate_floor=float(cfg.get("soft_gate_floor", 0.25)),
        runtime_mode=str(cfg.get("runtime_mode", "full")),
        candidate_top_n=int(cfg.get("candidate_top_n", 8)),
        global_assignment_mode=str(cfg.get("global_assignment_mode", "greedy")),
        global_assignment_top_m_candidates=int(cfg.get("global_assignment_top_m_candidates", 4)),
        submodule_weights={
            str(k): float(v)
            for k, v in dict(
                cfg.get(
                    "submodule_weights",
                    {
                        "logic_rule": 0.35,
                        "directional_consistency": 0.25,
                        "line_consistency": 0.25,
                        "global_assignment_prior": 0.15,
                    },
                )
            ).items()
        },
    ),
    "focus_score": lambda cfg: FocusScoreModule(window_size=int(cfg.get("window_size", 3))),
    "connectivity_heatmap": lambda cfg: ConnectivityHeatmapModule(
        decay=str(cfg.get("decay", "inverse_distance")),
        decay_gamma=float(cfg.get("decay_gamma", 0.35)),
    ),
    "difference_trend": lambda _cfg: DifferenceTrendModule(),
    "skip_patterns": lambda _cfg: SkipPatternsModule(),
    "mirror_sequences": lambda _cfg: MirrorSequencesModule(),
    "tail_analyzer": lambda cfg: TailAnalyzerModule(window_size=int(cfg.get("window_size", 3))),
    "neighborhood_association": lambda cfg: NeighborhoodAssociationModule(
        radius=int(cfg.get("radius", 1)),
        use_diagonal=bool(cfg.get("use_diagonal", True)),
        min_seed_count=int(cfg.get("min_seed_count", 1)),
        decay_by_distance=bool(cfg.get("decay_by_distance", True)),
        distance_decay_power=float(cfg.get("distance_decay_power", 1.0)),
        enabled_seed_families=list(cfg.get("enabled_seed_families", ["same_decade", "same_tail", "near_value"])),
        near_value_deltas=list(cfg.get("near_value_deltas", [1, 2, 10, 20])),
        enabled_neighbor_families=list(
            cfg.get("enabled_neighbor_families", ["same_decade", "same_tail", "near_value"])
        ),
        neighbor_value_deltas=list(cfg.get("neighbor_value_deltas", [1, 2, 10, 20])),
        score_mode=str(cfg.get("score_mode", "weighted_pattern_overlap")),
        seed_aggregation=str(cfg.get("seed_aggregation", "mean")),
        candidate_aggregation=str(cfg.get("candidate_aggregation", "mean")),
        neutral_score_when_no_seed=float(cfg.get("neutral_score_when_no_seed", 0.5)),
        floor_score=float(cfg.get("floor_score", 0.0)),
        ceil_score=float(cfg.get("ceil_score", 1.0)),
        relation_source=str(cfg.get("relation_source", "heuristic_family_profile_v1")),
    ),
}


def build_modules(module_settings: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, ScoringModule]:
    settings = module_settings or {}
    return {name: factory(settings.get(name, {})) for name, factory in MODULE_FACTORIES.items()}


# backward-compatible default module map (do not mutate at runtime)
MODULES: Dict[str, ScoringModule] = build_modules()
