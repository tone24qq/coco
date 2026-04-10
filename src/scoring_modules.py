from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Optional, Protocol, Tuple

from src.board_geometry import (
    anti_diagonal_cells,
    cell_on_anti_diagonal,
    cell_on_main_diagonal,
    main_diagonal_cells,
    relative_rank_in_line,
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

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        rows, cols = len(board), len(board[0])
        result: Dict[Cell, float] = {}
        for r, c in unopened_cells:
            neighbors = []
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] != -1:
                    neighbors.append(board[rr][cc])
            if not neighbors:
                result[(r, c)] = 0.5
                continue
            mean_abs_delta = sum(abs(v - target_number) for v in neighbors) / len(neighbors)
            result[(r, c)] = 1.0 / (1.0 + mean_abs_delta)
        return ModuleScoreResult(result, "logic_rule: 根據鄰近已開數字與 target 距離評分")


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

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        del target_number
        rows, cols = len(board), len(board[0])
        center_r = (rows - 1) / 2
        center_c = (cols - 1) / 2
        max_dist = max(center_r + center_c, 1.0)
        result: Dict[Cell, float] = {}
        for r, c in unopened_cells:
            dist = abs(r - center_r) + abs(c - center_c)
            result[(r, c)] = 1.0 - (dist / max_dist)
        return ModuleScoreResult(result, "prior_model: 使用位置先驗（中心偏好）評分")


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
    }
    components["directional_consistency"] = sum(components.values()) / len(components)
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
    }
    components["line_consistency"] = sum(components.values()) / len(components)
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


class DirectionalConsistencyModule:
    name = "directional_consistency"

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        scores: Dict[Cell, float] = {}
        details: Dict[Cell, Dict[str, float]] = {}
        for cell in unopened_cells:
            components = _directional_components(board, cell, target_number)
            scores[cell] = components["directional_consistency"]
            details[cell] = components
        return ModuleScoreResult(
            scores,
            "directional_consistency: 以左右上下順序、距離與列欄平滑性評估",
            details=details,
        )


class LineConsistencyModule:
    name = "line_consistency"

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        scores: Dict[Cell, float] = {}
        details: Dict[Cell, Dict[str, float]] = {}
        for cell in unopened_cells:
            components = _line_components(board, cell, target_number)
            scores[cell] = components["line_consistency"]
            details[cell] = components
        return ModuleScoreResult(
            scores,
            "line_consistency: 以整行整列與對角殘差/單調性/分位吻合度評估",
            details=details,
        )


class GlobalAssignmentPriorModule:
    name = "global_assignment_prior"

    def __init__(self, assignment_mode: str = "exact") -> None:
        self.assignment_mode = assignment_mode

    @staticmethod
    def _greedy_assignment_score(
        board: Board,
        cells: List[Cell],
        numbers: List[int],
    ) -> float:
        if not cells or not numbers:
            return NEUTRAL_SCORE
        ranked_pairs = []
        for number in numbers:
            for cell in cells:
                compat = _cell_number_compatibility(board, cell, number)
                ranked_pairs.append((compat, number, cell))
        ranked_pairs.sort(reverse=True, key=lambda x: x[0])
        used_cells: set[Cell] = set()
        used_numbers: set[int] = set()
        total = 0.0
        count = 0
        for compat, number, cell in ranked_pairs:
            if number in used_numbers or cell in used_cells:
                continue
            used_numbers.add(number)
            used_cells.add(cell)
            total += compat
            count += 1
            if len(used_cells) == len(cells):
                break
        if count == 0:
            return NEUTRAL_SCORE
        return total / count

    @staticmethod
    def _exact_assignment_score(
        board: Board,
        cells: List[Cell],
        numbers: List[int],
    ) -> Optional[float]:
        if linear_sum_assignment is None or not cells or not numbers or len(cells) != len(numbers):
            return None
        matrix = []
        for cell in cells:
            row_costs = []
            for number in numbers:
                compatibility = _cell_number_compatibility(board, cell, number)
                row_costs.append(1.0 - compatibility)
            matrix.append(row_costs)
        row_idx, col_idx = linear_sum_assignment(matrix)
        if len(row_idx) == 0:
            return None
        compat_total = 0.0
        for r_idx, c_idx in zip(row_idx.tolist(), col_idx.tolist()):
            compat_total += 1.0 - matrix[r_idx][c_idx]
        return compat_total / len(row_idx)

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        if len(unopened_cells) <= 1:
            return ModuleScoreResult(
                {cell: NEUTRAL_SCORE for cell in unopened_cells},
                "global_assignment_prior: 資訊不足，返回中性分數",
            )
        rows, cols = len(board), len(board[0])
        n_total = rows * cols
        scores: Dict[Cell, float] = {}
        details: Dict[Cell, Dict[str, float]] = {}
        for anchor in unopened_cells:
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
            assignment_score = None
            if self.assignment_mode == "exact":
                assignment_score = self._exact_assignment_score(board_with_anchor, others, pool)
                used_exact = int(assignment_score is not None)
            if assignment_score is None:
                assignment_score = self._greedy_assignment_score(board_with_anchor, others, pool)
                used_greedy = 1
            anchor_compat = _cell_number_compatibility(board, anchor, target_number)
            final_score = _clip(0.5 * assignment_score + 0.5 * anchor_compat)
            scores[anchor] = final_score
            details[anchor] = {
                "global_assignment_mode": 1.0 if self.assignment_mode == "exact" else 0.0,
                "used_exact_assignment": float(used_exact),
                "used_greedy_fallback": float(used_greedy),
                "global_assignment_score": final_score,
                "global_anchor_compatibility": anchor_compat,
                "global_remaining_assignment_score": assignment_score,
            }

        return ModuleScoreResult(
            scores,
            "global_assignment_prior: 固定 target 後估計剩餘數字全局唯一分配一致性",
            details=details,
        )


MODULE_FACTORIES = {
    "logic_rule": lambda _cfg: LogicRuleModule(),
    "pattern_model": lambda _cfg: PatternModelModule(),
    "prior_model": lambda _cfg: PriorModelModule(),
    "directional_consistency": lambda _cfg: DirectionalConsistencyModule(),
    "line_consistency": lambda _cfg: LineConsistencyModule(),
    "global_assignment_prior": lambda cfg: GlobalAssignmentPriorModule(
        assignment_mode=str(cfg.get("assignment_mode", "exact"))
    ),
}


def build_modules(module_settings: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, ScoringModule]:
    settings = module_settings or {}
    return {name: factory(settings.get(name, {})) for name, factory in MODULE_FACTORIES.items()}


# backward-compatible default module map (do not mutate at runtime)
MODULES: Dict[str, ScoringModule] = build_modules()
