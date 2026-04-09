from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple

Board = List[List[int]]
Cell = Tuple[int, int]


@dataclass
class ModuleScoreResult:
    scores: Dict[Cell, float]
    explanation: str
    details: Dict[Cell, Dict[str, float]] | None = None


class ScoringModule(Protocol):
    name: str

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        ...


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _neutral() -> float:
    return 0.5


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def _distance_score(values: List[int], target_number: int, n_total: int) -> float:
    if not values:
        return _neutral()
    min_d = min(abs(v - target_number) for v in values)
    mean_d = _mean([abs(v - target_number) for v in values])
    scale = max(1.0, n_total / 2.0)
    score = 1.0 - ((0.6 * min_d + 0.4 * mean_d) / scale)
    return _clamp_01(score)


def _order_score(values: List[int], target_number: int, expect_target_greater: bool) -> float:
    if not values:
        return _neutral()
    if expect_target_greater:
        ok = sum(1 for v in values if target_number >= v)
    else:
        ok = sum(1 for v in values if target_number <= v)
    return _clamp_01(ok / len(values))


def _balance_score(known_values: List[int], inserted_value: int, n_total: int) -> float:
    if not known_values:
        return _neutral()
    before = sorted(known_values)
    after = sorted(known_values + [inserted_value])
    before_span = max(before) - min(before) if len(before) > 1 else n_total
    after_span = max(after) - min(after) if len(after) > 1 else n_total
    before_std = _safe_std(before)
    after_std = _safe_std(after)
    span_gain = 1.0 - abs(after_span - before_span) / max(1.0, n_total)
    std_gain = 1.0 - abs(after_std - before_std) / max(1.0, n_total / 3.0)
    return _clamp_01(0.5 * span_gain + 0.5 * std_gain)


def _monotonicity_score(series: List[int]) -> float:
    if len(series) < 3:
        return _neutral()
    inc_viol = dec_viol = 0
    for a, b in zip(series, series[1:]):
        if b < a:
            inc_viol += 1
        if b > a:
            dec_viol += 1
    total = len(series) - 1
    best = min(inc_viol, dec_viol)
    return _clamp_01(1.0 - (best / max(1, total)))


def _residual_score(values: List[int], positions: List[int]) -> float:
    if len(values) < 3:
        return _neutral()
    x_mean = _mean([float(p) for p in positions])
    y_mean = _mean([float(v) for v in values])
    denom = sum((p - x_mean) ** 2 for p in positions)
    if abs(denom) < 1e-12:
        return _neutral()
    slope = sum((p - x_mean) * (v - y_mean) for p, v in zip(positions, values)) / denom
    intercept = y_mean - slope * x_mean
    residual = _mean([abs((slope * p + intercept) - v) for p, v in zip(positions, values)])
    value_span = max(values) - min(values)
    scale = max(1.0, value_span)
    return _clamp_01(1.0 - residual / scale)


def _percentile_fit(series: List[int], target: int, pos: int, length: int, n_total: int) -> float:
    if length <= 1:
        return _neutral()
    expected_quantile = pos / (length - 1)
    expected_value = 1.0 + expected_quantile * (n_total - 1)
    err = abs(target - expected_value)
    scale = max(1.0, n_total / 2.0)
    return _clamp_01(1.0 - err / scale)


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


class DirectionalConsistencyModule:
    name = "directional_consistency"

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        rows, cols = len(board), len(board[0])
        n_total = rows * cols
        result: Dict[Cell, float] = {}
        details: Dict[Cell, Dict[str, float]] = {}
        for r, c in unopened_cells:
            left_vals = [board[r][x] for x in range(0, c) if board[r][x] != -1]
            right_vals = [board[r][x] for x in range(c + 1, cols) if board[r][x] != -1]
            up_vals = [board[x][c] for x in range(0, r) if board[x][c] != -1]
            down_vals = [board[x][c] for x in range(r + 1, rows) if board[x][c] != -1]
            row_vals = [board[r][x] for x in range(cols) if board[r][x] != -1]
            col_vals = [board[x][c] for x in range(rows) if board[x][c] != -1]

            cell_detail = {
                "left_order_score": _order_score(left_vals, target_number, expect_target_greater=True),
                "right_order_score": _order_score(right_vals, target_number, expect_target_greater=False),
                "up_order_score": _order_score(up_vals, target_number, expect_target_greater=True),
                "down_order_score": _order_score(down_vals, target_number, expect_target_greater=False),
                "left_distance_score": _distance_score(left_vals, target_number, n_total),
                "right_distance_score": _distance_score(right_vals, target_number, n_total),
                "up_distance_score": _distance_score(up_vals, target_number, n_total),
                "down_distance_score": _distance_score(down_vals, target_number, n_total),
                "row_balance_score": _balance_score(row_vals, target_number, n_total),
                "col_balance_score": _balance_score(col_vals, target_number, n_total),
            }
            directional = _mean(list(cell_detail.values()))
            cell_detail["directional_score"] = directional
            result[(r, c)] = directional
            details[(r, c)] = cell_detail

        return ModuleScoreResult(result, "directional_consistency: 左右上下＋行列平衡一致性", details=details)


class LineConsistencyModule:
    name = "line_consistency"

    def _collect_line(
        self,
        board: Board,
        coords: List[Cell],
        target_number: int,
        candidate: Cell,
    ) -> Tuple[List[int], List[int]]:
        vals: List[int] = []
        pos: List[int] = []
        for idx, (r, c) in enumerate(coords):
            v = board[r][c]
            if (r, c) == candidate:
                v = target_number
            if v == -1:
                continue
            vals.append(int(v))
            pos.append(idx)
        return vals, pos

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        rows, cols = len(board), len(board[0])
        n_total = rows * cols
        result: Dict[Cell, float] = {}
        details: Dict[Cell, Dict[str, float]] = {}
        for r, c in unopened_cells:
            row_coords = [(r, x) for x in range(cols)]
            col_coords = [(x, c) for x in range(rows)]
            main_coords = [(i, i) for i in range(min(rows, cols))] if r == c else []
            anti_coords = [(i, cols - 1 - i) for i in range(min(rows, cols))] if r + c == cols - 1 else []

            row_vals, row_pos = self._collect_line(board, row_coords, target_number, (r, c))
            col_vals, col_pos = self._collect_line(board, col_coords, target_number, (r, c))
            main_vals, main_pos = (
                self._collect_line(board, main_coords, target_number, (r, c)) if main_coords else ([], [])
            )
            anti_vals, anti_pos = (
                self._collect_line(board, anti_coords, target_number, (r, c)) if anti_coords else ([], [])
            )
            diag_vals = main_vals if main_coords else anti_vals
            diag_index = 0
            diag_len = 1
            if main_coords:
                diag_index = main_coords.index((r, c))
                diag_len = len(main_coords)
            elif anti_coords:
                diag_index = anti_coords.index((r, c))
                diag_len = len(anti_coords)

            detail = {
                "row_residual_score": _residual_score(row_vals, row_pos),
                "col_residual_score": _residual_score(col_vals, col_pos),
                "main_diag_score": _residual_score(main_vals, main_pos) if main_coords else _neutral(),
                "anti_diag_score": _residual_score(anti_vals, anti_pos) if anti_coords else _neutral(),
                "row_monotonicity_score": _monotonicity_score(row_vals),
                "col_monotonicity_score": _monotonicity_score(col_vals),
                "diag_monotonicity_score": _monotonicity_score(diag_vals),
                "row_percentile_fit": _percentile_fit(row_vals, target_number, c, cols, n_total),
                "col_percentile_fit": _percentile_fit(col_vals, target_number, r, rows, n_total),
                "diag_percentile_fit": _percentile_fit(
                    diag_vals,
                    target_number,
                    diag_index,
                    diag_len,
                    n_total,
                ),
            }
            line_score = _mean(list(detail.values()))
            detail["line_score"] = line_score
            result[(r, c)] = line_score
            details[(r, c)] = detail
        return ModuleScoreResult(result, "line_consistency: 整行整列與對角一致性", details=details)


class GlobalAssignmentPriorModule:
    name = "global_assignment_prior"

    def _cell_number_compat(self, board: Board, cell: Cell, number: int) -> float:
        rows, cols = len(board), len(board[0])
        r, c = cell
        neighbors: List[int] = []
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < rows and 0 <= cc < cols:
                v = board[rr][cc]
                if v != -1:
                    neighbors.append(v)
        if not neighbors:
            return _neutral()
        span = rows * cols
        mean_abs = _mean([abs(number - x) for x in neighbors])
        return _clamp_01(1.0 - mean_abs / max(1.0, span / 2.0))

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        rows, cols = len(board), len(board[0])
        n_total = rows * cols
        known = {v for row in board for v in row if v != -1}
        remaining_numbers = [x for x in range(1, n_total + 1) if x not in known]
        if target_number not in remaining_numbers:
            return ModuleScoreResult(
                {cell: _neutral() for cell in unopened_cells},
                "global_assignment_prior: target不在剩餘集合，回傳中性",
            )

        result: Dict[Cell, float] = {}
        details: Dict[Cell, Dict[str, float]] = {}
        for cell in unopened_cells:
            others = [c for c in unopened_cells if c != cell]
            other_nums = [x for x in remaining_numbers if x != target_number]
            compat_pairs: List[Tuple[float, Cell, int]] = []
            for oc in others:
                for num in other_nums:
                    compat_pairs.append((self._cell_number_compat(board, oc, num), oc, num))
            compat_pairs.sort(reverse=True, key=lambda x: x[0])
            used_cells: set[Cell] = set()
            used_nums: set[int] = set()
            agg_scores: List[float] = []
            for sc, oc, num in compat_pairs:
                if oc in used_cells or num in used_nums:
                    continue
                used_cells.add(oc)
                used_nums.add(num)
                agg_scores.append(sc)
                if len(used_cells) >= len(others):
                    break
            target_fit = self._cell_number_compat(board, cell, target_number)
            assignment_quality = _mean(agg_scores) if agg_scores else _neutral()
            global_score = _clamp_01(0.65 * target_fit + 0.35 * assignment_quality)
            result[cell] = global_score
            details[cell] = {
                "target_compatibility": target_fit,
                "greedy_assignment_quality": assignment_quality,
                "global_assignment_score": global_score,
            }

        return ModuleScoreResult(result, "global_assignment_prior: 固定target後估計剩餘唯一分配品質", details=details)


MODULES: Dict[str, ScoringModule] = {
    "logic_rule": LogicRuleModule(),
    "pattern_model": PatternModelModule(),
    "prior_model": PriorModelModule(),
    "directional_consistency": DirectionalConsistencyModule(),
    "line_consistency": LineConsistencyModule(),
    "global_assignment_prior": GlobalAssignmentPriorModule(),
}
