from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple

Board = List[List[int]]
Cell = Tuple[int, int]


@dataclass
class ModuleScoreResult:
    scores: Dict[Cell, float]
    explanation: str


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


MODULES: Dict[str, ScoringModule] = {
    "logic_rule": LogicRuleModule(),
    "pattern_model": PatternModelModule(),
    "prior_model": PriorModelModule(),
}
