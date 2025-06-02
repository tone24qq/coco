# brain2.py

import numpy as np
import math
import logging
from collections import deque, Counter
from typing import List, Tuple, Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BaseModuleConfig(BaseModel):
    """
    基础模块配置：- enabled: bool - weight: float
    """
    enabled: bool = Field(default=True, description="模块启用/禁用开关")
    weight: float = Field(default=1.0, ge=0.0, description="模块权重")

    class Config:
        validate_assignment = True


class MathUtils:
    @staticmethod
    def normalize_value(
        value: float, min_val: float, max_val: float, clamp: bool = True
    ) -> float:
        if math.isclose(min_val, max_val):
            if math.isclose(value, min_val):
                return 0.5
            return 0.0 if value < min_val else 1.0
        norm = (value - min_val) / (max_val - min_val)
        return float(max(0.0, min(1.0, norm))) if clamp else float(norm)


class BoardAnalyzerUtils:
    @staticmethod
    def get_neighborhood_values(
        grid: np.ndarray,
        r: int,
        c: int,
        radius: int = 1,
        eight_connectivity: bool = True,
        val_func=None,
        include_center: bool = False,
    ) -> List[float]:
        if val_func is None:
            val_func = lambda x: float(x) if x != -1 else None

        rows, cols = grid.shape
        neighbors: List[float] = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if not include_center and dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not eight_connectivity and radius == 1 and (abs(dr) + abs(dc) != 1):
                        continue
                    val = val_func(grid[nr, nc])
                    if val is not None:
                        neighbors.append(val)
        return neighbors


def EXT_GM4_Spatial_Auto_Corr_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM4",
) -> np.ndarray:
    """
    GM4 – 空间自相关性
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers = list(
        BoardAnalyzerUtils.get_neighborhood_values(grid, 0, 0, radius=0)
    )
    # 取中位数或平均数作为假设值
    if potential_numbers:
        hypo_val = float(np.median(potential_numbers))
    else:
        max_val_board = max(rows, cols)
        hypo_val = (1.0 + float(max_val_board)) / 2.0 if max_val_board > 0 else 0.5

    max_norm = float(max(rows, cols))
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue

            neighbor_values = BoardAnalyzerUtils.get_neighborhood_values(
                grid, r, c, radius=1, eight_connectivity=True, val_func=lambda x: float(x) if x != -1 else None
            )
            if not neighbor_values:
                scores[r, c] = 0.5
                continue

            mean_nb = np.mean(neighbor_values)
            diff = abs(hypo_val - mean_nb)
            norm_diff = MathUtils.normalize_value(diff, 0, max_norm, clamp=True)
            scores[r, c] = 1.0 - norm_diff

    return scores * config.weight


def EXT_GM5_Line_Completion_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM5",
) -> np.ndarray:
    """
    GM5 – 线段补全
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0 or min(rows, cols) < 1:
        return scores

    potential_numbers = [
        v for v in range(1, rows * cols + 1) if v not in grid
    ]
    if not potential_numbers:
        return scores

    score_map = {
        "identical_3": 0.6,
        "arithmetic_3_mend": 0.7,
        "arithmetic_3_extend": 0.5,
        "arithmetic_3_mend_high": 0.9,
    }

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue

            max_cell_score = 0.0
            for p in potential_numbers:
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    # 修复等值三连 / 等差三连
                    r1, c1 = r - dr, c - dc
                    r2, c2 = r + dr, c + dc
                    if 0 <= r1 < rows and 0 <= c1 < cols and 0 <= r2 < rows and 0 <= c2 < cols:
                        v1 = grid[r1, c1]
                        v2 = grid[r2, c2]
                        if v1 != -1 and v2 != -1:
                            if v1 == p and v2 == p:
                                max_cell_score = max(max_cell_score, score_map["identical_3"])
                            if (v1 + v2) == 2 * p and abs(p - v1) > 0:
                                sc = score_map["arithmetic_3_mend"]
                                if (v1 + p + v2) / 3 > (rows * cols) / 2:
                                    sc = max(sc, score_map["arithmetic_3_mend_high"])
                                max_cell_score = max(max_cell_score, sc)

                    # 等差延伸情况
                    r1e, c1e = r + dr, c + dc
                    r2e, c2e = r + 2 * dr, c + 2 * dc
                    if 0 <= r1e < rows and 0 <= c1e < cols and 0 <= r2e < rows and 0 <= c2e < cols:
                        v1e = grid[r1e, c1e]
                        v2e = grid[r2e, c2e]
                        if v1e != -1 and v2e != -1:
                            if p == v1e and p == v2e:
                                max_cell_score = max(max_cell_score, score_map["identical_3"])
                            if (p + v2e) == 2 * v1e and abs(v1e - p) > 0:
                                max_cell_score = max(max_cell_score, score_map["arithmetic_3_extend"])

                    r1e2, c1e2 = r - 2 * dr, c - 2 * dc
                    r2e2, c2e2 = r - dr, c - dc
                    if 0 <= r1e2 < rows and 0 <= c1e2 < cols and 0 <= r2e2 < rows and 0 <= c2e2 < cols:
                        v1e2 = grid[r1e2, c1e2]
                        v2e2 = grid[r2e2, c2e2]
                        if v1e2 != -1 and v2e2 != -1:
                            if v1e2 == v2e2 and v1e2 == p:
                                max_cell_score = max(max_cell_score, score_map["identical_3"])
                            if (v1e2 + p) == 2 * v2e2 and abs(v2e2 - v1e2) > 0:
                                max_cell_score = max(max_cell_score, score_map["arithmetic_3_extend"])

                scores[r, c] = MathUtils.normalize_value(max_cell_score, 0, 1.0, clamp=True)

    return scores * config.weight


def EXT_GM6_Symmetry_Potential_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM6",
) -> np.ndarray:
    """
    GM6 – 对称潜力
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue

            # 水平翻转
            if grid[r, cols - 1 - c] != -1:
                scores[r, c] = max(scores[r, c], 1.0)
            # 垂直翻转
            if grid[rows - 1 - r, c] != -1:
                scores[r, c] = max(scores[r, c], 1.0)
            # 对角线翻转
            if grid[c, r] != -1:
                scores[r, c] = max(scores[r, c], 1.0)
            if grid[rows - 1 - c, cols - 1 - r] != -1:
                scores[r, c] = max(scores[r, c], 1.0)

    if np.any(scores > 0):
        mn = float(np.min(scores[grid == -1]))
        mx = float(np.max(scores[grid == -1]))
        if not math.isclose(mx, mn):
            scores = (scores - mn) / (mx - mn)
        else:
            scores = np.zeros_like(scores)
    return scores * config.weight


def EXT_GM7_Numeric_Gaps_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM7",
) -> np.ndarray:
    """
    GM7 – 数值间隙
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue

            max_score = 0.0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r1, c1 = r + dr, c + dc
                r2, c2 = r - dr, c - dc
                if 0 <= r1 < rows and 0 <= c1 < cols and 0 <= r2 < rows and 0 <= c2 < cols:
                    v1 = grid[r1, c1]
                    v2 = grid[r2, c2]
                    if v1 != -1 and v2 != -1:
                        gap = abs(v1 - v2)
                        if gap == 2:
                            max_score = max(max_score, 1.0)
            scores[r, c] = max_score

    return scores * config.weight


def EXT_GM8_Edge_Affinity_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM8",
) -> np.ndarray:
    """
    GM8 – 边缘亲和
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue

            # 角落
            if (r == 0 or r == rows - 1) and (c == 0 or c == cols - 1):
                scores[r, c] = 1.0
            # 边缘
            elif r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                scores[r, c] = 0.7
            else:
                scores[r, c] = 0.3

    return scores * config.weight


def EXT_GM9_Center_Control_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM9",
) -> np.ndarray:
    """
    GM9 – 中心控制
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    center_r, center_c = rows // 2, cols // 2
    max_dist = center_r + center_c

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            dist = abs(r - center_r) + abs(c - center_c)
            scores[r, c] = 1.0 - (dist / max_dist)

    return scores * config.weight


def EXT_GM10_BlockingValue_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM10",
) -> np.ndarray:
    """
    GM10 – 阻挡价值
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue

            low_penalty = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r1, c1 = r + dr, c + dc
                r2, c2 = r - dr, c - dc
                if 0 <= r1 < rows and 0 <= c1 < cols and 0 <= r2 < rows and 0 <= c2 < cols:
                    v1 = grid[r1, c1]
                    v2 = grid[r2, c2]
                    if v1 != -1 and v2 != -1 and v1 == v2:
                        low_penalty = True
            scores[r, c] = 0.0 if low_penalty else 1.0

    return scores * config.weight


def EXT_GM11_PairCorrelation_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM11",
) -> np.ndarray:
    """
    GM11 – 对偶相关
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue

            best_corr = 0.0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r1, c1 = r + dr, c + dc
                if 0 <= r1 < rows and 0 <= c1 < cols and grid[r1, c1] != -1:
                    val = grid[r1, c1]
                    # 假设对偶相关度就是 1/(|val - avg_val|+1)
                    avg_val = np.mean(grid[grid != -1]) if np.any(grid != -1) else 0.0
                    corr = 1.0 / (abs(val - avg_val) + 1.0)
                    best_corr = max(best_corr, corr)
            scores[r, c] = best_corr

    return scores * config.weight


def EXT_GM12_IslandAnalysis_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM12",
) -> np.ndarray:
    """
    GM12 – 岛屿分析
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    visited = np.zeros((rows, cols), dtype=bool)

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1 or visited[r, c]:
                continue

            # BFS 查找空格连通区
            queue = deque([(r, c)])
            component = [(r, c)]
            visited[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and not visited[nr, nc]
                        and grid[nr, nc] == -1
                    ):
                        visited[nr, nc] = True
                        component.append((nr, nc))
                        queue.append((nr, nc))

            area = float(len(component))
            total = float(rows * cols)
            norm_area = MathUtils.normalize_value(area, 0, total, clamp=True)
            for (ri, ci) in component:
                scores[ri, ci] = norm_area

    return scores * config.weight