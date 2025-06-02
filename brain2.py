# brain2.py
# 第二部分：包含 GM4–GM12 各 scoring 模块 (全部向量化) 与对应 Config。

import numpy as np
import math
import logging
from typing import List, Dict, Tuple, Any, Set

from pydantic import BaseModel, Field

from brain1 import MathUtils, BoardAnalyzerUtils, BaseModuleConfig

logger = logging.getLogger(__name__)


class SpatialAutocorrelationConfig(BaseModuleConfig):
    """
    (GM4) 空间自相关性配置
    """
    autocorrelation_type: str = Field(
        default="positive",
        pattern="^(positive|negative)$",
        description="偏好正 (positive) 或负 (negative) 自相关",
    )
    neighborhood_radius: int = Field(default=1, ge=1)
    use_median_for_hypothetical: bool = Field(
        default=True,
        description="True: 用潜在数字中位数; False: 用平均数",
    )


class LineCompletionConfig(BaseModuleConfig):
    """
    (GM5) 线段补全配置
    """
    target_line_length: int = Field(default=3, ge=3)
    score_identical_3: float = Field(default=0.6, ge=0.0)
    score_arithmetic_3_mend: float = Field(default=0.7, ge=0.0)
    score_arithmetic_3_extend: float = Field(default=0.5, ge=0.0)
    enable_quality_enhancement: bool = Field(default=True)
    score_arithmetic_3_mend_high_val_bonus: float = Field(
        default=0.2, ge=0.0, description="高值等差修复额外奖励"
    )
    high_value_threshold_factor_gm5: float = Field(
        default=0.66, ge=0.0, le=1.0, description="高值门槛占比"
    )


class SymmetryPotentialConfig(BaseModuleConfig):
    """
    (GM6) 对称性潜力配置
    """
    score_horizontal: float = Field(default=0.7, ge=0.0)
    score_vertical: float = Field(default=0.7, ge=0.0)
    score_point_center: float = Field(default=0.8, ge=0.0)
    score_main_diagonal: float = Field(default=0.6, ge=0.0)
    score_anti_diagonal: float = Field(default=0.6, ge=0.0)
    strict_square_for_diagonal: bool = Field(
        default=True,
        description="对角线对称是否严格要求方形",
    )


class NumericGapsConfig(BaseModuleConfig):
    """
    (GM7) 数值间隙配置
    """
    score_arithmetic_1_gap_fill: float = Field(default=0.9, ge=0.0)
    score_arithmetic_generic_mend: float = Field(default=0.7, ge=0.0)
    score_arithmetic_generic_extend: float = Field(default=0.5, ge=0.0)
    enable_quality_enhancement_gm7: bool = Field(default=True)
    score_gap_fill_high_val_bonus: float = Field(
        default=0.1, ge=0.0, description="高值间隙填充额外奖励"
    )
    high_value_threshold_factor_gm7: float = Field(
        default=0.66, ge=0.0, le=1.0
    )


class EdgeAffinityConfig(BaseModuleConfig):
    """
    (GM8) 边缘亲和配置
    """
    affinity_mode: str = Field(
        default="prefer_edge",
        pattern="^(prefer_edge|avoid_edge)$",
        description="边缘亲和模式",
    )
    corner_bonus_prefer: float = Field(default=0.2, ge=0.0)
    corner_penalty_avoid: float = Field(default=0.2, ge=0.0)


class CenterControlConfig(BaseModuleConfig):
    """
    (GM9) 中心控制配置
    """
    affinity_mode: str = Field(
        default="prefer_center",
        pattern="^(prefer_center|avoid_center)$",
        description="中心控制模式",
    )


class BlockingValueConfig(BaseModuleConfig):
    """
    (GM10) 阻挡价值配置
    """
    undesirable_sequences_list: List[List[int]] = Field(
        default_factory=lambda: [[1, 1, 1], [2, 2, 2]],
        description="不良序列列表",
    )
    score_if_safe: float = Field(default=0.9, ge=0.0, le=1.0)
    score_if_unsafe: float = Field(default=0.1, ge=0.0, le=1.0)
    check_line_length: int = Field(default=3, ge=2)


class PairCorrelationConfig(BaseModuleConfig):
    """
    (GM11) 对偶相关性配置
    """
    favorable_pairs: Dict[Tuple[int, int], float] = Field(
        default_factory=lambda: {
            (3, 7): 0.8,
            (7, 3): 0.8,
            (1, 2): 0.6,
            (2, 1): 0.6,
            (10, 20): 0.7,
            (20, 10): 0.7,
        },
        description="有利数对映射",
    )


class IslandAnalysisConfig(BaseModuleConfig):
    """
    (GM12) 岛屿分析配置
    """
    w_size: float = Field(default=0.4, ge=0.0, le=1.0)
    w_compactness: float = Field(default=0.3, ge=0.0, le=1.0)
    w_avg_value: float = Field(default=0.3, ge=0.0, le=1.0)


def EXT_GM4_Spatial_Auto_Corr_Vec(
    grid: np.ndarray,
    config: SpatialAutocorrelationConfig,
    request_id: str | None = "N/A_GM4",
) -> np.ndarray:
    """
    (GM4–空间自相关性)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    max_val = rows * cols
    revealed_mask = grid != -1
    revealed_vals = grid[revealed_mask]
    potential_vals = [v for v in range(1, max_val + 1) if v not in revealed_vals.tolist()]

    if potential_vals:
        hypo_val = (
            float(np.median(potential_vals))
            if config.use_median_for_hypothetical
            else float(np.mean(potential_vals))
        )
    else:
        hypo_val = float(max_val) / 2.0

    coords = np.stack(np.where(grid == -1), axis=1)
    for (r, c) in coords:
        neigh_vals = BoardAnalyzerUtils.get_neighborhood_values(
            grid,
            r,
            c,
            radius=config.neighborhood_radius,
            eight_connectivity=True,
            val_func=lambda x: float(x) if x != -1 else None,
            include_center=False,
        )
        if not neigh_vals:
            scores[r, c] = 0.5
            continue
        mean_neigh = float(np.mean(neigh_vals))
        diff = abs(hypo_val - mean_neigh)
        max_possible = float(max_val)
        norm_diff = MathUtils.normalize_value(diff, 0.0, max_possible, clamp=True)
        scores[r, c] = (1.0 - norm_diff) if config.autocorrelation_type == "positive" else norm_diff

    return scores * config.weight


def EXT_GM5_Line_Completion_Vec(
    grid: np.ndarray,
    config: LineCompletionConfig,
    request_id: str | None = "N/A_GM5",
) -> np.ndarray:
    """
    (GM5–线段补全)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    coords = np.stack(np.where(grid == -1), axis=1)
    for (r, c) in coords:
        lines = [
            grid[r, :].tolist(),
            grid[:, c].tolist(),
        ]

        diag1 = []
        k = 0
        while r - k >= 0 and c - k >= 0:
            diag1.append(grid[r - k, c - k])
            k += 1
        diag1.reverse()
        k = 1
        while r + k < rows and c + k < cols:
            diag1.append(grid[r + k, c + k])
            k += 1
        lines.append(diag1)

        diag2 = []
        k = 0
        while r - k >= 0 and c + k < cols:
            diag2.append(grid[r - k, c + k])
            k += 1
        diag2.reverse()
        k = 1
        while r + k < rows and c - k >= 0:
            diag2.append(grid[r + k, c - k])
            k += 1
        lines.append(diag2)

        best_score = 0.0
        for line in lines:
            sequences = BoardAnalyzerUtils.find_sequences_in_line(
                line,
                min_len=config.target_line_length,
                check_arithmetic=True,
                check_geometric=False,
                allow_gaps=0,
            )
            if sequences:
                for seq in sequences:
                    base = (
                        config.score_identical_3
                        if len(set(seq)) == 1
                        else config.score_arithmetic_3_mend
                    )
                    if (
                        config.enable_quality_enhancement
                        and max(seq) > config.high_value_threshold_factor_gm5 * (rows * cols)
                    ):
                        base += config.score_arithmetic_3_mend_high_val_bonus
                    best_score = max(best_score, base)
        scores[r, c] = best_score

    return scores * config.weight


def EXT_GM6_Symmetry_Potential_Vec(
    grid: np.ndarray,
    config: SymmetryPotentialConfig,
    request_id: str | None = "N/A_GM6",
) -> np.ndarray:
    """
    (GM6–对称潜力)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    center_r = (rows - 1) / 2.0
    center_c = (cols - 1) / 2.0

    coords = np.stack(np.where(grid == -1), axis=1)
    for (r, c) in coords:
        sc = 0.0
        if grid[r, cols - 1 - c] != -1:
            sc += config.score_horizontal
        if grid[rows - 1 - r, c] != -1:
            sc += config.score_vertical

        sym_r = int(round(2 * center_r - r))
        sym_c = int(round(2 * center_c - c))
        if 0 <= sym_r < rows and 0 <= sym_c < cols and grid[sym_r, sym_c] != -1:
            sc += config.score_point_center

        if grid[c, r] != -1:
            sc += config.score_main_diagonal

        mirror_r = cols - 1 - c
        mirror_c = rows - 1 - r
        if 0 <= mirror_r < rows and 0 <= mirror_c < cols and grid[mirror_r, mirror_c] != -1:
            sc += config.score_anti_diagonal

        scores[r, c] = sc

    max_score = (
        config.score_horizontal
        + config.score_vertical
        + config.score_point_center
        + config.score_main_diagonal
        + config.score_anti_diagonal
    )
    if max_score > 0:
        scores = scores / max_score

    return scores * config.weight


def EXT_GM7_Numeric_Gaps_Vec(
    grid: np.ndarray,
    config: NumericGapsConfig,
    request_id: str | None = "N/A_GM7",
) -> np.ndarray:
    """
    (GM7–数值间隙)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    coords = np.stack(np.where(grid == -1), axis=1)
    for (r, c) in coords:
        neighborhood = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbor_val = grid[nr, nc]
                if neighbor_val != -1:
                    neighborhood.append(int(neighbor_val))
        if len(neighborhood) < 2:
            scores[r, c] = 0.0
            continue

        best = 0.0
        for i in range(len(neighborhood)):
            for j in range(i + 1, len(neighborhood)):
                a = neighborhood[i]
                b = neighborhood[j]
                if abs(a - b) == 2:
                    mid = (a + b) // 2
                    if mid not in neighborhood:
                        best = max(best, config.score_arithmetic_1_gap_fill)
                else:
                    diff = b - a
                    length = abs(diff) + 1
                    if length >= config.target_line_length:
                        best = max(best, config.score_arithmetic_generic_mend)
                    else:
                        best = max(best, config.score_arithmetic_generic_extend)

        if (
            max(neighborhood) > config.high_value_threshold_factor_gm7 * (rows * cols)
            and config.enable_quality_enhancement_gm7
        ):
            best += config.score_gap_fill_high_val_bonus
            best = min(best, 1.0)

        scores[r, c] = best

    return scores * config.weight


def EXT_GM8_Edge_Affinity_Vec(
    grid: np.ndarray,
    config: EdgeAffinityConfig,
    request_id: str | None = "N/A_GM8",
) -> np.ndarray:
    """
    (GM8–边缘亲和)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            on_edge = r == 0 or r == rows - 1 or c == 0 or c == cols - 1
            on_corner = (r == 0 or r == rows - 1) and (c == 0 or c == cols - 1)
            if config.affinity_mode == "prefer_edge":
                if on_corner:
                    scores[r, c] = config.corner_bonus_prefer
                elif on_edge:
                    scores[r, c] = config.corner_bonus_prefer / 2.0
            else:
                if on_corner or on_edge:
                    scores[r, c] = max(0.0, 1.0 - config.corner_penalty_avoid)
                else:
                    scores[r, c] = 1.0

    return scores * config.weight


def EXT_GM9_Center_Control_Vec(
    grid: np.ndarray,
    config: CenterControlConfig,
    request_id: str | None = "N/A_GM9",
) -> np.ndarray:
    """
    (GM9–中心控制)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    center_r = (rows - 1) / 2.0
    center_c = (cols - 1) / 2.0

    idxs = np.indices((rows, cols))
    rr = idxs[0].astype(float)
    cc = idxs[1].astype(float)
    dists = np.sqrt((rr - center_r) ** 2 + (cc - center_c) ** 2)

    min_d, max_d = float(np.min(dists)), float(np.max(dists))
    norm_d = (
        (dists - min_d) / (max_d - min_d)
        if not math.isclose(max_d, min_d)
        else np.zeros_like(dists)
    )

    raw_scores = (1.0 - norm_d) if config.affinity_mode == "prefer_center" else norm_d

    mask = grid == -1
    scores[mask] = raw_scores[mask]

    return scores * config.weight


def EXT_GM10_BlockingValue_Vec(
    grid: np.ndarray,
    config: BlockingValueConfig,
    request_id: str | None = "N/A_GM10",
) -> np.ndarray:
    """
    (GM10–阻挡价值)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    L = config.check_line_length

    def check_line_for_undesirable(line: List[int]) -> bool:
        n = len(line)
        for i in range(n - L + 1):
            segment = line[i : i + L]
            for bad in config.undesirable_sequences_list:
                if segment == bad:
                    return True
        return False

    coords = np.stack(np.where(grid == -1), axis=1)
    for (r, c) in coords:
        unsafe = False

        row_list = grid[r, :].tolist()
        if check_line_for_undesirable(row_list):
            unsafe = True

        if not unsafe:
            col_list = grid[:, c].tolist()
            if check_line_for_undesirable(col_list):
                unsafe = True

        if not unsafe:
            diag1 = []
            k = 0
            while r - k >= 0 and c - k >= 0:
                diag1.append(grid[r - k, c - k])
                k += 1
            diag1.reverse()
            k = 1
            while r + k < rows and c + k < cols:
                diag1.append(grid[r + k, c + k])
                k += 1
            if check_line_for_undesirable(diag1):
                unsafe = True

        if not unsafe:
            diag2 = []
            k = 0
            while r - k >= 0 and c + k < cols:
                diag2.append(grid[r - k, c + k])
                k += 1
            diag2.reverse()
            k = 1
            while r + k < rows and c - k >= 0:
                diag2.append(grid[r + k, c - k])
                k += 1
            if check_line_for_undesirable(diag2):
                unsafe = True

        scores[r, c] = config.score_if_unsafe if unsafe else config.score_if_safe

    return scores * config.weight


def EXT_GM11_PairCorrelation_Vec(
    grid: np.ndarray,
    config: PairCorrelationConfig,
    request_id: str | None = "N/A_GM11",
) -> np.ndarray:
    """
    (GM11–对偶相关)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    max_val = rows * cols
    revealed_vals = grid[grid != -1]
    potential_vals = [v for v in range(1, max_val + 1) if v not in revealed_vals.tolist()]

    if not potential_vals:
        return scores

    coords = np.stack(np.where(grid == -1), axis=1)
    for (r, c) in coords:
        best = 0.0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                neigh = int(grid[nr, nc])
                for h in potential_vals:
                    pair = (neigh, h)
                    if pair in config.favorable_pairs:
                        best = max(best, config.favorable_pairs[pair])
        scores[r, c] = best

    return scores * config.weight


def EXT_GM12_IslandAnalysis_Vec(
    grid: np.ndarray,
    config: IslandAnalysisConfig,
    request_id: str | None = "N/A_GM12",
) -> np.ndarray:
    """
    (GM12–岛屿分析)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    revealed_mask = grid != -1

    def get_connected_component_size(r0: int, c0: int) -> int:
        visited: Set[Tuple[int, int]] = set()
        queue = [(r0, c0)]
        visited.add((r0, c0))
        size = 1
        for (r_i, c_i) in queue:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r_i + dr, c_i + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and (nr, nc) not in visited
                    and grid[nr, nc] != -1
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
                    size += 1
        return size

    coords = np.stack(np.where(grid == -1), axis=1)
    raw_vals = []
    for (r, c) in coords:
        comp_sizes = []
        comp_values = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                size = get_connected_component_size(nr, nc)
                comp_sizes.append(size)
                visited: Set[Tuple[int, int]] = set()
                queue = [(nr, nc)]
                visited.add((nr, nc))
                vals = [grid[nr, nc]]
                for (rr_i, cc_i) in queue:
                    for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        rrr, ccc = rr_i + ddr, cc_i + ddc
                        if (
                            0 <= rrr < rows
                            and 0 <= ccc < cols
                            and (rrr, ccc) not in visited
                            and grid[rrr, ccc] != -1
                        ):
                            visited.add((rrr, ccc))
                            queue.append((rrr, ccc))
                            vals.append(grid[rrr, ccc])
                comp_values.append(float(np.mean(vals)))
        if comp_sizes:
            size_val = float(max(comp_sizes))
            compactness_val = size_val / (len(comp_sizes) + 1.0)
            avg_val = float(np.mean(comp_values)) if comp_values else 0.0
        else:
            size_val = 0.0
            compactness_val = 0.0
            avg_val = 0.0
        raw_vals.append((r, c, size_val, compactness_val, avg_val))

    if raw_vals:
        sizes = np.array([v[2] for v in raw_vals], dtype=float)
        comps = np.array([v[3] for v in raw_vals], dtype=float)
        avgs = np.array([v[4] for v in raw_vals], dtype=float)
        mn_s, mx_s = float(np.min(sizes)), float(np.max(sizes))
        mn_c, mx_c = float(np.min(comps)), float(np.max(comps))
        mn_a, mx_a = float(np.min(avgs)), float(np.max(avgs))

        for idx, (r, c, s_val, c_val, a_val) in enumerate(raw_vals):
            ns = 0.0 if math.isclose(mx_s, mn_s) else (s_val - mn_s) / (mx_s - mn_s)
            nc = 0.0 if math.isclose(mx_c, mn_c) else (c_val - mn_c) / (mx_c - mn_c)
            na = 0.0 if math.isclose(mx_a, mn_a) else (a_val - mn_a) / (mx_a - mn_a)
            sc = config.w_size * ns + config.w_compactness * nc + config.w_avg_value * na
            scores[r, c] = sc

    return scores * config.weight