# brain3.py
# 第三部分：包含 GM13–GM26 高阶评分模块，全部向量化或“针对空格小范围循环”来保持性能。

import numpy as np
import math
import logging
from typing import List, Dict, Tuple, Set, Any

from pydantic import BaseModel, Field

from brain1 import MathUtils, BoardAnalyzerUtils, BaseModuleConfig

logger = logging.getLogger(__name__)


class SequenceDiversityConfig(BaseModuleConfig):
    """
    (GM13) 序列多样性配置
    """
    short_sequence_len: int = Field(default=3, ge=2, description="短序列长度")


class RiskAssessmentConfig(BaseModuleConfig):
    """
    (GM14) 风险评估配置
    """
    flexibility_metric_mode: str = Field(
        default="subsequent_moves",
        pattern="^(subsequent_moves|product_moves_empty_cells)$",
        description="灵活性度量模式",
    )


class InformationGainConfig(BaseModuleConfig):
    """
    (GM15) 信息增益配置
    """
    entropy_scope: str = Field(
        default="global_full",
        pattern="^(global_full|global_filled_only)$",
        description="熵计算范围",
    )


class HarmonicCentralityConfig(BaseModuleConfig):
    """
    (GM16) 调和中心性配置
    """
    node_definition: str = Field(
        default="all_cells",
        pattern="^(all_cells|empty_cells_only|filled_cells_only)$",
        description="节点类型",
    )


class LocalEntropyMinimizationConfig(BaseModuleConfig):
    """
    (GM17) 局部熵最小化配置
    """
    radius: int = Field(default=1, ge=1)


class RLValueEstimationConfig(BaseModuleConfig):
    """
    (GM18) RL 值估计配置
    """
    feature_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "identical_3": 1.0,
            "arithmetic_3": 0.7,
            "board_density_factor": 0.2,
            "central_control_boost": 0.1,
            "edge_affinity_boost": 0.05,
        },
        description="特征权重映射",
    )


class SkipPatternConfig(BaseModuleConfig):
    """
    (GM19) 跳格模式配置
    """
    min_occurrences_for_pattern_factor: float = Field(
        default=0.05, ge=0.0, le=1.0, description="模式最小出现比例"
    )
    base_pattern_definition: str = Field(
        default="left_to_right_top_to_bottom", description="扫描模式定义"
    )


class SkipPatternConfidenceConfig(BaseModuleConfig):
    """
    (GM20) 跳格置信度配置
    """
    min_occurrences_for_pattern_factor_gm20: float = Field(
        default=0.05, ge=0.0, le=1.0
    )
    arithmetic_enhancement_bonus: float = Field(
        default=0.4, ge=0.0, description="等差增强因子"
    )
    internal_gap_fill_bonus: float = Field(
        default=0.1, ge=0.0, description="内部 gap 填充奖励"
    )


class ClusterBalanceConfig(BaseModuleConfig):
    """
    (GM21) 群集平衡配置（示例占位）
    """
    pass


class CoOccurrenceConfig(BaseModuleConfig):
    """
    (GM22) 共现性配置（示例占位）
    """
    pass


class MotifDetectionConfig(BaseModuleConfig):
    """
    (GM23) 模式检测配置（示例占位）
    """
    pass


class TemporalCoherenceConfig(BaseModuleConfig):
    """
    (GM24) 时序连贯性配置（示例占位）
    """
    pass


class StrategicDepthConfig(BaseModuleConfig):
    """
    (GM25) 策略深度配置（示例占位）
    """
    pass


class ContextualFlexibilityConfig(BaseModuleConfig):
    """
    (GM26) 情境灵活性配置（示例占位）
    """
    pass


def EXT_GM13_Sequence_Diversity_Vec(
    grid: np.ndarray,
    config: SequenceDiversityConfig,
    request_id: str | None = "N/A_GM13",
) -> np.ndarray:
    """
    (GM13–序列多样性)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    L = config.short_sequence_len

    max_val = rows * cols
    revealed_vals = grid[grid != -1]
    potential_vals = [v for v in range(1, max_val + 1) if v not in revealed_vals.tolist()]
    if not potential_vals:
        return scores

    heuristic_max = float(4 * 2 * (1 if L == 3 else L))
    coords = np.stack(np.where(grid == -1), axis=1)
    for (r, c) in coords:
        max_count = 0
        for h in potential_vals:
            temp = grid.copy()
            temp[r, c] = h
            sigs: Set[Tuple[str, Tuple[int, int], int]] = set()
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                for offset in range(L):
                    seq_vals: List[int] = []
                    valid = True
                    for k in range(L):
                        rr = r + (k - offset) * dr
                        cc = c + (k - offset) * dc
                        if not (0 <= rr < rows and 0 <= cc < cols):
                            valid = False
                            break
                        seq_vals.append(int(temp[rr, cc]))
                    if not valid:
                        continue
                    diffs = [seq_vals[i + 1] - seq_vals[i] for i in range(L - 1)]
                    if all(math.isclose(diffs[i], diffs[0]) for i in range(len(diffs))) and not math.isclose(diffs[0], 0):
                        sigs.add(("arithmetic", (dr, dc), int(diffs[0])))
                    if len(set(seq_vals)) == 1 and seq_vals[0] != -1:
                        sigs.add(("identical", (dr, dc), seq_vals[0]))
            count = len(sigs)
            if count > max_count:
                max_count = count
        scores[r, c] = MathUtils.normalize_value(float(max_count), 0.0, heuristic_max, clamp=True)

    return scores * config.weight


def EXT_GM14_Risk_Assessment_Vec(
    grid: np.ndarray,
    config: RiskAssessmentConfig,
    request_id: str | None = "N/A_GM14",
) -> np.ndarray:
    """
    (GM14–风险评估)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    max_val = rows * cols
    original_revealed = set(grid[grid != -1].tolist())
    all_vals = set(range(1, max_val + 1))

    coords = np.stack(np.where(grid == -1), axis=1)
    for (r, c) in coords:
        available = list(all_vals - original_revealed)
        if not available:
            scores[r, c] = 0.0
            continue
        # subsequent_moves 模式
        if config.flexibility_metric_mode == "subsequent_moves":
            legal_after = len(available) - 1
            scores[r, c] = float(legal_after) / float(max_val) if max_val > 0 else 0.0
        else:
            legal_after = len(available) - 1
            rem_empty = int(np.sum(grid == -1) - 1)
            raw = float(legal_after * rem_empty)
            norm = float(max_val * rows * cols) if rows * cols > 0 else 1.0
            scores[r, c] = raw / norm

    return scores * config.weight


def EXT_GM15_Information_Gain_Vec(
    grid: np.ndarray,
    config: InformationGainConfig,
    request_id: str | None = "N/A_GM15",
) -> np.ndarray:
    """
    (GM15–信息增益)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    max_val = rows * cols
    revealed_vals = grid[grid != -1].tolist()
    if config.entropy_scope == "global_full":
        base_entropy = float(MathUtils.get_entropy(grid.flatten().tolist()))
    else:
        base_entropy = float(MathUtils.get_entropy(revealed_vals))

    potential_vals = [v for v in range(1, max_val + 1) if v not in revealed_vals]

    coords = np.stack(np.where(grid == -1), axis=1)
    for (r, c) in coords:
        max_delta = 0.0
        for h in potential_vals:
            temp = grid.copy()
            temp[r, c] = h
            if config.entropy_scope == "global_full":
                ent = float(MathUtils.get_entropy(temp.flatten().tolist()))
            else:
                filled = temp[temp != -1].tolist()
                ent = float(MathUtils.get_entropy(filled))
            delta = abs(ent - base_entropy)
            if delta > max_delta:
                max_delta = delta
        norm_delta = max_delta / math.log2(max_val) if max_val > 1 else max_delta
        scores[r, c] = norm_delta

    return scores * config.weight


def EXT_GM16_Harmonic_Centrality_Vec(
    grid: np.ndarray,
    config: HarmonicCentralityConfig,
    request_id: str | None = "N/A_GM16",
) -> np.ndarray:
    """
    (GM16–调和中心性)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    if config.node_definition == "all_cells":
        node_mask = grid == grid  # 所有位置都为节点
    elif config.node_definition == "empty_cells_only":
        node_mask = grid == -1
    else:
        node_mask = grid != -1

    node_coords = np.stack(np.where(node_mask), axis=1)
    if node_coords.size == 0:
        return scores

    idxs = np.indices((rows, cols))
    rr = idxs[0][..., None]
    cc = idxs[1][..., None]

    r_nodes = node_coords[:, 0]
    c_nodes = node_coords[:, 1]
    rn = r_nodes[None, None, :]
    cn = c_nodes[None, None, :]

    dist = np.abs(rr - rn) + np.abs(cc - cn)
    dist = np.where(dist == 0, np.inf, dist)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(dist == np.inf, 0.0, 1.0 / dist)
    harmonic = np.nansum(inv, axis=2)

    scores[grid == -1] = harmonic[grid == -1]
    max_h = float(np.max(scores))
    if max_h > 0:
        scores = scores / max_h

    return scores * config.weight


def EXT_GM17_Local_Entropy_Vec(
    grid: np.ndarray,
    config: LocalEntropyMinimizationConfig,
    request_id: str | None = "N/A_GM17",
) -> np.ndarray:
    """
    (GM17–局部熵最小化)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    r = config.radius

    coords = np.stack(np.where(grid == -1), axis=1)
    for (i, j) in coords:
        block = grid[
            max(0, i - r) : min(rows, i + r + 1),
            max(0, j - r) : min(cols, j + r + 1),
        ].flatten().tolist()
        if not block:
            continue
        ent = MathUtils.get_entropy(block)
        max_ent = math.log2(len(block)) if len(block) > 1 else 1.0
        val = 1.0 - (ent / max_ent if max_ent > 0 else 0.0)
        scores[i, j] = val

    return scores * config.weight


def EXT_GM18_RL_Value_Estimation_Vec(
    grid: np.ndarray,
    config: RLValueEstimationConfig,
    request_id: str | None = "N/A_GM18",
) -> np.ndarray:
    """
    (GM18–RL 值估计)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    filled = int(np.sum(grid != -1))
    total = rows * cols
    density = filled / total if total > 0 else 0.0

    from brain2 import EXT_GM8_Edge_Affinity_Vec, EXT_GM9_Center_Control_Vec

    gm8_raw = EXT_GM8_Edge_Affinity_Vec(grid, BaseModuleConfig(enabled=True, weight=1.0))
    gm9_raw = EXT_GM9_Center_Control_Vec(grid, BaseModuleConfig(enabled=True, weight=1.0))

    coords = np.stack(np.where(grid == -1), axis=1)
    for (i, j) in coords:
        has_identical_3 = 0
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            seq = []
            for k in range(-2, 3):
                rr = i + dr * k
                cc = j + dc * k
                seq.append(grid[rr, cc] if 0 <= rr < rows and 0 <= cc < cols else -1)
            for start in range(3):
                window = seq[start : start + 3]
                if window.count(window[0]) == 3 and window[0] != -1:
                    has_identical_3 = 1
                    break
            if has_identical_3:
                break

        has_arithmetic_3 = 0
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            seq = []
            for k in range(-2, 3):
                rr = i + dr * k
                cc = j + dc * k
                seq.append(grid[rr, cc] if 0 <= rr < rows and 0 <= cc < cols else -1)
            for start in range(3):
                window = seq[start : start + 3]
                if -1 not in window:
                    diffs = [window[1] - window[0], window[2] - window[1]]
                    if math.isclose(diffs[0], diffs[1]):
                        has_arithmetic_3 = 1
                        break
            if has_arithmetic_3:
                break

        f_density = density
        f_center = gm9_raw[i, j]
        f_edge = gm8_raw[i, j]

        val = (
            f_density * config.feature_weights.get("board_density_factor", 0.0)
            + f_center * config.feature_weights.get("central_control_boost", 0.0)
            + f_edge * config.feature_weights.get("edge_affinity_boost", 0.0)
            + has_identical_3 * config.feature_weights.get("identical_3", 0.0)
            + has_arithmetic_3 * config.feature_weights.get("arithmetic_3", 0.0)
        )
        scores[i, j] = val

    mask = grid == -1
    if np.any(scores[mask] > 0):
        mn = float(np.min(scores[mask]))
        mx = float(np.max(scores[mask]))
        if not math.isclose(mx, mn):
            scores[mask] = (scores[mask] - mn) / (mx - mn)
        else:
            scores[mask] = 0.0

    return scores * config.weight


def EXT_GM19_SkipPattern_Vec(
    grid: np.ndarray,
    config: SkipPatternConfig,
    request_id: str | None = "N/A_GM19",
) -> np.ndarray:
    """
    (GM19–跳格模式)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    coords = np.stack(np.where(grid == -1), axis=1)
    for (r, c) in coords:
        skip_pairs = 0
        total_pairs = 0

        def analyze_line(line: List[int]):
            nonlocal skip_pairs, total_pairs
            idxs = [i for i, v in enumerate(line) if v != -1]
            for i in range(len(idxs) - 1):
                total_pairs += 1
                if idxs[i + 1] - idxs[i] >= 2:
                    skip_pairs += 1

        analyze_line(grid[r, :].tolist())
        analyze_line(grid[:, c].tolist())

        diag1 = []
        rr, cc = r, c
        while rr >= 0 and cc >= 0:
            diag1.append(grid[rr, cc])
            rr -= 1
            cc -= 1
        diag1.reverse()
        rr, cc = r + 1, c + 1
        while rr < rows and cc < cols:
            diag1.append(grid[rr, cc])
            rr += 1
            cc += 1
        analyze_line(diag1)

        diag2 = []
        rr, cc = r, c
        while rr >= 0 and cc < cols:
            diag2.append(grid[rr, cc])
            rr -= 1
            cc += 1
        diag2.reverse()
        rr, cc = r + 1, c - 1
        while rr < rows and cc >= 0:
            diag2.append(grid[rr, cc])
            rr += 1
            cc -= 1
        analyze_line(diag2)

        if total_pairs > 0 and (skip_pairs / total_pairs) >= config.min_occurrences_for_pattern_factor:
            scores[r, c] = 1.0
        else:
            scores[r, c] = 0.0

    return scores * config.weight


def EXT_GM20_SkipPattern_Confidence_Vec(
    grid: np.ndarray,
    config: SkipPatternConfidenceConfig,
    request_id: str | None = "N/A_GM20",
) -> np.ndarray:
    """
    (GM20–跳格置信度)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    coords = np.stack(np.where(grid == -1), axis=1)
    for (r, c) in coords:
        skip_pairs = 0
        total_pairs = 0

        def analyze_line_for_confidence(line: List[int]) -> bool:
            nonlocal skip_pairs, total_pairs
            idxs = [i for i, v in enumerate(line) if v != -1]
            for i in range(len(idxs) - 1):
                total_pairs += 1
                if idxs[i + 1] - idxs[i] >= 2:
                    skip_pairs += 1
            for i in range(len(line) - 2):
                window = line[i : i + 3]
                if -1 not in window:
                    diffs = [window[1] - window[0], window[2] - window[1]]
                    if math.isclose(diffs[0], diffs[1]) and not math.isclose(diffs[0], 0):
                        return True
            return False

        has_arith3 = False
        arr = grid[r, :].tolist()
        if analyze_line_for_confidence(arr):
            has_arith3 = True
        if not has_arith3:
            arr = grid[:, c].tolist()
            if analyze_line_for_confidence(arr):
                has_arith3 = True
        if not has_arith3:
            diag1 = []
            rr, cc = r, c
            while rr >= 0 and cc >= 0:
                diag1.append(grid[rr, cc])
                rr -= 1
                cc -= 1
            diag1.reverse()
            rr, cc = r + 1, c + 1
            while rr < rows and cc < cols:
                diag1.append(grid[rr, cc])
                rr += 1
                cc += 1
            if analyze_line_for_confidence(diag1):
                has_arith3 = True
        if not has_arith3:
            diag2 = []
            rr, cc = r, c
            while rr >= 0 and cc < cols:
                diag2.append(grid[rr, cc])
                rr -= 1
                cc += 1
            diag2.reverse()
            rr, cc = r + 1, c - 1
            while rr < rows and cc >= 0:
                diag2.append(grid[rr, cc])
                rr += 1
                cc -= 1
            if analyze_line_for_confidence(diag2):
                has_arith3 = True

        ratio = (skip_pairs / total_pairs) if total_pairs > 0 else 0.0
        if ratio >= config.min_occurrences_for_pattern_factor_gm20:
            val = ratio + config.arithmetic_enhancement_bonus
            if has_arith3:
                val += config.internal_gap_fill_bonus
        else:
            val = 0.0
        scores[r, c] = min(val, 1.0)

    return scores * config.weight


def EXT_GM21_ClusterBalance_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: str | None = "N/A_GM21",
) -> np.ndarray:
    """
    (GM21–群集平衡) [示例占位]
    """
    return np.zeros_like(grid, dtype=float)


def EXT_GM22_CoOccurrence_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: str | None = "N/A_GM22",
) -> np.ndarray:
    """
    (GM22–共现性) [示例占位]
    """
    return np.zeros_like(grid, dtype=float)


def EXT_GM23_MotifDetection_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: str | None = "N/A_GM23",
) -> np.ndarray:
    """
    (GM23–模式检测) [示例占位]
    """
    return np.zeros_like(grid, dtype=float)


def EXT_GM24_TemporalCoherence_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: str | None = "N/A_GM24",
) -> np.ndarray:
    """
    (GM24–时序连贯性) [示例占位]
    """
    return np.zeros_like(grid, dtype=float)


def EXT_GM25_StrategicDepth_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: str | None = "N/A_GM25",
) -> np.ndarray:
    """
    (GM25–策略深度) [示例占位]
    """
    return np.zeros_like(grid, dtype=float)


def EXT_GM26_ContextualFlexibility_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: str | None = "N/A_GM26",
) -> np.ndarray:
    """
    (GM26–情境灵活性) [示例占位]
    """
    return np.zeros_like(grid, dtype=float)