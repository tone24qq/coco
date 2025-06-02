# brain3.py

import numpy as np
import math
import logging
from collections import deque, Counter
from typing import List, Tuple, Any, Optional, Dict

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BaseModuleConfig(BaseModel):
    """
    基础模块配置：通用字段（enabled、weight），以及各模块可能使用的额外参数。
    """
    enabled: bool = Field(default=True, description="模块启用/禁用开关")
    weight: float = Field(default=1.0, ge=0.0, description="模块权重")
    # GM17 用到的局部熵半径
    radius: Optional[int] = Field(default=1, ge=1, description="局部熵半径")
    # GM24 用到的历史揭露序列
    historical_sequence: Optional[List[Tuple[int, int]]] = Field(
        default=None, description="揭露历史顺序"
    )


def EXT_GM13_Sequence_Diversity_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM13",
) -> np.ndarray:
    """
    GM13 – 序列多样性
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    max_val = rows * cols

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            potential_vals = list(set(range(1, max_val + 1)) - set(grid.flatten()))
            best_div = 0.0
            for val in potential_vals:
                # 检查四个方向的连线
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    seq_count = 1
                    # 向前
                    rr, cc = r - dr, c - dc
                    while 0 <= rr < rows and 0 <= cc < cols and grid[rr, cc] == val:
                        seq_count += 1
                        rr -= dr
                        cc -= dc
                    # 向后
                    rr, cc = r + dr, c + dc
                    while 0 <= rr < rows and 0 <= cc < cols and grid[rr, cc] == val:
                        seq_count += 1
                        rr += dr
                        cc += dc
                    best_div = max(best_div, float(seq_count))
            scores[r, c] = best_div

    mask = (grid == -1)
    if np.any(mask):
        mn = float(np.min(scores[mask]))
        mx = float(np.max(scores[mask]))
        if not math.isclose(mn, mx):
            normed = (scores - mn) / (mx - mn)
        else:
            normed = np.zeros_like(scores)
        scores = np.where(mask, normed, 0.0)
    else:
        scores[:] = 0.0

    return scores * config.weight


def EXT_GM14_Risk_Assessment_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM14",
) -> np.ndarray:
    """
    GM14 – 风险评估
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    total_empty = int(np.sum(grid == -1))

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            rem_empty = total_empty - 1
            # 示例风险值：剩余空格比率
            scores[r, c] = float(rem_empty) / (float(total_empty) if total_empty > 0 else 1.0)

    mask = (grid == -1)
    if np.any(mask):
        mn = float(np.min(scores[mask]))
        mx = float(np.max(scores[mask]))
        if not math.isclose(mn, mx):
            normed = (scores - mn) / (mx - mn)
        else:
            normed = np.zeros_like(scores)
        scores = np.where(mask, normed, 0.0)
    else:
        scores[:] = 0.0

    return scores * config.weight


def EXT_GM15_Information_Gain_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM15",
) -> np.ndarray:
    """
    GM15 – 信息增益
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    def entropy_of_array(arr: np.ndarray) -> float:
        vals = arr[arr != -1]
        if vals.size == 0:
            return 0.0
        counts = Counter(vals.tolist())
        total = float(len(vals))
        ent = 0.0
        for cnt in counts.values():
            p = cnt / total
            ent -= p * math.log2(p) if p > 0 else 0.0
        return ent

    base_entropy = entropy_of_array(grid)

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            best_ig = 0.0
            for val in set(range(1, rows * cols + 1)) - set(grid.flatten()):
                grid[r, c] = val
                new_ent = entropy_of_array(grid)
                ig = base_entropy - new_ent
                best_ig = max(best_ig, ig)
            grid[r, c] = -1
            scores[r, c] = best_ig

    mask = (grid == -1)
    if np.any(mask):
        mn = float(np.min(scores[mask]))
        mx = float(np.max(scores[mask]))
        if not math.isclose(mn, mx):
            normed = (scores - mn) / (mx - mn)
        else:
            normed = np.zeros_like(scores)
        scores = np.where(mask, normed, 0.0)
    else:
        scores[:] = 0.0

    return scores * config.weight


def EXT_GM16_Harmonic_Centrality_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM16",
) -> np.ndarray:
    """
    GM16 – 调和中心性
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    # 在函数内部导入 PuzzleTensorOps，避免循环导入
    from new_module import PuzzleTensorOps

    pto = PuzzleTensorOps(grid)
    # 假设 PuzzleTensorOps.get_indices() 返回两个 shape=(rows,cols) 的数组
    rr, cc = pto.get_indices()
    mask = (grid == -1)
    coords = np.stack(np.where(mask), axis=1)
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    for (r, c) in coords:
        # 计算该空格到所有已揭示位置的距离倒数和
        all_nodes = np.stack(np.where(grid != -1), axis=1)
        if all_nodes.size == 0:
            scores[r, c] = 0.0
            continue
        dists = np.abs(all_nodes[:, 0] - r) + np.abs(all_nodes[:, 1] - c)
        inv = np.sum(1.0 / (dists + 1e-6))
        scores[r, c] = inv

    mask2 = (grid == -1)
    if np.any(mask2):
        mn = float(np.min(scores[mask2]))
        mx = float(np.max(scores[mask2]))
        if not math.isclose(mn, mx):
            normed = (scores - mn) / (mx - mn)
        else:
            normed = np.zeros_like(scores)
        scores = np.where(mask2, normed, 0.0)
    else:
        scores[:] = 0.0

    return scores * config.weight


def EXT_GM17_Local_Entropy_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM17",
) -> np.ndarray:
    """
    GM17 – 局部熵最小化
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    # 在函数内部导入 PuzzleTensorOps，避免循环导入
    from new_module import PuzzleTensorOps

    pto = PuzzleTensorOps(grid)
    ent = pto.local_entropy(radius=config.radius)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    mask = (grid == -1)

    if np.any(mask):
        mn = float(np.min(ent[mask]))
        mx = float(np.max(ent[mask]))
        if not math.isclose(mn, mx):
            normed = (ent - mn) / (mx - mn)
        else:
            normed = np.zeros_like(ent)
        scores[mask] = 1.0 - normed[mask]
    else:
        scores[:] = 0.0

    return scores * config.weight


def EXT_GM18_RL_Value_Estimation_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM18",
) -> np.ndarray:
    """
    GM18 – RL 值估计
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    # 示例：混合 GM1、GM7、GM9 的得分
    # 这里只做示例性调用，真实逻辑可更复杂
    from brain1 import EXT_GM1_Proximity_Vec, EXT_GM2_Heterogeneity_Vec
    from brain2 import EXT_GM7_Numeric_Gaps_Vec, EXT_GM9_Center_Control_Vec

    gm1 = EXT_GM1_Proximity_Vec(grid, config, request_id)
    gm7 = EXT_GM7_Numeric_Gaps_Vec(grid, config, request_id)
    gm9 = EXT_GM9_Center_Control_Vec(grid, config, request_id)

    combined = gm1 + gm7 + gm9
    mask = (grid == -1)
    if np.any(mask):
        mn = float(np.min(combined[mask]))
        mx = float(np.max(combined[mask]))
        if not math.isclose(mn, mx):
            normed = (combined - mn) / (mx - mn)
        else:
            normed = np.zeros_like(combined)
        scores[mask] = normed[mask]
    else:
        scores[:] = 0.0

    return scores * config.weight


def EXT_GM19_SkipPattern_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM19",
) -> np.ndarray:
    """
    GM19 – 跳格模式
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            max_score = 0.0
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                seq_indices = []
                for k in range(-2, 3):
                    rr, cc = r + dr * k, c + dc * k
                    if 0 <= rr < rows and 0 <= cc < cols:
                        seq_indices.append(grid[rr, cc])
                    else:
                        seq_indices.append(-1)
                filled = [v for v in seq_indices if v != -1]
                if len(filled) < 2:
                    continue
                diffs = []
                for i in range(len(filled) - 1):
                    diffs.append(abs(filled[i + 1] - filled[i]))
                if any(d > 1 for d in diffs):
                    max_score = 1.0
            scores[r, c] = max_score

    return scores * config.weight


def EXT_GM20_SkipPattern_Confidence_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM20",
) -> np.ndarray:
    """
    GM20 – 跳格置信度
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    base_scores = EXT_GM19_SkipPattern_Vec(grid, config, request_id)
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            if base_scores[r, c] > 0:
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    neighbors = []
                    for k in [-1, 1]:
                        rr, cc = r + dr * k, c + dc * k
                        if 0 <= rr < rows and 0 <= cc < cols:
                            neighbors.append(grid[rr, cc])
                    if len(neighbors) == 2 and neighbors[0] != -1 and neighbors[1] != -1:
                        if (neighbors[0] + neighbors[1]) // 2 == (
                            neighbors[0] + neighbors[1]
                        ) / 2:
                            scores[r, c] = min(1.0, base_scores[r, c] + 0.3)
                            break

    return scores * config.weight


def EXT_GM21_ClusterBalance_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM21",
) -> np.ndarray:
    """
    GM21 – 群集平衡
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    # 在函数内部导入 PuzzleTensorOps，避免循环依赖
    from new_module import PuzzleTensorOps

    rows, cols = grid.shape
    region_rows, region_cols = 3, 3
    r_step = math.ceil(rows / region_rows)
    c_step = math.ceil(cols / region_cols)

    region_counts = np.zeros((region_rows, region_cols), dtype=int)
    revealed_coords = np.stack(np.where(grid != -1), axis=1)
    for (r, c) in revealed_coords:
        rr = min(r // r_step, region_rows - 1)
        cc = min(c // c_step, region_cols - 1)
        region_counts[rr, cc] += 1

    counts_list = region_counts.flatten().tolist()
    mean_cnt = float(np.mean(counts_list))
    var_cnt = float(np.mean([(x - mean_cnt) ** 2 for x in counts_list]))
    std_cnt = math.sqrt(var_cnt)

    mask_revealed = (grid != -1)
    pto = PuzzleTensorOps(grid)
    comp_sizes = pto.connected_component_sizes(mask=mask_revealed)

    scores = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            tmp_counts = region_counts.copy()
            rr = min(r // r_step, region_rows - 1)
            cc = min(c // c_step, region_cols - 1)
            tmp_counts[rr, cc] += 1
            cnts = tmp_counts.flatten().tolist()
            mean2 = float(np.mean(cnts))
            var2 = float(np.mean([(x - mean2) ** 2 for x in cnts]))
            std2 = math.sqrt(var2)
            bal_score = max(0.0, std_cnt - std2)
            scores[r, c] = bal_score

    mask_empty = (grid == -1)
    if np.any(mask_empty):
        mn = float(np.min(scores[mask_empty]))
        mx = float(np.max(scores[mask_empty]))
        if not math.isclose(mn, mx):
            normed = (scores - mn) / (mx - mn)
        else:
            normed = np.zeros_like(scores)
        scores = np.where(mask_empty, normed, 0.0)
    else:
        scores[:] = 0.0

    return scores * config.weight


def EXT_GM22_CoOccurrence_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM22",
) -> np.ndarray:
    """
    GM22 – 共现性
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    revealed_coords = np.stack(np.where(grid != -1), axis=1)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            total_co = 0.0
            for (rr, cc) in revealed_coords:
                dist = abs(r - rr) + abs(c - cc)
                total_co += 1.0 / (dist + 1.0)
            scores[r, c] = total_co

    mask_empty = (grid == -1)
    if np.any(mask_empty):
        mn = float(np.min(scores[mask_empty]))
        mx = float(np.max(scores[mask_empty]))
        if not math.isclose(mn, mx):
            normed = (scores - mn) / (mx - mn)
        else:
            normed = np.zeros_like(scores)
        scores = np.where(mask_empty, normed, 0.0)
    else:
        scores[:] = 0.0

    return scores * config.weight


def EXT_GM23_MotifDetection_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM23",
) -> np.ndarray:
    """
    GM23 – 模式检测
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    motifs = []
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        for r in range(rows):
            for c in range(cols):
                coords = [(r + k * dr, c + k * dc) for k in range(3)]
                if all(0 <= rr < rows and 0 <= cc < cols for rr, cc in coords):
                    motifs.append(coords)

    for motif in motifs:
        vals = [grid[rr, cc] for rr, cc in motif]
        if vals.count(vals[0]) == 2 and -1 in vals:
            target_val = vals[0]
            for idx, (rr, cc) in enumerate(motif):
                if grid[rr, cc] == -1:
                    scores[rr, cc] = max(scores[rr, cc], 1.0)

    mask_empty = (grid == -1)
    if np.any(mask_empty):
        mn = float(np.min(scores[mask_empty]))
        mx = float(np.max(scores[mask_empty]))
        if not math.isclose(mn, mx):
            normed = (scores - mn) / (mx - mn)
        else:
            normed = np.zeros_like(scores)
        scores = np.where(mask_empty, normed, 0.0)
    else:
        scores[:] = 0.0

    return scores * config.weight


def EXT_GM24_TemporalCoherence_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM24",
) -> np.ndarray:
    """
    GM24 – 时序连贯
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    seq = config.historical_sequence or []
    if not seq:
        return scores

    last = seq[-1]
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            if (r, c) == last:
                scores[r, c] = 1.0

    mask_empty = (grid == -1)
    if np.any(mask_empty):
        mn = float(np.min(scores[mask_empty]))
        mx = float(np.max(scores[mask_empty]))
        if not math.isclose(mn, mx):
            normed = (scores - mn) / (mx - mn)
        else:
            normed = np.zeros_like(scores)
        scores = np.where(mask_empty, normed, 0.0)
    else:
        scores[:] = 0.0

    return scores * config.weight


def EXT_GM25_StrategicDepth_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM25",
) -> np.ndarray:
    """
    GM25 – 策略深度
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    # 简化示例：每个空格赋固定分值
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            scores[r, c] = 0.5

    mask_empty = (grid == -1)
    if np.any(mask_empty):
        mn = float(np.min(scores[mask_empty]))
        mx = float(np.max(scores[mask_empty]))
        if not math.isclose(mn, mx):
            normed = (scores - mn) / (mx - mn)
        else:
            normed = np.zeros_like(scores)
        scores = np.where(mask_empty, normed, 0.0)
    else:
        scores[:] = 0.0

    return scores * config.weight


def EXT_GM26_ContextualFlexibility_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM26",
) -> np.ndarray:
    """
    GM26 – 情境灵活性
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    total_empty = int(np.sum(grid == -1))
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            scores[r, c] = float(total_empty) / (rows * cols)

    mask_empty = (grid == -1)
    if np.any(mask_empty):
        mn = float(np.min(scores[mask_empty]))
        mx = float(np.max(scores[mask_empty]))
        if not math.isclose(mn, mx):
            normed = (scores - mn) / (mx - mn)
        else:
            normed = np.zeros_like(scores)
        scores = np.where(mask_empty, normed, 0.0)
    else:
        scores[:] = 0.0

    return scores * config.weight