# brain1.py

import numpy as np
import math
import logging
from collections import Counter
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
    def sigmoid(x: float, k: float = 1.0) -> float:
        try:
            clamped = max(-700.0, min(700.0, -k * x))
            return 1.0 / (1.0 + math.exp(clamped))
        except OverflowError:
            return 0.0 if (-k * x) > 0 else 1.0

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

    @staticmethod
    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @staticmethod
    def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def get_entropy(values: List[Any]) -> float:
        if not values:
            return 0.0
        counts = Counter(values)
        total = len(values)
        ent = 0.0
        for cnt in counts.values():
            p = cnt / total
            if p > 0:
                ent -= p * math.log2(p)
        return ent


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

    @staticmethod
    def get_value_gradient_at_cell(
        grid: np.ndarray,
        r: int,
        c: int,
        val_func=None,
    ) -> Tuple[float, float]:
        if val_func is None:
            val_func = lambda x: float(x) if x != -1 else 0.0

        rows, cols = grid.shape

        def safe_val(rr: int, cc: int) -> float:
            if 0 <= rr < rows and 0 <= cc < cols:
                return val_func(grid[rr, cc])
            return 0.0

        gx = (
            safe_val(r - 1, c + 1)
            + 2 * safe_val(r, c + 1)
            + safe_val(r + 1, c + 1)
            - safe_val(r - 1, c - 1)
            - 2 * safe_val(r, c - 1)
            - safe_val(r + 1, c - 1)
        )
        gy = (
            safe_val(r + 1, c - 1)
            + 2 * safe_val(r + 1, c)
            + safe_val(r + 1, c + 1)
            - safe_val(r - 1, c - 1)
            - 2 * safe_val(r - 1, c)
            - safe_val(r - 1, c + 1)
        )
        return gx, gy

    @staticmethod
    def find_sequences_in_line(
        line: List[int],
        min_len: int = 3,
        check_arithmetic: bool = True,
        check_geometric: bool = False,
        allow_gaps: int = 0,
    ) -> List[List[int]]:
        sequences: List[List[int]] = []
        n = len(line)
        if n == 0:
            return sequences

        processed = []
        for x in line:
            if x == -1:
                processed.append(None)
            else:
                processed.append(float(x))

        for i in range(n):
            if processed[i] is None:
                continue
            start_val = processed[i]
            if check_arithmetic:
                for j in range(i + 1, n):
                    gaps = 0
                    for k in range(i + 1, j):
                        if processed[k] is None:
                            gaps += 1
                    if gaps > allow_gaps:
                        continue
                    if processed[j] is None:
                        continue
                    diff = processed[j] - start_val
                    if math.isclose(diff, 0) and not math.isclose(start_val, 0):
                        continue
                    seq_vals = [int(start_val), int(processed[j])]
                    last_val = processed[j]
                    last_idx = j
                    gap_count = 0
                    for k in range(j + 1, n):
                        if processed[k] is None:
                            gap_count += 1
                            if gap_count > allow_gaps:
                                break
                            continue
                        steps = k - last_idx
                        expected = last_val + diff * (steps / (gap_count + 1))
                        if math.isclose(processed[k], expected):
                            seq_vals.append(int(processed[k]))
                            last_val = processed[k]
                            last_idx = k
                            gap_count = 0
                        else:
                            break
                    if len(seq_vals) >= min_len:
                        sequences.append(seq_vals)
            if check_geometric and not math.isclose(start_val, 0.0):
                for j in range(i + 1, n):
                    gaps = 0
                    for k in range(i + 1, j):
                        if processed[k] is None:
                            gaps += 1
                    if gaps > allow_gaps:
                        continue
                    if processed[j] is None or math.isclose(processed[j], 0.0):
                        continue
                    ratio = processed[j] / start_val
                    if math.isclose(ratio, 1.0) and not math.isclose(start_val, processed[j]):
                        continue
                    seq_vals = [int(start_val), int(processed[j])]
                    last_val = processed[j]
                    last_idx = j
                    gap_count = 0
                    for k in range(j + 1, n):
                        if processed[k] is None:
                            gap_count += 1
                            if gap_count > allow_gaps:
                                break
                            continue
                        steps = k - last_idx
                        expected = last_val * (ratio ** (steps / (gap_count + 1)))
                        if math.isclose(processed[k], expected, rel_tol=1e-6):
                            seq_vals.append(int(processed[k]))
                            last_val = processed[k]
                            last_idx = k
                            gap_count = 0
                        else:
                            break
                    if len(seq_vals) >= min_len:
                        sequences.append(seq_vals)
        return sequences


def EXT_GM1_Proximity_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM1",
) -> np.ndarray:
    """
    GM1 – 加权近邻
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    revealed_mask = grid != -1
    if not np.any(revealed_mask):
        return scores

    revealed_coords = np.stack(np.where(revealed_mask), axis=1)
    idxs = np.indices((rows, cols))
    rr = idxs[0][..., None]
    cc = idxs[1][..., None]

    r_rep = revealed_coords[:, 0]
    c_rep = revealed_coords[:, 1]

    dist = np.abs(rr - r_rep) + np.abs(cc - c_rep)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight_matrix = 1.0 / (1.0 + dist.astype(float))
    base_scores = np.nansum(weight_matrix, axis=2)
    scores[grid == -1] = base_scores[grid == -1]

    if np.any(scores[grid == -1]):
        min_val = float(np.nanmin(scores[grid == -1]))
        max_val = float(np.nanmax(scores[grid == -1]))
    else:
        min_val, max_val = 0.0, 1.0

    norm = np.vectorize(lambda v: MathUtils.normalize_value(v, min_val, max_val))
    normalized = norm(scores)
    return normalized * config.weight


def EXT_GM2_Heterogeneity_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM2",
) -> np.ndarray:
    """
    GM2 – 局部异质性
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    padded = np.pad(grid, pad_width=1, mode="constant", constant_values=-1)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            sub = padded[r : r + 3, c : c + 3]
            values = sub[sub != -1].astype(float)
            if values.size == 0:
                scores[r, c] = 0.0
            else:
                scores[r, c] = float(np.std(values))

    if np.any(scores > 0):
        mn = float(np.min(scores[grid == -1]))
        mx = float(np.max(scores[grid == -1]))
        np.seterr(divide="ignore", invalid="ignore")
        normed = (scores - mn) / (mx - mn) if not math.isclose(mx, mn) else np.zeros_like(scores)
        scores = np.nan_to_num(normed)
    return scores * config.weight


def EXT_GM3_PotentialField_Vec(
    grid: np.ndarray,
    config: BaseModuleConfig,
    request_id: Optional[str] = "N/A_GM3",
) -> np.ndarray:
    """
    GM3 – 势场
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    revealed_mask = grid != -1
    if not np.any(revealed_mask):
        return scores
    revealed_coords = np.stack(np.where(revealed_mask), axis=1)
    revealed_vals = grid[revealed_mask].astype(float)

    max_val = rows * cols
    existing = set(grid[revealed_mask].tolist())
    hypo_values = [v for v in range(1, max_val + 1) if v not in existing]
    if not hypo_values:
        return scores

    idxs = np.indices((rows, cols))
    r_idx = idxs[0][..., None]
    c_idx = idxs[1][..., None]

    r_rep = revealed_coords[:, 0][None, None, :]
    c_rep = revealed_coords[:, 1][None, None, :]
    val_rep = revealed_vals[None, None, :]

    dist2 = (r_idx - r_rep) ** 2 + (c_idx - c_rep) ** 2
    dist2 = np.where(dist2 == 0, 1e-6, dist2)

    influences = []
    for h in hypo_values:
        diff = val_rep - float(h)
        influence = np.nansum(diff / dist2, axis=2)
        influences.append(np.abs(influence))

    influences = np.stack(influences, axis=2)
    max_inf = np.nanmax(influences, axis=2)
    scores[grid == -1] = max_inf[grid == -1]

    if np.any(scores > 0):
        mn = float(np.min(scores[grid == -1]))
        mx = float(np.max(scores[grid == -1]))
        if not math.isclose(mx, mn):
            scores = (scores - mn) / (mx - mn)
        else:
            scores = np.zeros_like(scores)
    return scores * config.weight