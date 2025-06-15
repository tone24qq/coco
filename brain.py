# app/brain.py

import numpy as np
import math
import logging
from collections import Counter
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)

class MathUtils:
    """Provides utility functions for mathematical operations."""
    def sigmoid(self, x: float, k: float = 1.0) -> float:
        try:
            clamped_x = max(-700.0, min(700.0, -k * x))
            return 1 / (1 + math.exp(clamped_x))
        except OverflowError:
            return 0.0 if -k * x > 0 else 1.0

    def normalize_value(self, value: float, min_val: float, max_val: float, clamp: bool = True) -> float:
        if math.isclose(max_val, min_val):
            if math.isclose(value, min_val):
                return 0.5
            return 0.0 if value < min_val else 1.0
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized)) if clamp else normalized

    def manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

class BoardAnalyzerUtils:
    """Provides utility functions for board analysis."""
    def get_neighborhood_values(
        self,
        grid: np.ndarray,
        r: int,
        c: int,
        radius: int = 1,
        eight_connectivity: bool = True,
        val_func: callable = lambda x: float(x) if x != -1 else None,
        include_center: bool = False
    ) -> List[float]:
        neighbors: List[float] = []
        rows, cols = grid.shape
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if not include_center and dr == 0 and dc == 0:
                    continue
                if not eight_connectivity and abs(dr) + abs(dc) != 1:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    val = val_func(grid[nr, nc])
                    if val is not None:
                        neighbors.append(val)
        return neighbors

    def get_arithmetic_or_geometric_sequences(
        self,
        line: np.ndarray,
        min_len: int = 3,
        allow_gaps: int = 1
    ) -> List[List[int]]:
        sequences: List[List[int]] = []
        n = len(line)
        for i in range(n):
            if line[i] == -1:
                continue
            for j in range(i + 1, n):
                if line[j] == -1:
                    # gap-start sequence
                    gap_count = 0
                    for k in range(j, n):
                        if line[k] == -1:
                            gap_count += 1
                        else:
                            if gap_count <= allow_gaps:
                                diff = line[k] - line[i]
                                if diff == 0 and line[i] != 0:
                                    break
                                seq_vals = [line[i], line[k]]
                                curr_gap = gap_count
                                for l in range(k + 1, n):
                                    if line[l] == -1:
                                        curr_gap += 1
                                        if curr_gap > allow_gaps:
                                            break
                                        continue
                                    expected = seq_vals[-1] + diff
                                    if math.isclose(line[l], expected):
                                        seq_vals.append(line[l])
                                        curr_gap = 0
                                    else:
                                        break
                                if len(seq_vals) >= min_len:
                                    sequences.append(seq_vals)
                            break
                else:
                    # direct pair-start sequence
                    diff = line[j] - line[i]
                    if diff == 0 and line[i] != 0:
                        continue
                    seq_vals = [line[i], line[j]]
                    for k in range(j + 1, n):
                        if line[k] == -1:
                            continue
                        expected = seq_vals[-1] + diff
                        if math.isclose(line[k], expected):
                            seq_vals.append(line[k])
                        else:
                            break
                    if len(seq_vals) >= min_len:
                        sequences.append(seq_vals)
        return sequences

    def get_card_max_value_from_gridDimensions(self, grid_shape: Tuple[int, int]) -> int:
        rows, cols = grid_shape
        return rows * cols if rows > 0 and cols > 0 else 0

    def get_legal_values_for_placement(self, grid: np.ndarray) -> set[int]:
        rows, cols = grid.shape
        max_val = self.get_card_max_value_from_gridDimensions((rows, cols))
        all_vals = set(range(1, max_val + 1))
        used = {int(v) for v in grid.flatten() if v != -1 and v > 0}
        return all_vals - used

    def check_sequences(
        self,
        board: np.ndarray,
        min_len: int = 3,
        allow_gaps: int = 1
    ) -> bool:
        """
        Check for any arithmetic or geometric sequence of length >= min_len
        in any row or column.
        """
        rows, cols = board.shape
        # rows
        for r in range(rows):
            if self.get_arithmetic_or_geometric_sequences(board[r], min_len, allow_gaps):
                return True
        # columns
        for c in range(cols):
            if self.get_arithmetic_or_geometric_sequences(board[:, c], min_len, allow_gaps):
                return True
        return False  # no sequence found

def EXT_GM20_Skip_Pattern_Confidence_Vec(
    grid: np.ndarray,
    request_id: Optional[str] = "N/A"
) -> np.ndarray:
    """
    Compute confidence scores based on skip patterns.
    """
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    revealed = [
        {"value": int(grid[r, c]), "r": r, "c": c}
        for r in range(rows) for c in range(cols)
        if grid[r, c] != -1 and grid[r, c] > 0
    ]
    if not revealed:
        return scores

    base_positions = {
        val: ((val - 1) // cols, (val - 1) % cols)
        for val in range(1, rows * cols + 1)
    }
    skip_vectors: Dict[int, Tuple[int, int]] = {}
    for info in revealed:
        val, r, c = info["value"], info["r"], info["c"]
        exp_r, exp_c = base_positions[val]
        skip_vectors[val] = (r - exp_r, c - exp_c)

    if not skip_vectors:
        return scores

    counts = Counter(skip_vectors.values())
    min_occ = max(1, int(len(skip_vectors) * 0.05))
    patterns = []
    for vec, cnt in counts.most_common():
        if cnt >= min_occ:
            vals = sorted([v for v, sv in skip_vectors.items() if sv == vec])
            strength = MathUtils().normalize_value(float(cnt), float(min_occ), float(len(skip_vectors)))
            patterns.append({"skip": vec, "values": vals, "strength": strength})
        else:
            break
    if not patterns:
        return scores

    legal = BoardAnalyzerUtils().get_legal_values_for_placement(grid)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            best_score = 0.0
            for num in legal:
                if num not in base_positions:
                    continue
                br, bc = base_positions[num]
                for pat in patterns:
                    dr, dc = pat["skip"]
                    if (br + dr, bc + dc) == (r, c):
                        factor = 0.5
                        vals = pat["values"]
                        if vals:
                            seq = sorted(vals + [num])
                            diffs = np.diff(seq)
                            if len(set(diffs)) == 1 and diffs[0] != 0:
                                factor += 0.4
                            elif len(seq) >= 3 and min(vals) < num < max(vals):
                                factor += 0.1
                        score = pat["strength"] * factor
                        best_score = max(best_score, score)
            scores[r, c] = MathUtils().normalize_value(best_score, 0, 1.0)
    return scores