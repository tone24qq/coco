# brain.py
import math
import logging
from collections import Counter
from typing import List, Tuple, Callable, Optional, Dict, Any

import numpy as np
from scipy.stats import entropy  # 目前未直接用到，但保留以便後續統計需求

# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------
class MathUtils:
    """Utility functions for common mathematical operations."""

    def sigmoid(self, x: float, k: float = 1.0) -> float:
        """Clamped sigmoid to避免 overflow."""
        try:
            clamped_x = max(-700.0, min(700.0, -k * x))
            return 1 / (1 + math.exp(clamped_x))
        except OverflowError:
            return 0.0 if -k * x > 0 else 1.0

    def normalize_value(
        self, value: float, min_val: float, max_val: float, clamp: bool = True
    ) -> float:
        """Normalize value to [0, 1]."""
        if math.isclose(max_val, min_val, rel_tol=1e-9):
            return 0.5 if math.isclose(value, min_val, rel_tol=1e-9) else (
                0.0 if value < min_val else 1.0
            )
        normalized = (value - min_val) / (max_val - min_val + 1e-10)
        return max(0.0, min(1.0, normalized)) if clamp else normalized

    def manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """Compute Manhattan distance between two (row, col) points."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# ---------------------------------------------------------------------
# Board analysis helpers
# ---------------------------------------------------------------------
class BoardAnalyzerUtils:
    """Utility collection for scratch-card grid analysis."""

    # ---------------------------- neighborhood -----------------------
    def get_neighborhood_values(
        self,
        grid: np.ndarray,
        r: int,
        c: int,
        radius: int = 1,
        eight_connectivity: bool = True,
        val_func: Callable[[int], Optional[float]] = lambda x: float(x) if x != -1 else None,
        include_center: bool = False,
    ) -> List[float]:
        """
        Collect values surrounding grid[r, c] in a square radius.

        Returns a list of `val_func`-processed neighbor values (skips None).
        """
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
                    processed_val = val_func(grid[nr, nc])
                    if processed_val is not None:
                        neighbors.append(processed_val)
        return neighbors

    # ---------------------------- sequence check ---------------------
    def check_sequences(
        self,
        board: np.ndarray,
        original_grid: np.ndarray,
        min_len: int = 3,
        allow_gaps: int = 1,
    ) -> bool:
        """
        Return True if `board` contains at least one arithmetic/geometric
        sequence (len ≥ `min_len`) in any row/col/diag, tolerating
        up to `allow_gaps` masked cells (-1).
        """
        rows, cols = board.shape

        # rows
        for r in range(rows):
            if self.get_arithmetic_or_geometric_sequences(board[r], min_len, allow_gaps):
                return True
        # cols
        for c in range(cols):
            if self.get_arithmetic_or_geometric_sequences(board[:, c], min_len, allow_gaps):
                return True
        # diagonals (both directions)
        for offset in range(-(rows - min_len), cols - min_len + 1):
            if self.get_arithmetic_or_geometric_sequences(
                np.diagonal(board, offset), min_len, allow_gaps
            ):
                return True
            if self.get_arithmetic_or_geometric_sequences(
                np.diagonal(np.fliplr(board), offset), min_len, allow_gaps
            ):
                return True
        return False

    # ---------------------------- sequence finder --------------------
    def get_arithmetic_or_geometric_sequences(
        self,
        line: np.ndarray,
        min_len: int = 3,
        allow_gaps: int = 1,
    ) -> List[List[int]]:
        """
        Detect arithmetic/geometric subsequences in a 1-D array `line`.

        Returns list of subsequences (as value lists).
        """
        sequences: List[List[int]] = []
        n = len(line)

        for i in range(n):
            if line[i] == -1:
                continue
            for j in range(i + 1, n):
                if line[j] == -1:
                    # handle gaps before second value
                    temp_gap = 0
                    for k in range(j, n):
                        if line[k] == -1:
                            temp_gap += 1
                        else:
                            if temp_gap <= allow_gaps:
                                diff = line[k] - line[i]
                                if diff == 0:  # 排除常數序列
                                    break
                                seq_vals = [line[i], line[k]]
                                gap_cnt = temp_gap
                                for l in range(k + 1, n):
                                    if line[l] == -1:
                                        gap_cnt += 1
                                        if gap_cnt > allow_gaps:
                                            break
                                        continue
                                    expected = seq_vals[-1] + diff
                                    if math.isclose(line[l], expected, rel_tol=1e-9):
                                        seq_vals.append(line[l])
                                        gap_cnt = 0
                                    else:
                                        break
                                if len(seq_vals) >= min_len:
                                    sequences.append(seq_vals)
                            break
                else:
                    diff = line[j] - line[i]
                    if diff == 0:
                        continue
                    seq_vals = [line[i], line[j]]
                    gap_cnt = 0
                    for k in range(j + 1, n):
                        if line[k] == -1:
                            gap_cnt += 1
                            if gap_cnt > allow_gaps:
                                break
                            continue
                        expected = seq_vals[-1] + diff
                        if math.isclose(line[k], expected, rel_tol=1e-9):
                            seq_vals.append(line[k])
                            gap_cnt = 0
                        else:
                            break
                    if len(seq_vals) >= min_len:
                        sequences.append(seq_vals)

        return sequences

    # ---------------------------- misc helpers -----------------------
    def get_card_max_value_from_gridDimensions(self, grid_shape: Tuple[int, int]) -> int:
        """Return rows×cols (max possible face value)."""
        rows, cols = grid_shape
        return rows * cols if rows and cols else 0

    def get_legal_values_for_placement(self, grid: np.ndarray) -> set[int]:
        """Return unused numbers > 0 that can still appear on the board."""
        rows, cols = grid.shape
        all_vals = set(range(1, self.get_card_max_value_from_gridDimensions((rows, cols)) + 1))
        used = set(int(v) for v in grid.flatten() if v != -1 and v > 0)
        return all_vals - used

# ---------------------------------------------------------------------
# EXT_GM20 – skip-pattern confidence heuristic
# ---------------------------------------------------------------------
def EXT_GM20_Skip_Pattern_Confidence_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    """
    Produce a confidence map based on dominant skip vectors of revealed numbers.
    """
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    # collect revealed cells
    revealed: List[Dict[str, int]] = [
        {"value": int(grid[r, c]), "r": r, "c": c}
        for r in range(rows)
        for c in range(cols)
        if grid[r, c] != -1 and grid[r, c] > 0
    ]
    if not revealed:
        return scores

    utils = BoardAnalyzerUtils()
    math_utils = MathUtils()

    max_val = utils.get_card_max_value_from_gridDimensions((rows, cols))
    base_pos = {k: ((k - 1) // cols, (k - 1) % cols) for k in range(1, max_val + 1)}

    # compute skip vectors
    skip_vecs = {
        info["value"]: (
            info["r"] - base_pos[info["value"]][0],
            info["c"] - base_pos[info["value"]][1],
        )
        for info in revealed
        if info["value"] in base_pos
    }
    if not skip_vecs:
        return scores

    counts = Counter(skip_vecs.values())
    min_occ = max(1, int(len(skip_vecs) * 0.05))

    dominant_patterns: List[Dict[str, Any]] = []
    for vec, cnt in counts.most_common():
        if cnt < min_occ:
            break
        pattern_vals = sorted(v for v, sv in skip_vecs.items() if sv == vec)
        strength = math_utils.normalize_value(cnt, min_occ, len(skip_vecs)) * 1.1
        dominant_patterns.append({"skip": vec, "values": pattern_vals, "strength": strength})

    if not dominant_patterns:
        return scores

    legal_nums = utils.get_legal_values_for_placement(grid)

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            best_conf = 0.0
            for num in legal_nums:
                if num not in base_pos:
                    continue
                base_r, base_c = base_pos[num]
                for pat in dominant_patterns:
                    dr, dc = pat["skip"]
                    if base_r + dr == r and base_c + dc == c:
                        enh = 0.5
                        if len(pat["values"]) >= 1:
                            seq = sorted(pat["values"] + [num])
                            if len(seq) >= 2 and len(set(np.diff(seq))) == 1:
                                enh += 0.5
                            elif len(seq) >= 3 and min(seq) < num < max(seq):
                                enh += 0.15
                        best_conf = max(best_conf, pat["strength"] * enh)
            scores[r, c] = math_utils.normalize_value(best_conf, 0, 1.0)

    return scores

# ---------------------------------------------------------------------
# (optional) quick self-test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    test_grid = np.array([
        [13,  2, -1, 18,  9],
        [ 3, 15,  6, -1,  8],
        [ 4,  5, 10, 14,  7],
        [20,  1, 11, -1, 16],
    ])
    utils = BoardAnalyzerUtils()
    print("Legal:", utils.get_legal_values_for_placement(test_grid))
    print("Skip-pattern confidence:\n", EXT_GM20_Skip_Pattern_Confidence_Vec(test_grid))