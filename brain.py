import numpy as np
import math
from collections import Counter
import logging
from typing import List, Tuple, Callable, Optional, Dict, Any
from scipy.stats import entropy

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

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
        if clamp:
            return max(0.0, min(1.0, normalized))
        return normalized

    def manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

class BoardAnalyzerUtils:
    def get_neighborhood_values(
        self,
        grid: np.ndarray,
        r: int,
        c: int,
        radius: int = 1,
        eight_connectivity: bool = True,
        val_func: Callable[[int], Optional[float]] = lambda x: float(x) if x != -1 else None,
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
                    processed_val = val_func(grid[nr, nc])
                    if processed_val is not None:
                        neighbors.append(processed_val)
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
                    continue
                diff = line[j] - line[i]
                if diff == 0 and line[i] != 0:
                    continue
                current_seq_values = [line[i], line[j]]
                gap_count = 0
                for k in range(j + 1, n):
                    if line[k] == -1:
                        gap_count += 1
                        if gap_count > allow_gaps:
                            break
                        continue
                    expected_next = current_seq_values[-1] + diff
                    if math.isclose(line[k], expected_next):
                        current_seq_values.append(line[k])
                        gap_count = 0
                    elif line[k] != -1:
                        break
                if len(current_seq_values) >= min_len:
                    sequences.append(current_seq_values)
        return sequences

    def get_card_max_value_from_gridDimensions(self, grid_shape: Tuple[int, int]) -> int:
        rows, cols = grid_shape
        return rows * cols if rows > 0 and cols > 0 else 0

    def get_legal_values_for_placement(self, grid: np.ndarray) -> set[int]:
        if grid.size == 0:
            return set()
        rows, cols = grid.shape
        all_possible = set(range(1, self.get_card_max_value_from_gridDimensions((rows, cols)) + 1))
        used_values = set(int(v) for v in grid.flatten() if v != -1 and v > 0)
        return all_possible - used_values

def EXT_GM20_Skip_Pattern_Confidence_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    revealed = [{"value": int(grid[r, c]), "r": r, "c": c}
                for r in range(rows) for c in range(cols) if grid[r, c] != -1 and grid[r, c] > 0]

    if not revealed:
        return scores

    max_val = BoardAnalyzerUtils().get_card_max_value_from_gridDimensions((rows, cols))
    base_positions = {k: ((k-1) // cols, (k-1) % cols) for k in range(1, max_val + 1)}

    skip_vectors = {}
    for info in revealed:
        val = info["value"]
        if val in base_positions:
            expected_r, expected_c = base_positions[val]
            skip_vectors[val] = (info["r"] - expected_r, info["c"] - expected_c)

    if not skip_vectors:
        return scores

    counts = Counter(skip_vectors.values())
    min_occ = max(1, int(len(skip_vectors) * 0.05))
    patterns = []
    for skip, count in counts.most_common():
        if count >= min_occ:
            vals = sorted([val for val, v in skip_vectors.items() if v == skip])
            strength = MathUtils().normalize_value(count, min_occ, len(skip_vectors), clamp=True)
            patterns.append({"skip": skip, "values": vals, "strength": strength})
        else:
            break

    if not patterns:
        return scores

    candidates = BoardAnalyzerUtils().get_legal_values_for_placement(grid)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            max_score = 0.0
            for val in candidates:
                if val not in base_positions:
                    continue
                base_r, base_c = base_positions[val]
                for pat in patterns:
                    dr, dc = pat["skip"]
                    if (base_r + dr == r) and (base_c + dc == c):
                        enhance = 0.5
                        temp_seq = sorted(pat["values"] + [val])
                        if len(temp_seq) >= 2:
                            diffs = np.diff(temp_seq)
                            if len(set(diffs)) == 1 and diffs[0] != 0:
                                enhance += 0.4
                            elif len(temp_seq) >= 3 and min(pat["values"]) < val < max(pat["values"]):
                                enhance += 0.1
                        score = pat["strength"] * enhance
                        max_score = max(max_score, score)
            scores[r, c] = MathUtils().normalize_value(max_score, 0, 1.0, clamp=True)

    return scores