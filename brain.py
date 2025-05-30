import numpy as np
import math
from collections import Counter, deque
import logging
from typing import Callable, Any, Hashable


# --- Logging Configuration
logger = logging.getLogger(__name__)

# === Helper Utilities ===

class MathUtils:
    """提供通用數學工具,所有模組統一計算風格"""

    def sigmoid(self, x: float, k: float = 1.0) -> float:
        """安全型 sigmoid,避免 overflow"""
        try:
            clamped_x = max(-700.0, min(700.0, -k * x))
            return 1 / (1 + math.exp(clamped_x))
        except OverflowError:
            return 0.0 if -k * x > 0 else 1.0

    def normalize_value(self, value: float, min_val: float, max_val: float, clamp: bool = True) -> float:
        """
        Normalizes a value to the [0, 1] range.
        Handles cases where min_val equals max_val to prevent division by zero.
        Addresses Requirement 2.c (reasonable score distribution).
        強化:處理 min_val 和 max_val相等時,根據 value 與其的關係返回0.0,0.5,或1.0,更
        精確地處理邊界情況。
        """
        if math.isclose(max_val, min_val):
            if math.isclose(value, min_val):
                return 0.5
            elif value < min_val:
                return 0.0
            else:  # value > max_val (which is min_val)
                return 1.0

        normalized = (value - min_val) / (max_val - min_val)
        if clamp:
            return max(0.0, min(1.0, normalized))
        return normalized

    def manhattan_distance(self, p1: tuple[int, int], p2: tuple[int, int]) -> int:
        """Calculates Manhattan distance between two points (r, c)."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def euclidean_distance(self, p1: tuple[int, int], p2: tuple[int, int]) -> float:
        """Calculates Euclidean distance between two points (r, c)."""
        return math.sqrt(((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

    def get_entropy(self, values: list[Hashable]) -> float:
        """Calculates Shannon entropy for a list of values."""
        if not values:
            return 0.0
        counts = Counter(values)
        total_count = len(values)
        entropy = 0.0
        for count in counts.values():
            probability = count / total_count
            entropy -= probability * math.log2(probability)
        return entropy

class BoardAnalyzerUtils:
    """
    Provides common board analysis utility functions.
    Used by modules to inspect grid neighborhoods, gradients, etc.
    """

    def get_neighborhood_values(
        self,
        grid: np.ndarray,
        r: int,
        c: int,
        radius: int = 1,
        eight_connectivity: bool = True,
        val_func: Callable[[int], float | None] = lambda x_val: float(x_val) if x_val != -1 else None,
        include_center: bool = False
    ) -> list[float]:
        """
        Retrieves values from the neighborhood of a cell.
        Supports configurable radius, connectivity, and value processing.
        """
        neighbors: list[float] = []
        rows, cols = grid.shape
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if not include_center and dr == 0 and dc == 0:
                    continue
                if not eight_connectivity:
                    if radius == 1 and abs(dr) + abs(dc) != 1:
                        continue
                    elif radius > 1 and abs(dr) + abs(dc) > radius:
                        continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    processed_val = val_func(grid[nr, nc])
                    if processed_val is not None:
                        neighbors.append(processed_val)
        return neighbors

    def get_value_gradient_at_cell(
        self,
        grid: np.ndarray,
        r: int,
        c: int,
        val_func: Callable[[int], float] = lambda x_val: float(x_val) if x_val != -1 else 0.0
    ) -> tuple[float, float]:
        """
        Calculates an approximate gradient (Sobel-like) at a cell.
        Useful for modules analyzing value changes.
        """
        rows, cols = grid.shape

        def safe_val(r_in: int, c_in: int) -> float:
            if 0 <= r_in < rows and 0 <= c_in < cols:
                return val_func(grid[r_in, c_in])
            return 0.0

        gx = (safe_val(r - 1, c + 1) + 2 * safe_val(r, c + 1) + safe_val(r + 1, c + 1)) - \
             (safe_val(r - 1, c - 1) + 2 * safe_val(r, c - 1) + safe_val(r + 1, c - 1))
        gy = (safe_val(r + 1, c - 1) + 2 * safe_val(r + 1, c) + safe_val(r + 1, c + 1)) - \
             (safe_val(r - 1, c - 1) + 2 * safe_val(r - 1, c) + safe_val(r - 1, c + 1))
        return gx, gy

    def find_sequences_in_line(
        self,
        line: list[int],
        min_len: int = 3,
        check_arithmetic: bool = True,
        check_geometric: bool = False,
        allow_gaps: int = 0
    ) -> list[list[int]]:
        """
        Finds arithmetic or geometric sequences in a 1D list of numbers,
        supporting gaps and returning sequence elements.
        強化:提升算術序列檢測的彈性,能識別更多複雜的算術序列模式(負公差,跨零點,常
        數序列的明確處理)。
        同時返回找到的序列、類型和公差/比率。
        """
        sequences: list[list[int]] = []
        n = len(line)
        if n < min_len:
            return sequences

        for i in range(n):
            if line[i] == -1:
                continue

            # Arithmetic sequence check
            if check_arithmetic:
                for j in range(i + 1, n):
                    if line[j] == -1:
                        if allow_gaps > 0:
                            temp_gap_count = 0
                            for k_gap_check in range(j, n):
                                if line[k_gap_check] == -1:
                                    temp_gap_count += 1
                                else:
                                    if temp_gap_count <= allow_gaps:
                                        diff = line[k_gap_check] - line[i]
                                        if diff == 0 and line[i] != 0: # Not a strict arithmetic sequence
                                            break
                                        current_seq_values = [line[i], line[k_gap_check]]
                                        # current_seq_indices = [i, k_gap_check] # Indices not used in current return
                                        potential_gap_count_inner = temp_gap_count
                                        for l_extend in range(k_gap_check + 1, n):
                                            if line[l_extend] == -1:
                                                potential_gap_count_inner += 1
                                                if potential_gap_count_inner > allow_gaps:
                                                    break
                                                continue
                                            expected_next = current_seq_values[-1] + diff
                                            if math.isclose(line[l_extend], expected_next):
                                                current_seq_values.append(line[l_extend])
                                                # current_seq_indices.append(l_extend)
                                                potential_gap_count_inner = 0
                                            elif line[l_extend] != -1: # Sequence broken
                                                break
                                        if len(current_seq_values) >= min_len:
                                            sequences.append(current_seq_values)
                                    break # Done trying to establish diff from j after gap
                            break # Done with outer j loop for this i if initial gap finding started
                        continue # Move to next j if allow_gaps is 0 or no number found after gap

                    diff = line[j] - line[i]
                    if diff == 0 and line[i] != 0: # Exclude constant non-zero sequences as arithmetic by default
                        continue

                    current_seq_values = [line[i], line[j]]
                    # current_seq_indices = [i, j]
                    potential_gap_count = 0
                    for k in range(j + 1, n):
                        if line[k] == -1:
                            potential_gap_count += 1
                            if potential_gap_count > allow_gaps:
                                break
                            continue
                        expected_next = current_seq_values[-1] + diff
                        if math.isclose(line[k], expected_next):
                            current_seq_values.append(line[k])
                            # current_seq_indices.append(k)
                            potential_gap_count = 0
                        elif line[k] != -1: # Sequence broken by a different number
                            break
                    if len(current_seq_values) >= min_len:
                        sequences.append(current_seq_values)

            # Geometric sequence check
            if check_geometric and line[i] != 0:
                current_seq_values = [line[i]]
                # current_seq_indices = [i]
                potential_gap_count = 0
                ratio = None
                for j in range(i + 1, n):
                    if line[j] == -1:
                        potential_gap_count += 1
                        if potential_gap_count > allow_gaps:
                            break
                        continue

                    if line[j] == 0: # Geometric sequences with zero are tricky
                        break

                    if ratio is None:
                        if line[i] == 0 and line[j] != 0: # 0, non-zero cannot be start of geom seq
                            break
                        if line[i] != 0:
                             # If ratio isn't integer-like and not a trivial division
                            if line[j] % line[i] != 0 and line[i] % line[j] != 0 and \
                               not math.isclose(line[j] / line[i], round(line[j] / line[i])):
                                break
                            ratio = line[j] / line[i]
                        else: # line[i] == 0 and line[j] == 0
                            ratio = 1.0 # Treat as constant 0s sequence

                        if math.isclose(ratio, 1.0) and line[i] != line[j]: # Avoid constant sequences if not identical
                            continue
                    
                    expected_next_float = float(current_seq_values[-1]) * (ratio if ratio is not None else 0.0) # Ensure ratio is not None
                    if math.isclose(float(line[j]), expected_next_float):
                        # current_seq_indices.append(j)
                        current_seq_values.append(line[j])
                        potential_gap_count = 0
                    elif line[j] != -1:
                        break
                if len(current_seq_values) >= min_len:
                    sequences.append(current_seq_values)
        return sequences

    def get_card_max_value_from_grid_dimensions(self, grid_shape: tuple[int, int]) -> int:
        """Calculates the maximum possible number on the card based on its dimensions."""
        rows, cols = grid_shape
        if rows == 0 or cols == 0:
            return 0
        return rows * cols

    def get_all_possible_numbers_for_grid(self, grid_shape: tuple[int, int]) -> set[int]:
        """
        Returns a set of all numbers that could theoretically appear on a grid of given dimensions.
        """
        max_val = self.get_card_max_value_from_grid_dimensions(grid_shape)
        if max_val == 0:
            return set()
        return set(range(1, max_val + 1))

    def get_legal_values_for_placement(self, grid: np.ndarray) -> set[int]:
        """
        Determines the set of numbers that can be legally placed onto an empty cell in the grid.
        This adheres to the rule: numbers are 1 to R*C and no positive number can be repeated.
        (Requirement 1.c)
        """
        if grid.size == 0:
            return set()
        rows, cols = grid.shape
        all_possible_on_this_grid = self.get_all_possible_numbers_for_grid((rows, cols))
        used_positive_values_on_board = set(int(v) for v in grid.flatten() if v != -1 and v > 0)
        legal_placements = all_possible_on_this_grid - used_positive_values_on_board
        return legal_placements

# === Brain Core Dispatch Area ===

REGISTERED_MODULES_BRAIN: dict[str, Callable[[np.ndarray, str | None], np.ndarray]] = {}

def get_module_score(module_name: str, grid: np.ndarray, **kwargs: Any) -> np.ndarray:
    """
    Retrieves and executes a specific scoring module from the registry.
    Args:
        module_name: The registered name of the module to execute.
        grid: The input numpy array representing the game board.
        kwargs: Additional keyword arguments for the module.
    Returns:
        A numpy array containing the scores for each cell, as computed by the module.
        Returns a zero array of the same shape if the module is not found or an error occurs.
    """
    effective_request_id = kwargs.get("request_id", "N/A_brain_dispatch")
    if module_name not in REGISTERED_MODULES_BRAIN:
        logger.error(
            f"Module {module_name} not found in REGISTERED_MODULES_BRAIN.",
            extra={'request_id': effective_request_id}
        )
        rows, cols = grid.shape
        return np.zeros((rows, cols), dtype=float)

    module_func = REGISTERED_MODULES_BRAIN[module_name]
    logger.info(
        f"Executing module: {module_name}",
        extra={'request_id': effective_request_id}
    )
    try:
        score_grid = module_func(grid, request_id=effective_request_id, **kwargs) # Pass full kwargs
        return score_grid
    except Exception as e:
        logger.error(
            f"Error executing module {module_name}: {e}",
            exc_info=True,
            extra={'request_id': effective_request_id}
        )
        rows, cols = grid.shape
        return np.zeros((rows, cols), dtype=float)

# --- Scoring Module Implementations ---

# 1. EXT_A2_Weighted_Proximity_Vec (加權鄰近性)
def EXT_A2_Weighted_Proximity_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (A2-加權鄰近性)
    核心規則:評估空格周圍已填數字的接近程度及其值的影響。
    目的:偏好靠近高價值數字或數字密集區域的空格。
    啟發式類型:空間鄰近性
    輸出詮釋:分數越高表示鄰近效應越強(受周圍數字的值與密度影響)
    強化:增加對負值(-1)的處理,使其不計入鄰近數字,並微調距離衰減因子和價值權重。
    """
    effective_request_id = request_id or "N/A_brain_A2"
    logger.debug("Executing EXT_A2_Weighted_Proximity_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    radius = 2
    value_weight_factor = 0.15
    distance_decay_factor = 1.8

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue
            proximity_score = 0.0
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0:  # Skip center cell
                        continue
                    nr, nc = r_idx + dr, c_idx + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                        dist = MathUtils().manhattan_distance((r_idx, c_idx), (nr, nc))
                        if dist == 0: # Should ideally not happen if not center
                            dist = 1 # Safeguard
                        score_contribution = (grid[nr, nc] * value_weight_factor) / (dist**distance_decay_factor)
                        proximity_score += score_contribution
            
            max_val_on_grid = BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols))
            if max_val_on_grid == 0:
                max_val_on_grid = 1.0
            
            num_neighbors_in_radius = (2 * radius + 1)**2 - 1
            heuristic_max_score = num_neighbors_in_radius * max_val_on_grid * value_weight_factor / (1**distance_decay_factor)

            if heuristic_max_score > 0:
                scores[r_idx, c_idx] = MathUtils().normalize_value(proximity_score, 0, heuristic_max_score, clamp=True)
            else:
                scores[r_idx, c_idx] = 0.0
    return scores

# 2. EXT_M3_Local_Heterogeneity_Vec(局部異質性)
def EXT_M3_Local_Heterogeneity_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (M3 - 局部異質性)
    核心規則:評估空格周圍數字的多樣性。
    目的:偏好周圍數字分佈更隨機、更少重複的空格。
    啟發式類型:分佈統計(基於熵)
    輸出詮釋:分數越高表示周圍環境的數字異質性越高(熵越大)
    強化:精確計算理論最大熵以進行歸一化,確保歸一化結果的穩定性。
    """
    effective_request_id = request_id or "N/A_brain_M3"
    logger.debug("Executing EXT_M3_Local_Heterogeneity_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    radius = 1
    min_neighbors_for_robust_score = 2
    all_possible_values_in_game = BoardAnalyzerUtils().get_all_possible_numbers_for_grid(grid.shape)

    if not all_possible_values_in_game:
        return scores

    if len(all_possible_values_in_game) > 1:
        max_theoretical_entropy = math.log2(len(all_possible_values_in_game))
    elif len(all_possible_values_in_game) == 1:
        max_theoretical_entropy = math.log2(2) # Avoid log2(1)=0
    else:
        max_theoretical_entropy = 1.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue
            
            neighbor_values = BoardAnalyzerUtils().get_neighborhood_values(
                grid, r_idx, c_idx, radius=radius, eight_connectivity=True,
                val_func=lambda x_val: int(x_val) if x_val != -1 else None,
                include_center=False
            )

            if len(neighbor_values) < min_neighbors_for_robust_score:
                scores[r_idx, c_idx] = 0.0
                continue

            current_entropy = MathUtils().get_entropy(neighbor_values)
            if max_theoretical_entropy > 0:
                normalized_score = current_entropy / max_theoretical_entropy
                scores[r_idx, c_idx] = MathUtils().normalize_value(normalized_score, 0, 1, clamp=True)
            else:
                scores[r_idx, c_idx] = 0.0
    return scores

# 3. EXT_D3_Potential_Field_Vec(位勢場分析)
def EXT_D3_Potential_Field_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (D3-位勢場分析)
    核心規則:將盤面上的數字視為「電荷」,空格則根據其位置的「綜合位勢」來評分。
    目的:偏好位於受高價值數字「吸引」或低價值數字「排斥」(如果設計如此)區域的空格。
    此處簡化為僅正向吸引。
    啟發式類型:物理類比(類似靜電場或重力場)
    輸出詮釋:分數越高表示該空格受到周圍數字的正向「位勢影響」越大
    強化:優化位勢衰減計算,確保在處理大網格和極端值時的穩健性。
    """
    effective_request_id = request_id or "N/A_brain_D3"
    logger.debug("Executing EXT_D3_Potential_Field_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    decay_exponent = 1.5
    max_influence_radius = 3
    max_possible_val_on_grid = BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols))
    if max_possible_val_on_grid == 0:
        return scores

    num_cells_in_radius_approx = (2 * max_influence_radius + 1)**2 - 1
    heuristic_max_potential = num_cells_in_radius_approx * (max_possible_val_on_grid / (1**decay_exponent))
    if heuristic_max_potential == 0:
        heuristic_max_potential = 1.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue
            current_cell_potential = 0.0
            for nr in range(rows):
                for nc in range(cols):
                    if grid[nr, nc] != -1:  # If it's a filled cell (a "charge")
                        num_val = grid[nr, nc]
                        if num_val <= 0: # Consider only positive charges
                            continue
                        dist = MathUtils().manhattan_distance((r_idx, c_idx), (nr, nc))
                        if dist == 0: # Should not happen if only scoring empty cells
                            continue
                        if dist > max_influence_radius:
                            continue
                        potential_contribution = num_val / (dist**decay_exponent)
                        current_cell_potential += potential_contribution
            scores[r_idx, c_idx] = MathUtils().normalize_value(current_cell_potential, 0, heuristic_max_potential, clamp=True)
    return scores

# 4. EXT_F10_Discontinuity_Vec(不連續性修復/序列完成度)
def EXT_F10_Discontinuity_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (F10-不連續性修復/序列完成度)
    核心規則:評估在空格填入數字後,是否能修復或完成某個方向上的數字序列(例如等差)。
    目的:偏好那些能夠「承先啟後」,使斷裂的序列得以延續或形成的空格。
    啟發式類型:序列與模式識別
    輸出詮釋:分數越高表示該空格填入某個合法數字後,能形成或延長的序列越長/越重要
    強化:大幅提升算術序列檢測的深度和靈活性,加入對更複雜的算術序列判斷。
    """
    effective_request_id = request_id or "N/A_brain_F10"
    logger.debug("Executing EXT_F10_Discontinuity_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    legal_values_for_placement = BoardAnalyzerUtils().get_legal_values_for_placement(grid)
    if not legal_values_for_placement:
        return scores

    min_sequence_len_to_score = 3
    heuristic_max_len = float(max(rows, cols))
    if heuristic_max_len < min_sequence_len_to_score:
        heuristic_max_len = float(min_sequence_len_to_score)

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue
            max_len_contribution_for_this_cell = 0.0
            for val_to_try in legal_values_for_placement:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = val_to_try
                current_val_max_len = 0.0

                # Check Row
                row_line = list(temp_grid[r_idx, :])
                sequences_in_row = BoardAnalyzerUtils().find_sequences_in_line(
                    row_line, min_len=min_sequence_len_to_score, allow_gaps=1, check_arithmetic=True
                )
                for seq in sequences_in_row:
                    if val_to_try in seq:
                        current_val_max_len = max(current_val_max_len, len(seq))

                # Check Column
                col_line = list(temp_grid[:, c_idx])
                sequences_in_col = BoardAnalyzerUtils().find_sequences_in_line(
                    col_line, min_len=min_sequence_len_to_score, allow_gaps=1, check_arithmetic=True
                )
                for seq in sequences_in_col:
                    if val_to_try in seq:
                        current_val_max_len = max(current_val_max_len, len(seq))
                
                # Check Diagonals
                diag1_line = list(np.diag(temp_grid, k=c_idx - r_idx))
                sequences_in_diag1 = BoardAnalyzerUtils().find_sequences_in_line(
                    diag1_line, min_len=min_sequence_len_to_score, allow_gaps=1, check_arithmetic=True
                )
                for seq in sequences_in_diag1:
                    if val_to_try in seq:
                        current_val_max_len = max(current_val_max_len, len(seq))

                flipped_temp_grid = np.fliplr(temp_grid)
                flipped_c_idx = cols - 1 - c_idx
                diag2_line = list(np.diag(flipped_temp_grid, k=flipped_c_idx - r_idx))
                sequences_in_diag2 = BoardAnalyzerUtils().find_sequences_in_line(
                    diag2_line, min_len=min_sequence_len_to_score, allow_gaps=1, check_arithmetic=True
                )
                for seq in sequences_in_diag2:
                    if val_to_try in seq:
                        current_val_max_len = max(current_val_max_len, len(seq))
                
                if current_val_max_len >= min_sequence_len_to_score:
                    max_len_contribution_for_this_cell = max(max_len_contribution_for_this_cell, current_val_max_len)

            if heuristic_max_len > 0:
                scores[r_idx, c_idx] = MathUtils().normalize_value(max_len_contribution_for_this_cell, 0, heuristic_max_len, clamp=True)
            else:
                scores[r_idx, c_idx] = 0.0
    return scores

# 5. EXT_P7_Pathfinding_Value_Vec(路徑尋找價值)
def EXT_P7_Pathfinding_Value_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (P7-路徑尋找價值)
    核心規則:評估在空格填入數字後,形成連接到其他現有數字的路徑的價值。
    目的:偏好那些能夠「橋接」盤面區域,或連接到高價值目標的空格。
    啟發式類型:連通性與圖論
    輸出詮釋:分數越高表示該空格填入某數字後,能形成更有價值的路徑(考慮路徑長度與連接 到的數字大小)
    強化:修正 BFS 探索鄰居的方向,確保計算路徑時的正確性。
    """
    effective_request_id = request_id or "N/A_brain_P7"
    logger.debug("Executing EXT_P7_Pathfinding_Value_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    legal_values_for_placement = BoardAnalyzerUtils().get_legal_values_for_placement(grid)
    if not legal_values_for_placement:
        return scores

    max_path_search_depth = 4
    path_value_decay_factor = 1.0
    max_possible_val_on_grid = BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols))
    if max_possible_val_on_grid == 0:
        max_possible_val_on_grid = 1.0
    
    heuristic_max_path_score = ((2 * max_path_search_depth + 1)**2 * max_possible_val_on_grid / (1**path_value_decay_factor))
    if heuristic_max_path_score == 0:
        heuristic_max_path_score = 1.0

    for r_start in range(rows):
        for c_start in range(cols):
            if grid[r_start, c_start] != -1:  # Only score empty cells
                continue
            max_score_for_this_cell = 0.0
            # val_to_try is not directly used in this version of pathfinding, path explores from empty cell
            # If val_to_try were to influence path (e.g. path must be arithmetic with val_to_try), logic would change
            
            q: deque[tuple[tuple[int, int], int]] = deque([((r_start, c_start), 0)]) # ((r,c), path_length)
            visited_for_bfs: set[tuple[int,int]] = set([(r_start, c_start)])
            current_placement_path_score = 0.0
            
            head_count = 0
            max_bfs_steps = rows * cols # Max distinct cells to visit

            while q and head_count < max_bfs_steps:
                head_count += 1
                (curr_r, curr_c), path_len = q.popleft()

                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # 4-connectivity
                    next_r, next_c = curr_r + dr, curr_c + dc

                    if 0 <= next_r < rows and 0 <= next_c < cols:
                        if grid[next_r, next_c] != -1: # Reached an existing number
                            reached_val = grid[next_r, next_c]
                            effective_path_len = path_len + 1 # Path from start to (curr_r,curr_c) is path_len, then one more step to (next_r,next_c)
                            current_placement_path_score += reached_val / (effective_path_len**path_value_decay_factor)
                            # Do not add to queue or visited, this is a terminal number for this path segment
                        elif (next_r, next_c) not in visited_for_bfs and \
                             grid[next_r, next_c] == -1 and \
                             path_len + 1 < max_path_search_depth:
                            visited_for_bfs.add((next_r, next_c))
                            q.append(((next_r, next_c), path_len + 1))
            
            # This module's original logic implies placing any `val_to_try` and then finding paths.
            # The provided BFS explores from the empty cell itself to find existing numbers.
            # If the intention was to evaluate pathing *after* placing each `val_to_try`,
            # `current_placement_path_score` would need to be inside the `val_to_try` loop.
            # The current structure evaluates one path score from the empty cell (r_start, c_start).
            max_score_for_this_cell = current_placement_path_score # Assign the calculated score

            scores[r_start, c_start] = MathUtils().normalize_value(max_score_for_this_cell, 0, heuristic_max_path_score, clamp=True)
    return scores

# 6. EXT_R5_Resource_Control_Vec(資源控制)
def EXT_R5_Resource_Control_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (R5-資源控制)
    核心規則:從資源控制角度評估填補位置的策略價值。資源可包括行/列的完成度、對高價值數字的獲取潜力等。
    目的:偏好那些能夠鞏固盤面控制權,或獲取潛在高價值數字的空格。
    啟發式類型:策略與控制
    輸出詮釋:分數越高表示該空格在填入數字後,對資源的控制(如行列完成度、高價值數字佔 據)越強
    強化:更精確地計算行/列完成度,並調整權重以反映策略偏好。
    """
    effective_request_id = request_id or "N/A_brain_R5"
    logger.debug("Executing EXT_R5_Resource_Control_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    max_possible_val_on_grid = BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols))
    if max_possible_val_on_grid == 0:
        max_possible_val_on_grid = 1.0
    
    hypothetical_high_val_placed = 0.0
    if potential_numbers_to_place:
        hypothetical_high_val_placed = float(np.max(potential_numbers_to_place))


    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue

            num_filled_in_row = np.count_nonzero(grid[r_idx, :] != -1)
            row_completeness_score = (num_filled_in_row + 1) / cols if cols > 0 else 0.0

            num_filled_in_col = np.count_nonzero(grid[:, c_idx] != -1)
            col_completeness_score = (num_filled_in_col + 1) / rows if rows > 0 else 0.0
            
            value_capture_score = 0.0
            if hypothetical_high_val_placed > 0 and max_possible_val_on_grid > 0:
                 value_capture_score = MathUtils().normalize_value(hypothetical_high_val_placed, 1, max_possible_val_on_grid, clamp=True)

            w_row = 0.3
            w_col = 0.3
            w_val = 0.4
            combined_score = (w_row * row_completeness_score +
                              w_col * col_completeness_score +
                              w_val * value_capture_score)
            scores[r_idx, c_idx] = MathUtils().normalize_value(combined_score, 0, 1.0, clamp=True)
    return scores

# 7. EXT_GM1_Row_Control_Vec(行控制力)
def EXT_GM1_Row_Control_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM1-行控制力)
    核心規則:評估在特定空格填入數字後,對該行的完成度、數值總和或序列形成的貢獻。
    目的:偏好那些能增強單行控制力或形成有價值行模式的填補。
    啟發式類型:線性結構控制(行)
    輸出詮釋:分數越高表示對該行的潛在控制力或完成度越強
    強化:改善序列完成度的判斷,利用`find_sequences_in_line`進行更全面的算術序列檢測。
    """
    effective_request_id = request_id or "N/A_brain_GM1"
    logger.debug("Executing EXT_GM1_Row_Control_Vec", extra={'request_id': effective_request_id})
    
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    avg_potential_num_to_place = 0.0
    if potential_numbers_to_place:
        avg_potential_num_to_place = float(np.mean(potential_numbers_to_place))

    max_val_board = BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols))
    if max_val_board == 0:
        max_val_board = 1.0

    for r_idx in range(rows):
        current_row_values_list = [val for val in grid[r_idx, :] if val != -1]
        num_filled_in_row = len(current_row_values_list)
        sum_current_row_values = sum(current_row_values_list)

        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue

            density_score = (num_filled_in_row + 1.0) / cols if cols > 0 else 0.0
            
            potential_row_sum = sum_current_row_values + avg_potential_num_to_place
            heuristic_max_row_sum = float(cols * max_val_board)
            sum_score = 0.0
            if heuristic_max_row_sum > 0:
                sum_score = MathUtils().normalize_value(potential_row_sum, 0, heuristic_max_row_sum, clamp=True)

            seq_score = 0.0
            temp_row_line = list(grid[r_idx, :])
            # For sequence scoring, it's better to try each potential number if performance allows,
            # or use a representative value like avg_potential_num_to_place as a heuristic
            temp_row_line[c_idx] = int(round(avg_potential_num_to_place)) # Use rounded average for sequence check
            sequences_found = BoardAnalyzerUtils().find_sequences_in_line(
                temp_row_line, min_len=3, allow_gaps=1, check_arithmetic=True
            )
            max_seq_len_with_avg = 0
            for seq in sequences_found:
                if int(round(avg_potential_num_to_place)) in seq: # Check if the placed value is part of this sequence
                    max_seq_len_with_avg = max(max_seq_len_with_avg, len(seq))
            
            if max_seq_len_with_avg >= 3:
                seq_score = MathUtils().normalize_value(float(max_seq_len_with_avg), 3, cols, clamp=True)
            elif max_seq_len_with_avg > 0:
                seq_score = 0.25 # Small bonus

            w_density = 0.4
            w_sum = 0.3
            w_seq = 0.3
            combined_score = (w_density * density_score +
                              w_sum * sum_score +
                              w_seq * seq_score)
            scores[r_idx, c_idx] = MathUtils().normalize_value(combined_score, 0, 1.0, clamp=True)
    return scores

# 8. EXT_GM2_Col_Flow_Vec (列流動性/列控制力)
def EXT_GM2_Col_Flow_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM2 - 列流動性/列控制力)
    核心規則:評估在特定空格填入數字後,對該列的完成度、數值總和或序列形成的貢獻。
    目的:偏好那些能增強單列控制力或形成有價值列模式的填補。
    啟發式類型:線性結構控制(列)
    輸出詮釋:分數越高表示對該列的潛在控制力或完成度越強
    強化:改善序列完成度的判斷,利用`find_sequences_in_line`進行更全面的算術序列檢測。
    """
    effective_request_id = request_id or "N/A_brain_GM2"
    logger.debug("Executing EXT_GM2_Col_Flow_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    avg_potential_num_to_place = 0.0
    if potential_numbers_to_place:
        avg_potential_num_to_place = float(np.mean(potential_numbers_to_place))

    max_val_board = BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols))
    if max_val_board == 0:
        max_val_board = 1.0

    for c_idx in range(cols):
        current_col_values_list = [val for val in grid[:, c_idx] if val != -1]
        num_filled_in_col = len(current_col_values_list)
        sum_current_col_values = sum(current_col_values_list)

        for r_idx in range(rows):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue

            density_score = (num_filled_in_col + 1.0) / rows if rows > 0 else 0.0

            potential_col_sum = sum_current_col_values + avg_potential_num_to_place
            heuristic_max_col_sum = float(rows * max_val_board)
            sum_score = 0.0
            if heuristic_max_col_sum > 0:
                sum_score = MathUtils().normalize_value(potential_col_sum, 0, heuristic_max_col_sum, clamp=True)

            seq_score = 0.0
            temp_col_line = list(grid[:, c_idx])
            temp_col_line[r_idx] = int(round(avg_potential_num_to_place))
            sequences_found = BoardAnalyzerUtils().find_sequences_in_line(
                temp_col_line, min_len=3, allow_gaps=1, check_arithmetic=True
            )
            max_seq_len_with_avg = 0
            for seq in sequences_found:
                if int(round(avg_potential_num_to_place)) in seq:
                    max_seq_len_with_avg = max(max_seq_len_with_avg, len(seq))
            
            if max_seq_len_with_avg >= 3:
                seq_score = MathUtils().normalize_value(float(max_seq_len_with_avg), 3, rows, clamp=True)
            elif max_seq_len_with_avg > 0:
                seq_score = 0.25

            w_density = 0.4
            w_sum = 0.3
            w_seq = 0.3
            combined_score = (w_density * density_score +
                              w_sum * sum_score +
                              w_seq * seq_score)
            scores[r_idx, c_idx] = MathUtils().normalize_value(combined_score, 0, 1.0, clamp=True)
    return scores

# 9. EXT_GM3_Adv_Connected_Comp_Vec (高級連通元件分析-空格區域)
def EXT_GM3_Adv_Connected_Comp_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM3 - 高級連通元件分析-空格區域)
    核心規則:分析空格所屬的連續空格區域的大小。
    目的:偏好那些屬於較大連續空格區域的空格,這些區域可能提供更大的填補潛力或形成大型結構的機會。
    啟發式類型:連通元件分析(針對空格)
    輸出詮釋:分數越高表示該空格屬於一個面積越大的連續空格區域(分數經盤面總大小正規化)
    強化:確保 BFS遍歷的正確性,並優化對已訪問單元的標記。
    """
    effective_request_id = request_id or "N/A_brain_GM3"
    logger.debug("Executing EXT_GM3_Adv_Connected_Comp_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    visited_overall = np.zeros_like(grid, dtype=bool)

    for r_start in range(rows):
        for c_start in range(cols):
            if visited_overall[r_start, c_start] or grid[r_start, c_start] != -1:
                continue

            component_cells: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(r_start, c_start)])
            visited_bfs_current_component: set[tuple[int, int]] = set([(r_start, c_start)])
            visited_overall[r_start, c_start] = True
            
            while q:
                r_curr, c_curr = q.popleft()
                component_cells.append((r_curr, c_curr))
                for dr_bfs, dc_bfs in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r_curr + dr_bfs, c_curr + dc_bfs
                    if 0 <= nr < rows and 0 <= nc < cols and \
                       grid[nr, nc] == -1 and \
                       not visited_overall[nr, nc] and \
                       (nr, nc) not in visited_bfs_current_component:
                        visited_overall[nr, nc] = True
                        visited_bfs_current_component.add((nr, nc))
                        q.append((nr, nc))
            
            area_size = float(len(component_cells))
            total_cells = float(rows * cols)
            norm_area_size = 0.0
            if total_cells > 0:
                norm_area_size = MathUtils().normalize_value(area_size, 0, total_cells, clamp=True)
            
            for r_comp, c_comp in component_cells:
                scores[r_comp, c_comp] = norm_area_size
    return scores

# 10. EXT_GM4_Spatial_Auto_Corr_Vec (空間自相關性分析)
def EXT_GM4_Spatial_Auto_Corr_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM4 - 空間自相關性分析)
    核心規則:評估在空格填入一個假設的「平均」潛在數字後,該數字與其周圍現有數字的相似程度。
    目的:鼓勵形成數值聚集(正自相關)或數值交錯(負自相關,但此處偏好正自相關)。
    此版本偏好正自相關,即填入的數字與周圍鄰居的平均值相似時得分較高。
    啟發式類型:空間統計
    輸出詮釋:分數越高表示填入一個「典型」數字後,能更好地融入周圍環境,形成數值上的聚集。
    強化:優化假設值的選擇,並更精確地計算歸一化差異。
    """
    effective_request_id = request_id or "N/A_brain_GM4"
    logger.debug("Executing EXT_GM4_Spatial_Auto_Corr_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    hypothetical_val_to_place: float
    if potential_numbers:
        hypothetical_val_to_place = float(np.median(potential_numbers))
    else:
        max_board_val = BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols))
        hypothetical_val_to_place = (1.0 + float(max_board_val)) / 2.0 if max_board_val > 0 else 0.5
    
    max_val_on_grid_for_norm = float(BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols)))
    if max_val_on_grid_for_norm == 0:
        max_val_on_grid_for_norm = 1.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue
            
            neighbor_values = BoardAnalyzerUtils().get_neighborhood_values(
                grid, r_idx, c_idx, radius=1, eight_connectivity=True,
                val_func=lambda x: float(x) if x != -1 else None,
                include_center=False
            )
            if not neighbor_values:
                scores[r_idx, c_idx] = 0.5 # Neutral score
                continue

            mean_neighbors = float(np.mean(neighbor_values))
            diff_hypothetical_to_mean_neighbors = abs(hypothetical_val_to_place - mean_neighbors)
            norm_diff = MathUtils().normalize_value(diff_hypothetical_to_mean_neighbors, 0, max_val_on_grid_for_norm, clamp=True)
            positive_autocorr_score = 1.0 - norm_diff
            scores[r_idx, c_idx] = positive_autocorr_score
    return scores

# 11. EXT_GM5_Line_Completion_Vec(線段補全)
def EXT_GM5_Line_Completion_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM5-線段補全)
    核心規則:評估空格對於完成特定方向(行、列、對角線)上具有特定構成(如等差、等值)的短線段(例如長度為3)之潛力。
    目的:偏好那些能夠「臨門一腳」完成有意義短線段的空格。
    啟發式類型:模式匹配(短線段)
    輸出詮釋:分數越高表示該空格填入某數字後,越能完成一個預定義的短線段模式。
    強化:細化對算術序列的評分,對高價值序列給予額外獎勵。
    """
    effective_request_id = request_id or "N/A_brain_GM5"
    logger.debug("Executing EXT_GM5_Line_Completion_Vec", extra={'request_id': effective_request_id})
    
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0 or min(rows, cols) < 1: # Need at least 1 cell, for lines of 3, need more.
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not potential_numbers_to_place:
        return scores

    line_completion_score_map = {
        "identical_3": 0.6,
        "arithmetic_3_mend": 0.7,
        "arithmetic_3_extend": 0.5,
        "arithmetic_3_mend_high_val": 0.9,
    }
    max_board_val = BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols))
    high_val_threshold = max_board_val * 0.7 if max_board_val > 0 else 10.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue
            max_completion_score_for_cell = 0.0
            for p_val in potential_numbers_to_place:
                # Check all 4 directions for line completion
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]: # H, V, Diag, Anti-Diag
                    # Case 1: Mending N1 - p_val - N2
                    r_n1, c_n1 = r_idx - dr, c_idx - dc
                    r_n2, c_n2 = r_idx + dr, c_idx + dc
                    if 0 <= r_n1 < rows and 0 <= c_n1 < cols and \
                       0 <= r_n2 < rows and 0 <= c_n2 < cols:
                        val_n1 = grid[r_n1, c_n1]
                        val_n2 = grid[r_n2, c_n2]
                        if val_n1 != -1 and val_n2 != -1:
                            if val_n1 == p_val and val_n2 == p_val:
                                max_completion_score_for_cell = max(max_completion_score_for_cell, line_completion_score_map["identical_3"])
                            if (val_n1 + val_n2) == 2 * p_val and abs(p_val - val_n1) > 0:
                                score = line_completion_score_map["arithmetic_3_mend"]
                                if (val_n1 + p_val + val_n2) / 3.0 > high_val_threshold:
                                    score = max(score, line_completion_score_map.get("arithmetic_3_mend_high_val", score))
                                max_completion_score_for_cell = max(max_completion_score_for_cell, score)
                    
                    # Case 2: Extending p_val - N1 - N2
                    r_n1_ext1, c_n1_ext1 = r_idx + dr, c_idx + dc
                    r_n2_ext1, c_n2_ext1 = r_idx + 2 * dr, c_idx + 2 * dc
                    if 0 <= r_n1_ext1 < rows and 0 <= c_n1_ext1 < cols and \
                       0 <= r_n2_ext1 < rows and 0 <= c_n2_ext1 < cols:
                        val_n1_ext1 = grid[r_n1_ext1, c_n1_ext1]
                        val_n2_ext1 = grid[r_n2_ext1, c_n2_ext1]
                        if val_n1_ext1 != -1 and val_n2_ext1 != -1:
                            if p_val == val_n1_ext1 and p_val == val_n2_ext1:
                                max_completion_score_for_cell = max(max_completion_score_for_cell, line_completion_score_map["identical_3"])
                            if (p_val + val_n2_ext1) == 2 * val_n1_ext1 and abs(val_n1_ext1 - p_val) > 0:
                                max_completion_score_for_cell = max(max_completion_score_for_cell, line_completion_score_map["arithmetic_3_extend"])

                    # Case 3: Extending N1 - N2 - p_val
                    r_n1_ext2, c_n1_ext2 = r_idx - 2 * dr, c_idx - 2 * dc
                    r_n2_ext2, c_n2_ext2 = r_idx - dr, c_idx - dc
                    if 0 <= r_n1_ext2 < rows and 0 <= c_n1_ext2 < cols and \
                       0 <= r_n2_ext2 < rows and 0 <= c_n2_ext2 < cols:
                        val_n1_ext2 = grid[r_n1_ext2, c_n1_ext2]
                        val_n2_ext2 = grid[r_n2_ext2, c_n2_ext2]
                        if val_n1_ext2 != -1 and val_n2_ext2 != -1:
                            if val_n1_ext2 == val_n2_ext2 and val_n1_ext2 == p_val:
                                max_completion_score_for_cell = max(max_completion_score_for_cell, line_completion_score_map["identical_3"])
                            if (val_n1_ext2 + p_val) == 2 * val_n2_ext2 and abs(val_n2_ext2 - val_n1_ext2) > 0:
                                max_completion_score_for_cell = max(max_completion_score_for_cell, line_completion_score_map["arithmetic_3_extend"])
            scores[r_idx, c_idx] = MathUtils().normalize_value(max_completion_score_for_cell, 0, 1.0, clamp=True)
    return scores

# 12. EXT_GM6_Symmetry_Potential_Vec (對稱性潛力)
def EXT_GM6_Symmetry_Potential_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM6-對稱性潛力)
    核心規則:評估在空格填入數字後,盤面形成的對稱性程度(水平、垂直、中心、主對角線、反主對角線)。
    目的:偏好那些能夠創造或增強盤面對稱性的填補。
    啟發式類型:幾何與模式識別
    輸出詮釋:分數越高表示若在該空格填入特定數字,能與對稱位置上已存在的相同數字形成對稱。
    強化:精確處理對稱性檢查的邊界條件,特別是對角線對稱。
    """
    effective_request_id = request_id or "N/A_brain_GM6"
    logger.debug("Executing EXT_GM6_Symmetry_Potential_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not potential_numbers_to_place:
        return scores

    symmetry_scores_map = {
        "horizontal": 0.7,
        "vertical": 0.7,
        "point_center": 0.8,
        "main_diagonal": 0.6,
        "anti_diagonal": 0.6,
    }
    if rows == cols:
        symmetry_scores_map["main_diagonal"] = 0.7
        symmetry_scores_map["anti_diagonal"] = 0.7
    
    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue
            max_symmetry_score_for_cell = 0.0
            for p_val in potential_numbers_to_place:
                current_pval_max_sym = 0.0
                # 1. Horizontal
                sr_h, sc_h = r_idx, cols - 1 - c_idx
                if sc_h != c_idx and 0 <= sr_h < rows and 0 <= sc_h < cols and grid[sr_h, sc_h] == p_val:
                    current_pval_max_sym = max(current_pval_max_sym, symmetry_scores_map["horizontal"])
                # 2. Vertical
                sr_v, sc_v = rows - 1 - r_idx, c_idx
                if sr_v != r_idx and 0 <= sr_v < rows and 0 <= sc_v < cols and grid[sr_v, sc_v] == p_val:
                    current_pval_max_sym = max(current_pval_max_sym, symmetry_scores_map["vertical"])
                # 3. Point (Center)
                sr_p, sc_p = rows - 1 - r_idx, cols - 1 - c_idx
                if (sr_p != r_idx or sc_p != c_idx) and \
                   0 <= sr_p < rows and 0 <= sc_p < cols and grid[sr_p, sc_p] == p_val:
                    current_pval_max_sym = max(current_pval_max_sym, symmetry_scores_map["point_center"])
                # 4. Main Diagonal
                if rows == cols:
                    sr_d1, sc_d1 = c_idx, r_idx
                    if (sr_d1 != r_idx or sc_d1 != c_idx) and \
                       0 <= sr_d1 < rows and 0 <= sc_d1 < cols and grid[sr_d1, sc_d1] == p_val:
                        current_pval_max_sym = max(current_pval_max_sym, symmetry_scores_map["main_diagonal"])
                # 5. Anti-Diagonal
                if rows == cols: # For N x N matrix, symmetric to (r,c) is (N-1-c, N-1-r)
                    sr_d2, sc_d2 = (rows - 1) - c_idx, (rows - 1) - r_idx
                    if (sr_d2 != r_idx or sc_d2 != c_idx) and \
                       0 <= sr_d2 < rows and 0 <= sc_d2 < cols and grid[sr_d2, sc_d2] == p_val:
                        current_pval_max_sym = max(current_pval_max_sym, symmetry_scores_map["anti_diagonal"])
                
                if current_pval_max_sym > max_symmetry_score_for_cell:
                    max_symmetry_score_for_cell = current_pval_max_sym
            scores[r_idx, c_idx] = MathUtils().normalize_value(max_symmetry_score_for_cell, 0, 1.0, clamp=True)
    return scores

# 13. EXT_GM7_Numeric_Gaps_Vec (數值間隙填充)
def EXT_GM7_Numeric_Gaps_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM7 - 數值間隙填充)
    核心規則:識別並評估在局部區域或序列中,填補數字「間隙」的價值。
    特別是尋找能填入使之成為公差為1的連續數列的間隙。
    目的:偏好那些能填補序列中明顯缺失數字的空格。
    啟發式類型:序列與模式識別(間隙填充)
    輸出詮釋:分數越高表示該空格若填入特定數字,越能完美地填補一個數值間隙(尤其是公差為1的序列)。
    強化:增加對任意公差算術序列間隙填充的識別,並根據序列特徴給予不同分數。
    """
    effective_request_id = request_id or "N/A_brain_GM7"
    logger.debug("Executing EXT_GM7_Numeric_Gaps_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not potential_numbers_to_place:
        return scores

    gap_fill_scores_map = {
        "arithmetic_1_gap_fill": 0.9,
        "arithmetic_generic_mend": 0.7,
        "arithmetic_generic_extend": 0.5,
        "arithmetic_gap_fill_high_val": 0.95,
        # "arithmetic_gap_fill_long_seq_potential": 0.85, # Not directly used yet
    }
    max_board_val = BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols))
    high_val_threshold = max_board_val * 0.7 if max_board_val > 0 else 10.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue
            max_cell_gap_score = 0.0
            for p_val in potential_numbers_to_place:
                # Iterate over 4 directions
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    # Case 1: p_val mends N1 - p_val - N2
                    r_n1, c_n1 = r_idx - dr, c_idx - dc
                    r_n2, c_n2 = r_idx + dr, c_idx + dc
                    if 0 <= r_n1 < rows and 0 <= c_n1 < cols and \
                       0 <= r_n2 < rows and 0 <= c_n2 < cols:
                        val_n1 = grid[r_n1, c_n1]
                        val_n2 = grid[r_n2, c_n2]
                        if val_n1 != -1 and val_n2 != -1:
                            if val_n1 == p_val - 1 and val_n2 == p_val + 1:
                                score = gap_fill_scores_map["arithmetic_1_gap_fill"]
                                if (val_n1 + p_val + val_n2) / 3.0 > high_val_threshold:
                                    score = max(score, gap_fill_scores_map.get("arithmetic_gap_fill_high_val", score))
                                max_cell_gap_score = max(max_cell_gap_score, score)
                            elif (val_n1 + val_n2) == 2 * p_val and abs(p_val - val_n1) > 0:
                                max_cell_gap_score = max(max_cell_gap_score, gap_fill_scores_map["arithmetic_generic_mend"])
                    
                    # Case 2: p_val extends p_val - N1 - N2
                    r_n1_ext1, c_n1_ext1 = r_idx + dr, c_idx + dc
                    r_n2_ext1, c_n2_ext1 = r_idx + 2 * dr, c_idx + 2 * dc
                    if 0 <= r_n1_ext1 < rows and 0 <= c_n1_ext1 < cols and \
                       0 <= r_n2_ext1 < rows and 0 <= c_n2_ext1 < cols:
                        val_n1_ext1 = grid[r_n1_ext1, c_n1_ext1]
                        val_n2_ext1 = grid[r_n2_ext1, c_n2_ext1]
                        if val_n1_ext1 != -1 and val_n2_ext1 != -1:
                            if (val_n1_ext1 - p_val) == (val_n2_ext1 - val_n1_ext1) and (val_n1_ext1 - p_val) != 0:
                                max_cell_gap_score = max(max_cell_gap_score, gap_fill_scores_map["arithmetic_generic_extend"])
                    
                    # Case 3: p_val extends N1 - N2 - p_val
                    r_n1_ext2, c_n1_ext2 = r_idx - 2 * dr, c_idx - 2 * dc
                    r_n2_ext2, c_n2_ext2 = r_idx - dr, c_idx - dc
                    if 0 <= r_n1_ext2 < rows and 0 <= c_n1_ext2 < cols and \
                       0 <= r_n2_ext2 < rows and 0 <= c_n2_ext2 < cols:
                        val_n1_ext2 = grid[r_n1_ext2, c_n1_ext2]
                        val_n2_ext2 = grid[r_n2_ext2, c_n2_ext2]
                        if val_n1_ext2 != -1 and val_n2_ext2 != -1:
                            if (val_n2_ext2 - val_n1_ext2) == (p_val - val_n2_ext2) and (val_n2_ext2 - val_n1_ext2) != 0:
                                max_cell_gap_score = max(max_cell_gap_score, gap_fill_scores_map["arithmetic_generic_extend"])
            scores[r_idx, c_idx] = MathUtils().normalize_value(max_cell_gap_score, 0, 1.0, clamp=True)
    return scores

# 14. EXT_GM8_Edge_Affinity_Vec (邊緣親和度)
def EXT_GM8_Edge_Affinity_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM8-邊緣親和度)
    核心規則:評估空格與盤面邊緣或角落的接近程度及其策略意義。
    目的:根據策略配置,偏好靠近或遠離邊緣/角落的空格。
    啟發式類型:位置與邊界分析
    輸出詮釋:分數高低取決於設定(偏好/避開邊緣)。預設偏好邊緣,越靠近邊緣/角落分數越高。
    強化:優化最大最小距離的計算,確保歸一化的準確性,並微調角落獎勵/懲罰。
    """
    effective_request_id = request_id or "N/A_brain_GM8"
    logger.debug("Executing EXT_GM8_Edge_Affinity_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    affinity_mode = "prefer_edge" 
    corner_bonus_prefer = 0.2
    corner_penalty_avoid = 0.2

    max_min_dist_to_edge_row = (rows - 1) // 2 if rows > 0 else 0
    max_min_dist_to_edge_col = (cols - 1) // 2 if cols > 0 else 0
    overall_max_of_min_distances = float(min(max_min_dist_to_edge_row, max_min_dist_to_edge_col))

    if overall_max_of_min_distances == 0 and (rows > 1 or cols > 1): # e.g. a 1xN or Nx1 line
        overall_max_of_min_distances = 0.5 
    elif rows <= 1 and cols <= 1: # For 1x1 grid
        overall_max_of_min_distances = 1.0 


    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue

            dist_to_top_edge = r_idx
            dist_to_bottom_edge = rows - 1 - r_idx
            dist_to_left_edge = c_idx
            dist_to_right_edge = cols - 1 - c_idx
            min_dist = float(min(dist_to_top_edge, dist_to_bottom_edge, dist_to_left_edge, dist_to_right_edge))
            
            is_corner = (r_idx == 0 or r_idx == rows - 1) and \
                        (c_idx == 0 or c_idx == cols - 1)
            
            current_score = 0.0
            normalized_dist = 0.0
            if overall_max_of_min_distances > 0:
                normalized_dist = min_dist / overall_max_of_min_distances
                normalized_dist = min(1.0, normalized_dist) # Clamp
            elif min_dist == 0: # all cells are on an edge
                 normalized_dist = 0.0
            else: # Should not happen
                 normalized_dist = 1.0

            if affinity_mode == "prefer_edge":
                current_score = 1.0 - normalized_dist
                if is_corner and min_dist == 0:
                    current_score += corner_bonus_prefer
            elif affinity_mode == "avoid_edge":
                current_score = normalized_dist
                if is_corner and min_dist == 0:
                    current_score -= corner_penalty_avoid
            
            scores[r_idx, c_idx] = MathUtils().normalize_value(current_score, -corner_penalty_avoid, 1.0 + corner_bonus_prefer, clamp=True)
    return scores

# 15. EXT_GM9_Center_Control_Vec(中心控制偏好)
def EXT_GM9_Center_Control_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM9-中心控制偏好)
    核心規則:評估空格與盤面中心的接近程度及其策略意義。
    目的:根據策略配置,偏好靠近或遠離盤面中心區域的空格。
    啟發式類型:位置與中心性分析
    輸出詮釋:分數高低取決於設定(偏好/避開中心)。預設偏好中心,越靠近中心分數越高。
    強化:精確計算中心點和最大距離,避免除以零的情況。
    """
    effective_request_id = request_id or "N/A_brain_GM9"
    logger.debug("Executing EXT_GM9_Center_Control_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    affinity_mode = "prefer_center"
    center_r = (rows - 1) / 2.0
    center_c = (cols - 1) / 2.0

    # Max Euclidean distance from corner (0,0) to center
    max_dist_to_center = MathUtils().euclidean_distance((0.0, 0.0), (center_r, center_c))
    
    if max_dist_to_center == 0: # Handles 1x1 grid or if somehow calculation is 0
        max_dist_to_center = 1.0 # Avoid division by zero

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue
            
            current_dist_to_center = MathUtils().euclidean_distance((float(r_idx), float(c_idx)), (center_r, center_c))
            normalized_dist = 0.0
            if max_dist_to_center > 0: # Should always be true now
                 normalized_dist = MathUtils().normalize_value(current_dist_to_center, 0, max_dist_to_center, clamp=True)
            # For 1x1 grid, dist is 0, max_dist is 1.0 (or 0 then set to 1.0). norm_dist = 0.
            # normalize_value handles val=min=max returning 0.5, but here min=0.
            
            current_score = 0.0
            if affinity_mode == "prefer_center":
                current_score = 1.0 - normalized_dist
            elif affinity_mode == "avoid_center":
                current_score = normalized_dist
            
            scores[r_idx, c_idx] = MathUtils().normalize_value(current_score, 0, 1.0, clamp=True)
    return scores

# 16. EXT_GM10_Blocking_Value_Vec (阻斷價值評估)
def EXT_GM10_Blocking_Value_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM10-阻斷價值評估)
    核心規則:評估在空格填入數字是否能有效「阻止」或「避免」形成預定義的不良模式或序列。
    目的:偏好那些不會導致形成不良結構的填補,或者理想情況下能主動阻止潛在不良結構形成的填補。
    啟發式類型:防禦性策略與模式規避
    輸出詮釋:分數越高表示在該空格填入數字後,越不可能形成已知的不良模式。
    強化:增加對不良模式的定義彈性,並更精確地檢查所有潛在的不良序列。
    """
    effective_request_id = request_id or "N/A_brain_GM10"
    logger.debug("Executing EXT_GM10_Blocking_Value_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not potential_numbers_to_place:
        return scores

    UNDESIRABLE_SEQUENCES = [
        [1, 1, 1],
        [2, 2, 2],
    ]
    
    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue
            
            max_safety_score_for_cell = 0.0 # Default to low if all placements are bad
            if not potential_numbers_to_place: # Should not happen if checked above, but defensive
                scores[r_idx, c_idx] = 0.5 # Neutral
                continue

            for p_val in potential_numbers_to_place:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val
                forms_undesirable_pattern = False

                # Check lines of length 3 passing through (r_idx, c_idx)
                for dr_line, dc_line in [(0, 1), (1, 0), (1, 1), (1, -1)]: # H, V, D1, D2
                    for offset in range(-2, 1): # Start of a 3-cell window relative to (r_idx, c_idx)
                        current_line_values: list[int] = []
                        valid_line = True
                        line_coords: list[tuple[int,int]] = []

                        for i in range(3): # For a 3-cell line
                            # Effective position of cell 'i' in the line
                            # If offset = 0, p_val is at line[0]. (r_idx,c_idx) is temp_grid[r_idx + 0*dr, c_idx + 0*dc]
                            # If offset = -1, p_val is at line[1]. (r_idx,c_idx) is temp_grid[r_idx + 1*dr, c_idx + 1*dc]
                            # If offset = -2, p_val is at line[2]. (r_idx,c_idx) is temp_grid[r_idx + 2*dr, c_idx + 2*dc]
                            # This seems to be about where the line *starts* relative to the point
                            # The p_val is at (r_idx, c_idx).
                            # We need to check lines like: X-X-P, X-P-X, P-X-X
                            # The current logic: offset means the *start* of the 3-cell window relative to (0,0) *in the direction vector's frame*
                            # And then (r_idx, c_idx) is the cell being evaluated.
                            # (r_idx + (offset + i) * dr_line, c_idx + (offset + i) * dc_line) means (r_idx,c_idx) is the origin of the directional check.

                            # A simpler way: check 3 windows centered around the placed p_val
                            # Window 1: [p_val, N1, N2]
                            # Window 2: [N-1, p_val, N1]
                            # Window 3: [N-2, N-1, p_val]
                            # The current loop (offset from -2 to 0) means:
                            # offset = -2: line starts at (r_idx - 2*dr, c_idx - 2*dc), p_val is the 3rd element
                            # offset = -1: line starts at (r_idx - 1*dr, c_idx - 1*dc), p_val is the 2nd element
                            # offset =  0: line starts at (r_idx,      c_idx     ), p_val is the 1st element
                            # This covers all cases where p_val is part of a 3-segment line.

                            check_r, check_c = r_idx + (offset + i) * dr_line, c_idx + (offset + i) * dc_line
                            
                            if 0 <= check_r < rows and 0 <= check_c < cols:
                                line_coords.append((check_r, check_c))
                                current_line_values.append(int(temp_grid[check_r, check_c]))
                            else:
                                valid_line = False
                                break
                        
                        if valid_line and len(current_line_values) == 3:
                            # Ensure the p_val at (r_idx, c_idx) is part of this specific line segment check
                            # The construction with offset ensures (r_idx, c_idx) is involved if p_val makes it so.
                            # Specifically, (r_idx, c_idx) = (r_idx + (offset + k)*dr, c_idx + (offset+k)*dc)
                            # for k such that offset+k = 0. So k = -offset. This k must be in [0,1,2].
                            # So -offset in [0,1,2] -> offset in [-2, -1, 0]. This is already handled by loop.
                            
                            # The PDF had: if (r_idx, c_idx) not in line_coords: continue
                            # This check is implicitly handled by how current_line_values is built around temp_grid which has p_val at (r_idx, c_idx)
                            # and the sliding window `offset`.

                            for undesirable_seq in UNDESIRABLE_SEQUENCES:
                                if len(undesirable_seq) == 3 and current_line_values == undesirable_seq:
                                    forms_undesirable_pattern = True
                                    break # from undesirable_seq loop
                            if forms_undesirable_pattern:
                                break # from offset loop
                    if forms_undesirable_pattern:
                        break # from dr_line, dc_line loop
                
                current_score_for_pval = 0.9 if not forms_undesirable_pattern else 0.1
                if current_score_for_pval > max_safety_score_for_cell:
                    max_safety_score_for_cell = current_score_for_pval
            
            scores[r_idx, c_idx] = max_safety_score_for_cell
    return scores

# 17. EXT_GM11_Pair_Correlation_Vec (數字配對關聯分析)
def EXT_GM11_Pair_Correlation_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM11-數字配對關聯分析)
    核心規則:分析特定數字對(pair)共同出現或以特定相對位置(此處為鄰近)出現的頻率與價值。
    目的:偏好那些能夠形成已知有利數字配對的填補。
    啟發式類型:關聯性分析(局部)
    輸出詮釋:分數越高表示在該空格填入特定數字後,能與周圍已存在的數字形成更多或更高價值的有利配對。
    強化:擴展有利配對的定義,並確保最大分數的歸一化是穩健的。
    """
    effective_request_id = request_id or "N/A_brain_GM11"
    logger.debug("Executing EXT_GM11_Pair_Correlation_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not potential_numbers_to_place:
        return scores

    card_max_val = BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows,cols))
    mid_val = max(1, card_max_val // 2)

    FAVORABLE_PAIRS_SCORES = {
        (3, 7): 0.8, (7, 3): 0.8,
        (1, 2): 0.6, (2, 1): 0.6,
        (10, 20): 0.7, (20, 10): 0.7,
        (5, 10): 0.5, (10, 5): 0.5,
        (mid_val, mid_val + 1): 0.4,
        (mid_val + 1, mid_val): 0.4,
    }
    max_single_pair_score = 0.0
    if FAVORABLE_PAIRS_SCORES:
        max_single_pair_score = max(FAVORABLE_PAIRS_SCORES.values()) if FAVORABLE_PAIRS_SCORES else 0.0
    
    heuristic_max_total_pair_score = 8.0 * max_single_pair_score if max_single_pair_score > 0 else 1.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue
            max_accumulated_score_for_cell = 0.0
            for p_val in potential_numbers_to_place:
                current_pval_accumulated_score = 0.0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r_idx + dr, c_idx + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbor_val = grid[nr, nc]
                            if neighbor_val != -1:
                                if (p_val, int(neighbor_val)) in FAVORABLE_PAIRS_SCORES:
                                    current_pval_accumulated_score += FAVORABLE_PAIRS_SCORES[(p_val, int(neighbor_val))]
                if current_pval_accumulated_score > max_accumulated_score_for_cell:
                    max_accumulated_score_for_cell = current_pval_accumulated_score
            scores[r_idx, c_idx] = MathUtils().normalize_value(max_accumulated_score_for_cell, 0, heuristic_max_total_pair_score, clamp=True)
    return scores

# 18. EXT_GM12_Island_Analysis_Vec(島嶼分析)
def EXT_GM12_Island_Analysis_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM12 - 島嶼分析)
    核心規則:分析由已填數字形成的「島嶼」的特性,如大小、緊湊度和平均值。
    目的:根據策略,可能偏好大型、緊湊或包含高價值數字的島嶼。
    此處假設偏好較大、較緊湊、平均值較高的數字島嶼。
    啟發式類型:連通元件與區域形態分析(針對已填數字)
    輸出詮釋:分數越高表示該格屬於一個更優(大、緊湊、高平均值)的數字島嶼。空格得0分。
    強化:優化 BFS 遍歷以確保島嶼分析的準確性,並調整歸一化參數。
    """
    effective_request_id = request_id or "N/A_brain_GM12"
    logger.debug("Executing EXT_GM12_Island_Analysis_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    visited_island_search = np.zeros_like(grid, dtype=bool)
    max_val_on_board = BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols))
    if max_val_on_board == 0:
        max_val_on_board = 1.0

    w_size = 0.4
    w_compactness = 0.3
    w_avg_value = 0.3

    for r_start in range(rows):
        for c_start in range(cols):
            if grid[r_start, c_start] != -1 and not visited_island_search[r_start, c_start]:
                current_island_cells: list[tuple[int, int]] = []
                current_island_values: list[int] = []
                q: deque[tuple[int, int]] = deque([(r_start, c_start)])
                visited_island_search[r_start, c_start] = True
                min_r_bbox, max_r_bbox = r_start, r_start
                min_c_bbox, max_c_bbox = c_start, c_start

                while q:
                    r_curr, c_curr = q.popleft()
                    current_island_cells.append((r_curr, c_curr))
                    current_island_values.append(int(grid[r_curr, c_curr]))
                    min_r_bbox = min(min_r_bbox, r_curr)
                    max_r_bbox = max(max_r_bbox, r_curr)
                    min_c_bbox = min(min_c_bbox, c_curr)
                    max_c_bbox = max(max_c_bbox, c_curr)

                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # 4-connectivity
                        nr, nc = r_curr + dr, c_curr + dc
                        if 0 <= nr < rows and 0 <= nc < cols and \
                           grid[nr, nc] != -1 and not visited_island_search[nr, nc]:
                            visited_island_search[nr, nc] = True
                            q.append((nr, nc))
                
                island_size = float(len(current_island_cells))
                avg_value_island = 0.0
                if island_size > 0:
                    avg_value_island = sum(current_island_values) / island_size
                
                bbox_height = float(max_r_bbox - min_r_bbox + 1)
                bbox_width = float(max_c_bbox - min_c_bbox + 1)
                bbox_area = bbox_height * bbox_width
                compactness = 0.0
                if bbox_area > 0:
                    compactness = island_size / bbox_area
                
                norm_size = MathUtils().normalize_value(island_size, 1, float(rows * cols), clamp=True)
                norm_compactness = MathUtils().normalize_value(compactness, 0, 1.0, clamp=True)
                norm_avg_value = MathUtils().normalize_value(avg_value_island, 1, max_val_on_board, clamp=True)

                island_score = (w_size * norm_size +
                                w_compactness * norm_compactness +
                                w_avg_value * norm_avg_value)
                final_island_score = MathUtils().normalize_value(island_score, 0, 1.0, clamp=True)

                for r_cell, c_cell in current_island_cells:
                    scores[r_cell, c_cell] = final_island_score
            elif grid[r_start, c_start] == -1:
                scores[r_start, c_start] = 0.0 # Empty cells get 0
                visited_island_search[r_start, c_start] = True # Mark as visited to avoid re-check

    return scores

# 19. EXT_GM13_Sequence_Diversity_Vec (序列多樣性)
def EXT_GM13_Sequence_Diversity_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM13-序列多樣性)
    核心規則:評估填補位置是否有助於形成多樣化的短序列(例如,不同方向、不同類型),而非僅專注於單一長序列。
    目的:鼓勵在盤面上形成多個不同類型或方向的短數字序列,增加盤面的「活性」或「機會」。
    啟發式類型:模式識別與組合多樣性
    輸出詮釋:分數越高表示在該空格填入特定數字後,能參與形成的獨特短序列種類越多。
    強化:更全面地識別不同類型的短序列,並確保多樣性分數的準確性。
    """
    effective_request_id = request_id or "N/A_brain_GM13"
    logger.debug("Executing EXT_GM13_Sequence_Diversity_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not potential_numbers_to_place:
        return scores

    short_sequence_len = 3
    heuristic_max_distinct_sequences = 8.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue
            max_diversity_count_for_cell = 0
            for p_val in potential_numbers_to_place:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val
                found_sequence_signatures: set[tuple[str, tuple[int,int], int|float]] = set()

                for dr_dir, dc_dir in [(0, 1), (1, 0), (1, 1), (1, -1)]: # H, V, D1, D2
                    for i_offset in range(short_sequence_len): # p_val is at index i_offset in the window
                        current_sequence_values: list[int] = []
                        valid_segment = True
                        for k_seq in range(short_sequence_len):
                            check_r = r_idx + (k_seq - i_offset) * dr_dir
                            check_c = c_idx + (k_seq - i_offset) * dc_dir
                            if 0 <= check_r < rows and 0 <= check_c < cols:
                                current_sequence_values.append(int(temp_grid[check_r, check_c]))
                            else:
                                valid_segment = False
                                break
                        
                        if valid_segment and len(current_sequence_values) == short_sequence_len:
                            s = current_sequence_values
                            if all(val != -1 for val in s): # Ensure all are numbers
                                # Arithmetic
                                diff1 = s[1] - s[0]
                                diff2 = s[2] - s[1]
                                if diff1 == diff2 and diff1 != 0:
                                    found_sequence_signatures.add(("arithmetic", (dr_dir, dc_dir), diff1))
                                # Identical
                                if s[0] == s[1] and s[1] == s[2] and s[0] != -1 : # Ensure not three -1s
                                    found_sequence_signatures.add(("identical", (dr_dir, dc_dir), s[0]))
                
                current_pval_diversity_count = len(found_sequence_signatures)
                if current_pval_diversity_count > max_diversity_count_for_cell:
                    max_diversity_count_for_cell = current_pval_diversity_count
            
            scores[r_idx, c_idx] = MathUtils().normalize_value(float(max_diversity_count_for_cell), 0, heuristic_max_distinct_sequences, clamp=True)
    return scores

# 20. EXT_GM14_Risk_Assessment_Vec (風險評估)
def EXT_GM14_Risk_Assessment_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM14 - 風險評估)
    核心規則:評估某個填補動作的潛在「風險」,例如是否會導致後續選擇過少(降低盤面靈活 性)。
    目的:偏好那些能保持盤面較高靈活性的填補。低風險=高分數。
    啟發式類型:盤面狀態評估(未來選擇性)
    輸出詮釋:分數越高表示填入該數字後,盤面剩餘的合法填補選項越多,風險越低。
    強化:更精確地評估後續合法移動的數量作為靈活性指標。
    """
    effective_request_id = request_id or "N/A_brain_GM14"
    logger.debug("Executing EXT_GM14_Risk_Assessment_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    initial_potential_numbers = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not initial_potential_numbers:
        return scores # No numbers to place initially

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue
            
            max_flexibility_score_for_cell = 0.0
            for p_val in initial_potential_numbers: # Try only values currently legal for original grid
                temp_grid = grid.copy()
                # Check if p_val is still legal for this specific cell (r_idx, c_idx) if it wasn't generally
                # This is implicitly handled as p_val comes from get_legal_values_for_placement on original grid
                # And we only score empty cells.
                temp_grid[r_idx, c_idx] = p_val
                subsequent_legal_moves = len(BoardAnalyzerUtils().get_legal_values_for_placement(temp_grid))
                current_flexibility = float(subsequent_legal_moves)
                if current_flexibility > max_flexibility_score_for_cell:
                    max_flexibility_score_for_cell = current_flexibility
            
            current_max_heuristic_flex = float(rows * cols - 1) # Max legal values after 1 placement
            if current_max_heuristic_flex == 0: # e.g. 1x1 grid
                current_max_heuristic_flex = 1.0
            
            scores[r_idx, c_idx] = MathUtils().normalize_value(max_flexibility_score_for_cell, 0, current_max_heuristic_flex, clamp=True)
    return scores

# 21. EXT_GM15_Information_Gain_Vec (資訊增益評估)
def EXT_GM15_Information_Gain_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM15-資訊增益評估)
    核心規則:評估填入數字後,對盤面整體結構「有序性」的提升(例如,熵的降低)。
    目的:偏好那些能使盤面狀態更「確定」或「有序」的填補。
    啟發式類型:資訊理論啟發(基於全局熵變)
    輸出詮釋:分數越高表示填入該數字後,盤面整體熵降低得越多(即資訊增益越大,盤面越 有序)。
    強化:精確計算全局熵變化,並確保歸一化範圍的正確性。
    """
    effective_request_id = request_id or "N/A_brain_GM15"
    logger.debug("Executing EXT_GM15_Information_Gain_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not potential_numbers_to_place:
        return scores

    initial_grid_values = [int(val) for val in grid.flatten()]
    entropy_before = MathUtils().get_entropy(initial_grid_values)
    
    num_symbols = rows * cols + 1 # Max R*C numbers + the -1 symbol
    max_possible_entropy_change = math.log2(num_symbols) if num_symbols > 1 else 1.0
    if max_possible_entropy_change == 0: # Should not happen for num_symbols > 1
        max_possible_entropy_change = 1.0


    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue
            
            max_entropy_reduction_for_cell = -float('inf')
            for p_val in potential_numbers_to_place:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val
                temp_grid_values = [int(val) for val in temp_grid.flatten()]
                entropy_after = MathUtils().get_entropy(temp_grid_values)
                entropy_reduction = entropy_before - entropy_after
                if entropy_reduction > max_entropy_reduction_for_cell:
                    max_entropy_reduction_for_cell = entropy_reduction
            
            if max_entropy_reduction_for_cell == -float('inf'): # No valid p_val or no reduction
                max_entropy_reduction_for_cell = 0.0
            
            # Normalize positive reductions from 0 to max_possible_entropy_change
            scores[r_idx, c_idx] = MathUtils().normalize_value(max_entropy_reduction_for_cell, 0, max_possible_entropy_change, clamp=True)
    return scores

# 22. EXT_GM16_Harmonic_Centrality_Vec (調和中心性)
def EXT_GM16_Harmonic_Centrality_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM16 - 調和中心性)
    核心規則:應用圖論中的調和中心性概念,評估盤面上各空格節點的重要性。
    調和中心性是一個節點到所有其他節點距離倒數的總和。
    目的:偏好那些在盤面「網絡」中更具中心性的空格。
    啟發式類型: 圖論中心性
    輸出詮釋:分數越高表示該空格在圖結構中越「中心」(平均而言離其他格子越近)。
    強化:確保在計算調和中心性時對單元格的處理正確,並處理邊界情況。
    """
    effective_request_id = request_id or "N/A_brain_GM16"
    logger.debug("Executing EXT_GM16_Harmonic_Centrality_Vec", extra={'request_id': effective_request_id})
    
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0 or (rows * cols) <= 1: # Needs more than 1 cell
        return scores

    max_hc_heuristic = float(rows * cols - 1) # Max possible if dist 1 to all others
    if max_hc_heuristic == 0:
        max_hc_heuristic = 1.0

    for r_eval in range(rows):
        for c_eval in range(cols):
            if grid[r_eval, c_eval] != -1:  # Only score empty cells
                continue
            
            current_harmonic_centrality = 0.0
            num_other_nodes = 0
            for r_other in range(rows):
                for c_other in range(cols):
                    if r_eval == r_other and c_eval == c_other:
                        continue
                    # Considers all cells (empty or filled) as nodes for centrality
                    dist = MathUtils().manhattan_distance((r_eval, c_eval), (r_other, c_other))
                    if dist > 0:
                        current_harmonic_centrality += 1.0 / dist
                    num_other_nodes += 1
            
            if num_other_nodes == 0: # Should be caught by (rows*cols) <= 1
                scores[r_eval, c_eval] = 0.0
            else:
                scores[r_eval, c_eval] = MathUtils().normalize_value(current_harmonic_centrality, 0, max_hc_heuristic, clamp=True)
    return scores

# 23. EXT_GM17_Entropy_Minimization_Vec (局部熵最小化)
def EXT_GM17_Entropy_Minimization_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM17 - 局部熵最小化)
    核心規則:評估填入數字後,盤面局部鄰域「熵」(無序度)的降低程度。
    目的:偏好那些能使其直接周圍環境更有規律、更「有序」的填補。
    啟發式類型:資訊理論啟發(基於局部熵變)
    輸出詮釋:分數越高表示填入該數字後,其局部鄰域的熵降低得越多(局部更有序)。
    強化:精確計算局部熵變化,包括空單元格作為符號,以更全面地評估有序性。
    """
    effective_request_id = request_id or "N/A_brain_GM17"
    logger.debug("Executing EXT_GM17_Entropy_Minimization_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not potential_numbers_to_place:
        return scores

    radius = 1
    num_cells_in_neighborhood = (2 * radius + 1)**2
    # Max entropy change is related to log2 of number of distinct symbols in neighborhood
    # Using log2(num_cells_in_neighborhood) as a rough upper bound for change
    max_local_entropy_change = math.log2(num_cells_in_neighborhood) if num_cells_in_neighborhood > 1 else 1.0
    if max_local_entropy_change == 0:
        max_local_entropy_change = 1.0
    
    def val_func_for_entropy(x_val: int) -> int:
        return int(x_val) # Includes -1 as a symbol

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue

            # Calculate entropy of local neighborhood *before* placement (r_idx,c_idx is -1)
            # Make a temporary grid copy to ensure (r_idx, c_idx) is treated as -1 for this calculation
            temp_grid_before = grid.copy()
            temp_grid_before[r_idx, c_idx] = -1 # Explicitly ensure it's -1 for 'before' state
            values_before_placement_local = BoardAnalyzerUtils().get_neighborhood_values(
                temp_grid_before, r_idx, c_idx, radius=radius, eight_connectivity=True,
                val_func=val_func_for_entropy, include_center=True
            )
            entropy_before_local = MathUtils().get_entropy(values_before_placement_local)

            max_entropy_reduction_for_cell = -float('inf')
            for p_val in potential_numbers_to_place:
                temp_grid_local_place = grid.copy() # Start from original grid
                temp_grid_local_place[r_idx, c_idx] = p_val
                
                values_after_placement_local = BoardAnalyzerUtils().get_neighborhood_values(
                    temp_grid_local_place, r_idx, c_idx, radius=radius, eight_connectivity=True,
                    val_func=val_func_for_entropy, include_center=True
                )
                entropy_after_local = MathUtils().get_entropy(values_after_placement_local)
                entropy_reduction = entropy_before_local - entropy_after_local
                if entropy_reduction > max_entropy_reduction_for_cell:
                    max_entropy_reduction_for_cell = entropy_reduction
            
            if max_entropy_reduction_for_cell == -float('inf'):
                max_entropy_reduction_for_cell = 0.0
            
            scores[r_idx, c_idx] = MathUtils().normalize_value(max_entropy_reduction_for_cell, 0, max_local_entropy_change, clamp=True)
    return scores

# 24. EXT_GM18_RL_Value_Est_Vec (類強化學習價值估計)
def EXT_GM18_RL_Value_Est_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM18-類強化學習價值估計)
    核心規則:基於一組預定義的「理想特徴」來評估某個填補動作的啟發式長期潜在價值。
    此為簡化版,模擬從歷史數據學習到的偏好。
    目的:偏好那些能夠使盤面展現更多理想特徵(如形成特定序列、達到特定盤面密度等)的填補。
    啟發式類型:狀態價值啟發(基於盤面特徵計數)
    輸出詮釋:分數越高表示填入該數字後,盤面呈現的理想特徵越多,預期長期回報越大。
    強化:細化特徵權重,並增加更多對盤面整體結構的考量。
    """
    effective_request_id = request_id or "N/A_brain_GM18"
    logger.debug("Executing EXT_GM18_RL_Value_Est_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not potential_numbers_to_place:
        return scores

    FEATURE_WEIGHTS = {
        "identical_3": 1.0,
        "arithmetic_3": 0.7,
        "board_density_factor": 0.2,
        "central_control_boost": 0.15,
        "edge_affinity_boost": 0.05,
    }
    # Max heuristic: 4 directions * (identical + arithmetic) + density + central + edge
    max_heuristic_feature_score = (4 * (FEATURE_WEIGHTS["identical_3"] + FEATURE_WEIGHTS["arithmetic_3"])) + \
                                  FEATURE_WEIGHTS["board_density_factor"] + \
                                  FEATURE_WEIGHTS["central_control_boost"] + \
                                  FEATURE_WEIGHTS["edge_affinity_boost"]
    if max_heuristic_feature_score == 0:
        max_heuristic_feature_score = 1.0
    
    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue
            max_feature_score_for_cell = 0.0
            for p_val in potential_numbers_to_place:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val
                current_features_score = 0.0

                # Features 1 & 2: Lines of 3
                for dr_line, dc_line in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    for offset in range(-2, 1): # Window start to ensure p_val is part of line
                        line_values: list[int] = []
                        is_valid_line = True
                        involved_pval = False 
                        
                        current_line_coords = [] # For checking p_val involvement
                        for i in range(3):
                            check_r, check_c = r_idx + (offset + i) * dr_line, c_idx + (offset + i) * dc_line
                            current_line_coords.append((check_r, check_c))
                            if 0 <= check_r < rows and 0 <= check_c < cols:
                                line_values.append(int(temp_grid[check_r, check_c]))
                            else:
                                is_valid_line = False
                                break
                        
                        # Ensure p_val at (r_idx, c_idx) is part of the current 3-cell line segment
                        if (r_idx,c_idx) in current_line_coords:
                            involved_pval = True

                        if is_valid_line and involved_pval and len(line_values) == 3 and all(v != -1 for v in line_values):
                            s = line_values
                            if s[0] == s[1] and s[1] == s[2]:
                                current_features_score += FEATURE_WEIGHTS["identical_3"]
                            elif (s[1] - s[0]) == (s[2] - s[1]) and (s[1] - s[0]) != 0:
                                current_features_score += FEATURE_WEIGHTS["arithmetic_3"]
                
                # Feature 3: Board density
                num_filled_after_placement = np.count_nonzero(temp_grid != -1)
                density_after_placement = num_filled_after_placement / (rows * cols) if (rows * cols) > 0 else 0.0
                current_features_score += FEATURE_WEIGHTS["board_density_factor"] * density_after_placement

                # Conceptual Features (Centrality, Edge Affinity)
                if rows > 1 and cols > 1:
                    center_r, center_c = (rows - 1) / 2.0, (cols - 1) / 2.0
                    dist_to_center = MathUtils().euclidean_distance((float(r_idx), float(c_idx)), (center_r, center_c))
                    max_dist_center_heuristic = MathUtils().euclidean_distance((0.0,0.0), (center_r, center_c))
                    if max_dist_center_heuristic == 0: max_dist_center_heuristic = 1.0 # Avoid div by zero for 1x1 like case

                    current_features_score += FEATURE_WEIGHTS["central_control_boost"] * \
                        (1 - MathUtils().normalize_value(dist_to_center, 0, max_dist_center_heuristic, clamp=True))
                    
                    dist_to_edge = min(r_idx, rows - 1 - r_idx, c_idx, cols - 1 - c_idx)
                    max_min_dist_to_edge = min((rows - 1) // 2, (cols - 1) // 2)
                    if max_min_dist_to_edge > 0 :
                         current_features_score += FEATURE_WEIGHTS["edge_affinity_boost"] * \
                            (1- MathUtils().normalize_value(float(dist_to_edge), 0, float(max_min_dist_to_edge), clamp=True))
                    elif max_min_dist_to_edge == 0 and dist_to_edge == 0 : # On edge for 1xN or Nx1 grid
                        current_features_score += FEATURE_WEIGHTS["edge_affinity_boost"] * 1.0


                if current_features_score > max_feature_score_for_cell:
                    max_feature_score_for_cell = current_features_score
            
            scores[r_idx, c_idx] = MathUtils().normalize_value(max_feature_score_for_cell, 0, max_heuristic_feature_score, clamp=True)
    return scores

# 25. EXT_GM19_Masked_Number_Skip_Pattern_Vec(遮罩數字跳格模式向量)
def EXT_GM19_Masked_Number_Skip_Pattern_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM19-遮罩數字跳格模式向量)
    核心規則:分析已揭示數字的「跳格模式」(其實際位置與預期基礎位置的偏差),並對符合主導跳格模式的空格進行評分。
    啟發式類型:空間模式匹配(基於全局偏移量)
    輸出詮釋:分數越高表示該空格若填入特定數字,能與盤面上觀察到的主要「跳格」規律性最為吻合。
    強化:精確計算跳格模式並評估其強度,確保模式識別的準確性。
    """
    effective_request_id = request_id or "N/A_brain_GM19"
    logger.debug("Executing EXT_GM19_Masked_Number_Skip_Pattern_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    revealed_numbers_info: list[dict[str, int]] = [
        {'value': int(grid[r, c]), 'r': r, 'c': c}
        for r in range(rows) for c in range(cols)
        if grid[r, c] != -1 and grid[r, c] > 0
    ]
    if not revealed_numbers_info:
        return scores

    expected_max_number_on_card = BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols))
    base_positions: dict[int, tuple[int, int]] = {}
    for k_val in range(1, expected_max_number_on_card + 1):
        base_r = (k_val - 1) // cols
        base_c = (k_val - 1) % cols
        if base_r < rows: # Ensure base position is within grid
            base_positions[k_val] = (base_r, base_c)

    skip_vectors: dict[int, tuple[int, int]] = {}
    for rn_info in revealed_numbers_info:
        val = rn_info['value']
        if val in base_positions:
            expected_r, expected_c = base_positions[val]
            skip_vectors[val] = (rn_info['r'] - expected_r, rn_info['c'] - expected_c)
    
    if not skip_vectors:
        return scores

    dominant_skip_patterns_strength: dict[tuple[int, int], float] = {}
    skip_vector_tuples_list = list(skip_vectors.values())
    if not skip_vector_tuples_list: return scores # Should be caught by `if not skip_vectors`

    counts = Counter(skip_vector_tuples_list)
    min_occurrences_for_pattern = max(1, int(len(skip_vector_tuples_list) * 0.05))

    for skip_vec_tuple, count_val in counts.most_common():
        if count_val >= min_occurrences_for_pattern:
            pattern_strength = MathUtils().normalize_value(
                float(count_val), float(min_occurrences_for_pattern), float(len(skip_vector_tuples_list)), clamp=True
            )
            dominant_skip_patterns_strength[skip_vec_tuple] = pattern_strength
        else:
            break
    
    if not dominant_skip_patterns_strength:
        return scores

    potential_numbers_to_place_set = BoardAnalyzerUtils().get_legal_values_for_placement(grid)
    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:
                continue
            cell_max_pattern_score = 0.0
            for p_val_test in potential_numbers_to_place_set:
                if p_val_test not in base_positions:
                    continue
                base_r_test, base_c_test = base_positions[p_val_test]
                for current_skip_pattern, pattern_str in dominant_skip_patterns_strength.items():
                    skip_dr, skip_dc = current_skip_pattern
                    predicted_r = base_r_test + skip_dr
                    predicted_c = base_c_test + skip_dc
                    if predicted_r == r_idx and predicted_c == c_idx:
                        current_score_fit = pattern_str
                        if current_score_fit > cell_max_pattern_score:
                            cell_max_pattern_score = current_score_fit
            scores[r_idx, c_idx] = cell_max_pattern_score
    return scores

# 26. EXT_GM20_Skip_Pattern_Confidence_Vec(跳格模式信心度/規律性增強)
def EXT_GM20_Skip_Pattern_Confidence_Vec(grid: np.ndarray, request_id: str | None = "N/A", **kwargs: Any) -> np.ndarray:
    """
    (GM20-跳格模式信心度/規律性增強)
    核心規則:評估在空格填入數字是否能增強或完成已觀察到的全局跳格規律性,
    特別是當這個填補能使遵循跳格模式的數字序列更完整或更具算術規律性時。
    啟發式類型:序列完成與模式確認(基於全局偏移量)
    輸出詮釋:分數越高表示填入該數字不僅符合跳格模式的幾何位置,
    且能使該模式下的數字序列在算術/序列意義上更為「自信」或「完整」。
    強化:引入「算術序列增強」邏輯,確保跳格模式與數字序列的算術關係結合評分。
    """
    effective_request_id = request_id or "N/A_brain_GM20"
    logger.debug("Executing EXT_GM20_Skip_Pattern_Confidence_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    revealed_numbers_info_gm20: list[dict[str, int]] = []
    for r_gm20 in range(rows):
        for c_gm20 in range(cols):
            if grid[r_gm20, c_gm20] != -1 and grid[r_gm20, c_gm20] > 0:
                revealed_numbers_info_gm20.append({'value': int(grid[r_gm20, c_gm20]), 'r': r_gm20, 'c': c_gm20})
    
    if not revealed_numbers_info_gm20: return scores

    expected_max_num_gm20 = BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols))
    base_pos_gm20: dict[int, tuple[int, int]] = {
        k: ((k - 1) // cols, (k - 1) % cols) 
        for k in range(1, expected_max_num_gm20 + 1) 
        if ((k - 1) // cols) < rows
    }
    
    skip_vecs_initial_gm20: dict[int, tuple[int, int]] = {}
    for rn_gm20 in revealed_numbers_info_gm20:
        val_gm20 = rn_gm20['value']
        if val_gm20 in base_pos_gm20:
            skip_vecs_initial_gm20[val_gm20] = (
                rn_gm20['r'] - base_pos_gm20[val_gm20][0], 
                rn_gm20['c'] - base_pos_gm20[val_gm20][1]
            )
    
    dominant_patterns_details_gm20: list[dict[str, Any]] = []
    if skip_vecs_initial_gm20:
        skip_tuples_list_gm20 = list(skip_vecs_initial_gm20.values())
        counts_gm20 = Counter(skip_tuples_list_gm20)
        min_occ_gm20 = max(1, int(len(skip_tuples_list_gm20) * 0.05))
        for skip_v_gm20, count_v_gm20 in counts_gm20.most_common():
            if count_v_gm20 >= min_occ_gm20:
                pattern_vals_gm20 = sorted([
                    val for val, sv_tuple in skip_vecs_initial_gm20.items() if sv_tuple == skip_v_gm20
                ])
                p_strength_gm20 = MathUtils().normalize_value(
                    float(count_v_gm20), float(min_occ_gm20), float(len(skip_tuples_list_gm20)), clamp=True
                )
                dominant_patterns_details_gm20.append({
                    'skip': skip_v_gm20, 
                    'values': pattern_vals_gm20, 
                    'strength': p_strength_gm20
                })
            else:
                break
    
    if not dominant_patterns_details_gm20: return scores

    potential_nums_to_place_gm20 = BoardAnalyzerUtils().get_legal_values_for_placement(grid)
    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_confidence_score_for_cell_gm20 = 0.0
            for p_val_test in potential_nums_to_place_gm20:
                if p_val_test not in base_pos_gm20: continue
                base_r_t, base_c_t = base_pos_gm20[p_val_test]
                current_max_conf_for_pval = 0.0

                for pattern_detail in dominant_patterns_details_gm20:
                    pat_skip_dr, pat_skip_dc = pattern_detail['skip']
                    pat_existing_vals = pattern_detail['values'] # sorted list
                    pat_strength = pattern_detail['strength']

                    predicted_r_for_pval = base_r_t + pat_skip_dr
                    predicted_c_for_pval = base_c_t + pat_skip_dc

                    if predicted_r_for_pval == r_idx and predicted_c_for_pval == c_idx: # Geometrically fits
                        enhancement_factor = 0.5 # Base for geometric fit
                        
                        if len(pat_existing_vals) >= 1:
                            temp_sequence_with_pval = sorted(pat_existing_vals + [p_val_test])
                            if len(temp_sequence_with_pval) >= 2:
                                diffs_in_temp_seq = np.diff(temp_sequence_with_pval)
                                if len(diffs_in_temp_seq) > 0:
                                    is_arithmetic_now = len(set(diffs_in_temp_seq)) == 1
                                    first_diff = diffs_in_temp_seq[0]
                                    if is_arithmetic_now and first_diff != 0:
                                        enhancement_factor += 0.4 # Strong enhancement
                                        # Bonus if p_val_test is between min/max and fills internal gap
                                        if len(pat_existing_vals) >=2 and \
                                           min(pat_existing_vals) < p_val_test < max(pat_existing_vals):
                                            enhancement_factor += 0.1 

                        current_conf = pat_strength * enhancement_factor
                        if current_conf > current_max_conf_for_pval:
                            current_max_conf_for_pval = current_conf
                
                if current_max_conf_for_pval > max_confidence_score_for_cell_gm20:
                    max_confidence_score_for_cell_gm20 = current_max_conf_for_pval
            
            scores[r_idx, c_idx] = MathUtils().normalize_value(max_confidence_score_for_cell_gm20, 0, 1.0, clamp=True) # Max possible enhanced strength is ~1.0
    return scores

# --- Module Registration
REGISTERED_MODULES_BRAIN = {
    "EXT_A2_Weighted_Proximity_Vec": EXT_A2_Weighted_Proximity_Vec,
    "EXT_M3_Local_Heterogeneity_Vec": EXT_M3_Local_Heterogeneity_Vec,
    "EXT_D3_Potential_Field_Vec": EXT_D3_Potential_Field_Vec,
    "EXT_F10_Discontinuity_Vec": EXT_F10_Discontinuity_Vec,
    "EXT_P7_Pathfinding_Value_Vec": EXT_P7_Pathfinding_Value_Vec,
    "EXT_R5_Resource_Control_Vec": EXT_R5_Resource_Control_Vec,
    "EXT_GM1_Row_Control_Vec": EXT_GM1_Row_Control_Vec,
    "EXT_GM2_Col_Flow_Vec": EXT_GM2_Col_Flow_Vec,
    "EXT_GM3_Adv_Connected_Comp_Vec": EXT_GM3_Adv_Connected_Comp_Vec,
    "EXT_GM4_Spatial_Auto_Corr_Vec": EXT_GM4_Spatial_Auto_Corr_Vec,
    "EXT_GM5_Line_Completion_Vec": EXT_GM5_Line_Completion_Vec,
    "EXT_GM6_Symmetry_Potential_Vec": EXT_GM6_Symmetry_Potential_Vec,
    "EXT_GM7_Numeric_Gaps_Vec": EXT_GM7_Numeric_Gaps_Vec,
    "EXT_GM8_Edge_Affinity_Vec": EXT_GM8_Edge_Affinity_Vec,
    "EXT_GM9_Center_Control_Vec": EXT_GM9_Center_Control_Vec,
    "EXT_GM10_Blocking_Value_Vec": EXT_GM10_Blocking_Value_Vec,
    "EXT_GM11_Pair_Correlation_Vec": EXT_GM11_Pair_Correlation_Vec,
    "EXT_GM12_Island_Analysis_Vec": EXT_GM12_Island_Analysis_Vec,
    "EXT_GM13_Sequence_Diversity_Vec": EXT_GM13_Sequence_Diversity_Vec,
    "EXT_GM14_Risk_Assessment_Vec": EXT_GM14_Risk_Assessment_Vec,
    "EXT_GM15_Information_Gain_Vec": EXT_GM15_Information_Gain_Vec,
    "EXT_GM16_Harmonic_Centrality_Vec": EXT_GM16_Harmonic_Centrality_Vec,
    "EXT_GM17_Entropy_Minimization_Vec": EXT_GM17_Entropy_Minimization_Vec,
    "EXT_GM18_RL_Value_Est_Vec": EXT_GM18_RL_Value_Est_Vec,
    "EXT_GM19_Masked_Number_Skip_Pattern_Vec": EXT_GM19_Masked_Number_Skip_Pattern_Vec,
    "EXT_GM20_Skip_Pattern_Confidence_Vec": EXT_GM20_Skip_Pattern_Confidence_Vec,
}

# Verification (Optional - for testing brain.py directly)
if __name__ == '__main__':
    print("Verifying brain.py structure...")
    # Configure basic logging for testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    dummy_grid_empty = np.array([
        [-1, -1, -1],
        [-1, -1, -1],
        [-1, -1, -1]
    ])
    dummy_grid_filled = np.array([
        [1, 2, -1],
        [-1, 10, 5], # Changed 1 to 10 to avoid duplicate for some tests
        [3, -1, 4]
    ])
    print(f"Created dummy grid (empty):\n{dummy_grid_empty}")
    print(f"Created dummy grid (filled):\n{dummy_grid_filled}")

    module_to_test = "EXT_A2_Weighted_Proximity_Vec"
    print(f"\nTesting get_module_score with '{module_to_test}' on filled grid...")
    try:
        scores = get_module_score(module_to_test, dummy_grid_filled, request_id="test_A2")
        print(f"Successfully called {module_to_test}. Output:\n{scores}")
        assert isinstance(scores, np.ndarray), "Return type is not np.ndarray"
        assert scores.shape == dummy_grid_filled.shape, "Return shape does not match grid shape"
        assert scores.dtype == float, "Return dtype is not float"
    except Exception as e:
        print(f"Error testing {module_to_test}: {e}")
        logger.exception(f"Exception during {module_to_test} test")


    print("\nTesting EXT_GM1_Row_Control_Vec with a specific scenario...")
    grid_gm1_test = np.array([
        [1, -1, 3],
        [-1, 5, -1],
        [7, -1, 9]
    ])
    try:
        scores_gm1 = get_module_score("EXT_GM1_Row_Control_Vec", grid_gm1_test, request_id="test_GM1")
        print(f"Scores for EXT_GM1_Row_Control_Vec:\n{scores_gm1}")
    except Exception as e:
        print(f"Error testing EXT_GM1_Row_Control_Vec: {e}")
        logger.exception("Exception during EXT_GM1_Row_Control_Vec test")

    print("\nTesting EXT_F10_Discontinuity_Vec for sequence completion...")
    grid_f10_test = np.array([
        [2, -1, 6], # Potential arithmetic: 2, 4, 6
        [-1, -1, -1],
        [10, -1, 8]  # Potential arithmetic: 10, 9, 8 or 10, 12, 14 (if 12 placed) etc.
    ])
    try:
        scores_f10 = get_module_score("EXT_F10_Discontinuity_Vec", grid_f10_test, request_id="test_F10")
        print(f"Scores for EXT_F10_Discontinuity_Vec:\n{scores_f10}")
    except Exception as e:
        print(f"Error testing EXT_F10_Discontinuity_Vec: {e}")
        logger.exception("Exception during EXT_F10_Discontinuity_Vec test")
    
    non_existent_module = "EXT_XXX_NonExistentModule"
    print(f"\nTesting get_module_score with non-existent module '{non_existent_module}'...")
    try:
        scores_non_existent = get_module_score(non_existent_module, dummy_grid_filled, request_id="test_XXX")
        # This part should ideally not be reached if error handling in get_module_score is correct (returns zeros)
        print(f"Output for non-existent module (should be zeros):\n{scores_non_existent}")
        assert np.all(scores_non_existent == 0), "Expected zeros for non-existent module"
    except Exception as e: # Should not raise an exception here if get_module_score handles it
        print(f"Error testing non-existent module: {e}")
        logger.exception("Exception during non-existent module test")

    print("\nListing all registered modules:")
    for i, name in enumerate(REGISTERED_MODULES_BRAIN.keys()):
        print(f"{i + 1}. {name}")
    print(f"\nTotal modules registered: {len(REGISTERED_MODULES_BRAIN)}")

    print("\nTesting all registered modules with the empty grid (basic run test):")
    all_modules_passed = True
    for module_name in REGISTERED_MODULES_BRAIN:
        print(f"  Testing module: {module_name}")
        try:
            test_scores = get_module_score(module_name, dummy_grid_empty, request_id=f"test_empty_{module_name}")
            assert isinstance(test_scores, np.ndarray), f"{module_name} did not return np.ndarray"
            assert test_scores.shape == dummy_grid_empty.shape, f"{module_name} return shape mismatch"
            assert test_scores.dtype == float, f"{module_name} return dtype is not float"
            print(f"    {module_name} passed basic checks.")
        except Exception as e:
            print(f"    ERROR testing {module_name}: {e}")
            logger.exception(f"Exception during {module_name} (empty grid) test")
            all_modules_passed = False
    
    if all_modules_passed:
        print("All registered modules passed basic execution checks with empty grid.")
    else:
        print("Some modules failed basic execution checks with empty grid.")

    print("\nbrain.py verification complete.")