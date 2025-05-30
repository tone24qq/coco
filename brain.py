import numpy as np
import math
from collections import Counter, deque
import logging
import random
from typing import List, Dict, Tuple, Callable, Optional, Any

#--- Logging Configuration
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
                # 如果你只是想留註解,請改成這樣
                return 0.0
            else: # value > max_val (which is min_val)
                return 1.0

        normalized = (value - min_val) / (max_val - min_val)
        if clamp:
            return max(0.0, min(1.0, normalized))
        return normalized

    def manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """Calculates Manhattan distance between two points (r, c)."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def euclidean_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float: #參數類型也可能是float
        """Calculates Euclidean distance between two points (r, c)."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def get_entropy(self, values: List[Any]) -> float:
        """Calculates Shannon entropy for a list of values."""
        if not values:
            return 0.0
        counts = Counter(values)
        total_count = len(values)
        entropy = 0.0
        for count in counts.values():
            probability = count / total_count
            if probability > 0: # 避免 math.log2(0)
                entropy -= probability * math.log2(probability)
        return entropy

class BoardAnalyzerUtils:
    """
    Provides common board analysis utility functions.
    Used by modules to inspect grid neighborhoods, gradients, etc.
    """
    def get_neighborhood_values(self, grid: np.ndarray, r: int, c: int, radius: int = 1,
                                eight_connectivity: bool = True,
                                val_func: Callable[[int], Optional[float]] = lambda x_val: float(x_val) if x_val != -1 else None,
                                include_center: bool = False) -> List[float]:
        """
        Retrieves values from the neighborhood of a cell.
        Supports configurable radius, connectivity, and value processing.
        """
        neighbors: List[float] = []
        rows, cols = grid.shape
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if not include_center and dr == 0 and dc == 0:
                    continue
                if not eight_connectivity:
                    if radius == 1 and abs(dr) + abs(dc) != 1:
                        continue
                    # elif radius > 1 and abs(dr) + abs(dc) > radius: # This condition might be too restrictive for larger radii if only 4-connectivity is desired for a specific radius
                    # For general N-radius 4-connectivity, it's usually defined as abs(dr) + abs(dc) <= radius and (dr==0 or dc==0 if strictly cardinal)
                    # But the original code for radius > 1 with eight_connectivity=False was:
                    # elif radius > 1 and abs(dr)+abs(dc)>radius: continue #This seems to be an error, it should be related to 4-connectivity
                    # For 4-connectivity (von Neumann neighborhood) usually it's abs(dr) + abs(dc) == 1 for radius 1
                    # or abs(dr) + abs(dc) <= radius and (dr * dc == 0) if you want a "cross" shape for larger radii.
                    # Given the context, if not eight_connectivity, it implies 4-connectivity.
                    # For radius 1, abs(dr) + abs(dc) != 1 correctly filters out diagonals and center.
                    # For radius > 1, if 4-connectivity means only cells on axes, then:
                    if radius > 1 and dr != 0 and dc != 0: # Exclude diagonals for 4-connectivity
                        continue
                nr, nc = r + dr, c + dc

                if 0 <= nr < rows and 0 <= nc < cols:
                    cell_value = grid[nr, nc] # Store to avoid multiple lookups
                    if callable(val_func): # Ensure val_func is callable
                        processed_val = val_func(cell_value)
                        if processed_val is not None:
                            neighbors.append(processed_val)
                    elif cell_value != -1: # Default behavior if val_func is not as expected or for simpler cases
                         neighbors.append(float(cell_value))

        return neighbors

    def get_value_gradient_at_cell(self, grid: np.ndarray, r: int, c: int,
                                     val_func: Callable[[int], float] = lambda x_val: float(x_val) if x_val != -1 else 0.0) -> Tuple[float, float]:
        """
        Calculates an approximate gradient (Sobel-like) at a cell. Useful for modules
        analyzing value changes.
        """
        rows, cols = grid.shape

        def safe_val(r_in, c_in):
            if 0 <= r_in < rows and 0 <= c_in < cols:
                return val_func(grid[r_in, c_in])
            return 0.0

        gx = (safe_val(r - 1, c + 1) + 2 * safe_val(r, c + 1) + safe_val(r + 1, c + 1)) - \
             (safe_val(r - 1, c - 1) + 2 * safe_val(r, c - 1) + safe_val(r + 1, c - 1))
        gy = (safe_val(r + 1, c - 1) + 2 * safe_val(r + 1, c) + safe_val(r + 1, c + 1)) - \
             (safe_val(r - 1, c - 1) + 2 * safe_val(r - 1, c) + safe_val(r - 1, c + 1))
        return gx, gy

    def find_sequences_in_line(self, line: List[int], min_len: int = 3,
                               check_arithmetic: bool = True, check_geometric: bool = False,
                               allow_gaps: int = 0) -> List[List[int]]: #Return type was List[List[int]], but code appends List[int] or List[float]
        """
        Finds arithmetic or geometric sequences in a 1D list of numbers,
        supporting gaps and returning sequence elements.
        強化:提升算術序列檢測的彈性,能識別更多複雜的算術序列模式(負公差,跨零點,常
        數序列的明確處理)。
        同時返回找到的序列、類型和公差/比率。
        """
        sequences: List[List[Any]] = [] # Changed to List[Any] to accommodate floats from geometric
        n = len(line)
        if n < min_len:
            return sequences

        for i in range(n):
            if line[i] == -1:
                continue

            # Arithmetic sequence check
            if check_arithmetic:
                # Iterate through possible common differences for sequences starting at i
                for j in range(i + 1, n):
                    current_seq_values: List[Union[int, float]] # For type checker
                    current_seq_indices: List[int]

                    if line[j] == -1:
                        if allow_gaps > 0:
                            temp_gap_count = 0
                            # Try to find the next non-gap number to establish diff
                            for k_loop_var in range(j, n):
                                if line[k_loop_var] == -1:
                                    temp_gap_count += 1
                                else:
                                    if temp_gap_count <= allow_gaps:
                                        diff = line[k_loop_var] - line[i]
                                        if diff == 0 and line[i] != 0: # Not a strict arithmetic sequence for general purpose
                                            break 
                                        current_seq_values = [line[i], line[k_loop_var]]
                                        current_seq_indices = [i, k_loop_var]
                                        potential_gap_count_inner = temp_gap_count
                                        # Extend sequence
                                        for l_idx in range(k_loop_var + 1, n):
                                            if line[l_idx] == -1:
                                                potential_gap_count_inner += 1
                                                if potential_gap_count_inner > allow_gaps:
                                                    break
                                                continue
                                            expected_next = current_seq_values[-1] + diff
                                            if math.isclose(float(line[l_idx]), float(expected_next)):
                                                current_seq_values.append(line[l_idx])
                                                current_seq_indices.append(l_idx)
                                                potential_gap_count_inner = 0 # Reset gap count
                                            elif line[l_idx] != -1: # Sequence broken by a different number
                                                break
                                        if len(current_seq_values) >= min_len:
                                            sequences.append(list(current_seq_values)) # Ensure it's a list copy
                                    break # Done trying to establish diff from k_loop_var
                            else: # Inner k_loop_var loop didn't break
                                break # No non-gap number found after j to establish diff, break from j loop
                            continue # To next j, as current j was a gap
                        else: # allow_gaps is 0, so line[j] == -1 breaks sequence
                            break 

                    # line[j] is not -1, establish diff
                    diff = line[j] - line[i]
                    if diff == 0 and line[i] != 0: # Exclude constant non-zero sequences as arithmetic by default
                        continue # Try next j to form a sequence with line[i]

                    current_seq_values = [line[i], line[j]]
                    current_seq_indices = [i, j]
                    potential_gap_count = 0
                    for k in range(j + 1, n):
                        if line[k] == -1:
                            potential_gap_count += 1
                            if potential_gap_count > allow_gaps:
                                break
                            continue
                        expected_next = current_seq_values[-1] + diff
                        if math.isclose(float(line[k]), float(expected_next)):
                            current_seq_values.append(line[k])
                            current_seq_indices.append(k)
                            potential_gap_count = 0 # Reset gap count after finding a valid number
                        elif line[k] != -1: # Sequence broken by a different number
                            break
                    if len(current_seq_values) >= min_len:
                        sequences.append(list(current_seq_values)) # Ensure it's a list copy
            
            # Geometric sequence check
            if check_geometric and line[i] != 0: # Starting with 0 can be tricky for ratio unless it's all 0s
                current_geo_seq_values: List[Union[int, float]] = [float(line[i])]
                current_geo_seq_indices: List[int] = [i]
                potential_gap_count = 0
                ratio: Optional[float] = None

                for j in range(i + 1, n):
                    if line[j] == -1:
                        potential_gap_count += 1
                        if potential_gap_count > allow_gaps:
                            break
                        continue
                    
                    # Handling zero in geometric sequence
                    if line[j] == 0:
                        # If ratio is already established and not 0, sequence broken
                        # If ratio is None, and current_geo_seq_values[-1] is 0, can continue with ratio 0 or 1
                        if ratio is not None and not math.isclose(ratio,0.0): 
                            break
                        if current_geo_seq_values[-1] == 0: # 0, 0 sequence
                            ratio = 1.0 # Or handle as a special case of constant zeros
                        else: # Non-zero, 0. Ratio must be 0.
                            ratio = 0.0
                    
                    current_val_float = float(line[j])
                    last_val_float = float(current_geo_seq_values[-1])

                    if ratio is None:
                        if math.isclose(last_val_float, 0.0): # Should have been line[i] == 0
                             # if current_val_float is also 0, ratio can be 1 (constant 0s)
                             # if current_val_float is non-zero, this is problematic (0, X) unless we allow infinite ratio
                            if math.isclose(current_val_float, 0.0):
                                ratio = 1.0 # for 0,0,0 sequence
                            else:
                                break # Cannot start a geometric sequence from 0 to non-zero with finite ratio
                        else:
                            ratio = current_val_float / last_val_float
                            # Optional: Check if ratio is integer-like if dealing with integer sequences primarily
                            # if not math.isclose(ratio, round(ratio)) and (line[j] % line[i] != 0 and line[i] % line[j] != 0):
                            #    break

                    # Avoid constant sequences unless they are all identical (which `isclose` will handle)
                    # The original code had: `if math.isclose(ratio, 1.0) and line[i] != line[j]: continue`
                    # This was outside the ratio establishment, which is problematic.
                    # If ratio is 1, it's a constant sequence.
                    
                    expected_next_float = last_val_float * ratio
                    if math.isclose(current_val_float, expected_next_float):
                        current_geo_seq_values.append(current_val_float)
                        current_geo_seq_indices.append(j)
                        potential_gap_count = 0
                    elif line[j] != -1: # Sequence broken by a different number
                        break
                
                is_constant_sequence = True
                if len(current_geo_seq_values)>1:
                    first_val = current_geo_seq_values[0]
                    for val_in_seq in current_geo_seq_values:
                        if not math.isclose(val_in_seq, first_val):
                            is_constant_sequence = False
                            break
                
                # Add if long enough AND (not constant OR if constant, it's a valid geometric sequence e.g. 0,0,0 or X,X,X)
                # The definition of "geometric" sometimes excludes constant sequences where ratio is 1, unless value is 0 or 1.
                # Here, we assume constant sequences (ratio 1) are fine if they meet length.
                if len(current_geo_seq_values) >= min_len:
                    sequences.append(list(current_geo_seq_values))
        return sequences


    def get_card_max_value_from_grid_dimensions(self, grid_shape: Tuple[int, int]) -> int:
        """Calculates the maximum possible number on the card based on its dimensions."""
        rows, cols = grid_shape
        if rows == 0 or cols == 0: return 0
        return rows * cols

    def get_all_possible_numbers_for_grid(self, grid_shape: Tuple[int, int]) -> set[int]:
        """
        Returns a set of all numbers that could theoretically appear on a grid of given
        dimensions.
        """
        max_val = self.get_card_max_value_from_grid_dimensions(grid_shape)
        if max_val == 0:
            return set()
        return set(range(1, max_val + 1))

    def get_legal_values_for_placement(self, grid: np.ndarray) -> set[int]:
        """
        Determines the set of numbers that can be legally placed onto an empty cell in the grid.
        This adheres to the rule: numbers are 1 to R*C and no positive number can be
        repeated.
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

REGISTERED_MODULES_BRAIN: Dict[str, Callable] = {}

def get_module_score(module_name: str, grid: np.ndarray, **kwargs) -> np.ndarray:
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
        logger.error(f"Module {module_name} not found in REGISTERED_MODULES_BRAIN.",
                     extra={'request_id': effective_request_id})
        rows, cols = grid.shape
        return np.zeros((rows, cols), dtype=float)

    module_func = REGISTERED_MODULES_BRAIN[module_name]
    logger.info(f"Executing module: {module_name}", extra={'request_id': effective_request_id})

    try:
        # Ensure 'request_id' is passed if the module expects it
        if "request_id" in module_func.__code__.co_varnames: # Check if module accepts request_id
             score_grid = module_func(grid, request_id=effective_request_id, **kwargs)
        else:
             score_grid = module_func(grid, **kwargs)
        return score_grid
    except Exception as e:
        logger.error(f"Error executing module {module_name}: {e}", exc_info=True,
                     extra={'request_id': effective_request_id})
        rows, cols = grid.shape
        return np.zeros((rows, cols), dtype=float)

#--- Scoring Module Implementations ---

# 1. EXT_A2_Weighted_Proximity_Vec (加權鄰近性)
def EXT_A2_Weighted_Proximity_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_A2"
    logger.debug("Executing EXT_A2_Weighted_Proximity_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    radius = kwargs.get("radius", 2)
    value_weight_factor = kwargs.get("value_weight_factor", 0.15)
    distance_decay_factor = kwargs.get("distance_decay_factor", 1.8)

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:
                continue
            proximity_score = 0.0
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0: continue
                    nr, nc = r_idx + dr, c_idx + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                        dist = MathUtils().manhattan_distance((r_idx, c_idx), (nr, nc))
                        if dist == 0: dist = 1 # Should not happen if dr,dc !=0,0
                        score_contribution = (grid[nr, nc] * value_weight_factor) / (dist ** distance_decay_factor)
                        proximity_score += score_contribution
            
            max_val_on_grid = float(BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols)))
            if max_val_on_grid == 0: max_val_on_grid = 1.0
            num_neighbors_in_radius = (2 * radius + 1)**2 - 1
            heuristic_max_score = num_neighbors_in_radius * max_val_on_grid * value_weight_factor / (1**distance_decay_factor) if num_neighbors_in_radius > 0 else 1.0
            
            if heuristic_max_score > 0:
                scores[r_idx, c_idx] = MathUtils().normalize_value(proximity_score, 0, heuristic_max_score, clamp=True)
            else:
                scores[r_idx, c_idx] = 0.0
    return scores

# 2. EXT_M3_Local_Heterogeneity_Vec(局部異質性)
def EXT_M3_Local_Heterogeneity_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_M3"
    logger.debug("Executing EXT_M3_Local_Heterogeneity_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    radius = kwargs.get("radius", 1)
    min_neighbors_for_robust_score = kwargs.get("min_neighbors_for_robust_score", 2)
    all_possible_values_in_game = BoardAnalyzerUtils().get_all_possible_numbers_for_grid(grid.shape)
    if not all_possible_values_in_game: return scores

    if len(all_possible_values_in_game) > 1:
        max_theoretical_entropy = math.log2(len(all_possible_values_in_game))
    elif len(all_possible_values_in_game) == 1:
        max_theoretical_entropy = math.log2(2) # Avoid log2(1)=0
    else: 
        max_theoretical_entropy = 1.0 

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:
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
def EXT_D3_Potential_Field_Vec(grid: np.ndarray, request_id: Optional [str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_D3"
    logger.debug("Executing EXT_D3_Potential_Field_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    decay_exponent = kwargs.get("decay_exponent", 1.5)
    max_influence_radius = kwargs.get("max_influence_radius",3)
    max_possible_val_on_grid = float(BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols)))
    if max_possible_val_on_grid == 0: return scores 

    num_cells_in_radius_approx = (2 * max_influence_radius + 1)**2 - 1
    heuristic_max_potential = num_cells_in_radius_approx * (max_possible_val_on_grid / (1**decay_exponent)) if num_cells_in_radius_approx >0 else 1.0
    if heuristic_max_potential == 0: heuristic_max_potential = 1.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            current_cell_potential = 0.0
            for nr in range(rows):
                for nc in range(cols):
                    if grid[nr, nc] != -1:
                        num_val = grid[nr, nc]
                        if num_val <= 0: continue
                        dist = MathUtils().manhattan_distance((r_idx, c_idx), (nr, nc))
                        if dist == 0: continue 
                        if dist > max_influence_radius: continue
                        potential_contribution = num_val / (dist ** decay_exponent)
                        current_cell_potential += potential_contribution
            scores[r_idx, c_idx] = MathUtils().normalize_value(current_cell_potential, 0, heuristic_max_potential, clamp=True)
    return scores

# 4. EXT_F10_Discontinuity_Vec(不連續性修復/序列完成度)
def EXT_F10_Discontinuity_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_F10"
    logger.debug("Executing EXT_F10_Discontinuity_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores
    legal_values_for_placement = BoardAnalyzerUtils().get_legal_values_for_placement(grid)
    if not legal_values_for_placement: return scores

    min_sequence_len_to_score = kwargs.get("min_sequence_len_to_score",3)
    allow_gaps_for_f10 = kwargs.get("allow_gaps_for_f10",1) # Specific name to avoid collision

    heuristic_max_len = float(max(rows, cols))
    if heuristic_max_len < min_sequence_len_to_score: # Ensure max_len is at least min_len
        heuristic_max_len = float(min_sequence_len_to_score)
    if heuristic_max_len == 0: heuristic_max_len = 1.0 # Avoid division by zero

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_len_contribution_for_this_cell = 0.0
            for val_to_try in legal_values_for_placement:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = val_to_try # This should be int
                current_val_max_len = 0.0
                
                # Check Row
                row_line = [int(x) for x in temp_grid[r_idx, :]]
                sequences_in_row = BoardAnalyzerUtils().find_sequences_in_line(row_line, min_len=min_sequence_len_to_score, allow_gaps=allow_gaps_for_f10, check_arithmetic=True)
                for seq in sequences_in_row:
                    if val_to_try in seq: 
                        current_val_max_len = max(current_val_max_len, len(seq))
                # Check Column
                col_line = [int(x) for x in temp_grid[:, c_idx]]
                sequences_in_col = BoardAnalyzerUtils().find_sequences_in_line(col_line, min_len=min_sequence_len_to_score, allow_gaps=allow_gaps_for_f10, check_arithmetic=True)
                for seq in sequences_in_col:
                    if val_to_try in seq:
                        current_val_max_len = max(current_val_max_len, len(seq))
                # Check Diagonals
                diag1_line = [int(x) for x in np.diag(temp_grid, k=c_idx - r_idx)]
                sequences_in_diag1 = BoardAnalyzerUtils().find_sequences_in_line(diag1_line, min_len=min_sequence_len_to_score, allow_gaps=allow_gaps_for_f10, check_arithmetic=True)
                for seq in sequences_in_diag1:
                    if val_to_try in seq:
                        current_val_max_len = max(current_val_max_len, len(seq))
                
                flipped_temp_grid = np.fliplr(temp_grid)
                flipped_c_idx = cols - 1 - c_idx
                diag2_line = [int(x) for x in np.diag(flipped_temp_grid, k=flipped_c_idx - r_idx)]
                sequences_in_diag2 = BoardAnalyzerUtils().find_sequences_in_line(diag2_line, min_len=min_sequence_len_to_score, allow_gaps=allow_gaps_for_f10, check_arithmetic=True)
                for seq in sequences_in_diag2:
                    if val_to_try in seq:
                        current_val_max_len = max(current_val_max_len, len(seq))
                
                if current_val_max_len >= min_sequence_len_to_score:
                    max_len_contribution_for_this_cell = max(max_len_contribution_for_this_cell, current_val_max_len)
            
            scores[r_idx, c_idx] = MathUtils().normalize_value(max_len_contribution_for_this_cell, 0, heuristic_max_len, clamp=True)
    return scores

# 5. EXT_P7_Pathfinding_Value_Vec(路徑尋找價值)
def EXT_P7_Pathfinding_Value_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_P7"
    logger.debug("Executing EXT_P7_Pathfinding_Value_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores
    legal_values_for_placement = BoardAnalyzerUtils().get_legal_values_for_placement(grid)
    if not legal_values_for_placement: return scores

    max_path_search_depth = kwargs.get("max_path_search_depth", 4)
    path_value_decay_factor = kwargs.get("path_value_decay_factor",1.0)
    max_possible_val_on_grid = float(BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols)))
    if max_possible_val_on_grid == 0: max_possible_val_on_grid = 1.0
    
    # Max heuristic assuming connecting to max_val at dist 1 from (2*depth+1)^2 cells (a square area)
    # This is a very rough upper bound.
    heuristic_max_path_score = ((2 * max_path_search_depth + 1)**2 * max_possible_val_on_grid / (1**path_value_decay_factor)) if max_path_search_depth >0 else 1.0
    if heuristic_max_path_score == 0: heuristic_max_path_score = 1.0 

    for r_start in range(rows):
        for c_start in range(cols):
            if grid[r_start, c_start] != -1: continue
            max_score_for_this_cell = 0.0
            # val_to_try is not used in BFS path scoring itself, path goes through empty to existing numbers
            # The original code iterated val_to_try but didn't use it in scoring.
            # If the idea was to score based on paths FOR val_to_try, the logic would need to change.
            # Assuming the existing logic is intended: score empty cells based on paths to existing numbers.
            
            current_placement_path_score = 0.0
            q = deque([((r_start, c_start), 0)]) # ((r,c), path_len from start)
            visited_for_bfs = set([(r_start, c_start)])
            
            head_count = 0 
            # max_bfs_steps limit can be tricky. rows*cols is a reasonable limit for paths without cycles on unweighted grid.
            max_bfs_steps = rows * cols 
            
            while q and head_count < max_bfs_steps:
                head_count += 1
                (curr_r, curr_c), path_len = q.popleft()
                
                # Explore neighbors
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    next_r, next_c = curr_r + dr, curr_c + dc
                    if 0 <= next_r < rows and 0 <= next_c < cols:
                        if grid[next_r, next_c] != -1: # Found an existing number
                            reached_val = grid[next_r, next_c]
                            effective_path_len = path_len + 1 # Path from (r_start, c_start) to (next_r, next_c)
                            if effective_path_len > 0: # Ensure no division by zero if path_len somehow leads to 0
                                current_placement_path_score += reached_val / (effective_path_len ** path_value_decay_factor)
                            # Do not add to queue, this is a terminal number for this path segment
                        elif (next_r, next_c) not in visited_for_bfs and \
                             grid[next_r, next_c] == -1 and \
                             path_len + 1 < max_path_search_depth: # Path can go through other empty cells
                            visited_for_bfs.add((next_r, next_c))
                            q.append(((next_r, next_c), path_len + 1))
            
            # The original code had val_to_try loop here, but score was independent of it.
            # If score should depend on val_to_try, this logic needs rethink.
            # For now, assume the score is for the cell itself.
            max_score_for_this_cell = current_placement_path_score 
            scores[r_start, c_start] = MathUtils().normalize_value(max_score_for_this_cell, 0, heuristic_max_path_score, clamp=True)
    return scores

# 6. EXT_R5_Resource_Control_Vec(資源控制)
def EXT_R5_Resource_Control_Vec(grid: np.ndarray, request_id: Optional [str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_R5"
    logger.debug("Executing EXT_R5_Resource_Control_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores
    
    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    max_possible_val_on_grid = float(BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols)))
    if max_possible_val_on_grid == 0: max_possible_val_on_grid = 1.0
    
    hypothetical_high_val_placed = 0.0
    if potential_numbers_to_place:
        hypothetical_high_val_placed = float(np.max(potential_numbers_to_place))
    
    w_row = kwargs.get("w_row",0.3)
    w_col = kwargs.get("w_col",0.3)
    w_val = kwargs.get("w_val",0.4)

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            
            num_filled_in_row = np.count_nonzero(grid[r_idx, :] != -1)
            row_completeness_score = (num_filled_in_row + 1) / cols if cols > 0 else 0
            
            num_filled_in_col = np.count_nonzero(grid[:, c_idx] != -1)
            col_completeness_score = (num_filled_in_col + 1) / rows if rows > 0 else 0
            
            value_capture_score = 0.0
            if hypothetical_high_val_placed > 0 and max_possible_val_on_grid > 0:
                value_capture_score = MathUtils().normalize_value(hypothetical_high_val_placed, 1, max_possible_val_on_grid, clamp=True)
            
            combined_score = (w_row * row_completeness_score + 
                              w_col * col_completeness_score + 
                              w_val * value_capture_score)
            # Weights might not sum to 1, so normalize the combined_score
            # Assuming max possible for each component is 1. Max combined_score is sum of weights.
            scores[r_idx, c_idx] = MathUtils().normalize_value(combined_score, 0, w_row + w_col + w_val if (w_row + w_col + w_val) > 0 else 1.0, clamp=True)
    return scores

# 7. EXT_GM1_Row_Control_Vec(行控制力)
def EXT_GM1_Row_Control_Vec(grid: np.ndarray, request_id: Optional [str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM1"
    logger.debug("Executing EXT_GM1_Row_Control_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    avg_potential_num_to_place = 0.0
    if potential_numbers_to_place:
        avg_potential_num_to_place = float(np.mean(potential_numbers_to_place))

    max_val_board = float(BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols)))
    if max_val_board == 0: max_val_board = 1.0
    
    w_density = kwargs.get("w_density_gm1", 0.4) # Unique weight names
    w_sum_gm1 = kwargs.get("w_sum_gm1", 0.3)
    w_seq_gm1 = kwargs.get("w_seq_gm1", 0.3)
    min_len_for_seq_score = kwargs.get("min_len_for_seq_score_gm1", 3)


    for r_idx in range(rows):
        current_row_values_list = [val for val in grid[r_idx, :] if val != -1]
        num_filled_in_row = len(current_row_values_list)
        sum_current_row_values = sum(current_row_values_list)
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            
            density_score = (num_filled_in_row + 1.0) / cols if cols > 0 else 0.0
            
            potential_row_sum = sum_current_row_values + avg_potential_num_to_place
            heuristic_max_row_sum = float(cols * max_val_board) if cols > 0 else 1.0
            if heuristic_max_row_sum == 0: heuristic_max_row_sum = 1.0
            sum_score = MathUtils().normalize_value(potential_row_sum, 0, heuristic_max_row_sum, clamp=True)
            
            seq_score = 0.0
            # For sequence score, it's better to test with actual potential values
            max_seq_len_contribution = 0
            if potential_numbers_to_place:
                for p_val_test in potential_numbers_to_place:
                    temp_row_line = [int(x) for x in grid[r_idx, :]]
                    temp_row_line[c_idx] = int(p_val_test) # find_sequences_in_line expects List[int]
                    sequences_found = BoardAnalyzerUtils().find_sequences_in_line(temp_row_line, min_len=min_len_for_seq_score, allow_gaps=1, check_arithmetic=True)
                    current_max_for_pval = 0
                    for seq in sequences_found:
                        if p_val_test in seq: # Check if the placed value is part of the new sequence
                           current_max_for_pval = max(current_max_for_pval, len(seq))
                    max_seq_len_contribution = max(max_seq_len_contribution, current_max_for_pval)
            
            if max_seq_len_contribution >= min_len_for_seq_score:
                seq_score = MathUtils().normalize_value(float(max_seq_len_contribution), min_len_for_seq_score, cols if cols >0 else min_len_for_seq_score, clamp=True)
            elif max_seq_len_contribution > 0: 
                seq_score = 0.25 # Small bonus
            
            combined_score = (w_density * density_score + 
                              w_sum_gm1 * sum_score + 
                              w_seq_gm1 * seq_score)
            max_possible_combined = w_density + w_sum_gm1 + w_seq_gm1
            scores[r_idx, c_idx] = MathUtils().normalize_value(combined_score, 0, max_possible_combined if max_possible_combined > 0 else 1.0, clamp=True)
    return scores

# 8. EXT_GM2_Col_Flow_Vec (列流動性/列控制力)
def EXT_GM2_Col_Flow_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM2"
    logger.debug("Executing EXT_GM2_Col_Flow_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores
    
    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    avg_potential_num_to_place = 0.0
    if potential_numbers_to_place:
        avg_potential_num_to_place = float(np.mean(potential_numbers_to_place))

    max_val_board = float(BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols)))
    if max_val_board == 0: max_val_board = 1.0

    w_density_gm2 = kwargs.get("w_density_gm2",0.4)
    w_sum_gm2 = kwargs.get("w_sum_gm2",0.3)
    w_seq_gm2 = kwargs.get("w_seq_gm2",0.3)
    min_len_for_seq_score_gm2 = kwargs.get("min_len_for_seq_score_gm2",3)

    for c_idx in range(cols):
        current_col_values_list = [val for val in grid[:, c_idx] if val != -1]
        num_filled_in_col = len(current_col_values_list)
        sum_current_col_values = sum(current_col_values_list)
        for r_idx in range(rows):
            if grid[r_idx, c_idx] != -1: continue
            
            density_score = (num_filled_in_col + 1.0) / rows if rows > 0 else 0.0
            
            potential_col_sum = sum_current_col_values + avg_potential_num_to_place
            heuristic_max_col_sum = float(rows * max_val_board) if rows > 0 else 1.0
            if heuristic_max_col_sum == 0: heuristic_max_col_sum = 1.0
            sum_score = MathUtils().normalize_value(potential_col_sum, 0, heuristic_max_col_sum, clamp=True)
            
            seq_score = 0.0
            max_seq_len_contribution = 0
            if potential_numbers_to_place:
                for p_val_test in potential_numbers_to_place:
                    temp_col_line = [int(x) for x in grid[:, c_idx]]
                    temp_col_line[r_idx] = int(p_val_test)
                    sequences_found = BoardAnalyzerUtils().find_sequences_in_line(temp_col_line, min_len=min_len_for_seq_score_gm2, allow_gaps=1, check_arithmetic=True)
                    current_max_for_pval = 0
                    for seq in sequences_found:
                        if p_val_test in seq:
                           current_max_for_pval = max(current_max_for_pval, len(seq))
                    max_seq_len_contribution = max(max_seq_len_contribution, current_max_for_pval)

            if max_seq_len_contribution >= min_len_for_seq_score_gm2:
                seq_score = MathUtils().normalize_value(float(max_seq_len_contribution), min_len_for_seq_score_gm2, rows if rows > 0 else min_len_for_seq_score_gm2, clamp=True)
            elif max_seq_len_contribution > 0:
                seq_score = 0.25
            
            combined_score = (w_density_gm2 * density_score + 
                              w_sum_gm2 * sum_score + 
                              w_seq_gm2 * seq_score)
            max_possible_combined = w_density_gm2 + w_sum_gm2 + w_seq_gm2
            scores[r_idx, c_idx] = MathUtils().normalize_value(combined_score, 0, max_possible_combined if max_possible_combined >0 else 1.0, clamp=True)
    return scores

# 9. EXT_GM3_Adv_Connected_Comp_Vec (高級連通元件分析-空格區域)
def EXT_GM3_Adv_Connected_Comp_Vec(grid: np.ndarray, request_id: Optional [str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM3"
    logger.debug("Executing EXT_GM3_Adv_Connected_Comp_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    visited_overall = np.zeros_like(grid, dtype=bool)
    for r_start in range(rows):
        for c_start in range(cols):
            if visited_overall[r_start, c_start] or grid[r_start, c_start] != -1:
                continue
            
            component_cells: List[Tuple[int, int]] = []
            q = deque([(r_start, c_start)])
            # visited_bfs_current_component was used to avoid adding same cell multiple times in THIS bfs run
            # visited_overall marks it for ALL future component searches
            visited_overall[r_start, c_start] = True 
            component_cells.append((r_start, c_start)) # Add start cell to component

            head_bfs = 0
            while q:
                head_bfs +=1
                if head_bfs > rows*cols : break # Safety break for BFS
                r_curr, c_curr = q.popleft()
                # component_cells.append((r_curr, c_curr)) # Already added when put in queue or at start
                
                for dr_bfs, dc_bfs in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r_curr + dr_bfs, c_curr + dc_bfs
                    if 0 <= nr < rows and 0 <= nc < cols and \
                       grid[nr, nc] == -1 and \
                       not visited_overall[nr, nc]: # Check overall visited
                        visited_overall[nr, nc] = True # Mark globally
                        component_cells.append((nr,nc)) # Add to current component
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
def EXT_GM4_Spatial_Auto_Corr_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM4"
    logger.debug("Executing EXT_GM4_Spatial_Auto_Corr_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    hypothetical_val_to_place: float
    if potential_numbers:
        hypothetical_val_to_place = float(np.median(potential_numbers))
    else:
        max_board_val = float(BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols)))
        hypothetical_val_to_place = (1.0 + max_board_val) / 2.0 if max_board_val > 0 else 0.5

    max_val_on_grid_for_norm = float(BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols)))
    if max_val_on_grid_for_norm == 0: max_val_on_grid_for_norm = 1.0

    radius_gm4 = kwargs.get("radius_gm4", 1) # Parameter for neighborhood

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            neighbor_values = BoardAnalyzerUtils().get_neighborhood_values(
                grid, r_idx, c_idx, radius=radius_gm4, eight_connectivity=True,
                val_func=lambda x: float(x) if x != -1 else None,
                include_center=False
            )
            if not neighbor_values:
                scores[r_idx, c_idx] = 0.5 # Neutral score
                continue
            
            mean_neighbors = np.mean(neighbor_values)
            diff_hypothetical_to_mean_neighbors = abs(hypothetical_val_to_place - mean_neighbors)
            norm_diff = MathUtils().normalize_value(diff_hypothetical_to_mean_neighbors, 0, max_val_on_grid_for_norm, clamp=True)
            positive_autocorr_score = 1.0 - norm_diff
            scores[r_idx, c_idx] = positive_autocorr_score
    return scores

# 11. EXT_GM5_Line_Completion_Vec(線段補全)
def EXT_GM5_Line_Completion_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM5"
    logger.debug("Executing EXT_GM5_Line_Completion_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0 or min(rows,cols) < 1 : return scores # Needs at least 1 for min_len=1

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores

    line_completion_score_map_default = {
        "identical_3": 0.6, "arithmetic_3_mend": 0.7,
        "arithmetic_3_extend": 0.5, "arithmetic_3_mend_high_val": 0.9,
    }
    line_completion_score_map = kwargs.get("line_completion_score_map_gm5", line_completion_score_map_default)

    max_board_val = float(BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols)))
    high_val_threshold = kwargs.get("high_val_threshold_gm5", (max_board_val * 0.7 if max_board_val > 0 else 10.0))

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_completion_score_for_cell = 0.0
            for p_val in potential_numbers_to_place:
                current_max_for_pval = 0.0 # Max score for this specific p_val in this cell
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]: # 4 main directions
                    # Case 1: Mending a line -> N1 - p_val - N2
                    r_n1, c_n1 = r_idx - dr, c_idx - dc
                    r_n2, c_n2 = r_idx + dr, c_idx + dc
                    if 0 <= r_n1 < rows and 0 <= c_n1 < cols and \
                       0 <= r_n2 < rows and 0 <= c_n2 < cols:
                        val_n1, val_n2 = grid[r_n1, c_n1], grid[r_n2, c_n2]
                        if val_n1 != -1 and val_n2 != -1:
                            if val_n1 == p_val and val_n2 == p_val:
                                current_max_for_pval = max(current_max_for_pval, line_completion_score_map["identical_3"])
                            # Arithmetic check for N1, p_val, N2
                            if (val_n1 + val_n2) == 2 * p_val and abs(p_val - val_n1) > 0: # Ensure not constant unless p_val is mean
                                score_key = "arithmetic_3_mend"
                                if (val_n1 + p_val + val_n2) / 3.0 > high_val_threshold:
                                    score_key = "arithmetic_3_mend_high_val"
                                current_max_for_pval = max(current_max_for_pval, line_completion_score_map.get(score_key, line_completion_score_map["arithmetic_3_mend"]))
                    
                    # Case 2: Extending a line -> p_val - N1 - N2
                    r_n1_ext1, c_n1_ext1 = r_idx + dr, c_idx + dc
                    r_n2_ext1, c_n2_ext1 = r_idx + 2 * dr, c_idx + 2 * dc
                    if 0 <= r_n1_ext1 < rows and 0 <= c_n1_ext1 < cols and \
                       0 <= r_n2_ext1 < rows and 0 <= c_n2_ext1 < cols:
                        val_n1_ext1, val_n2_ext1 = grid[r_n1_ext1, c_n1_ext1], grid[r_n2_ext1, c_n2_ext1]
                        if val_n1_ext1 != -1 and val_n2_ext1 != -1:
                            if p_val == val_n1_ext1 and p_val == val_n2_ext1:
                                current_max_for_pval = max(current_max_for_pval, line_completion_score_map["identical_3"])
                            # Arithmetic check for p_val, N1, N2
                            if (p_val + val_n2_ext1) == 2 * val_n1_ext1 and abs(val_n1_ext1 - p_val) > 0:
                                current_max_for_pval = max(current_max_for_pval, line_completion_score_map["arithmetic_3_extend"])

                    # Case 3: Extending a line (other end) -> N1 - N2 - p_val
                    r_n1_ext2, c_n1_ext2 = r_idx - 2 * dr, c_idx - 2 * dc
                    r_n2_ext2, c_n2_ext2 = r_idx - dr, c_idx - dc
                    if 0 <= r_n1_ext2 < rows and 0 <= c_n1_ext2 < cols and \
                       0 <= r_n2_ext2 < rows and 0 <= c_n2_ext2 < cols:
                        val_n1_ext2, val_n2_ext2 = grid[r_n1_ext2, c_n1_ext2], grid[r_n2_ext2, c_n2_ext2]
                        if val_n1_ext2 != -1 and val_n2_ext2 != -1:
                            if val_n1_ext2 == val_n2_ext2 and val_n1_ext2 == p_val:
                                current_max_for_pval = max(current_max_for_pval, line_completion_score_map["identical_3"])
                            # Arithmetic check for N1, N2, p_val
                            if (val_n1_ext2 + p_val) == 2 * val_n2_ext2 and abs(val_n2_ext2 - val_n1_ext2) > 0:
                                current_max_for_pval = max(current_max_for_pval, line_completion_score_map["arithmetic_3_extend"])
                max_completion_score_for_cell = max(max_completion_score_for_cell, current_max_for_pval)
            scores[r_idx, c_idx] = MathUtils().normalize_value(max_completion_score_for_cell, 0, 1.0, clamp=True) # Scores from map are already ~0-1
    return scores

# 12. EXT_GM6_Symmetry_Potential_Vec (對稱性潛力)
def EXT_GM6_Symmetry_Potential_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM6"
    logger.debug("Executing EXT_GM6_Symmetry_Potential_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores

    symmetry_scores_map_default = {
        "horizontal": 0.7, "vertical": 0.7, "point_center": 0.8,
        "main_diagonal": 0.6, "anti_diagonal": 0.6,
    }
    symmetry_scores_map = kwargs.get("symmetry_scores_map_gm6", symmetry_scores_map_default)
    if rows == cols: # More emphasis on diagonal symmetries for square grids
        symmetry_scores_map["main_diagonal"] = max(symmetry_scores_map.get("main_diagonal",0.6), 0.7)
        symmetry_scores_map["anti_diagonal"] = max(symmetry_scores_map.get("anti_diagonal",0.6), 0.7)


    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_symmetry_score_for_cell = 0.0
            for p_val in potential_numbers_to_place:
                current_max_for_pval = 0.0
                # 1. Horizontal Symmetry: (r_idx, c_idx) vs (r_idx, cols-1-c_idx)
                sr_h, sc_h = r_idx, cols - 1 - c_idx
                if sc_h != c_idx and 0 <= sr_h < rows and 0 <= sc_h < cols and grid[sr_h, sc_h] == p_val:
                    current_max_for_pval = max(current_max_for_pval, symmetry_scores_map["horizontal"])
                
                # 2. Vertical Symmetry: (r_idx, c_idx) vs (rows-1-r_idx, c_idx)
                sr_v, sc_v = rows - 1 - r_idx, c_idx
                if sr_v != r_idx and 0 <= sr_v < rows and 0 <= sc_v < cols and grid[sr_v, sc_v] == p_val:
                    current_max_for_pval = max(current_max_for_pval, symmetry_scores_map["vertical"])

                # 3. Point (Center) Symmetry: (r_idx, c_idx) vs (rows-1-r_idx, cols-1-c_idx)
                sr_p, sc_p = rows - 1 - r_idx, cols - 1 - c_idx
                if (sr_p != r_idx or sc_p != c_idx) and \
                   0 <= sr_p < rows and 0 <= sc_p < cols and grid[sr_p, sc_p] == p_val:
                    current_max_for_pval = max(current_max_for_pval, symmetry_scores_map["point_center"])

                # 4. Main Diagonal Symmetry (\): (r_idx, c_idx) vs (c_idx, r_idx)
                if rows == cols: 
                    sr_d1, sc_d1 = c_idx, r_idx
                    if (sr_d1 != r_idx or sc_d1 != c_idx) and \
                       0 <= sr_d1 < rows and 0 <= sc_d1 < cols and grid[sr_d1, sc_d1] == p_val:
                        current_max_for_pval = max(current_max_for_pval, symmetry_scores_map["main_diagonal"])
                
                # 5. Anti-Diagonal Symmetry (/): (r_idx, c_idx) vs (cols-1-c_idx, rows-1-r_idx) - matrix reflection
                if rows == cols: 
                    # For a point (r,c) its reflection across anti-diagonal (y=-x+N-1) is ( (N-1)-c, (N-1)-r )
                    sr_d2, sc_d2 = (rows - 1) - c_idx, (cols - 1) - r_idx 
                    if (sr_d2 != r_idx or sc_d2 != c_idx) and \
                       0 <= sr_d2 < rows and 0 <= sc_d2 < cols and grid[sr_d2, sc_d2] == p_val:
                        current_max_for_pval = max(current_max_for_pval, symmetry_scores_map["anti_diagonal"])
                max_symmetry_score_for_cell = max(max_symmetry_score_for_cell, current_max_for_pval)
            
            scores[r_idx, c_idx] = MathUtils().normalize_value(max_symmetry_score_for_cell, 0, 1.0, clamp=True) # Max of map is ~0.8
    return scores

# 13. EXT_GM7_Numeric_Gaps_Vec (數值間隙填充)
def EXT_GM7_Numeric_Gaps_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM7"
    logger.debug("Executing EXT_GM7_Numeric_Gaps_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores

    gap_fill_scores_map_default = {
        "arithmetic_1_gap_fill": 0.9, "arithmetic_generic_mend": 0.7,
        "arithmetic_generic_extend": 0.5, "arithmetic_gap_fill_high_val": 0.95,
        "arithmetic_gap_fill_long_seq_potential": 0.85,
    }
    gap_fill_scores_map = kwargs.get("gap_fill_scores_map_gm7", gap_fill_scores_map_default)
    max_board_val = float(BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols)))
    high_val_threshold = kwargs.get("high_val_threshold_gm7", (max_board_val * 0.7 if max_board_val > 0 else 10.0))

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_cell_gap_score = 0.0
            for p_val in potential_numbers_to_place:
                current_max_for_pval = 0.0
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    # Case 1: p_val mends a gap: N1 - p_val - N2
                    r_n1, c_n1 = r_idx - dr, c_idx - dc
                    r_n2, c_n2 = r_idx + dr, c_idx + dc
                    if 0 <= r_n1 < rows and 0 <= c_n1 < cols and \
                       0 <= r_n2 < rows and 0 <= c_n2 < cols:
                        val_n1, val_n2 = grid[r_n1, c_n1], grid[r_n2, c_n2]
                        if val_n1 != -1 and val_n2 != -1:
                            score_to_add = 0.0
                            if val_n1 == p_val - 1 and val_n2 == p_val + 1: # Diff 1 gap
                                score_to_add = gap_fill_scores_map["arithmetic_1_gap_fill"]
                                if (val_n1 + p_val + val_n2) / 3.0 > high_val_threshold:
                                    score_to_add = max(score_to_add, gap_fill_scores_map.get("arithmetic_gap_fill_high_val", score_to_add))
                            elif (val_n1 + val_n2) == 2 * p_val and abs(p_val - val_n1) > 0: # Generic arithmetic mend
                                score_to_add = gap_fill_scores_map["arithmetic_generic_mend"]
                            current_max_for_pval = max(current_max_for_pval, score_to_add)

                    # Case 2: p_val extends a sequence: p_val - N1 - N2
                    r_n1_ext1, c_n1_ext1 = r_idx + dr, c_idx + dc
                    r_n2_ext1, c_n2_ext1 = r_idx + 2 * dr, c_idx + 2 * dc
                    if 0 <= r_n1_ext1 < rows and 0 <= c_n1_ext1 < cols and \
                       0 <= r_n2_ext1 < rows and 0 <= c_n2_ext1 < cols:
                        val_n1_ext1, val_n2_ext1 = grid[r_n1_ext1, c_n1_ext1], grid[r_n2_ext1, c_n2_ext1]
                        if val_n1_ext1 != -1 and val_n2_ext1 != -1:
                            if (val_n1_ext1 - p_val) == (val_n2_ext1 - val_n1_ext1) and (val_n1_ext1 - p_val) != 0:
                                current_max_for_pval = max(current_max_for_pval, gap_fill_scores_map["arithmetic_generic_extend"])
                    
                    # Case 3: p_val extends a sequence: N1 - N2 - p_val
                    r_n1_ext2, c_n1_ext2 = r_idx - 2 * dr, c_idx - 2 * dc
                    r_n2_ext2, c_n2_ext2 = r_idx - dr, c_idx - dc
                    if 0 <= r_n1_ext2 < rows and 0 <= c_n1_ext2 < cols and \
                       0 <= r_n2_ext2 < rows and 0 <= c_n2_ext2 < cols:
                        val_n1_ext2, val_n2_ext2 = grid[r_n1_ext2, c_n1_ext2], grid[r_n2_ext2, c_n2_ext2]
                        if val_n1_ext2 != -1 and val_n2_ext2 != -1:
                            if (val_n2_ext2 - val_n1_ext2) == (p_val - val_n2_ext2) and (val_n2_ext2 - val_n1_ext2) != 0:
                                current_max_for_pval = max(current_max_for_pval, gap_fill_scores_map["arithmetic_generic_extend"])
                max_cell_gap_score = max(max_cell_gap_score, current_max_for_pval)
            scores[r_idx, c_idx] = MathUtils().normalize_value(max_cell_gap_score, 0, 1.0, clamp=True)
    return scores

# 14. EXT_GM8_Edge_Affinity_Vec (邊緣親和度)
def EXT_GM8_Edge_Affinity_Vec(grid: np.ndarray, request_id: Optional [str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM8"
    logger.debug("Executing EXT_GM8_Edge_Affinity_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    affinity_mode = kwargs.get("affinity_mode_gm8", "prefer_edge") # "prefer_edge" or "avoid_edge"
    corner_bonus_prefer = kwargs.get("corner_bonus_prefer_gm8", 0.2)
    corner_penalty_avoid = kwargs.get("corner_penalty_avoid_gm8", 0.2)

    max_min_dist_to_edge_row = (rows - 1) // 2 if rows > 0 else 0
    max_min_dist_to_edge_col = (cols - 1) // 2 if cols > 0 else 0
    overall_max_of_min_distances = float(min(max_min_dist_to_edge_row, max_min_dist_to_edge_col))

    if overall_max_of_min_distances == 0: 
        if rows > 1 or cols > 1: overall_max_of_min_distances = 0.5 # For line grids
        else: overall_max_of_min_distances = 1.0 # For single cell grid, to avoid div by zero

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            
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
                normalized_dist = min(1.0, normalized_dist) 
            elif min_dist == 0: # All cells on edge, including single cell
                normalized_dist = 0.0
            else: # Should not be reached if overall_max_of_min_distances is handled
                normalized_dist = 1.0 
            
            if affinity_mode == "prefer_edge":
                current_score = 1.0 - normalized_dist
                if is_corner and min_dist == 0: # min_dist == 0 implies it's on an edge
                    current_score += corner_bonus_prefer
            elif affinity_mode == "avoid_edge":
                current_score = normalized_dist
                if is_corner and min_dist == 0:
                    current_score -= corner_penalty_avoid
            
            # Normalize the final score considering bonus/penalty
            min_possible_score = -corner_penalty_avoid if affinity_mode == "avoid_edge" else 0.0
            max_possible_score = 1.0 + corner_bonus_prefer if affinity_mode == "prefer_edge" else 1.0
            if max_possible_score == min_possible_score : max_possible_score = min_possible_score + 1.0 # Avoid div by zero
            scores[r_idx, c_idx] = MathUtils().normalize_value(current_score, min_possible_score, max_possible_score, clamp=True)
    return scores

# 15. EXT_GM9_Center_Control_Vec(中心控制偏好)
def EXT_GM9_Center_Control_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM9"
    logger.debug("Executing EXT_GM9_Center_Control_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    affinity_mode = kwargs.get("affinity_mode_gm9", "prefer_center")
    center_r = (rows - 1) / 2.0
    center_c = (cols - 1) / 2.0
    
    # Max distance from a corner to the center
    max_dist_to_center = MathUtils().euclidean_distance((0.0, 0.0), (center_r, center_c))
    if max_dist_to_center == 0: # Handles 1x1 grid
        max_dist_to_center = 1.0 # Avoid division by zero for normalization

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            current_dist_to_center = MathUtils().euclidean_distance((float(r_idx), float(c_idx)), (center_r, center_c))
            
            normalized_dist = MathUtils().normalize_value(current_dist_to_center, 0, max_dist_to_center, clamp=True)
            # For 1x1 grid: dist=0, max_dist=1 -> norm_dist=0. Score prefer=1, avoid=0. Correct.
            
            current_score = 0.0
            if affinity_mode == "prefer_center":
                current_score = 1.0 - normalized_dist
            elif affinity_mode == "avoid_center":
                current_score = normalized_dist
            scores[r_idx, c_idx] = MathUtils().normalize_value(current_score, 0, 1.0, clamp=True) # Final score is already 0-1
    return scores

# 16. EXT_GM10_Blocking_Value_Vec (阻斷價值評估)
def EXT_GM10_Blocking_Value_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM10"
    logger.debug("Executing EXT_GM10_Blocking_Value_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores

    undesirable_sequences_default = [[1, 1, 1], [2, 2, 2]]
    UNDESIRABLE_SEQUENCES = kwargs.get("undesirable_sequences_gm10", undesirable_sequences_default)
    
    line_length_to_check = 3 # Assuming we check for length 3 undesirable sequences

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_safety_score_for_cell = 0.0 # Start with 0, means placement is bad or no options
            
            if not potential_numbers_to_place: # Should be caught earlier, but for safety
                scores[r_idx, c_idx] = 0.5 # Neutral if no options to evaluate
                continue

            for p_val in potential_numbers_to_place:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val
                forms_undesirable_pattern = False

                for dr_line, dc_line in [(0, 1), (1, 0), (1, 1), (1, -1)]: # 4 directions
                    # Check line_length_to_check windows that include (r_idx, c_idx) with p_val
                    for offset in range(-line_length_to_check + 1, 1): 
                        current_line_values = []
                        is_valid_segment = True
                        # Check if (r_idx, c_idx) is part of the segment defined by this offset
                        # The element at (r_idx, c_idx) is at index `-offset` in the conceptual window
                        if not (0 <= -offset < line_length_to_check) : continue

                        for i_val in range(line_length_to_check):
                            check_r, check_c = r_idx + (offset + i_val) * dr_line, \
                                               c_idx + (offset + i_val) * dc_line
                            if 0 <= check_r < rows and 0 <= check_c < cols:
                                current_line_values.append(int(temp_grid[check_r, check_c]))
                            else:
                                is_valid_segment = False
                                break
                        
                        if is_valid_segment and len(current_line_values) == line_length_to_check:
                            # Ensure the p_val at (r_idx, c_idx) is indeed what we are checking
                            # This is implicitly handled if the window contains (r_idx,c_idx)
                            # and current_line_values are from temp_grid
                            for undesirable_seq in UNDESIRABLE_SEQUENCES:
                                if len(undesirable_seq) == line_length_to_check and \
                                   current_line_values == undesirable_seq:
                                    forms_undesirable_pattern = True
                                    break 
                            if forms_undesirable_pattern: break 
                    if forms_undesirable_pattern: break 
                
                # Score for this p_val: high if safe, low if creates undesirable pattern
                current_score_for_pval = 0.9 if not forms_undesirable_pattern else 0.1 
                if current_score_for_pval > max_safety_score_for_cell:
                    max_safety_score_for_cell = current_score_for_pval
            
            scores[r_idx, c_idx] = max_safety_score_for_cell
    return scores

# 17. EXT_GM11_Pair_Correlation_Vec (數字配對關聯分析)
def EXT_GM11_Pair_Correlation_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM11"
    logger.debug("Executing EXT_GM11_Pair_Correlation_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores
    
    max_val_for_pairs = BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows,cols))
    mid_val_for_pairs = max(1, max_val_for_pairs // 2) if max_val_for_pairs >0 else 1

    favorable_pairs_scores_default = {
        (3, 7): 0.8, (7, 3): 0.8, (1, 2): 0.6, (2, 1): 0.6,
        (10, 20): 0.7, (20, 10): 0.7, (5, 10): 0.5, (10, 5): 0.5,
        (mid_val_for_pairs, mid_val_for_pairs + 1): 0.4,
        (mid_val_for_pairs + 1, mid_val_for_pairs): 0.4,
    }
    FAVORABLE_PAIRS_SCORES = kwargs.get("favorable_pairs_scores_gm11", favorable_pairs_scores_default)
    
    max_single_pair_score = 0.0
    if FAVORABLE_PAIRS_SCORES: # Check if dict is not empty
        max_single_pair_score = max(FAVORABLE_PAIRS_SCORES.values()) if FAVORABLE_PAIRS_SCORES else 0.0
    
    # Max possible sum if all 8 neighbors form max-scoring pairs
    heuristic_max_total_pair_score = 8.0 * max_single_pair_score if max_single_pair_score > 0 else 1.0
    if heuristic_max_total_pair_score == 0 : heuristic_max_total_pair_score = 1.0 # Avoid div by zero
    
    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_accumulated_score_for_cell = 0.0
            for p_val in potential_numbers_to_place:
                current_pval_accumulated_score = 0.0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
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
def EXT_GM12_Island_Analysis_Vec(grid: np.ndarray, request_id: Optional [str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM12"
    logger.debug("Executing EXT_GM12_Island_Analysis_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float) # Scores are for filled cells, empty cells get 0
    if rows == 0 or cols == 0: return scores

    visited_island_search = np.zeros_like(grid, dtype=bool)
    max_val_on_board = float(BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols)))
    if max_val_on_board == 0: max_val_on_board = 1.0
    
    w_size = kwargs.get("w_size_gm12", 0.4)
    w_compactness = kwargs.get("w_compactness_gm12", 0.3)
    w_avg_value_gm12 = kwargs.get("w_avg_value_gm12", 0.3) # Renamed to avoid conflict

    for r_start in range(rows):
        for c_start in range(cols):
            if grid[r_start, c_start] != -1 and not visited_island_search[r_start, c_start]:
                current_island_cells: List[Tuple[int, int]] = []
                current_island_values: List[int] = []
                q = deque([(r_start, c_start)])
                visited_island_search[r_start, c_start] = True # Mark visited before adding to component
                current_island_cells.append((r_start,c_start))
                current_island_values.append(int(grid[r_start,c_start]))

                min_r_bbox, max_r_bbox = r_start, r_start
                min_c_bbox, max_c_bbox = c_start, c_start
                
                head_bfs = 0
                while q:
                    head_bfs +=1
                    if head_bfs > rows*cols : break # Safety break
                    r_curr, c_curr = q.popleft()
                    # Values and cells are added when they are marked visited and added to queue or at start
                    min_r_bbox = min(min_r_bbox, r_curr)
                    max_r_bbox = max(max_r_bbox, r_curr)
                    min_c_bbox = min(min_c_bbox, c_curr)
                    max_c_bbox = max(max_c_bbox, c_curr)

                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r_curr + dr, c_curr + dc
                        if 0 <= nr < rows and 0 <= nc < cols and \
                           grid[nr, nc] != -1 and not visited_island_search[nr, nc]:
                            visited_island_search[nr, nc] = True
                            current_island_cells.append((nr,nc))
                            current_island_values.append(int(grid[nr,nc]))
                            q.append((nr, nc))
                
                island_size = float(len(current_island_cells))
                avg_value = 0.0
                if island_size > 0:
                    avg_value = sum(current_island_values) / island_size
                
                bbox_height = float(max_r_bbox - min_r_bbox + 1)
                bbox_width = float(max_c_bbox - min_c_bbox + 1)
                bbox_area = bbox_height * bbox_width
                compactness = 0.0
                if bbox_area > 0:
                    compactness = island_size / bbox_area
                
                norm_size = MathUtils().normalize_value(island_size, 1, float(rows * cols) if rows*cols >0 else 1.0, clamp=True)
                norm_compactness = MathUtils().normalize_value(compactness, 0, 1.0, clamp=True) # Already 0-1
                norm_avg_value = MathUtils().normalize_value(avg_value, 1, max_val_on_board, clamp=True)
                
                island_score_val = (w_size * norm_size + 
                                 w_compactness * norm_compactness + 
                                 w_avg_value_gm12 * norm_avg_value)
                max_possible_combined = w_size + w_compactness + w_avg_value_gm12
                final_island_score = MathUtils().normalize_value(island_score_val, 0, max_possible_combined if max_possible_combined >0 else 1.0, clamp=True)
                
                for r_cell, c_cell in current_island_cells:
                    scores[r_cell, c_cell] = final_island_score
            # For empty cells or already visited cells, score remains 0 (default for this module)
            # Mark visited to avoid re-evaluating empty cells if they were part of visited_island_search logic
            visited_island_search[r_start,c_start] = True

    return scores

# 19. EXT_GM13_Sequence_Diversity_Vec (序列多樣性)
def EXT_GM13_Sequence_Diversity_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM13"
    logger.debug("Executing EXT_GM13_Sequence_Diversity_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores

    short_sequence_len = kwargs.get("short_sequence_len_gm13",3)
    # Heuristic max: 4 directions * (mend, extend_left, extend_right) * (arithmetic, identical) = 4 * 3 * 2 = 24.
    # But many overlaps. 8.0 was original. Let's make it configurable.
    heuristic_max_distinct_sequences = kwargs.get("heuristic_max_distinct_sequences_gm13", 8.0)
    if heuristic_max_distinct_sequences == 0: heuristic_max_distinct_sequences = 1.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_diversity_count_for_cell = 0
            for p_val in potential_numbers_to_place:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val
                found_sequence_signatures = set() # Store signatures like ("arithmetic", (dr,dc), diff, start_idx_in_line)
                
                for dr_dir, dc_dir in [(0, 1), (1, 0), (1, 1), (1, -1)]: # 4 directions
                    # Check windows of short_sequence_len that involve the placed p_val
                    for i_offset in range(-short_sequence_len + 1, 1): # Offset of the window start relative to (r_idx, c_idx)
                                                                       # such that (r_idx,c_idx) is in the window
                        current_sequence_values = []
                        valid_segment = True
                        # Check if (r_idx,c_idx) is actually in this segment
                        # The p_val is at window index: -i_offset
                        if not (0 <= -i_offset < short_sequence_len): continue

                        for k_seq in range(short_sequence_len): # k_seq is index within the current window
                            check_r = r_idx + (i_offset + k_seq) * dr_dir
                            check_c = c_idx + (i_offset + k_seq) * dc_dir
                            if 0 <= check_r < rows and 0 <= check_c < cols:
                                current_sequence_values.append(int(temp_grid[check_r, check_c]))
                            else:
                                valid_segment = False
                                break
                        
                        if valid_segment and len(current_sequence_values) == short_sequence_len:
                            s = current_sequence_values
                            # Ensure all are numbers (p_val is a number, others from grid could be -1 if not careful with temp_grid)
                            # but temp_grid should have p_val, other cells are from original grid unless overwritten.
                            # The logic implies we are checking sequences formed in temp_grid.
                            if all(val != -1 for val in s):
                                # Arithmetic sequence (non-constant)
                                diffs = np.diff(s)
                                if len(diffs) == short_sequence_len -1 : # e.g. for len 3, 2 diffs
                                    is_arithmetic = all(d == diffs[0] for d in diffs)
                                    if is_arithmetic and diffs[0] != 0:
                                        # Signature: type, direction, diff, sorted tuple of values for uniqueness
                                        found_sequence_signatures.add(("arithmetic", (dr_dir, dc_dir), diffs[0], tuple(sorted(s))))
                                
                                # Identical sequence
                                is_identical = all(val == s[0] for val in s)
                                if is_identical and s[0] != -1 : # Ensure not sequence of -1 (if logic allows)
                                    found_sequence_signatures.add(("identical", (dr_dir, dc_dir), s[0])) # Value is the "diff"
                
                current_pval_diversity_count = len(found_sequence_signatures)
                if current_pval_diversity_count > max_diversity_count_for_cell:
                    max_diversity_count_for_cell = current_pval_diversity_count
            
            scores[r_idx, c_idx] = MathUtils().normalize_value(float(max_diversity_count_for_cell), 0, heuristic_max_distinct_sequences, clamp=True)
    return scores

# 20. EXT_GM14_Risk_Assessment_Vec (風險評估)
def EXT_GM14_Risk_Assessment_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM14"
    logger.debug("Executing EXT_GM14_Risk_Assessment_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    initial_potential_numbers = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not initial_potential_numbers: 
        return scores # No numbers to place initially, risk is high (score low) or undefined

    # Max possible legal moves is roughly rows*cols if board is empty
    # After one placement, it's rows*cols - 1.
    max_possible_flexibility = float(rows * cols -1) if rows*cols >1 else 1.0
    if max_possible_flexibility == 0 : max_possible_flexibility = 1.0 # Avoid div by zero

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_flexibility_score_for_cell = 0.0 
            
            # If current cell cannot be placed (e.g. no legal values for it, though loop is on initial_potential_numbers)
            # This loop iterates through numbers that were legal for the *original* grid.
            # We need to ensure p_val can be placed at (r_idx, c_idx) if grid was empty there.
            # The outer loop already ensures (r_idx,c_idx) is empty.
            
            num_evaluated_options = 0
            accumulated_flexibility = 0.0

            for p_val in initial_potential_numbers: # These are numbers legal for the *current* grid state
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val # Place one of the currently legal numbers
                
                subsequent_legal_options = BoardAnalyzerUtils().get_legal_values_for_placement(temp_grid)
                current_flexibility = float(len(subsequent_legal_options))
                accumulated_flexibility += current_flexibility
                num_evaluated_options +=1
            
            if num_evaluated_options > 0:
                 avg_flexibility = accumulated_flexibility / num_evaluated_options
            else: # No legal numbers to try placing in this empty cell (should not happen if initial_potential_numbers is not empty)
                 avg_flexibility = 0.0
            
            # Score is the average flexibility after placing a number
            scores[r_idx, c_idx] = MathUtils().normalize_value(avg_flexibility, 0, max_possible_flexibility, clamp=True)
    return scores

# 21. EXT_GM15_Information_Gain_Vec (資訊增益評估)
def EXT_GM15_Information_Gain_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM15"
    logger.debug("Executing EXT_GM15_Information_Gain_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores

    # Flatten grid includes -1s. Entropy considers all unique values.
    initial_grid_values = [int(val) for val in grid.flatten()]
    entropy_before = MathUtils().get_entropy(initial_grid_values)
    
    # Max possible entropy (for a single cell state) is log2(num_possible_values_for_a_cell + 1 for empty)
    # Max change is roughly entropy_before itself, or log2 of total symbols if we consider absolute max.
    # The number of symbols: 1 to R*C plus -1
    num_symbols = rows * cols + 1 
    max_possible_entropy_change = math.log2(num_symbols) if num_symbols > 1 else 1.0
    if max_possible_entropy_change == 0: max_possible_entropy_change = 1.0 # Avoid div by zero

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_entropy_reduction_for_cell = -float('inf') 
            
            for p_val in potential_numbers_to_place:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val
                temp_grid_values = [int(val) for val in temp_grid.flatten()]
                entropy_after = MathUtils().get_entropy(temp_grid_values)
                entropy_reduction = entropy_before - entropy_after # Higher reduction is better
                if entropy_reduction > max_entropy_reduction_for_cell:
                    max_entropy_reduction_for_cell = entropy_reduction
            
            if max_entropy_reduction_for_cell == -float('inf'): # No potential numbers or no change
                max_entropy_reduction_for_cell = 0.0
            
            # Normalize reduction. Range is roughly [-max_possible_entropy_change, max_possible_entropy_change]
            # We want positive reductions to be high score.
            # Score = (reduction - (-max_change)) / (max_change - (-max_change)) - not quite
            # Simpler: normalize positive reductions from 0 to max_possible_entropy_change
            scores[r_idx, c_idx] = MathUtils().normalize_value(max_entropy_reduction_for_cell, 0, max_possible_entropy_change, clamp=True)
    return scores

# 22. EXT_GM16_Harmonic_Centrality_Vec (調和中心性)
def EXT_GM16_Harmonic_Centrality_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM16"
    logger.debug("Executing EXT_GM16_Harmonic_Centrality_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0 or (rows * cols) <= 1: return scores 

    # Max possible harmonic centrality (heuristic): if a cell is at distance 1 from all N-1 other cells.
    # Sum(1/1) for N-1 cells = N-1
    max_hc_heuristic = float(rows * cols - 1)
    if max_hc_heuristic == 0: max_hc_heuristic = 1.0 

    for r_eval in range(rows):
        for c_eval in range(cols):
            if grid[r_eval, c_eval] != -1: continue # Only score empty cells
            current_harmonic_centrality = 0.0
            num_other_nodes = 0
            for r_other in range(rows):
                for c_other in range(cols):
                    if r_eval == r_other and c_eval == c_other: continue
                    # Consider other cells regardless of whether they are filled or empty for centrality
                    dist = MathUtils().manhattan_distance((r_eval, c_eval), (r_other, c_other))
                    if dist > 0:
                        current_harmonic_centrality += 1.0 / dist
                        num_other_nodes += 1
            
            if num_other_nodes == 0: # Should not happen if grid cells > 1
                scores[r_eval, c_eval] = 0.0
            else:
                # Normalization: Max possible HC is complex. A simple node connected to all others at dist 1
                # would have (N-1). A corner node in a line graph has smaller HC.
                # The heuristic max_hc_heuristic is an upper bound.
                scores[r_eval, c_eval] = MathUtils().normalize_value(current_harmonic_centrality, 0, max_hc_heuristic, clamp=True)
    return scores

# 23. EXT_GM17_Entropy_Minimization_Vec (局部熵最小化)
def EXT_GM17_Entropy_Minimization_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM17"
    logger.debug("Executing EXT_GM17_Entropy_Minimization_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores

    radius = kwargs.get("radius_gm17",1)
    num_cells_in_neighborhood = (2 * radius + 1)**2 
    # Max possible entropy for the neighborhood if all cells are different.
    # Max number of distinct symbols in neighborhood: num_cells_in_neighborhood (if all are unique numbers)
    # or fewer if values are constrained (e.g., R*C for board values, plus -1).
    # For normalization of change, log2(num_cells_in_neighborhood) is a reasonable upper bound for change if all were random then become same.
    max_local_entropy_change = math.log2(num_cells_in_neighborhood) if num_cells_in_neighborhood > 1 else 1.0
    if max_local_entropy_change == 0: max_local_entropy_change = 1.0

    def val_func_for_entropy_gm17(x_val: int) -> int: return int(x_val) # Includes -1 as a symbol

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            
            # Create a temp grid where current cell is empty (-1) to get "before" neighborhood
            grid_before_local = grid.copy()
            # grid_before_local[r_idx, c_idx] = -1 # Already empty, this line is not needed.
            
            values_before_placement_local = BoardAnalyzerUtils().get_neighborhood_values(
                grid_before_local, r_idx, c_idx, radius=radius, eight_connectivity=True, # Use grid_before_local
                val_func=val_func_for_entropy_gm17, include_center=True 
            )
            entropy_before_local = MathUtils().get_entropy(values_before_placement_local)
            max_entropy_reduction_for_cell = -float('inf')

            for p_val in potential_numbers_to_place:
                temp_grid_local_place = grid.copy() # Start from original grid
                temp_grid_local_place[r_idx, c_idx] = p_val
                
                values_after_placement_local = BoardAnalyzerUtils().get_neighborhood_values(
                    temp_grid_local_place, r_idx, c_idx, radius=radius, eight_connectivity=True,
                    val_func=val_func_for_entropy_gm17, include_center=True
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
def EXT_GM18_RL_Value_Est_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM18"
    logger.debug("Executing EXT_GM18_RL_Value_Est_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils().get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores

    feature_weights_default = {
        "identical_3": 1.0, "arithmetic_3": 0.7,
        "board_density_factor": 0.2,
        "central_control_boost": 0.15, "edge_affinity_boost": 0.05,
    }
    FEATURE_WEIGHTS = kwargs.get("feature_weights_gm18", feature_weights_default)
    
    # Max heuristic score: 4 directions * (max_line_score) + density_max + central_max + edge_max
    max_line_score = FEATURE_WEIGHTS.get("identical_3",0) + FEATURE_WEIGHTS.get("arithmetic_3",0) # Simplified
    max_heuristic_feature_score = (4 * max_line_score) + \
                                FEATURE_WEIGHTS.get("board_density_factor",0) + \
                                FEATURE_WEIGHTS.get("central_control_boost",0) + \
                                FEATURE_WEIGHTS.get("edge_affinity_boost",0)
    if max_heuristic_feature_score == 0: max_heuristic_feature_score = 1.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_feature_score_for_cell = 0.0
            for p_val in potential_numbers_to_place:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val
                current_features_score = 0.0
                
                # Feature 1 & 2: Lines of 3
                for dr_line, dc_line in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    for offset in range(-2, 1): 
                        line_values = []
                        is_valid_line = True
                        involved_pval_in_segment = False # Check if current p_val is in this specific 3-segment
                        
                        # Check if (r_idx, c_idx) which holds p_val is part of this segment
                        # p_val is at window index -offset.
                        if not (0 <= -offset < 3): continue


                        for i_val in range(3):
                            check_r, check_c = r_idx + (offset + i_val) * dr_line, c_idx + (offset + i_val) * dc_line
                            # if r_idx == check_r and c_idx == check_c: involved_pval_in_segment = True # p_val is in this segment
                            if 0 <= check_r < rows and 0 <= check_c < cols:
                                line_values.append(int(temp_grid[check_r, check_c]))
                            else:
                                is_valid_line = False
                                break
                        if is_valid_line and len(line_values) == 3 and all(v != -1 for v in line_values):
                            s = line_values
                            if s[0] == s[1] and s[1] == s[2]:
                                current_features_score += FEATURE_WEIGHTS.get("identical_3",0)
                            elif (s[1] - s[0]) == (s[2] - s[1]) and (s[1] - s[0]) != 0:
                                current_features_score += FEATURE_WEIGHTS.get("arithmetic_3",0)
                
                # Feature 3: Board density
                num_filled_after_placement = np.count_nonzero(temp_grid != -1)
                density_after_placement = num_filled_after_placement / (rows * cols) if (rows * cols) > 0 else 0.0
                current_features_score += FEATURE_WEIGHTS.get("board_density_factor",0) * density_after_placement
                
                # Feature 4 & 5: Central/Edge control
                if rows > 1 and cols > 1:
                    center_r, center_c = (rows - 1) / 2.0, (cols - 1) / 2.0
                    dist_to_center = MathUtils().euclidean_distance((float(r_idx), float(c_idx)), (center_r, center_c))
                    max_center_dist = MathUtils().euclidean_distance((0.0,0.0), (center_r,center_c))
                    if max_center_dist > 0:
                         current_features_score += FEATURE_WEIGHTS.get("central_control_boost",0) * \
                            (1 - MathUtils().normalize_value(dist_to_center, 0, max_center_dist, clamp=True))

                    dist_to_edge = min(r_idx, rows - 1 - r_idx, c_idx, cols - 1 - c_idx)
                    max_min_dist_to_edge = min((rows - 1) // 2, (cols - 1) // 2) if rows>0 and cols>0 else 0
                    if max_min_dist_to_edge > 0:
                        current_features_score += FEATURE_WEIGHTS.get("edge_affinity_boost",0) * \
                            (1 - MathUtils().normalize_value(float(dist_to_edge), 0, float(max_min_dist_to_edge), clamp=True))
                
                if current_features_score > max_feature_score_for_cell:
                    max_feature_score_for_cell = current_features_score
            scores[r_idx, c_idx] = MathUtils().normalize_value(max_feature_score_for_cell, 0, max_heuristic_feature_score, clamp=True)
    return scores

# 25. EXT_GM19_Masked_Number_Skip_Pattern_Vec(遮罩數字跳格模式向量)
def EXT_GM19_Masked_Number_Skip_Pattern_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM19"
    logger.debug("Executing EXT_GM19_Masked_Number_Skip_Pattern_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    revealed_numbers_info = [
        {'value': int(grid[r, c]), 'r': r, 'c': c}
        for r in range(rows) for c in range(cols)
        if grid[r, c] != -1 and grid[r, c] > 0
    ]
    if not revealed_numbers_info: return scores

    expected_max_number_on_card = BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols))
    base_positions: Dict[int, Tuple[int, int]] = {}
    for k_val in range(1, expected_max_number_on_card + 1):
        base_r = (k_val - 1) // cols if cols > 0 else 0 # Avoid div by zero if cols is 0 (though caught earlier)
        base_c = (k_val - 1) % cols if cols > 0 else 0
        if base_r < rows: 
            base_positions[k_val] = (base_r, base_c)
    
    skip_vectors: Dict[int, Tuple[int, int]] = {}
    for rn_info in revealed_numbers_info:
        val = rn_info['value']
        if val in base_positions:
            expected_r, expected_c = base_positions[val]
            skip_vectors[val] = (rn_info['r'] - expected_r, rn_info['c'] - expected_c)
    
    if not skip_vectors: return scores
    
    dominant_skip_patterns_strength: Dict[Tuple[int, int], float] = {}
    skip_vector_tuples_list = list(skip_vectors.values())
    if not skip_vector_tuples_list: return scores 
    
    counts = Counter(skip_vector_tuples_list)
    min_occurrences_for_pattern_gm19 = kwargs.get("min_occurrences_for_pattern_gm19", max(1, int(len(skip_vector_tuples_list) * 0.05)))
    
    for skip_vec_tuple, count_val in counts.most_common():
        if count_val >= min_occurrences_for_pattern_gm19:
            pattern_strength = MathUtils().normalize_value(float(count_val), float(min_occurrences_for_pattern_gm19), float(len(skip_vector_tuples_list)), clamp=True)
            dominant_skip_patterns_strength[skip_vec_tuple] = pattern_strength
        else: break
            
    if not dominant_skip_patterns_strength: return scores
    potential_numbers_to_place_set = BoardAnalyzerUtils().get_legal_values_for_placement(grid)

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            cell_max_pattern_score = 0.0
            for p_val_test in potential_numbers_to_place_set:
                if p_val_test not in base_positions: continue
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
def EXT_GM20_Skip_Pattern_Confidence_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A", **kwargs) -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM20"
    logger.debug("Executing EXT_GM20_Skip_Pattern_Confidence_Vec", extra={'request_id': effective_request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    revealed_numbers_info_gm20 = []
    for r_coord in range(rows):
        for c_coord in range(cols):
            if grid[r_coord, c_coord] != -1 and grid[r_coord, c_coord] > 0:
                revealed_numbers_info_gm20.append({'value': int(grid[r_coord, c_coord]), 'r': r_coord, 'c': c_coord})
    if not revealed_numbers_info_gm20: return scores

    expected_max_num_gm20 = BoardAnalyzerUtils().get_card_max_value_from_grid_dimensions((rows, cols))
    base_pos_gm20: Dict[int, Tuple[int, int]] = {
        k: ((k - 1) // cols if cols >0 else 0, (k - 1) % cols if cols >0 else 0) 
        for k in range(1, expected_max_num_gm20 + 1) 
        if ((k - 1) // cols if cols > 0 else 0) < rows
    }
    skip_vecs_initial_gm20: Dict[int, Tuple[int, int]] = {}
    for rn in revealed_numbers_info_gm20:
        val = rn['value']
        if val in base_pos_gm20:
            skip_vecs_initial_gm20[val] = (rn['r'] - base_pos_gm20[val][0], rn['c'] - base_pos_gm20[val][1])
    
    dominant_patterns_details_gm20: List[Dict[str, Any]] = []
    if skip_vecs_initial_gm20:
        skip_tuples_list_gm20 = list(skip_vecs_initial_gm20.values())
        counts_gm20 = Counter(skip_tuples_list_gm20)
        min_occ_gm20 = kwargs.get("min_occurrences_for_pattern_gm20", max(1, int(len(skip_tuples_list_gm20) * 0.05)))
        for skip_v, count_v in counts_gm20.most_common():
            if count_v >= min_occ_gm20:
                pattern_vals = sorted([val_key for val_key, sv_tuple in skip_vecs_initial_gm20.items() if sv_tuple == skip_v]) # Renamed val to val_key
                p_strength = MathUtils().normalize_value(float(count_v), float(min_occ_gm20), float(len(skip_tuples_list_gm20)), clamp=True)
                dominant_patterns_details_gm20.append({'skip': skip_v, 'values': pattern_vals, 'strength': p_strength})
            else: break
    if not dominant_patterns_details_gm20: return scores

    potential_nums_to_place_gm20 = BoardAnalyzerUtils().get_legal_values_for_placement(grid)
    
    arithmetic_enhancement_factor = kwargs.get("arithmetic_enhancement_factor_gm20",0.4)
    internal_gap_bonus_factor = kwargs.get("internal_gap_bonus_factor_gm20",0.1)


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
                    pat_existing_vals = pattern_detail['values']
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
                                    is_arithmetic_now = all(d == diffs_in_temp_seq[0] for d in diffs_in_temp_seq)
                                    first_diff = diffs_in_temp_seq[0]
                                    if is_arithmetic_now and first_diff != 0:
                                        enhancement_factor += arithmetic_enhancement_factor
                                        if len(temp_sequence_with_pval) >=3 and \
                                           min(pat_existing_vals) < p_val_test < max(pat_existing_vals):
                                            enhancement_factor += internal_gap_bonus_factor
                        current_conf = pat_strength * enhancement_factor
                        if current_conf > current_max_conf_for_pval:
                            current_max_conf_for_pval = current_conf
                if current_max_conf_for_pval > max_confidence_score_for_cell_gm20:
                    max_confidence_score_for_cell_gm20 = current_max_conf_for_pval
            scores[r_idx, c_idx] = MathUtils().normalize_value(max_confidence_score_for_cell_gm20, 0, 1.0, clamp=True) # Max possible confidence score can be > 1 due to enhancement_factor
    return scores


#--- Module Registration
# This is the single source of truth for REGISTERED_MODULES_BRAIN
_REGISTERED_MODULES_BRAIN_TEMP: Dict[str, Callable] = { # Removed type hint for Callable for brevity, was complex
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
REGISTERED_MODULES_BRAIN.update(_REGISTERED_MODULES_BRAIN_TEMP)


#Verification (Optional - for testing brain.py directly)
if __name__ == '__main__':
    # Basic logging setup for testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("Verifying brain.py structure...")
    dummy_grid = np.array([[1, 2, -1], [-1, 1, 5], [3, -1, -1]], dtype=int) # Ensure dtype for val_func
    print(f"Created dummy grid:\n{dummy_grid}")

    module_to_test = "EXT_A2_Weighted_Proximity_Vec"
    print(f"\nTesting get_module_score with '{module_to_test}'...")
    try:
        scores_result = get_module_score(module_to_test, dummy_grid, radius=1) # Example with kwarg
        print(f"Successfully called {module_to_test}. Output:\n{scores_result}")
        assert isinstance(scores_result, np.ndarray), "Return type is not np.ndarray"
        assert scores_result.shape == dummy_grid.shape, "Return shape does not match grid shape"
        assert scores_result.dtype == float, "Return dtype is not float"
    except Exception as e:
        print(f"An unexpected error occurred with {module_to_test}: {e}")
        logger.exception("Error during module test")


    print("\nTesting EXT_GM1_Row_Control_Vec with a specific scenario...")
    grid_gm1_test = np.array([
        [1, -1, 3],
        [-1, 5, -1],
        [7, -1, 9]
    ], dtype=int)
    try:
        scores_gm1 = get_module_score("EXT_GM1_Row_Control_Vec", grid_gm1_test)
        print(f"Scores for EXT_GM1_Row_Control_Vec:\n{scores_gm1}")
    except Exception as e:
        print(f"Error testing EXT_GM1_Row_Control_Vec: {e}")
        logger.exception("Error during EXT_GM1_Row_Control_Vec test")

    print("\nTesting EXT_F10_Discontinuity_Vec for sequence completion...")
    grid_f10_test = np.array([
        [2, -1, 6],
        [-1, -1, -1],
        [10, -1, 8]
    ], dtype=int)
    try:
        scores_f10 = get_module_score("EXT_F10_Discontinuity_Vec", grid_f10_test)
        print(f"Scores for EXT_F10_Discontinuity_Vec:\n{scores_f10}")
    except Exception as e:
        print(f"Error testing EXT_F10_Discontinuity_Vec: {e}")
        logger.exception("Error during EXT_F10_Discontinuity_Vec test")

    non_existent_module = "EXT_XXX_NonExistentModule"
    print(f"\nTesting get_module_score with non-existent module '{non_existent_module}'...")
    scores_non_existent = get_module_score(non_existent_module, dummy_grid)
    print(f"Output for non-existent module (should be zeros_like grid):\n{scores_non_existent}")
    assert np.all(scores_non_existent == 0), "Scores for non-existent module are not all zero."


    print("\nListing all registered modules:")
    for i, name in enumerate(REGISTERED_MODULES_BRAIN.keys()):
        print(f"{i + 1}. {name}")
    print(f"\nTotal modules registered: {len(REGISTERED_MODULES_BRAIN)}")

    print("\nbrain.py verification complete.")