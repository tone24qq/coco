# brain1.py
# 本文件自動生成，依據新大腦.pdf、給你2025资料在深度建议一次.pdf、极限强化.pdf 維度實現
# 主要包含 AI 評分模組的核心邏輯與數學實作。
# Optimized for performance using NumPy vectorization and Numba.

import numpy as np
import math
import new_module
from collections import Counter, deque
import logging
from typing import List, Dict, Tuple, Callable, Optional, Any, Set

# Numba import for JIT compilation
import numba
from numba import njit, prange, typed

# Pydantic models for configuration
from pydantic import BaseModel, Field

# 來源：新大腦.pdf - Logging Configuration (Page 1)
# 來源：给你2025资料在深度建议一次.pdf - 日誌與監控整合 (Page 1)
# 來源：main.py (用户需求) - 全局统一配置日志 (Point 4.c)
logger = logging.getLogger(__name__)

# 來源：新大腦.pdf - Helper Utilities (Page 1)
class MathUtils:
    """提供通用數學工具,所有模組統一計算風格"""

    @staticmethod
    @njit
    def sigmoid(x: float, k: float = 1.0) -> float:
        """安全型 sigmoid,避免 overflow"""
        # 來源：新大腦.pdf - MathUtils.sigmoid (Page 1)
        try:
            # PDF 中 $-k^{*}x$ 應為 -k*x
            clamped_x = max(-700.0, min(700.0, -k * x)) # [cite: 2]
            return 1.0 / (1.0 + math.exp(clamped_x))
        except OverflowError: # pragma: no cover (hard to trigger with clamping)
            # 來源：新大腦.pdf - MathUtils.sigmoid (Page 1)
            return 0.0 if -k * x > 0 else 1.0 # [cite: 3]

    @staticmethod
    @njit
    def normalize_value(
        value: float, min_val: float, max_val: float, clamp: bool = True
    ) -> float:
        """
        Normalizes a value to the [0, 1] range.
        Handles cases where min_val equals max_val to prevent division by zero. [cite: 4, 5]
        Addresses Requirement 2.c (reasonable score distribution). [cite: 6]
        來源：新大腦.pdf - MathUtils.normalize_value (Page 1)
        """
        if math.isclose(max_val, min_val):
            if math.isclose(value, min_val): # 來源：新大腦.pdf (Page 1)
                return 0.5
            elif value < min_val: # 來源：新大腦.pdf (Page 2)
                return 0.0 # [cite: 7]
            else:  # value > max_val (which is min_val)
                return 1.0
        normalized = (value - min_val) / (max_val - min_val)
        if clamp:
            return max(0.0, min(1.0, normalized))
        return normalized

    @staticmethod
    @njit
    def manhattan_distance(p1_r: int, p1_c: int, p2_r: int, p2_c: int) -> int:
        """Calculates Manhattan distance between two points (r, c).
        來源：新大腦.pdf - MathUtils.manhattan_distance (Page 2) [cite: 5, 9]
        Modified to accept r, c components directly for Numba.
        """
        return abs(p1_r - p2_r) + abs(p1_c - p2_c)

    @staticmethod
    @njit
    def euclidean_distance(p1_r: float, p1_c: float, p2_r: float, p2_c: float) -> float:
        """Calculates Euclidean distance between two points (r, c).
        來源：新大腦.pdf - MathUtils.euclidean_distance (Page 1) [cite: 6, 10]
        Modified to accept r, c components directly for Numba.
        """
        # 來源：新大腦.pdf - MathUtils.euclidean_distance (Page 2)
        return math.sqrt((p1_r - p2_r) ** 2 + (p1_c - p2_c) ** 2)

    @staticmethod
    # @njit # Counter is not supported by Numba directly. If critical, rewrite with basic loops.
    # For now, keeping as Python, assuming it's not the primary bottleneck across all modules.
    def get_entropy(values: List[Any]) -> float: # Numba typed list could be used if values are homogenous
        """Calculates Shannon entropy for a list of values.
        來源：新大腦.pdf - MathUtils.get_entropy (Page 2) [cite: 7, 11]
        """
        if not values:
            return 0.0
        
        # Numba-compatible Counter replacement
        counts_dict = {}
        for item in values:
            counts_dict[item] = counts_dict.get(item, 0) + 1
            
        total_count = len(values)
        entropy = 0.0
        for count in counts_dict.values():
            probability = count / total_count
            if probability > 0: # Avoid log(0) # [cite: 12]
                 entropy -= probability * math.log2(probability)
        return entropy

# 來源：新大腦.pdf - BoardAnalyzerUtils (Page 2) [cite: 8, 13]
class BoardAnalyzerUtils:
    """
    Provides common board analysis utility functions. [cite: 8, 13]
    Used by modules to inspect grid neighborhoods, gradients, etc. [cite: 8, 13]
    """

    @staticmethod
    # This function is hard to fully vectorize with NumPy due to the arbitrary val_func
    # and dynamic neighbor list construction. Numba is a good candidate.
    @njit
    def get_neighborhood_values_numba(
        grid: np.ndarray, # int array
        r: int,
        c: int,
        radius: int = 1,
        eight_connectivity: bool = True,
        # val_func signature changed for Numba: returns (value, is_valid)
        # Example: (float(x_val), True) if x_val != -1 else (0.0, False)
        # This is more complex to pass a generic val_func to Numba.
        # For now, let's assume a simplified val_func logic directly in Numba,
        # or require val_func to be a Numba-jitted function if passed.
        # The original val_func was: lambda x_val: float(x_val) if x_val != -1 else None
        # We will replicate: append float(grid[nr, nc]) if grid[nr,nc] != -1
        include_center: bool = False,
    ) -> numba.typed.List: # Returns a Numba typed list of floats
        """
        Retrieves values from the neighborhood of a cell (Numba-optimized).
        Supports configurable radius, connectivity. Processes values as float if not -1.
        """
        neighbors = numba.typed.List() # type: numba.typed.List[float]
        rows, cols = grid.shape
        for dr_val in range(-radius, radius + 1):
            for dc_val in range(-radius, radius + 1):
                if not include_center and dr_val == 0 and dc_val == 0: # [cite: 17]
                    continue
                if not eight_connectivity:
                    if radius == 1 and abs(dr_val) + abs(dc_val) != 1: # Only N, E, S, W
                        continue
                    # Complex non-eight_connectivity for radius > 1 logic from PDF omitted for clarity,
                    # as it was ambiguous. This will behave as 8-conn for radius > 1 if not eight_connectivity. [cite: 18, 19, 20, 21, 22, 23, 24, 25]
                
                nr, nc = r + dr_val, c + dc_val # [cite: 26]
                if 0 <= nr < rows and 0 <= nc < cols: # [cite: 26]
                    val_at_neighbor = grid[nr, nc]
                    if val_at_neighbor != -1:
                        neighbors.append(float(val_at_neighbor)) # [cite: 27]
        return neighbors
    
    @staticmethod
    def get_neighborhood_values( # Python wrapper for type hinting and TypedList to List conversion
        grid: np.ndarray,
        r: int,
        c: int,
        radius: int = 1,
        eight_connectivity: bool = True,
        val_func: Callable[[int], Optional[float]] = lambda x_val: float(x_val) if x_val != -1 else None, # Original signature
        include_center: bool = False,
    ) -> List[float]: # [cite: 14]
        """ Python wrapper for get_neighborhood_values_numba using the original val_func semantic """
        # This wrapper re-implements the logic if val_func is complex and cannot be easily passed to Numba.
        # Or, if val_func is simple (like the default), we can use a specialized Numba version.
        # The default val_func logic is simple enough to be embedded in the Numba version.
        # So, we directly call the Numba version that has this logic.
        # The Numba version now returns List[float64] essentially.
        # For type consistency in Python, convert Numba typed list to Python list.
        # 來源：给你2025资料在深度建议一次.pdf -通用型別提示更新範例 (Page 1)
        # 來源：新大腦.pdf - BoardAnalyzerUtils.get_neighborhood_values (Page 2) [cite: 9, 15, 16]

        # Simplified: directly use a Numba-fied version that handles the common val_func case
        numba_list = BoardAnalyzerUtils.get_neighborhood_values_numba(
            grid, r, c, radius, eight_connectivity, include_center
        )
        return list(numba_list)


    @staticmethod
    @njit
    def get_value_gradient_at_cell_numba(
        grid: np.ndarray, # int array
        r: int,
        c: int,
        # Default val_func logic: float(x_val) if x_val != -1 else 0.0
    ) -> Tuple[float, float]: # [cite: 28]
        """Calculates an approximate gradient (Sobel-like) at a cell (Numba-optimized). [cite: 11]"""
        rows, cols = grid.shape

        # Inner function for safe value access, Numba compatible
        # @njit # Not needed if this whole staticmethod is jitted
        def safe_val(r_in: int, c_in: int) -> float:
            if 0 <= r_in < rows and 0 <= c_in < cols:
                val = grid[r_in, c_in]
                return float(val) if val != -1 else 0.0 # [cite: 29]
            return 0.0

        gx = (safe_val(r - 1, c + 1) + 2 * safe_val(r, c + 1) + safe_val(r + 1, c + 1)) - \
             (safe_val(r - 1, c - 1) + 2 * safe_val(r, c - 1) + safe_val(r + 1, c - 1)) # [cite: 32]
        
        gy = (safe_val(r + 1, c - 1) + 2 * safe_val(r + 1, c) + safe_val(r + 1, c + 1)) - \
             (safe_val(r - 1, c - 1) + 2 * safe_val(r - 1, c) + safe_val(r - 1, c + 1)) # [cite: 33]
        
        return gx, gy

    @staticmethod
    def get_value_gradient_at_cell( # Python wrapper
        grid: np.ndarray,
        r: int,
        c: int,
        val_func: Callable[[int], float] = lambda x_val: float(x_val) if x_val != -1 else 0.0, # [cite: 28]
    ) -> Tuple[float, float]: # [cite: 11]
        """ Python wrapper for Numba-optimized gradient calculation.
            Note: The provided val_func is effectively hardcoded in the Numba version for performance.
            If a different val_func behavior is critical, the Numba version needs adaptation or this wrapper
            would need to pre-process the grid using the Python val_func.
        """
        # Assuming default val_func is used, call Numba version.
        if val_func(0) == 0.0 and val_func(1) == 1.0 and val_func(-1) == 0.0: # Heuristic check for default
             return BoardAnalyzerUtils.get_value_gradient_at_cell_numba(grid, r, c)
        else: # Fallback to slower Python version if val_func is custom and complex
            logger.warning("Custom val_func used in get_value_gradient_at_cell, Numba optimization bypassed for this call.")
            rows, cols = grid.shape
            def safe_val_py(r_in: int, c_in: int) -> float:
                if 0 <= r_in < rows and 0 <= c_in < cols:
                    return val_func(grid[r_in, c_in])
                return 0.0
            gx = (safe_val_py(r - 1, c + 1) + 2 * safe_val_py(r, c + 1) + safe_val_py(r + 1, c + 1)) - \
                 (safe_val_py(r - 1, c - 1) + 2 * safe_val_py(r, c - 1) + safe_val_py(r + 1, c - 1))
            gy = (safe_val_py(r + 1, c - 1) + 2 * safe_val_py(r + 1, c) + safe_val_py(r + 1, c + 1)) - \
                 (safe_val_py(r - 1, c - 1) + 2 * safe_val_py(r - 1, c) + safe_val_py(r - 1, c + 1))
            return gx, gy


    @staticmethod
    @njit
    def _is_close_numba(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    @staticmethod
    @njit
    def find_sequences_in_line_numba(
        line_arr: np.ndarray, # Expects 1D NumPy array of floats (with np.nan for gaps)
        min_len: int = 3,
        check_arithmetic: bool = True, # [cite: 34]
        check_geometric: bool = False, # [cite: 34]
        allow_gaps: int = 0,
    ) -> numba.typed.List: # Returns List[List[int]] (Numba typed lists)
        """
        Numba-optimized version of find_sequences_in_line.
        Uses np.nan to represent gaps internally.
        來源：新大腦.pdf - BoardAnalyzerUtils.find_sequences_in_line (Page 3-5) [cite: 35]
        """
        sequences_res = numba.typed.List() # type: numba.typed.List[numba.typed.List[np.int_]]
        n = len(line_arr)
        if n == 0: # [cite: 35]
            return sequences_res

        for i in range(n):
            if math.isnan(line_arr[i]): # Cannot start sequence with a gap [cite: 37]
                continue
            start_val = line_arr[i] # [cite: 38]

            # Arithmetic sequence check [cite: 39]
            if check_arithmetic:
                for j in range(i + 1, n):
                    gaps_between_i_j = 0
                    for k_gap_check in range(i + 1, j):
                        if math.isnan(line_arr[k_gap_check]): # [cite: 40]
                            gaps_between_i_j +=1
                    
                    if gaps_between_i_j > allow_gaps: # [cite: 41]
                        continue
                    if math.isnan(line_arr[j]): # [cite: 42, 43, 44]
                        continue 
                    
                    val_j = line_arr[j]
                    diff: float = val_j - start_val # [cite: 45, 46, 47, 48, 49]
                    
                    # PDF: "Avoid constant sequences unless they are all zeros" [cite: 50]
                    if BoardAnalyzerUtils._is_close_numba(diff, 0.0) and not BoardAnalyzerUtils._is_close_numba(start_val, 0.0):
                        continue

                    current_seq_values_list = numba.typed.List() # type: numba.typed.List[np.int_]
                    current_seq_values_list.append(int(round(start_val))) # [cite: 51]
                    
                    if gaps_between_i_j == 0: # [cite: 52]
                        current_seq_values_list.append(int(round(val_j)))

                    last_val_in_seq = val_j
                    last_idx_in_seq = j # [cite: 53]
                    potential_gap_count_after_j = 0

                    for k in range(j + 1, n):
                        val_k = line_arr[k]
                        if math.isnan(val_k): # [cite: 54]
                            potential_gap_count_after_j += 1
                            if potential_gap_count_after_j > allow_gaps: # [cite: 55]
                                break 
                            continue
                        
                        steps_from_last = (k - last_idx_in_seq) # [cite: 56]
                        expected_val_at_k = last_val_in_seq + diff * (steps_from_last / (potential_gap_count_after_j + 1.0)) # Ensure float division
                        
                        if BoardAnalyzerUtils._is_close_numba(val_k, expected_val_at_k): # [cite: 57]
                            current_seq_values_list.append(int(round(val_k)))
                            last_val_in_seq = val_k
                            last_idx_in_seq = k # [cite: 58]
                            potential_gap_count_after_j = 0
                        else: # [cite: 59]
                            break 

                    if len(current_seq_values_list) >= min_len:
                        sequences_res.append(current_seq_values_list)
            
            # Geometric sequence check [cite: 60]
            if check_geometric and not BoardAnalyzerUtils._is_close_numba(start_val, 0.0):
                for j in range(i + 1, n):
                    gaps_between_i_j = 0
                    for k_gap_check in range(i + 1, j): # [cite: 61]
                        if math.isnan(line_arr[k_gap_check]):
                            gaps_between_i_j +=1 # [cite: 62]
                    
                    if gaps_between_i_j > allow_gaps:
                        continue
                    
                    val_j = line_arr[j]
                    if math.isnan(val_j) or BoardAnalyzerUtils._is_close_numba(val_j, 0.0): # [cite: 63]
                        continue
                    if BoardAnalyzerUtils._is_close_numba(start_val, 0.0): continue # [cite: 64]

                    ratio_candidate = val_j / start_val # [cite: 66]
                    
                    # PDF: "Avoid constant sequences" [cite: 67]
                    if BoardAnalyzerUtils._is_close_numba(ratio_candidate, 1.0) and \
                       not BoardAnalyzerUtils._is_close_numba(start_val, val_j):
                        continue

                    current_seq_values_geo_list = numba.typed.List() # type: numba.typed.List[np.int_]
                    current_seq_values_geo_list.append(int(round(start_val))) # [cite: 68]
                    if gaps_between_i_j == 0:
                        current_seq_values_geo_list.append(int(round(val_j)))

                    last_val_in_seq = val_j # [cite: 69]
                    last_idx_in_seq = j
                    potential_gap_count_after_j = 0
                    ratio = ratio_candidate

                    for k in range(j + 1, n): # [cite: 70]
                        val_k = line_arr[k]
                        if math.isnan(val_k):
                            potential_gap_count_after_j += 1
                            if potential_gap_count_after_j > allow_gaps: # [cite: 71]
                                break
                            continue
                        
                        if BoardAnalyzerUtils._is_close_numba(val_k, 0.0) : break # [cite: 72]
                        
                        num_ratio_applications_float = (k - last_idx_in_seq) / (potential_gap_count_after_j + 1.0) # [cite: 73]
                        if not BoardAnalyzerUtils._is_close_numba(num_ratio_applications_float, round(num_ratio_applications_float)): # Check if it's an integer number of steps
                             break # Not a clean step [cite: 73]
                        num_ratio_applications = int(round(num_ratio_applications_float))
                        
                        if num_ratio_applications <=0 : break # Should not happen if k > last_idx_in_seq

                        expected_val_at_k = last_val_in_seq * (ratio ** num_ratio_applications) # [cite: 74]

                        if BoardAnalyzerUtils._is_close_numba(val_k, expected_val_at_k):
                            current_seq_values_geo_list.append(int(round(val_k))) # [cite: 75]
                            last_val_in_seq = val_k
                            last_idx_in_seq = k
                            potential_gap_count_after_j = 0
                        else: # [cite: 76]
                            break
                    
                    if len(current_seq_values_geo_list) >= min_len:
                        sequences_res.append(current_seq_values_geo_list) # [cite: 77]

        # Remove duplicate sequences (Python part, as Numba typed list of lists is tricky for set operations)
        # This part will be handled in the Python wrapper.
        return sequences_res

    @staticmethod
    def find_sequences_in_line(
        line: List[Union[int, float]], # Original type [cite: 34]
        min_len: int = 3,
        check_arithmetic: bool = True,
        check_geometric: bool = False,
        allow_gaps: int = 0,
    ) -> List[List[int]]: # Returns sequences of original integer values [cite: 35]
        """
        Python wrapper for Numba-optimized find_sequences_in_line.
        Handles conversion to NumPy array with NaNs and back to Python list of lists.
        """
        # Convert line to NumPy array with NaNs for gaps
        line_np = np.array([float(x) if x != -1 else np.nan for x in line], dtype=np.float64)

        numba_sequences = BoardAnalyzerUtils.find_sequences_in_line_numba(
            line_np, min_len, check_arithmetic, check_geometric, allow_gaps
        )
        
        # Convert Numba typed list of lists to Python list of lists and remove duplicates
        py_sequences = []
        temp_set_for_uniqueness = set()
        for numba_list_inner in numba_sequences:
            py_list_inner = list(numba_list_inner) # Convert Numba list to Python list
            # To ensure uniqueness for sequences regardless of order if found multiple times
            # (e.g. from different start points that yield the same sequence)
            # we convert the list to a tuple to add to a set.
            seq_tuple = tuple(py_list_inner)
            if seq_tuple not in temp_set_for_uniqueness:
                py_sequences.append(py_list_inner)
                temp_set_for_uniqueness.add(seq_tuple)
        
        return py_sequences # [cite: 78]

    @staticmethod
    @njit
    def get_card_max_value_from_grid_dimensions(grid_shape_rows: int, grid_shape_cols: int) -> int: # [cite: 16]
        """Calculates the maximum possible number on the card based on its dimensions. [cite: 16]"""
        if grid_shape_rows == 0 or grid_shape_cols == 0: # [cite: 79]
            return 0
        return grid_shape_rows * grid_shape_cols

    @staticmethod
    # @njit # Set comprehension with range might be slow to compile or less efficient than Python's set for small N
    def get_all_possible_numbers_for_grid(grid_shape: Tuple[int, int]) -> Set[int]: # [cite: 17]
        """Returns a set of all numbers that could theoretically appear on a grid of given
        dimensions. [cite: 17, 80]"""
        rows, cols = grid_shape
        max_val = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions(
            rows, cols
        ) # 來源：新大腦.pdf (Page 5)
        if max_val == 0:
            return set()
        return set(range(1, max_val + 1))

    @staticmethod
    # @njit # Set operations on potentially large sets, Numba might not offer speedup over CPython's optimized set ops.
    def get_legal_values_for_placement(grid: np.ndarray) -> Set[int]: # [cite: 18]
        """
        Determines the set of numbers that can be legally placed onto an empty cell in the grid. [cite: 81]
        This adheres to the rule: numbers are 1 to R*C and no positive number can be repeated. [cite: 82, 83, 19]
        (Requirement 1.c) [cite: 20]
        來源：新大鵝.pdf - BoardAnalyzerUtils.get_legal_values_for_placement (Page 5-6)
        """
        if grid.size == 0: # 來源：新大腦.pdf (Page 6)
            return set()
        rows, cols = grid.shape
        all_possible_on_this_grid = (
            BoardAnalyzerUtils.get_all_possible_numbers_for_grid((rows, cols))
        ) # [cite: 84]
        
        # Efficient way to get unique positive values from grid
        # grid.flatten() creates a copy. For very large grids, consider iterating.
        # However, for typical grid sizes, this is often fine and clear.
        used_positive_values_on_board = set()
        flat_grid = grid.ravel() # ravel() is often more memory-efficient than flatten()
        for v_val in flat_grid:
            if v_val != -1 and v_val > 0:
                used_positive_values_on_board.add(int(v_val))
                
        legal_placements = all_possible_on_this_grid - used_positive_values_on_board
        return legal_placements

# --- Pydantic Config Models for Modules ---
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - 通用強化思路 (參數動態化)
# 來源：给你2025资料在深度建议一次.pdf - 統一的配置管理, Pydantic V2 (Page 9, Page 1)

class BaseModuleConfig(BaseModel):
    enabled: bool = Field(default=True, description="Whether this module is enabled.") # [cite: 85]
    weight: float = Field(default=1.0, ge=0.0, description="Weight of this module's score in aggregation.") # [cite: 85]

class WeightedProximityConfig(BaseModuleConfig):
    # 來源：新大腦.pdf - EXT_A2 parameters (Page 7)
    # 來源：给你2025资料在深度建议一次.pdf - EXT_A2 Pydantic配置範例 (Page 2)
    radius: int = Field(default=2, ge=1, description="考慮的鄰域半徑")
    value_weight_factor: float = Field(default=0.1, ge=0.0, description="鄰居值的權重因子")
    distance_decay_factor: float = Field(default=1.5, gt=0.0, description="距離衰減因子")
    # 來源：新大腦.pdf - EXT_A2 Conceptual repulsion (Page 7)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - A2 斥力概念的細化
    enable_repulsion: bool = Field(default=False, description="是否啟用斥力概念") # [cite: 86]
    undesirable_pairs_config: Dict[Tuple[int, int], float] = Field(default_factory=dict, description="不良配對及其斥力因子, e.g. {(1,1): -0.2}") # [cite: 87]


class LocalHeterogeneityConfig(BaseModuleConfig):
    # 來源：新大腦.pdf - EXT_M3 parameters (Page 9)
    # 來源：给你2025资料在深度建议一次.pdf - EXT_M3 Pydantic配置範例 (Page 2 of previous response)
    radius: int = Field(default=1, ge=1, description="異質性計算的鄰域半徑")
    min_neighbors_for_robust_score: int = Field(default=2, ge=0, description="計算有效熵的最小鄰居數")
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - M3 熵以外的異質性度量
    diversity_metric: str = Field(default="entropy", pattern="^(entropy|gini|unique_count)$", description="異質性度量方法: entropy, gini, or unique_count")


class PotentialFieldConfig(BaseModuleConfig):
    # 來源：新大腦.pdf - EXT_D3 parameters (Page 10-11)
    decay_exponent: float = Field(default=1.5, gt=0.0, description="影響力隨距離衰減的指數 (e.g., 1 for 1/r, 2 for 1/r^2)") # [cite: 39, 88]
    max_influence_radius: int = Field(default=3, ge=1, description="考慮數字影響力的最大曼哈頓距離") # [cite: 39, 88]
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - D3 「電荷」性質的擴展
    enable_negative_charges: bool = Field(default=False, description="是否啟用負電荷（排斥力）概念")
    negative_charge_map: Dict[int, float] = Field(default_factory=dict, description="定義哪些數字視為負電荷及其強度（<0）")


class DiscontinuityRepairConfig(BaseModuleConfig):
    # 來源：新大腦.pdf - EXT_F10 parameters (Page 12)
    # 來源：给你2025资料在深度建议一次.pdf - EXT_F10 Pydantic配置範例 (Page 4)
    min_sequence_len_to_score: int = Field(default=3, ge=2, description="視為有效的最小序列長度")
    allow_gaps_in_sequence: int = Field(default=1, ge=0, description="序列中允許的最大間隙數") # [cite: 43]
    check_arithmetic: bool = Field(default=True, description="是否檢查等差序列")
    check_geometric: bool = Field(default=False, description="是否檢查等比序列") # [cite: 89]
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - F10 序列價值評估
    sequence_quality_weighting: bool = Field(default=False, description="是否對序列質量（如構成數字大小）進行額外加權")
    high_value_sequence_threshold_factor: float = Field(default=0.75, ge=0, le=1, description="序列平均值超過盤面最大值*此因子時視為高價值")


class PathfindingValueConfig(BaseModuleConfig):
    # 來源：新大腦.pdf - EXT_P7 parameters (Page 14)
    max_path_search_depth: int = Field(default=4, ge=1, description="搜尋路徑的最大長度") # [cite: 51]
    path_value_decay_factor: float = Field(default=1.0, ge=0.0, description="路徑長度對價值的衰減因子 (e.g., val / (len^decay))") # [cite: 51]
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - P7 BFS優化 / 只針對高價值
    target_value_threshold_factor: float = Field(default=0.5, ge=0, le=1, description="只尋找連接到值高於盤面最大值*此因子的路徑 (0=不篩選)")


class ResourceControlConfig(BaseModuleConfig):
    # 來源：新大腦.pdf - EXT_R5 parameters (Page 16-17) # [cite: 104]
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - R5「資源」定義的擴展
    w_row_completeness: float = Field(default=0.3, ge=0.0, le=1.0, description="行完成度分數的權重")
    w_col_completeness: float = Field(default=0.3, ge=0.0, le=1.0, description="列完成度分數的權重")
    w_value_capture: float = Field(default=0.4, ge=0.0, le=1.0, description="價值捕獲分數的權重")

class LineControlConfig(BaseModuleConfig): # For GM1 and GM2
    # 來源：新大腦.pdf - EXT_GM1/GM2 parameters (Page 18, 20)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM1/GM2 序列評估的增強
    w_density: float = Field(default=0.4, ge=0.0, le=1.0, description="密度分數權重")
    w_sum_score: float = Field(default=0.3, ge=0.0, le=1.0, description="總和分數權重")
    w_sequence_score: float = Field(default=0.3, ge=0.0, le=1.0, description="序列分數權重") # [cite: 105]
    use_advanced_sequence_detection: bool = Field(default=True, description="是否使用 BoardAnalyzerUtils.find_sequences_in_line 進行序列評估")
    min_len_for_sequence_score: int = Field(default=3, ge=2)
    allow_gaps_for_sequence_score: int = Field(default=1, ge=0)

class ConnectedComponentConfig(BaseModuleConfig): # For GM3
    # 來源：新大腦.pdf - EXT_GM3 parameters (Page 21-22)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM3 連通區域的「形狀」和「質量」
    consider_shape_factor: bool = Field(default=False, description="是否考慮連通區域的形狀因子（概念性）")
    shape_factor_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="形狀因子權重（如果啟用）")

class SpatialAutocorrelationConfig(BaseModuleConfig): # For GM4
    # 來源：新大腦.pdf - EXT_GM4 parameters (Page 23-24)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM4 自相關性方向 # [cite: 106]
    autocorrelation_type: str = Field(default="positive", pattern="^(positive|negative)$", description="偏好的自相關類型（positive: 聚集, negative: 交錯）")
    neighborhood_radius: int = Field(default=1, ge=1)
    use_median_for_hypothetical: bool = Field(default=True, description="是否使用潛在數字的中位數作為假設值，否則用平均值")

class LineCompletionConfig(BaseModuleConfig): # For GM5
    # 來源：新大腦.pdf - EXT_GM5 parameters (Page 24-25)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM5 線段長度和類型的擴展
    target_line_length: int = Field(default=3, ge=3, description="目標補全的線段長度")
    score_identical_3: float = Field(default=0.6, ge=0.0)
    score_arithmetic_3_mend: float = Field(default=0.7, ge=0.0)
    score_arithmetic_3_extend: float = Field(default=0.5, ge=0.0)
    # 來源：新大腦.pdf - EXT_GM5 Added: scoring for quality (conceptual) (Page 25)
    enable_quality_enhancement: bool = Field(default=True) # [cite: 107]
    score_arithmetic_3_mend_high_val_bonus: float = Field(default=0.2, ge=0.0, description="高價值等差序列修復額外獎勵")
    high_value_threshold_factor_gm5: float = Field(default=0.66, ge=0, le=1, description="平均值超過盤面最大值*此因子視為高價值")