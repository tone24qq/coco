# brain1.py
# Part 1 of 3: Contains shared utilities and the first set of AI scoring modules.
# Based on Brain.txt, which was generated according to 新大腦.pdf, 给你2025资料在深度建议一次.pdf, 极限强化.pdf

import numpy as np
import math
from collections import Counter, deque
import logging
from typing import List, Dict, Tuple, Callable, Optional, Any, Set

# 來源：新大腦.pdf - Logging Configuration (Page 1)
# 來源：给你2025资料在深度建议一次.pdf - 日誌與監控整合 (Page 1)
# 來源：main.py (用户需求) - 全局统一配置日志 (Point 4.c)
logger = logging.getLogger(__name__)

# 來源：新大腦.pdf - Helper Utilities (Page 1)
class MathUtils:
    """提供通用數學工具,所有模組統一計算風格"""

    @staticmethod
    def sigmoid(x: float, k: float = 1.0) -> float:
        """安全型 sigmoid,避免 overflow"""
        # 來源：新大腦.pdf - MathUtils.sigmoid (Page 1)
        try:
            # PDF 中 $-k^{*}x$ 應為 -k*x
            clamped_x = max(-700.0, min(700.0, -k * x))
            return 1 / (1 + math.exp(clamped_x))
        except OverflowError: # [cite: 15]
            # 來源：新大腦.pdf - MathUtils.sigmoid (Page 1)
            return 0.0 if -k * x > 0 else 1.0 # [cite: 16]

    @staticmethod
    def normalize_value(
        value: float, min_val: float, max_val: float, clamp: bool = True
    ) -> float:
        """
        Normalizes a value to the [0, 1] range. [cite: 17]
        Handles cases where min_val equals max_val to prevent division by zero. [cite: 17, 18]
        Addresses Requirement 2.c (reasonable score distribution). [cite: 19]
        來源：新大腦.pdf - MathUtils.normalize_value (Page 1)
        """
        if math.isclose(max_val, min_val):
            if math.isclose(value, min_val): # 來源：新大腦.pdf (Page 1)
                return 0.5
            elif value < min_val: # 來源：新大腦.pdf (Page 2)
                return 0.0 # [cite: 20]
            else:  # value > max_val (which is min_val)
                return 1.0
        normalized = (value - min_val) / (max_val - min_val)
        if clamp:
            return max(0.0, min(1.0, normalized))
        return normalized

    @staticmethod
    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int: # [cite: 21]
        """Calculates Manhattan distance between two points (r, c).
        來源：新大腦.pdf - MathUtils.manhattan_distance (Page 2) [cite: 5, 22]
        """
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @staticmethod
    def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculates Euclidean distance between two points (r, c).
        來源：新大腦.pdf - MathUtils.euclidean_distance (Page 1) [cite: 6, 23]
        """
        # 來源：新大腦.pdf - MathUtils.euclidean_distance (Page 2)
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def get_entropy(values: List[Any]) -> float:
        """Calculates Shannon entropy for a list of values.
        來源：新大腦.pdf - MathUtils.get_entropy (Page 2) [cite: 7, 24]
        """
        if not values:
            return 0.0
        counts = Counter(values)
        total_count = len(values)
        entropy = 0.0
        for count in counts.values():
            probability = count / total_count
            if probability > 0: # Avoid log(0) # [cite: 25]
                 entropy -= probability * math.log2(probability)
        return entropy


# 來源：新大腦.pdf - BoardAnalyzerUtils (Page 2) [cite: 8]
class BoardAnalyzerUtils:
    """
    Provides common board analysis utility functions. [cite: 26]
    Used by modules to inspect grid neighborhoods, gradients, etc. [cite: 8, 26]
    """

    @staticmethod
    # 來源：给你2025资料在深度建议一次.pdf -通用型別提示更新範例 (Page 1)
    # 來源：新大腦.pdf - BoardAnalyzerUtils.get_neighborhood_values (Page 2) [cite: 9]
    def get_neighborhood_values(
        grid: np.ndarray,
        r: int,
        c: int,
        radius: int = 1,
        eight_connectivity: bool = True,
        val_func: Callable[[int], float | None] = lambda x_val: float(x_val) # [cite: 27]
        if x_val != -1
        else None,
        include_center: bool = False,
    ) -> List[float]:
        """
        Retrieves values from the neighborhood of a cell. [cite: 28]
        Supports configurable radius, connectivity, and value processing. [cite: 28, 29]
        來源：新大腦.pdf - BoardAnalyzerUtils.get_neighborhood_values (Page 2)
        """
        neighbors: List[float] = []
        rows, cols = grid.shape
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if not include_center and dr == 0 and dc == 0: # [cite: 30]
                    continue
                if not eight_connectivity:
                    if radius == 1 and abs(dr) + abs(dc) != 1: # Only N, E, S, W
                        continue
                    # 來源：新大腦.pdf - BoardAnalyzerUtils.get_neighborhood_values (Page 2) # [cite: 31]
                    # Original PDF had a typo: abs(dr)+abs(dc)>radius; (semicolon)
                    # This condition for radius > 1 and not eight_connectivity is a bit ambiguous. [cite: 32]
                    # Assuming for non-eight_connectivity it means only cardinal directions up to `radius` distance, [cite: 32]
                    # or a diamond shape. The PDF's example implies a filter for specific patterns. [cite: 32]
                    # For simplicity and clarity, if not eight_connectivity and radius > 1, [cite: 33]
                    # we might interpret it as still cardinal but within the larger radius. [cite: 33]
                    # However, the PDF example `abs(dr)+abs(dc)>radius` seems to be for another case. [cite: 34]
                    # Given the ambiguity, sticking to the radius 1 case for non-eight_connectivity [cite: 35]
                    # or assuming it only applies if radius=1. [cite: 35]
                    # For radius > 1, non-eight_connectivity is less standard. [cite: 36]
                    # For now, this will behave as only 4-connectivity if radius=1 and not eight_connectivity. [cite: 37]
                    # If radius > 1 and not eight_connectivity, it will behave like eight_connectivity. [cite: 38]
                    # This part might need further clarification based on exact desired behavior for larger radii without 8-connectivity. [cite: 38]
                nr, nc = r + dr, c + dc # 來源：新大腦.pdf (Page 2) [cite: 39]
                if 0 <= nr < rows and 0 <= nc < cols: # 來源：新大腦.pdf (Page 2) [cite: 39]
                    processed_val = val_func(grid[nr, nc])
                    if processed_val is not None:
                        neighbors.append(processed_val) # [cite: 40]
        return neighbors

    @staticmethod
    # P來源：新大腦.pdf - BoardAnalyzerUtils.get_value_gradient_at_cell (Page 2-3) [cite: 11]
    def get_value_gradient_at_cell(
        grid: np.ndarray,
        r: int,
        c: int,
        val_func: Callable[[int], float] = lambda x_val: float(x_val)
        if x_val != -1
        else 0.0, # 來源：新大腦.pdf (Page 3) [cite: 41]
    ) -> Tuple[float, float]:
        """Calculates an approximate gradient (Sobel-like) at a cell. [cite: 11] Useful for modules
        analyzing value changes. [cite: 11]"""
        rows, cols = grid.shape

        def safe_val(r_in: int, c_in: int) -> float:
            if 0 <= r_in < rows and 0 <= c_in < cols:
                return val_func(grid[r_in, c_in]) # [cite: 42]
            return 0.0

        # Sobel operators
        # Gx = ( (top-right + 2*middle-right + bottom-right) -
        #        (top-left  + 2*middle-left  + bottom-left) )
        # Gy = ( (bottom-left + 2*bottom-middle + bottom-right) -
        #        (top-left    + 2*top-middle    + top-right) ) # [cite: 43]
        # 來源：新大腦.pdf - Gx, Gy calculation (Page 3)
        # Note: PDF formula for gx seems to have a factor of 1, e.g. [cite: 44]
        # "...)-1.(safe_val...)", assuming typo and it's a minus. [cite: 44]
        # And gy has "sate_val", corrected to "safe_val". [cite: 45]
        gx = (safe_val(r - 1, c + 1) + 2 * safe_val(r, c + 1) + safe_val(r + 1, c + 1)) - \
             (safe_val(r - 1, c - 1) + 2 * safe_val(r, c - 1) + safe_val(r + 1, c - 1))
        
        gy = (safe_val(r + 1, c - 1) + 2 * safe_val(r + 1, c) + safe_val(r + 1, c + 1)) - \
              (safe_val(r - 1, c - 1) + 2 * safe_val(r - 1, c) + safe_val(r - 1, c + 1)) # [cite: 46]
        
        return gx, gy

    @staticmethod
    # 來源：新大腦.pdf - BoardAnalyzerUtils.find_sequences_in_line (Page 3)
    def find_sequences_in_line(
        line: List[int | float], # Allow float for geometric intermediate steps
        min_len: int = 3,
        check_arithmetic: bool = True, # [cite: 47]
        check_geometric: bool = False,
        allow_gaps: int = 0,
    ) -> List[List[int]]: # Returns sequences of original integer values
        """
        Finds arithmetic or geometric sequences in a 1D list of numbers,
        supporting gaps and returning sequence elements.
        This is a more faithful implementation of the PDF's logic.
        來源：新大腦.pdf - BoardAnalyzerUtils.find_sequences_in_line (Page 3-5) [cite: 48]
        """
        sequences: List[List[int]] = []
        n = len(line)
        if n == 0: # handle empty line explicitly
            return sequences
        
        # Convert to float for internal processing, especially for geometric [cite: 49]
        # but keep track of original int values for the final sequence list. [cite: 49]
        # -1 (gap) will be handled as a special marker. [cite: 49]
        
        processed_line: List[float | None] = []
        for x in line:
            if x == -1:
                processed_line.append(None) # Using None for gaps internally
            else: # [cite: 50]
                processed_line.append(float(x))


        for i in range(n):
            if processed_line[i] is None: # Cannot start sequence with a gap
                continue

            start_val = processed_line[i]
            assert start_val is not None # Should be true due to previous continue # [cite: 51]

            # Arithmetic sequence check
            if check_arithmetic:
                # 來源：新大腦.pdf - Arithmetic sequence check (Page 3)
                # Iterate through all possible second elements to define a difference [cite: 52]
                for j in range(i + 1, n): # [cite: 52]
                    gaps_between_i_j = 0
                    k_gap_check = i + 1
                    while k_gap_check < j:
                        if processed_line[k_gap_check] is None: # [cite: 53]
                            gaps_between_i_j +=1
                        k_gap_check +=1
                    
                    if gaps_between_i_j > allow_gaps: # [cite: 54]
                        continue # Too many gaps to define initial difference with j

                    if processed_line[j] is None:
                        if j == i + 1 and allow_gaps == 0 : continue # Cannot define diff with immediate gap if no gaps allowed # [cite: 55]
                        if j > i + 1 and (j - (i + gaps_between_i_j) > 1) and allow_gaps < gaps_between_i_j +1 : continue # [cite: 56]
                        # If allow_gaps > 0, we might be able to find a diff with a later element. [cite: 56]
                        # This loop structure is for finding the *first* element to establish 'diff'. [cite: 56]
                    val_j = processed_line[j] # [cite: 57]
                    if val_j is None: continue # Still a gap, try next j

                    diff = val_j - start_val
                    num_steps_for_diff = (j - i) # Number of steps including gaps # [cite: 58]
                    
                    # Normalize diff if there were gaps between start_val and val_j [cite: 59]
                    # Example: line[i]=1, gap, gap, line[j]=7. [cite: 59]
                    # allow_gaps=2. num_steps=3. diff=6. Actual diff = 6/3 = 2. [cite: 59]
                    if num_steps_for_diff > 1 + gaps_between_i_j : # If there are actual numbers between i and j, this logic needs refinement. [cite: 60]
                        # The PDF implies diff is established by the first non-gap pair. [cite: 61]
                        # Let's stick to the PDF's simpler interpretation for now: [cite: 61]
                                                                 # diff is between line[i] and the first non-gap line[j] [cite: 61]
                        pass # No adjustment if diff is just between two numbers [cite: 62]

                    # PDF: "Avoid constant sequences unless they are all zeros"
                    # "Here, we exclude if common diff is 0 and non-zero point)"
                    # 來源：新大腦.pdf - Arithmetic constant sequence avoidance (Page 4) [cite: 63]
                    if math.isclose(diff, 0) and not math.isclose(start_val, 0):
                        continue

                    current_seq_indices = [i]
                    current_seq_values = [int(start_val)] # Store original int values # [cite: 64]
                    
                    # Add intermediate elements if they fit the pattern and account for gaps [cite: 65]
                    # This part is complex in the PDF, let's first establish the sequence with j [cite: 65]
                    if gaps_between_i_j == 0 : # j is the immediate next non-gap # [cite: 65]
                         current_seq_indices.append(j)
                         current_seq_values.append(int(val_j))

                    last_val_in_seq = val_j
                    last_idx_in_seq = j # [cite: 66]
                    potential_gap_count_after_j = 0

                    for k in range(j + 1, n):
                        val_k = processed_line[k]
                        if val_k is None: # [cite: 67]
                            potential_gap_count_after_j += 1
                            if potential_gap_count_after_j > allow_gaps:
                                break # Too many gaps # [cite: 68]
                            continue
                        
                        # Expected next value if there were no gaps from last_val_in_seq to val_k # [cite: 69]
                        steps_from_last = (k - last_idx_in_seq)
                        expected_val_at_k = last_val_in_seq + diff * (steps_from_last / (potential_gap_count_after_j + 1))
                        
                        if math.isclose(val_k, expected_val_at_k): # [cite: 70]
                            current_seq_indices.append(k)
                            current_seq_values.append(int(val_k))
                            last_val_in_seq = val_k # [cite: 71]
                            last_idx_in_seq = k # [cite: 71]
                            potential_gap_count_after_j = 0 # Reset gap count
                        else:
                            break # Sequence broken # [cite: 72]

                    if len(current_seq_values) >= min_len:
                        sequences.append(current_seq_values)


            # Geometric sequence check
            if check_geometric and not math.isclose(start_val, 0): # Start_val cannot be 0 for typical geometric # [cite: 73]
                # 來源：新大腦.pdf - Geometric sequence check (Page 4)
                for j in range(i + 1, n):
                    gaps_between_i_j = 0
                    k_gap_check = i + 1
                    while k_gap_check < j: # [cite: 74]
                        if processed_line[k_gap_check] is None:
                            gaps_between_i_j +=1
                        k_gap_check +=1 # [cite: 75]
                    
                    if gaps_between_i_j > allow_gaps:
                        continue

                    val_j = processed_line[j]
                    if val_j is None: continue # [cite: 76]
                    if math.isclose(val_j, 0): continue # Geometric sequence with zero is tricky

                    # PDF: "If ratio isn't integer-like and not a trivial division break"
                    # 來源：新大腦.pdf - Geometric ratio check (Page 5) [cite: 77]
                    if math.isclose(start_val, 0): continue # Should have been caught, but defensive

                    # Try to establish ratio [cite: 78]
                    # Using a tolerance for float comparisons might be needed if line can have floats [cite: 78]
                    # For int lines, we expect integer ratios or clean divisions. [cite: 78]
                    ratio_candidate = val_j / start_val # [cite: 79]
                    
                    # PDF: "Avoid constant sequences"
                    # 來源：新大腦.pdf - Geometric constant sequence avoidance (Page 5)
                    if math.isclose(ratio_candidate, 1.0) and not math.isclose(start_val, val_j): # If ratio is 1, values must be same # [cite: 80]
                        continue # This condition might be too strict if allow_gaps changes things

                    current_seq_indices = [i]
                    current_seq_values = [int(start_val)]
                    if gaps_between_i_j == 0 : # [cite: 81]
                         current_seq_indices.append(j)
                         current_seq_values.append(int(val_j))

                    last_val_in_seq = val_j # [cite: 82]
                    last_idx_in_seq = j
                    potential_gap_count_after_j = 0
                    ratio = ratio_candidate # Established ratio

                    for k in range(j + 1, n): # [cite: 83]
                        val_k = processed_line[k]
                        if val_k is None:
                            potential_gap_count_after_j += 1
                            if potential_gap_count_after_j > allow_gaps: # [cite: 84]
                                break
                            continue
                        
                        if math.isclose(val_k, 0) : break # Geometric sequence broken by zero # [cite: 85]

                        # Expected next value
                        # Number of actual steps of ratio application [cite: 86]
                        num_ratio_applications = (k - last_idx_in_seq) // (potential_gap_count_after_j + 1) # [cite: 86]
                        if (k - last_idx_in_seq) % (potential_gap_count_after_j + 1) != 0: # not a clean step
                            break 

                        expected_val_at_k = last_val_in_seq * (ratio ** num_ratio_applications) # [cite: 87]

                        if math.isclose(val_k, expected_val_at_k):
                            current_seq_indices.append(k)
                            current_seq_values.append(int(val_k)) # [cite: 88]
                            last_val_in_seq = val_k
                            last_idx_in_seq = k
                            potential_gap_count_after_j = 0
                        else: # [cite: 89]
                            break
                    
                    if len(current_seq_values) >= min_len:
                        sequences.append(current_seq_values) # [cite: 90]
        
        # Remove duplicate sequences that might have been found from different start points [cite: 91]
        # or due to the simplified looping structure compared to the PDF's intricate one. [cite: 91]
        unique_sequences = [] # [cite: 91]
        for seq in sequences:
            if seq not in unique_sequences:
                unique_sequences.append(seq)
        return unique_sequences

    @staticmethod
    # 來源：新大腦.pdf - BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions (Page 5) [cite: 16]
    def get_card_max_value_from_grid_dimensions(grid_shape: Tuple[int, int]) -> int:
        """Calculates the maximum possible number on the card based on its dimensions. [cite: 16]"""
        rows, cols = grid_shape # [cite: 92]
        if rows == 0 or cols == 0:
            return 0
        return rows * cols

    @staticmethod
    # 來源：新大腦.pdf - BoardAnalyzerUtils.get_all_possible_numbers_for_grid (Page 5) [cite: 17]
    def get_all_possible_numbers_for_grid(grid_shape: Tuple[int, int]) -> Set[int]:
        """Returns a set of all numbers that could theoretically appear on a grid of given
        dimensions. [cite: 17, 93]"""
        max_val = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions(
            grid_shape
        ) # 來源：新大腦.pdf (Page 5)
        if max_val == 0:
            return set()
        return set(range(1, max_val + 1))

    @staticmethod
    # 來源：新大腦.pdf - BoardAnalyzerUtils.get_legal_values_for_placement (Page 5) [cite: 18]
    def get_legal_values_for_placement(grid: np.ndarray) -> Set[int]:
        """
        Determines the set of numbers that can be legally placed onto an empty cell in the grid. [cite: 94]
        This adheres to the rule: numbers are 1 to R*C and no positive number can be repeated. [cite: 19, 95]
        (Requirement 1.c) [cite: 20, 96]
        來源：新大鵝.pdf - BoardAnalyzerUtils.get_legal_values_for_placement (Page 5-6)
        """
        if grid.size == 0: # 來源：新大腦.pdf (Page 6)
            return set()
        rows, cols = grid.shape
        all_possible_on_this_grid = (
            BoardAnalyzerUtils.get_all_possible_numbers_for_grid((rows, cols))
        ) # [cite: 97]
        used_positive_values_on_board = set(
            int(v) for v in grid.flatten() if v != -1 and v > 0
        )
        legal_placements = all_possible_on_this_grid - used_positive_values_on_board
        return legal_placements

# --- Pydantic Config Models for Modules ---
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - 通用強化思路 (參數動態化)
# 來源：给你2025资料在深度建议一次.pdf - 統一的配置管理, Pydantic V2 (Page 9, Page 1)
from pydantic import BaseModel, Field

class BaseModuleConfig(BaseModel):
    # Common config fields can go here if any # [cite: 98]
    enabled: bool = Field(default=True, description="Whether this module is enabled.")
    weight: float = Field(default=1.0, ge=0.0, description="Weight of this module's score in aggregation.")

class WeightedProximityConfig(BaseModuleConfig):
    # 來源：新大腦.pdf - EXT_A2 parameters (Page 7)
    # 來源：给你2025资料在深度建议一次.pdf - EXT_A2 Pydantic配置範例 (Page 2)
    radius: int = Field(default=2, ge=1, description="考慮的鄰域半徑")
    value_weight_factor: float = Field(default=0.1, ge=0.0, description="鄰居值的權重因子")
    distance_decay_factor: float = Field(default=1.5, gt=0.0, description="距離衰減因子")
    # 來源：新大腦.pdf - EXT_A2 Conceptual repulsion (Page 7)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - A2 斥力概念的細化
    enable_repulsion: bool = Field(default=False, description="是否啟用斥力概念") # [cite: 99]
    # Undesirable pairs could be more complex, e.g. [cite: 100]
    undesirable_pairs_config: Dict[Tuple[int, int], float] = Field(default_factory=dict, description="不良配對及其斥力因子, e.g. {(1,1): -0.2}")


class LocalHeterogeneityConfig(BaseModuleConfig):
    # 來源：新大腦.pdf - EXT_M3 parameters (Page 9)
    # 來源：给你2025资料在深度建议一次.pdf - EXT_M3 Pydantic配置範例 (Page 2 of previous response)
    radius: int = Field(default=1, ge=1, description="異質性計算的鄰域半徑")
    min_neighbors_for_robust_score: int = Field(default=2, ge=0, description="計算有效熵的最小鄰居數")
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - M3 熵以外的異質性度量
    diversity_metric: str = Field(default="entropy", pattern="^(entropy|gini|unique_count)$", description="異質性度量方法: entropy, gini, or unique_count")


class PotentialFieldConfig(BaseModuleConfig):
    # 來源：新大腦.pdf - EXT_D3 parameters (Page 10-11)
    decay_exponent: float = Field(default=1.5, gt=0.0, description="影響力隨距離衰減的指數 (e.g., 1 for 1/r, 2 for 1/r^2)") # [cite: 101]
    max_influence_radius: int = Field(default=3, ge=1, description="考慮數字影響力的最大曼哈頓距離") # [cite: 101]
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - D3 「電荷」性質的擴展
    enable_negative_charges: bool = Field(default=False, description="是否啟用負電荷（排斥力）概念")
    negative_charge_map: Dict[int, float] = Field(default_factory=dict, description="定義哪些數字視為負電荷及其強度（<0）")


class DiscontinuityRepairConfig(BaseModuleConfig):
    # 來源：新大腦.pdf - EXT_F10 parameters (Page 12)
    # 來源：给你2025资料在深度建议一次.pdf - EXT_F10 Pydantic配置範例 (Page 4)
    min_sequence_len_to_score: int = Field(default=3, ge=2, description="視為有效的最小序列長度")
    allow_gaps_in_sequence: int = Field(default=1, ge=0, description="序列中允許的最大間隙數") # [cite: 43]
    check_arithmetic: bool = Field(default=True, description="是否檢查等差序列")
    check_geometric: bool = Field(default=False, description="是否檢查等比序列") # [cite: 102]
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
    # 來源：新大腦.pdf - EXT_R5 parameters (Page 16-17) # [cite: 117]
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - R5「資源」定義的擴展
    w_row_completeness: float = Field(default=0.3, ge=0.0, le=1.0, description="行完成度分數的權重")
    w_col_completeness: float = Field(default=0.3, ge=0.0, le=1.0, description="列完成度分數的權重")
    w_value_capture: float = Field(default=0.4, ge=0.0, le=1.0, description="價值捕獲分數的權重")
    # Conceptual: Add weights for specific area control if implemented

class LineControlConfig(BaseModuleConfig): # For GM1 and GM2
    # 來源：新大腦.pdf - EXT_GM1/GM2 parameters (Page 18, 20)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM1/GM2 序列評估的增強
    w_density: float = Field(default=0.4, ge=0.0, le=1.0, description="密度分數權重")
    w_sum_score: float = Field(default=0.3, ge=0.0, le=1.0, description="總和分數權重")
    w_sequence_score: float = Field(default=0.3, ge=0.0, le=1.0, description="序列分數權重") # [cite: 118]
    use_advanced_sequence_detection: bool = Field(default=True, description="是否使用 BoardAnalyzerUtils.find_sequences_in_line 進行序列評估")
    min_len_for_sequence_score: int = Field(default=3, ge=2)
    allow_gaps_for_sequence_score: int = Field(default=1, ge=0) # Consistent with F10

class ConnectedComponentConfig(BaseModuleConfig): # For GM3
    # 來源：新大腦.pdf - EXT_GM3 parameters (Page 21-22)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM3 連通區域的「形狀」和「質量」
    consider_shape_factor: bool = Field(default=False, description="是否考慮連通區域的形狀因子（概念性）")
    shape_factor_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="形狀因子權重（如果啟用）")


# --- Scoring Module Implementations ---

# 來源：新大腦.pdf - 1. EXT_A2_Weighted_Proximity_Vec (Page 7) [cite: 21]
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_A2強化建議
# 來源：给你2025资料在深度建议一次.pdf - EXT_A2 Pydantic配置範例 (Page 2)
def EXT_A2_Weighted_Proximity_Vec(
    grid: np.ndarray,
    config: WeightedProximityConfig, # Now expects the Pydantic config object
    request_id: str | None = "N/A_A2_Proximity", # [cite: 104]
) -> np.ndarray:
    """
    (A2-加權鄰近性) [cite: 21]
    核心規則:評估空格周圍已填數字的接近程度及其值的影響。[cite: 21]
    目的:偏好靠近高價值數字或數字密集區域的空格。[cite:21]
    啟發式類型:空間鄰近性 [cite: 21]
    輸出詮釋:分數越高表示鄰近效應越強(受周圍數字的值與密度影響) [cite: 21]
    來源：新大腦.pdf - EXT_A2_Weighted_Proximity_Vec (Page 7)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_A2"
    logger.debug(
        f"Executing EXT_A2_Weighted_Proximity_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    ) # [cite: 105]

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    radius = config.radius
    value_weight_factor = config.value_weight_factor
    distance_decay_factor = config.distance_decay_factor
    
    # 來源：新大腦.pdf - EXT_A2 Self-adaptive weights (Conceptual) (Page 7)
    # 實現概念性自適應權重: 若盤面平均值高，增加 value_weight_factor [cite: 106]
    # This can be part of a more sophisticated config update mechanism or pre-calculation in analyzer [cite: 106]
    # For now, let's assume config provides the final factors. [cite: 107]
    # avg_grid_val_calc = np.mean(grid[grid != -1]) if np.count_nonzero(grid != -1) > 0 else 0
    # if avg_grid_val_calc > (BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols)) * 0.5):
    #     value_weight_factor *= 1.2 # Example dynamic adjustment

    max_val_on_grid = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions(
        (rows, cols)
    )
    if max_val_on_grid == 0:
        max_val_on_grid = 1.0

    num_neighbors_in_radius = (2 * radius + 1) ** 2 - 1
    heuristic_max_score = (
        num_neighbors_in_radius # [cite: 108]
        * max_val_on_grid
        * value_weight_factor
    ) # Min dist is 1, so 1**decay_factor is 1
    # 來源：新大腦.pdf (Page 8) [cite: 23] - original was / (1**distance_decay_factor)

    if heuristic_max_score <= 0: 
        heuristic_max_score = 1.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: # [cite: 109]
                continue

            proximity_score = 0.0
            
            # 來源：新大腦.pdf - EXT_A2 Conceptual repulsion (Page 7)
            # 這裡的斥力計算需要一個「假設填入的值」才能判斷是否與鄰居形成不良配對 [cite: 110]
            # 目前模組只評估空格本身，若要加入此斥力，需修改函式簽名或由 analyzer 傳入假設值 [cite: 110]
            # conceptual_placed_value = ... (needs to be determined or iterated) [cite: 110]
            # For now, skipping the PDF's direct repulsion logic for UNDESIRABLE_PAIRS [cite: 110]
            # as it requires a `some_proposed_val_for_this_cell`. [cite: 110]
            # The config `undesirable_pairs_config` is there for future enhancement. [cite: 111]

            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0: # 來源：新大腦.pdf (Page 8) [cite: 21, 112]
                        continue
                    
                    nr, nc = r_idx + dr, c_idx + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                        dist = MathUtils.manhattan_distance( # [cite: 113]
                            (r_idx, c_idx), (nr, nc)
                        )
                        # 來源：新大腦.pdf (Page 8) [cite: 22]
                        if dist == 0: dist = 1 # Safeguard # [cite: 114]

                        score_contribution = (
                            grid[nr, nc] * value_weight_factor
                        ) / (dist**distance_decay_factor) # 來源：新大腦.pdf (Page 8) [cite: 22, 115]
                        proximity_score += score_contribution
            
            if heuristic_max_score > 0: # 來源：新大腦.pdf (Page 8) [cite: 23]
                scores[r_idx, c_idx] = MathUtils.normalize_value(
                    proximity_score, 0, heuristic_max_score, clamp=True # [cite: 116]
                )
            else:
                scores[r_idx, c_idx] = 0.0
    return scores * config.weight

# 來源：新大腦.pdf - 2. EXT_M3_Local_Heterogeneity_Vec (Page 8)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_M3強化建議
def EXT_M3_Local_Heterogeneity_Vec(
    grid: np.ndarray,
    config: LocalHeterogeneityConfig, # Expects Pydantic config # [cite: 130]
    request_id: str | None = "N/A_M3_Heterogeneity", # [cite: 131]
) -> np.ndarray:
    """
    (M3 - 局部異質性)
    核心規則:評估空格周圍數字的多樣性。
    目的:偏好周圍數字分佈更隨機、更少重複的空格。
    啟發式類型:分佈統計(基於熵)
    輸出詮釋: 分數越高表示周圍環境的數字異質性越高(熵越大)
    來源：新大腦.pdf - EXT_M3_Local_Heterogeneity_Vec (Page 8-9)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_M3"
    logger.debug(
        f"Executing EXT_M3_Local_Heterogeneity_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape # [cite: 132]
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    radius = config.radius
    min_neighbors_for_robust_score = config.min_neighbors_for_robust_score
    
    # 來源：新大腦.pdf - EXT_M3 Conceptual: Self-adaptive min_neighbors (Page 9)
    # Example of self-adaptation (can be more complex)
    if rows * cols < 10: # For very small grids
        min_neighbors_for_robust_score = max(0, min(min_neighbors_for_robust_score, 1))


    all_possible_values_in_game = BoardAnalyzerUtils.get_all_possible_numbers_for_grid( # [cite: 133]
        grid.shape
    ) # 來源：新大腦.pdf (Page 9)
    if not all_possible_values_in_game:
        return scores 

    # 來源：新大腦.pdf - EXT_M3 Theoretical maximum entropy (Page 9) [cite: 26, 27, 28, 29, 30, 31, 32, 33]
    # The PDF has several notes on max_theoretical_entropy. [cite: 134]
    # Simplified logic: log2(N) if N > 1, else log2(2) or 1.0 to avoid log2(1)=0 or log2(0). [cite: 134]
    num_distinct_symbols = len(all_possible_values_in_game) # [cite: 135]
    if num_distinct_symbols > 1:
        max_theoretical_diversity_measure = math.log2(num_distinct_symbols)
    elif num_distinct_symbols == 1:
        max_theoretical_diversity_measure = math.log2(2) # Avoid log2(1)=0, provide some scale
    else: 
        max_theoretical_diversity_measure = 1.0 

    if max_theoretical_diversity_measure <= 0: max_theoretical_diversity_measure = 1.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: # [cite: 136]
                continue
            
            neighbor_values = BoardAnalyzerUtils.get_neighborhood_values(
                grid, r_idx, c_idx, radius=radius, eight_connectivity=True,
                val_func=lambda x_val: int(x_val) if x_val != -1 else None,
                include_center=False, # [cite: 137]
            ) # 來源：新大腦.pdf (Page 10) [cite: 34]

            if len(neighbor_values) < min_neighbors_for_robust_score: # 來源：新大腦.pdf (Page 10)
                scores[r_idx, c_idx] = 0.0
                continue

            current_diversity_value: float
            if config.diversity_metric == "entropy": # [cite: 138]
                current_diversity_value = MathUtils.get_entropy(neighbor_values) # 來源：新大腦.pdf (Page 10) [cite: 35]
            elif config.diversity_metric == "gini":
                # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - M3 熵以外的異質性度量 (基尼不純度)
                counts = Counter(neighbor_values)
                impurity = 1.0 # [cite: 139]
                for count_val in counts.values():
                    prob = count_val / len(neighbor_values)
                    impurity -= prob**2
                current_diversity_value = impurity 
                # Gini's max is (k-1)/k for k classes. For normalization,  [cite: 140]
                # we could normalize Gini against its own theoretical max based on num_distinct_symbols. [cite: 140]
                # For simplicity here, we are normalizing against log2(N) as a general diversity cap. [cite: 141]
                # This might not be ideal for Gini. A more proper normalization for Gini: [cite: 141, 142]
                # max_gini = (num_distinct_symbols -1) / num_distinct_symbols if num_distinct_symbols > 0 else 0 [cite: 142]
                # if max_gini > 0: normalized_gini = current_diversity_value / max_gini else 0 [cite: 142]
            elif config.diversity_metric == "unique_count":
                current_diversity_value = float(len(set(neighbor_values))) # [cite: 143]
                # Normalize unique_count against min(len(neighbor_values), num_distinct_symbols)
                max_possible_unique_in_neighborhood = min(len(neighbor_values), num_distinct_symbols)
                if max_possible_unique_in_neighborhood > 0 :
                    current_diversity_value = current_diversity_value / max_possible_unique_in_neighborhood
                else: # [cite: 144]
                    current_diversity_value = 0.0
                 # This direct normalization makes its range [0,1] already for unique_count ratio
                max_theoretical_diversity_measure = 1.0 # Adjust for unique_count ratio

            else: # Fallback to entropy
                current_diversity_value = MathUtils.get_entropy(neighbor_values) # [cite: 145]

            if max_theoretical_diversity_measure > 0:
                # 來源：新大腦.pdf - EXT_M3 Normalizing (Page 10) [cite: 36, 38]
                normalized_score = current_diversity_value / max_theoretical_diversity_measure
                scores[r_idx, c_idx] = MathUtils.normalize_value(
                    normalized_score, 0, 1, clamp=True # [cite: 146]
                )
            else:
                scores[r_idx, c_idx] = 0.0
    return scores * config.weight

# 來源：新大腦.pdf - 3. EXT_D3_Potential_Field_Vec (Page 10)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_D3強化建議
def EXT_D3_Potential_Field_Vec(
    grid: np.ndarray,
    config: PotentialFieldConfig,
    request_id: str | None = "N/A_D3_Potential", # [cite: 147]
) -> np.ndarray:
    """
    (D3-位勢場分析)
    核心規則:將盤面上的數字視為「電荷」,空格則根據其位置的「綜合位勢」來評分。
    目的:偏好位於受高價值數字「吸引」或低價值數字「排斥」(如果設計如此)區域的空格。此處簡化為僅正向吸引。
    啟發式類型:物理類比(類似靜電場或重力場)
    輸出詮釋:分數越高表示該空格受到周圍數字的正向「位勢影響」越大
    來源：新大腦.pdf - EXT_D3_Potential_Field_Vec (Page 10) [cite: 39]
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_D3"
    logger.debug(
        f"Executing EXT_D3_Potential_Field_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape # [cite: 148]
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    decay_exponent = config.decay_exponent # 來源：新大腦.pdf (Page 11) [cite: 39]
    max_influence_radius = config.max_influence_radius # 來源：新大腦.pdf (Page 11) [cite: 39]
    
    max_possible_val_on_grid = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions(
        (rows, cols)
    ) # 來源：新大腦.pdf (Page 11) [cite: 39]
    if max_possible_val_on_grid == 0: return scores

    # 來源：新大腦.pdf - EXT_D3 Heuristic maximum potential (Page 11) [cite: 40, 149]
    # Sum of max_value / (min_dist^decay) for all cells in radius. [cite: 150]
    # This is a very rough upper bound. [cite: 150]
    num_cells_in_radius_approx = (2 * max_influence_radius + 1)**2 - 1 # Max neighbors
    heuristic_max_potential = num_cells_in_radius_approx * (
        max_possible_val_on_grid / (1**decay_exponent) # Assuming min dist 1
    )
    if heuristic_max_potential <= 0: heuristic_max_potential = 1.0 # 來源：新大腦.pdf (Page 11) [cite: 41]

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:
                continue # [cite: 151]

            current_cell_potential = 0.0
            for nr in range(rows):
                for nc in range(cols):
                    if grid[nr, nc] != -1:  # If it's a filled cell (a "charge")
                        charge_val = float(grid[nr, nc]) # [cite: 152]
                        
                        # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - D3 「電荷」性質的擴展
                        if config.enable_negative_charges and int(charge_val) in config.negative_charge_map: # [cite: 153]
                            charge_val = config.negative_charge_map[int(charge_val)] # Use configured negative strength
                        elif charge_val <= 0 and not config.enable_negative_charges: # Original logic considered only positive # [cite: 154]
                            continue

                        dist = MathUtils.manhattan_distance((r_idx, c_idx), (nr, nc))
                        
                        if dist == 0: continue # Should not happen if only scoring empty cells
                        if dist > max_influence_radius: continue # 來源：新大腦.pdf (Page 11) [cite: 36, 155]

                        # Potential = charge_value / distance^decay_exponent
                        # 來源：新大腦.pdf (Page 11) [cite: 42]
                        potential_contribution = charge_val / (dist**decay_exponent) # [cite: 156]
                        current_cell_potential += potential_contribution
            
            scores[r_idx, c_idx] = MathUtils.normalize_value(
                current_cell_potential, 0, heuristic_max_potential, clamp=True
            ) # Note: if negative charges are strong, potential could be < 0. Normalization min_val might need adjustment. [cite: 157]
            # For now, assuming 0 as min, so strong repulsion would be clamped to 0. [cite: 158]
              # A bipolar normalization might be (-heuristic_max, heuristic_max) -> (0,1) [cite: 158]
              # or separate attractive/repulsive scores. [cite: 158]
            # Sticking to PDF's normalization for now. [cite: 159]

    return scores * config.weight

# 來源：新大腦.pdf - 4. EXT_F10_Discontinuity_Vec (Page 12)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_F10強化建議
# Config for this (DiscontinuityRepairConfig) was defined in PART 1
def EXT_F10_Discontinuity_Vec(
    grid: np.ndarray,
    config: DiscontinuityRepairConfig, # Expects Pydantic config
    request_id: str | None = "N/A_F10_Discontinuity",
) -> np.ndarray:
    """
    (F10-不連續性修復/序列完成度)
    核心規則:評估在空格填入數字後,是否能修復或完成某個方向上的數字序列(例如等差)。
    目的:偏好那些能夠「承先啟後」,使斷裂的序列得以延續或形成的空格。
    啟發式類型:序列與模式識別 # [cite: 160]
    輸出詮釋:分數越高表示該空格填入某個合法數字後,能形成或延長的序列越長/越重要
    來源：新大腦.pdf - EXT_F10_Discontinuity_Vec (Page 12)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_F10"
    logger.debug(
        f"Executing EXT_F10_Discontinuity_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores # [cite: 161]

    legal_values_for_placement = BoardAnalyzerUtils.get_legal_values_for_placement(grid) # 來源：新大腦.pdf (Page 12)
    if not legal_values_for_placement:
        return scores

    min_sequence_len_to_score = config.min_sequence_len_to_score
    
    # 來源：新大腦.pdf - EXT_F10 Heuristic max length for normalization (Page 12)
    heuristic_max_len = float(max(rows, cols))
    if heuristic_max_len < min_sequence_len_to_score: # 來源：新大腦.pdf (Page 12)
        heuristic_max_len = float(min_sequence_len_to_score)
    if heuristic_max_len <= 0: heuristic_max_len = 1.0 

    for r_idx in range(rows): # [cite: 162]
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 12)
                continue

            max_len_contribution_for_this_cell: float = 0.0 # 來源：新大腦.pdf (Page 12)

            for val_to_try in legal_values_for_placement:
                temp_grid = grid.copy() # [cite: 163]
                temp_grid[r_idx, c_idx] = val_to_try
                current_val_max_len: float = 0.0

                lines_to_check_data: List[Tuple[List[int], str]] = []
                # 1. Check Row # [cite: 164]
                # 來源：新大腦.pdf - EXT_F10 Check Row (Page 12)
                lines_to_check_data.append((list(temp_grid[r_idx, :]), "row"))
                # 2. Check Column
                # 來源：新大腦.pdf - EXT_F10 Check Column (Page 13)
                lines_to_check_data.append((list(temp_grid[:, c_idx]), "col"))
                # 3. Check Diagonals # [cite: 165]
                # 來源：新大腦.pdf - EXT_F10 Check Diagonals (Page 13)
                diag1_line = list(np.diag(temp_grid, k=c_idx - r_idx))
                lines_to_check_data.append((diag1_line, "diag1"))
                
                flipped_temp_grid = np.fliplr(temp_grid)
                flipped_c_idx = cols - 1 - c_idx # Max col index - current col index # [cite: 166]
                diag2_line = list(np.diag(flipped_temp_grid, k=flipped_c_idx - r_idx))
                lines_to_check_data.append((diag2_line, "diag2"))

                for line_values, line_type_debug in lines_to_check_data:
                    # 來源：新大腦.pdf - EXT_F10 find_sequences_in_line call (Page 13) [cite: 167]
                    # Using the more complete find_sequences_in_line from BoardAnalyzerUtils
                    sequences_found = BoardAnalyzerUtils.find_sequences_in_line(
                        line_values,
                        min_len=min_sequence_len_to_score, # [cite: 168]
                        check_arithmetic=config.check_arithmetic,
                        check_geometric=config.check_geometric,
                        allow_gaps=config.allow_gaps_in_sequence,
                    ) # [cite: 169]
                    for seq in sequences_found:
                        if val_to_try in seq:  # Check if the placed value is part of this new/extended sequence
                            # 來源：新大腦.pdf (Page 13)
                            seq_len = float(len(seq)) # [cite: 170]
                            # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - F10 序列價值評估
                            if config.sequence_quality_weighting:
                                avg_val_in_seq = sum(seq) / len(seq) if len(seq) > 0 else 0 # [cite: 171]
                                max_board_val = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows,cols))
                                if max_board_val > 0 and avg_val_in_seq > (max_board_val * config.high_value_sequence_threshold_factor): # [cite: 172]
                                    seq_len *= 1.2 # Example: Boost score for high-value sequences
                            current_val_max_len = max(current_val_max_len, seq_len)
                
                if current_val_max_len >= min_sequence_len_to_score: # [cite: 173]
                    max_len_contribution_for_this_cell = max(
                        max_len_contribution_for_this_cell, current_val_max_len
                    )
            
            if heuristic_max_len > 0: # 來源：新大腦.pdf (Page 13) [cite: 174]
                scores[r_idx, c_idx] = MathUtils.normalize_value(
                    max_len_contribution_for_this_cell,
                    0, # Min possible score for length contribution
                    heuristic_max_len, # [cite: 175]
                    clamp=True,
                )
            else: # 來源：新大腦.pdf (Page 14)
                scores[r_idx, c_idx] = 0.0
    return scores * config.weight

# 來源：新大腦.pdf - 5. EXT_P7_Pathfinding_Value_Vec (Page 14)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_P7強化建議
# Config for this (PathfindingValueConfig) was defined in PART 1
def EXT_P7_Pathfinding_Value_Vec(
    grid: np.ndarray, # [cite: 176]
    config: PathfindingValueConfig,
    request_id: str | None = "N/A_P7_Pathfinding", # [cite: 177]
) -> np.ndarray:
    """
    (P7-路徑尋找價值)
    核心規則:評估在空格填入數字後,形成連接到其他現有數字的路徑的價值。
    目的:偏好那些能夠「橋接」盤面區域,或連接到高價值目標的空格。
    啟發式類型:連通性與圖論
    輸出詮釋:分數越高表示該空格填入某數字後,能形成更有價值的路徑(考慮路徑長度與連接到的數字大小)
    來源：新大腦.pdf - EXT_P7_Pathfinding_Value_Vec (Page 14)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_P7"
    logger.debug(
        f"Executing EXT_P7_Pathfinding_Value_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape # [cite: 178]
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    legal_values_for_placement = BoardAnalyzerUtils.get_legal_values_for_placement(grid) # 來源：新大腦.pdf (Page 14)
    if not legal_values_for_placement:
        return scores

    max_path_search_depth = config.max_path_search_depth # 來源：新大腦.pdf (Page 14)
    path_value_decay_factor = config.path_value_decay_factor # 來源：新大腦.pdf (Page 14)
    
    max_possible_val_on_grid = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions(
        (rows, cols)
    ) # 來源：新大腦.pdf (Page 14) [cite: 179]
    if max_possible_val_on_grid == 0: max_possible_val_on_grid = 1.0

    # 來源：新大腦.pdf - EXT_P7 Heuristic max path score (Page 14-15)
    # A very loose upper bound: (max_depth_search_radius_squared_area) * max_val / (1^decay) [cite: 180]
    # The PDF uses (2*max_path_search_depth + 1)**2, which is area. [cite: 180]
    # Let's consider max connections. Max neighbors in BFS up to depth D is roughly sum of 4*i for i=1 to D. [cite: 180]
    # Simpler heuristic from PDF:
    heuristic_max_path_score = (
        (2 * max_path_search_depth + 1)**2 * max_possible_val_on_grid / (1**path_value_decay_factor)
    )
    if heuristic_max_path_score <= 0: heuristic_max_path_score = 1.0 # 來源：新大腦.pdf (Page 15)

    target_value_min_threshold = max_possible_val_on_grid * config.target_value_threshold_factor

    for r_start in range(rows):
        for c_start in range(cols):
            if grid[r_start, c_start] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 14) [cite: 181]
                continue
            
            max_score_for_this_cell: float = 0.0 # 來源：新大腦.pdf (Page 15)

            # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - P7 只針對高價值潛在數字 (conceptual)
            # For this implementation, we iterate all legal values as per PDF base logic [cite: 182]
            # Enhancement: filter legal_values_for_placement here if needed. [cite: 182]
            for val_to_try in legal_values_for_placement: # [cite: 183]
                # The PDF states: "The original grid is used to find *existing* numbers." [cite: 184]
                # "The path itself can traverse other empty cells." [cite: 184]
                # So, val_to_try is not actually placed on a temp_grid for BFS pathfinding logic. [cite: 185]
                # BFS explores from (r_start, c_start) through other empty cells to existing numbers. [cite: 185]
                # The value of val_to_try might influence the *decision* to place it there, [cite: 186]
                # but the path score itself is about connecting (r_start, c_start) to existing numbers. [cite: 186]
                # The PDF seems to calculate a score for (r_start, c_start) if val_to_try were placed, [cite: 187]
                # by summing up values of paths originating from it. [cite: 187]
                # The current logic in the PDF seems to iterate val_to_try but doesn't use it in BFS. [cite: 188]
                # Let's assume val_to_try is for future "what if this number is placed" scenarios, [cite: 189]
                # but for the path score, it's about the connectivity of the empty cell (r_start, c_start). [cite: 189]
                # The loop over val_to_try might be redundant if it's not used in path score calculation. [cite: 190]
                # Re-reading PDF: "The BFS explores from the cell (r_start, c_start) *as if* val_to_try is placed there." [cite: 191]
                # This implies val_to_try *is* relevant, perhaps as the starting "charge" or value of the path. [cite: 192]
                # However, the path score `reached_val / (effective_path_len ** ...)` uses `reached_val` (existing number). [cite: 193]
                # For now, I will follow the PDF structure where `val_to_try` is looped but not directly used in the score sum, [cite: 194]
                # which means the score for (r_start, c_start) will be the same regardless of `val_to_try`. [cite: 194]
                # This implies the outer loop for `val_to_try` for *this specific module's scoring as written in PDF* might be optimized out [cite: 195]
                # unless `val_to_try` is meant to affect `target_value_min_threshold` or pathing rules (which it currently doesn't). [cite: 195]
                # For "不可有任何簡化效能 只能增強", I will keep the loop. [cite: 196]

                current_placement_path_score: float = 0.0
                # ((r, c), current_path_length_from_start)
                q = deque([((r_start, c_start), 0)]) # 來源：新大腦.pdf (Page 15)
                # Visited for this specific BFS starting at (r_start, c_start)
                visited_for_bfs: Set[Tuple[int,int]] = set([(r_start, c_start)]) # 來源：新大腦.pdf (Page 15) [cite: 197]
                
                head_count = 0 # Safety break for BFS # 來源：新大腦.pdf (Page 15)
                # PDF: max_bfs_steps = rows* cols * len(legal_values_for_placement) - this can be huge [cite: 198]
                # Using a more constrained but still generous limit based on depth for practical reasons [cite: 198]
                max_bfs_steps_practical = (2 * max_path_search_depth + 1)**2 * 4 # Max cells in search area * avg degree
                
                paths_found_this_bfs: List[Tuple[int,int,int]] = [] # (val, len, count) for unique paths

                while q and head_count < max_bfs_steps_practical: # 來源：新大腦.pdf (Page 15) # [cite: 199]
                    head_count += 1
                    (curr_r, curr_c), path_len = q.popleft()

                    # Explore neighbors (4-connectivity)
                    # PDF typo: (0,1) (0,1) corrected to (0,1) (0,-1) (1,0) (-1,0) [cite: 200]
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # 來源：新大腦.pdf (Page 15)
                        next_r, next_c = curr_r + dr, curr_c + dc

                        if 0 <= next_r < rows and 0 <= next_c < cols: # [cite: 201]
                            # If neighbor is an *existing number* on the original grid
                            if grid[next_r, next_c] != -1: # 來源：新大腦.pdf (Page 15)
                                reached_val = int(grid[next_r, next_c]) # [cite: 202]
                                if reached_val < target_value_min_threshold and config.target_value_threshold_factor > 0:
                                    continue # Skip if below threshold (enhancement)

                                effective_path_len = path_len + 1 # Distance to this existing number # [cite: 203]
                                
                                # Path score contribution # [cite: 204]
                                path_score_contrib = reached_val / (effective_path_len**path_value_decay_factor)
                                current_placement_path_score += path_score_contrib
                                # 來源：新大腦.pdf - Do not add this to visited_for_bfs or queue (Page 15) [cite: 205]

                            # If neighbor is an *empty cell* (excluding starting cell if path_len is 0 implicitly by (curr_r,curr_c)) [cite: 206]
                            # and path is not too long, and not yet visited in this BFS [cite: 206]
                            elif (next_r, next_c) not in visited_for_bfs and \
                                 grid[next_r, next_c] == -1 and \
                                 path_len + 1 < max_path_search_depth: # 來源：新大腦.pdf (Page 15) # [cite: 207]
                                visited_for_bfs.add((next_r, next_c))
                                q.append(((next_r, next_c), path_len + 1))
                
                # The PDF structure implies max_score_for_this_cell is updated per val_to_try. [cite: 208]
                # If val_to_try is not used in current_placement_path_score, this loop is not varying the path score. [cite: 209]
                # For now, following structure, assuming val_to_try *could* be used in a more advanced version. [cite: 210]
                if current_placement_path_score > max_score_for_this_cell: # 來源：新大腦.pdf (Page 16) # [cite: 211]
                    max_score_for_this_cell = current_placement_path_score
            
            scores[r_start, c_start] = MathUtils.normalize_value(
                max_score_for_this_cell, 0, heuristic_max_path_score, clamp=True
            ) # 來源：新大腦.pdf (Page 16)
    return scores * config.weight

# 來源：新大腦.pdf - 6. EXT_R5_Resource_Control_Vec (Page 16) [cite: 212]
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_R5強化建議
# Config for this (ResourceControlConfig) was defined in PART 2
def EXT_R5_Resource_Control_Vec(
    grid: np.ndarray,
    config: ResourceControlConfig,
    request_id: str | None = "N/A_R5_ResourceCtrl",
) -> np.ndarray:
    """
    (R5-資源控制)
    核心規則:從資源控制角度評估填補位置的策略價值。資源可包括行/列的完成度、對高價值數字的獲取潜力等。
    目的:偏好那些能夠鞏固盤面控制權,或獲取潛在高價值數字的空格。
    啟發式類型:策略與控制
    輸出詮釋:分數越高表示該空格在填入數字後,對資源的控制(如行列完成度、高價值數字佔據)越強
    來源：新大腦.pdf - EXT_R5_Resource_Control_Vec (Page 16)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_R5" # [cite: 213]
    logger.debug(
        f"Executing EXT_R5_Resource_Control_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 16)
    
    max_possible_val_on_grid = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions(
        (rows, cols)
    ) # 來源：新大腦.pdf (Page 16) # [cite: 214]
    if max_possible_val_on_grid == 0: max_possible_val_on_grid = 1.0 # 來源：新大腦.pdf (Page 16)

    # 來源：新大腦.pdf - EXT_R5 hypothetical_high_val_placed (Page 16)
    hypothetical_high_val_placed: float = 0.0
    if potential_numbers_to_place:
        # Ensure potential_numbers_to_place is not empty before np.max
        hypothetical_high_val_placed = float(np.max(potential_numbers_to_place))


    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 17) # [cite: 215]
                continue

            # 1. Row Completeness Score
            # 來源：新大腦.pdf - EXT_R5 Row Completeness (Page 17)
            num_filled_in_row = np.count_nonzero(grid[r_idx, :] != -1)
            row_completeness_score = (num_filled_in_row + 1.0) / cols if cols > 0 else 0.0

            # 2. Column Completeness Score # [cite: 216]
            # 來源：新大腦.pdf - EXT_R5 Column Completeness (Page 17)
            num_filled_in_col = np.count_nonzero(grid[:, c_idx] != -1)
            col_completeness_score = (num_filled_in_col + 1.0) / rows if rows > 0 else 0.0
            
            # 3. Value Capture Score [cite: 217]
            # 來源：新大腦.pdf - EXT_R5 Value Capture Score (Page 17)
            value_capture_score: float = 0.0
            if hypothetical_high_val_placed > 0 and max_possible_val_on_grid > 0:
                # Normalizing the highest possible value we could place
                value_capture_score = MathUtils.normalize_value(
                    hypothetical_high_val_placed, 1, max_possible_val_on_grid, clamp=True # [cite: 218]
                )
            
            # Combine scores
            # 來源：新大腦.pdf - EXT_R5 Combine scores (Page 17)
            w_row = config.w_row_completeness
            w_col = config.w_col_completeness # [cite: 219]
            w_val = config.w_value_capture
            
            # Ensure weights sum to 1 for direct combination, or normalize afterwards
            # If weights don't sum to 1, the normalization below is crucial
            total_weight = w_row + w_col + w_val # [cite: 220]
            if total_weight <=0: total_weight = 1.0 # Avoid division by zero if all weights are 0

            combined_score = (
                w_row * row_completeness_score +
                w_col * col_completeness_score +
                w_val * value_capture_score
            ) / total_weight # Weighted average # [cite: 221]

            # The PDF normalizes again, which is good if component scores aren't strictly [0,1] or weights don't sum to 1. [cite: 222]
            # Since components are [0,1] and we did weighted average, combined_score is already [0,1]. [cite: 222]
            # But for robustness, an extra normalize_value is fine. Max for combined_score is 1.0 here. [cite: 222]
            # 來源：新大腦.pdf (Page 17) [cite: 223]
            scores[r_idx, c_idx] = MathUtils.normalize_value(combined_score, 0, 1.0, clamp=True)
            
    return scores * config.weight

# 來源：新大腦.pdf - 7. EXT_GM1_Row_Control_Vec (Page 17)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM1強化建議
# Config for this (LineControlConfig) was defined in PART 2
def EXT_GM1_Row_Control_Vec(
    grid: np.ndarray,
    config: LineControlConfig,
    request_id: str | None = "N/A_GM1_RowCtrl", # [cite: 224]
) -> np.ndarray:
    """
    (GM1-行控制力)
    核心規則:評估在特定空格填入數字後,對該行的完成度、數值總和或序列形成的貢獻。
    目的:偏好那些能增強單行控制力或形成有價值行模式的填補。
    啟發式類型:線性結構控制(行)
    輸出詮釋:分數越高表示對該行的潛在控制力或完成度越強
    來源：新大腦.pdf - EXT_GM1_Row_Control_Vec (Page 17)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM1"
    logger.debug(
        f"Executing EXT_GM1_Row_Control_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    ) # [cite: 225]

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 18)
    
    avg_potential_num_to_place: float = 0.0 # 來源：新大腦.pdf (Page 18)
    if potential_numbers_to_place:
        avg_potential_num_to_place = float(np.mean(potential_numbers_to_place))

    max_val_board = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols)) # 來源：新大腦.pdf (Page 18)
    if max_val_board == 0: max_val_board = 1.0

    for r_idx in range(rows): # [cite: 226]
        current_row_values_list_orig = [val for val in grid[r_idx, :] if val != -1]
        num_filled_in_row_orig = len(current_row_values_list_orig)
        sum_current_row_values_orig = sum(current_row_values_list_orig)

        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue # [cite: 227]
            
            # 1. Density Score: How full the row will be
            # 來源：新大腦.pdf - EXT_GM1 Density Score (Page 18)
            density_score = (num_filled_in_row_orig + 1.0) / cols if cols > 0 else 0.0

            # 2. Value Contribution Score (Sum Score)
            # 來源：新大腦.pdf - EXT_GM1 Value Contribution (Page 18)
            # Use avg_potential_num_to_place for the empty cell being scored [cite: 228]
            potential_row_sum = sum_current_row_values_orig + avg_potential_num_to_place
            heuristic_max_row_sum = float(cols * max_val_board) # Max possible row sum
            # 來源：新大鵝.pdf (Page 18)

            sum_score: float = 0.0
            if heuristic_max_row_sum > 0:
                sum_score = MathUtils.normalize_value( # [cite: 229]
                    potential_row_sum, 0, heuristic_max_row_sum, clamp=True
                )

            # 3. Sequence Completion Score
            # 來源：新大腦.pdf - EXT_GM1 Sequence Completion (Page 18)
            # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM1/GM2 序列評估的增強 (使用 BoardAnalyzerUtils.find_sequences_in_line) [cite: 230]
            seq_score: float = 0.0
            if config.use_advanced_sequence_detection:
                max_len_this_placement = 0.0
                if potential_numbers_to_place: # Only attempt if there are numbers to place # [cite: 231]
                    # Check sequence for average potential number
                    temp_grid_row_slice = grid[r_idx, :].copy()
                    temp_grid_row_slice[c_idx] = int(round(avg_potential_num_to_place)) # Use rounded average
                    
                    sequences = BoardAnalyzerUtils.find_sequences_in_line(
                        list(temp_grid_row_slice), # Must be list for find_sequences_in_line # [cite: 232]
                        min_len=config.min_len_for_sequence_score,
                        allow_gaps=config.allow_gaps_for_sequence_score
                    )
                    for s in sequences: # [cite: 233]
                        if int(round(avg_potential_num_to_place)) in s:
                           max_len_this_placement = max(max_len_this_placement, float(len(s)))
                
                if cols > 0: # Normalize by max possible length in row (cols) # [cite: 234]
                    seq_score = MathUtils.normalize_value(max_len_this_placement, 0, float(cols), clamp=True)

            else: # Original simplified mend logic from PDF
                # 來源：新大腦.pdf - EXT_GM1 Simplified mend logic (Page 19)
                if 0 < c_idx < cols - 1: # [cite: 235]
                    prev_val = grid[r_idx, c_idx - 1]
                    next_val = grid[r_idx, c_idx + 1]
                    if prev_val != -1 and next_val != -1:
                        if (prev_val + next_val) % 2 == 0: # [cite: 236]
                            mend_val = (prev_val + next_val) // 2
                            if mend_val in potential_numbers_to_place and abs(mend_val - prev_val) > 0:
                                seq_score = 0.75 # 來源：新大腦.pdf (Page 19) # [cite: 237]
                elif (c_idx == 0 and cols > 1 and grid[r_idx, c_idx + 1] != -1 and \
                      abs(grid[r_idx, c_idx + 1] - avg_potential_num_to_place) > 1e-6) or \
                      (c_idx == cols - 1 and cols > 1 and grid[r_idx, c_idx - 1] != -1 and \
                      abs(avg_potential_num_to_place - grid[r_idx, c_idx - 1]) > 1e-6): # 來源：新大腦.pdf (Page 19) # [cite: 238]
                      # Note: PDF had "... !=0", using > 1e-6 for float comparison robustness
                    seq_score = 0.25 # 來源：新大腦.pdf (Page 19) # [cite: 239]


            # Combine scores
            # 來源：新大腦.pdf - EXT_GM1 Combine scores (Page 19)
            w_density = config.w_density
            w_sum = config.w_sum_score
            w_seq = config.w_sequence_score
            total_weight = w_density + w_sum + w_seq # [cite: 240]
            if total_weight <= 0: total_weight = 1.0

            combined_score = (
                w_density * density_score + w_sum * sum_score + w_seq * seq_score
            ) / total_weight

            scores[r_idx, c_idx] = MathUtils.normalize_value(combined_score, 0, 1.0, clamp=True) # 來源：新大腦.pdf (Page 19) # [cite: 241]
            
    return scores * config.weight

# 來源：新大腦.pdf - 8. EXT_GM2_Col_Flow_Vec (Page 19)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM2強化建議
# Config for this (LineControlConfig) was defined in PART 2
def EXT_GM2_Col_Flow_Vec(
    grid: np.ndarray,
    config: LineControlConfig, # Reuses LineControlConfig
    request_id: str | None = "N/A_GM2_ColCtrl", # [cite: 242]
) -> np.ndarray:
    """
    (GM2 - 列流動性/列控制力)
    核心規則:評估在特定空格填入數字後,對該列的完成度、數值總和或序列形成的貢獻。
    目的:偏好那些能增強單列控制力或形成有價值列模式的填補。
    啟發式類型:線性結構控制(列)
    輸出詮釋:分數越高表示對該列的潛在控制力或完成度越強
    來源：新大腦.pdf - EXT_GM2_Col_Flow_Vec (Page 19-20)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM2"
    logger.debug(
        f"Executing EXT_GM2_Col_Flow_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape # [cite: 243]
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 20)
    
    avg_potential_num_to_place: float = 0.0 # 來源：新大腦.pdf (Page 20)
    if potential_numbers_to_place:
        avg_potential_num_to_place = float(np.mean(potential_numbers_to_place))

    max_val_board = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols)) # 來源：新大腦.pdf (Page 20)
    if max_val_board == 0: max_val_board = 1.0

    for c_idx in range(cols): # [cite: 244]
        current_col_values_list_orig = [val for val in grid[:, c_idx] if val != -1] # PDF typo: val != -11
        # 來源：新大腦.pdf (Page 20)
        num_filled_in_col_orig = len(current_col_values_list_orig)
        sum_current_col_values_orig = sum(current_col_values_list_orig)

        for r_idx in range(rows):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue # [cite: 245]

            # 1. Density Score
            # 來源：新大腦.pdf - EXT_GM2 Density Score (Page 20)
            density_score = (num_filled_in_col_orig + 1.0) / rows if rows > 0 else 0.0

            # 2. Value Contribution Score (Sum Score)
            # 來源：新大腦.pdf - EXT_GM2 Value Contribution (Page 20) # [cite: 246]
            potential_col_sum = sum_current_col_values_orig + avg_potential_num_to_place
            heuristic_max_col_sum = float(rows * max_val_board) # Max possible col sum
            # 來源：新大腦.pdf (Page 20)

            sum_score: float = 0.0
            if heuristic_max_col_sum > 0:
                sum_score = MathUtils.normalize_value( # [cite: 247]
                    potential_col_sum, 0, heuristic_max_col_sum, clamp=True
                )

            # 3. Sequence Completion Score
            # 來源：新大腦.pdf - EXT_GM2 Sequence Completion (Page 20)
            seq_score: float = 0.0
            if config.use_advanced_sequence_detection: # [cite: 248]
                max_len_this_placement = 0.0
                if potential_numbers_to_place:
                    temp_grid_col_slice = grid[:, c_idx].copy()
                    temp_grid_col_slice[r_idx] = int(round(avg_potential_num_to_place))
                    
                    sequences = BoardAnalyzerUtils.find_sequences_in_line( # [cite: 249]
                        list(temp_grid_col_slice),
                        min_len=config.min_len_for_sequence_score,
                        allow_gaps=config.allow_gaps_for_sequence_score
                    ) # [cite: 250]
                    for s in sequences:
                        if int(round(avg_potential_num_to_place)) in s:
                            max_len_this_placement = max(max_len_this_placement, float(len(s)))
                if rows > 0: # Normalize by max possible length in col (rows) # [cite: 251]
                    seq_score = MathUtils.normalize_value(max_len_this_placement, 0, float(rows), clamp=True)
            else: # Original simplified mend logic
                # 來源：新大腦.pdf - EXT_GM2 Simplified mend logic (Page 21)
                if 0 < r_idx < rows - 1: # [cite: 252]
                    prev_val = grid[r_idx - 1, c_idx]
                    next_val = grid[r_idx + 1, c_idx]
                    if prev_val != -1 and next_val != -1:
                        if (prev_val + next_val) % 2 == 0: # [cite: 253]
                            mend_val = (prev_val + next_val) // 2
                            if mend_val in potential_numbers_to_place and abs(mend_val - prev_val) > 0: # 來源：新大腦.pdf (Page 21) # [cite: 254]
                                seq_score = 0.75
                elif (r_idx == 0 and rows > 1 and grid[r_idx + 1, c_idx] != -1 and \
                      abs(grid[r_idx + 1, c_idx] - avg_potential_num_to_place) > 1e-6) or \
                      (r_idx == rows - 1 and rows > 1 and grid[r_idx - 1, c_idx] != -1 and \
                      abs(avg_potential_num_to_place - grid[r_idx - 1, c_idx]) > 1e-6): # 來源：新大腦.pdf (Page 21) # [cite: 255]
                      # Corrected PDF typo grid[r_idx-1, c_idx] != -1 and...
                    seq_score = 0.25 # [cite: 256]

            # Combine scores
            # 來源：新大腦.pdf - EXT_GM2 Combine scores (Page 21)
            w_density = config.w_density
            w_sum = config.w_sum_score
            w_seq = config.w_sequence_score
            total_weight = w_density + w_sum + w_seq # [cite: 257]
            if total_weight <= 0: total_weight = 1.0

            combined_score = (
                w_density * density_score + w_sum * sum_score + w_seq * seq_score
            ) / total_weight
            scores[r_idx, c_idx] = MathUtils.normalize_value(combined_score, 0, 1.0, clamp=True) # 來源：新大腦.pdf (Page 21) # [cite: 258]

    return scores * config.weight

# 來源：新大腦.pdf - 9. EXT_GM3_Adv_Connected_Comp_Vec (Page 21)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM3強化建議
# Config for this (ConnectedComponentConfig) was defined in PART 2
def EXT_GM3_Adv_Connected_Comp_Vec(
    grid: np.ndarray,
    config: ConnectedComponentConfig,
    request_id: str | None = "N/A_GM3_ConnComp", # [cite: 259]
) -> np.ndarray:
    """
    (GM3 - 高級連通元件分析-空格區域)
    核心規則:分析空格所屬的連續空格區域的大小。
    目的:偏好那些屬於較大連續空格區域的空格,這些區域可能提供更大的填補潛力或形成大型結構的機會。
    啟發式類型:連通元件分析(針對空格)
    輸出詮釋:分數越高表示該空格屬於一個面積越大的連續空格區域(分數經盤面總大小正規化)
    來源：新大腦.pdf - EXT_GM3_Adv_Connected_Comp_Vec (Page 21)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM3"
    logger.debug(
        f"Executing EXT_GM3_Adv_Connected_Comp_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape # [cite: 260]
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    visited_overall = np.zeros_like(grid, dtype=bool) # Tracks visited cells for any component search
    # 來源：新大腦.pdf (Page 22)

    for r_start in range(rows):
        for c_start in range(cols):
            if visited_overall[r_start, c_start] or grid[r_start, c_start] != -1:
                # Skip if already visited or not an empty cell [cite: 261]
                # 來源：新大腦.pdf (Page 22)
                continue

            # Start BFS for a new connected component of empty cells
            component_cells: List[Tuple[int, int]] = [] # PDF has typo: component_cells: List[Tuple[int, int]] = [ ] # [cite: 262]
            q = deque([(r_start, c_start)])
            # Visited in current BFS path (PDF typo: visited_bfs_current_component = set([(r_start, c_start)]) # Visited in current BFS path)
            visited_bfs_current_component: Set[Tuple[int,int]] = set([(r_start, c_start)]) 
            visited_overall[r_start, c_start] = True # Mark as globally visited

            while q: # 來源：新大腦.pdf (Page 22)
                r_curr, c_curr = q.popleft() # [cite: 263]
                component_cells.append((r_curr, c_curr))

                # Explore 4-connectivity neighbors
                # 來源：新大腦.pdf (Page 22) - PDF directions (0,1), (0,-1), (1,0), (-1,0)
                for dr_bfs, dc_bfs in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # [cite: 264]
                    nr, nc = r_curr + dr_bfs, c_curr + dc_bfs

                    if 0 <= nr < rows and 0 <= nc < cols and \
                       grid[nr, nc] == -1 and \
                       not visited_overall[nr, nc] and \
                       (nr, nc) not in visited_bfs_current_component: # Ensure not re-adding to q for current BFS # [cite: 265]
                        
                        visited_overall[nr, nc] = True # [cite: 266]
                        visited_bfs_current_component.add((nr, nc))
                        q.append((nr, nc))
            
            area_size = float(len(component_cells))
            
            # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM3 連通區域的「形狀」和「質量」 [cite: 267]
            shape_factor_score = 0.0
            if config.consider_shape_factor and area_size > 0:
                # Conceptual: Calculate compactness or other shape metric
                # For example, ratio of area to bounding box area [cite: 268]
                if component_cells: # [cite: 268]
                    min_r_bbox = min(r for r,c in component_cells)
                    max_r_bbox = max(r for r,c in component_cells)
                    min_c_bbox = min(c for r,c in component_cells)
                    max_c_bbox = max(c for r,c in component_cells) # [cite: 269]
                    bbox_area = (max_r_bbox - min_r_bbox + 1) * (max_c_bbox - min_c_bbox + 1)
                    if bbox_area > 0:
                        shape_factor_score = area_size / bbox_area # Compactness
            
            # Normalize area size against total number of cells in the grid [cite: 270]
            # 來源：新大腦.pdf (Page 22)
            total_cells = float(rows * cols)
            norm_area_size: float = 0.0
            if total_cells > 0:
                norm_area_size = MathUtils.normalize_value(area_size, 0, total_cells, clamp=True) # [cite: 271]
            
            # Combine base score with shape factor score
            final_component_score = norm_area_size
            if config.consider_shape_factor:
                final_component_score = (1.0 - config.shape_factor_weight) * norm_area_size + \
                                        config.shape_factor_weight * shape_factor_score # [cite: 272]
                final_component_score = MathUtils.normalize_value(final_component_score, 0, 1.0, clamp=True)


            # Assign this normalized area size score to all cells in the found component
            # 來源：新大腦.pdf (Page 23)
            for r_comp, c_comp in component_cells: # [cite: 273]
                scores[r_comp, c_comp] = final_component_score
                
    return scores * config.weight