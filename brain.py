# brain.py
# 本文件自動生成，依據新大腦.pdf、給你2025资料在深度建议一次.pdf、极限强化.pdf 維度實現
# 主要包含 AI 評分模組的核心邏輯與數學實作。

import numpy as np
import math
import new_module
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
        except OverflowError:
            # 來源：新大腦.pdf - MathUtils.sigmoid (Page 1)
            return 0.0 if -k * x > 0 else 1.0

    @staticmethod
    def normalize_value(
        value: float, min_val: float, max_val: float, clamp: bool = True
    ) -> float:
        """
        Normalizes a value to the [0, 1] range.
        Handles cases where min_val equals max_val to prevent division by zero. [cite: 3]
        Addresses Requirement 2.c (reasonable score distribution). [cite: 4]
        來源：新大腦.pdf - MathUtils.normalize_value (Page 1)
        """
        if math.isclose(max_val, min_val):
            if math.isclose(value, min_val): # 來源：新大腦.pdf (Page 1)
                return 0.5
            elif value < min_val: # 來源：新大腦.pdf (Page 2)
                return 0.0
            else:  # value > max_val (which is min_val)
                return 1.0
        normalized = (value - min_val) / (max_val - min_val)
        if clamp:
            return max(0.0, min(1.0, normalized))
        return normalized

    @staticmethod
    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """Calculates Manhattan distance between two points (r, c).
        來源：新大腦.pdf - MathUtils.manhattan_distance (Page 2) [cite: 5]
        """
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @staticmethod
    def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculates Euclidean distance between two points (r, c).
        來源：新大腦.pdf - MathUtils.euclidean_distance (Page 1) [cite: 6]
        """
        # 來源：新大腦.pdf - MathUtils.euclidean_distance (Page 2)
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def get_entropy(values: List[Any]) -> float:
        """Calculates Shannon entropy for a list of values.
        來源：新大腦.pdf - MathUtils.get_entropy (Page 2) [cite: 7]
        """
        if not values:
            return 0.0
        counts = Counter(values)
        total_count = len(values)
        entropy = 0.0
        for count in counts.values():
            probability = count / total_count
            if probability > 0: # Avoid log(0)
                 entropy -= probability * math.log2(probability)
        return entropy


# 來源：新大腦.pdf - BoardAnalyzerUtils (Page 2) [cite: 8]
class BoardAnalyzerUtils:
    """
    Provides common board analysis utility functions. [cite: 8]
    Used by modules to inspect grid neighborhoods, gradients, etc. [cite: 8]
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
        val_func: Callable[[int], float | None] = lambda x_val: float(x_val)
        if x_val != -1
        else None,
        include_center: bool = False,
    ) -> List[float]:
        """
        Retrieves values from the neighborhood of a cell. [cite: 9]
        Supports configurable radius, connectivity, and value processing. [cite: 9]
        來源：新大腦.pdf - BoardAnalyzerUtils.get_neighborhood_values (Page 2)
        """
        neighbors: List[float] = []
        rows, cols = grid.shape
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if not include_center and dr == 0 and dc == 0:
                    continue
                if not eight_connectivity:
                    if radius == 1 and abs(dr) + abs(dc) != 1: # Only N, E, S, W
                        continue
                    # 來源：新大腦.pdf - BoardAnalyzerUtils.get_neighborhood_values (Page 2)
                    # Original PDF had a typo: abs(dr)+abs(dc)>radius; (semicolon)
                    # This condition for radius > 1 and not eight_connectivity is a bit ambiguous.
                    # Assuming for non-eight_connectivity it means only cardinal directions up to `radius` distance,
                    # or a diamond shape. The PDF's example implies a filter for specific patterns.
                    # For simplicity and clarity, if not eight_connectivity and radius > 1,
                    # we might interpret it as still cardinal but within the larger radius.
                    # However, the PDF example `abs(dr)+abs(dc)>radius` seems to be for another case.
                    # Given the ambiguity, sticking to the radius 1 case for non-eight_connectivity
                    # or assuming it only applies if radius=1. For radius > 1, non-eight_connectivity is less standard.
                    # For now, this will behave as only 4-connectivity if radius=1 and not eight_connectivity.
                    # If radius > 1 and not eight_connectivity, it will behave like eight_connectivity.
                    # This part might need further clarification based on exact desired behavior for larger radii without 8-connectivity.

                nr, nc = r + dr, c + dc # 來源：新大腦.pdf (Page 2)
                if 0 <= nr < rows and 0 <= nc < cols: # 來源：新大腦.pdf (Page 2)
                    processed_val = val_func(grid[nr, nc])
                    if processed_val is not None:
                        neighbors.append(processed_val)
        return neighbors

    @staticmethod
    # P來源：新大腦.pdf - BoardAnalyzerUtils.get_value_gradient_at_cell (Page 2-3) [cite: 11]
    def get_value_gradient_at_cell(
        grid: np.ndarray,
        r: int,
        c: int,
        val_func: Callable[[int], float] = lambda x_val: float(x_val)
        if x_val != -1
        else 0.0, # 來源：新大腦.pdf (Page 3)
    ) -> Tuple[float, float]:
        """Calculates an approximate gradient (Sobel-like) at a cell. [cite: 11] Useful for modules
        analyzing value changes. [cite: 11]"""
        rows, cols = grid.shape

        def safe_val(r_in: int, c_in: int) -> float:
            if 0 <= r_in < rows and 0 <= c_in < cols:
                return val_func(grid[r_in, c_in])
            return 0.0

        # Sobel operators
        # Gx = ( (top-right + 2*middle-right + bottom-right) -
        #        (top-left  + 2*middle-left  + bottom-left) )
        # Gy = ( (bottom-left + 2*bottom-middle + bottom-right) -
        #        (top-left    + 2*top-middle    + top-right) )
        # 來源：新大腦.pdf - Gx, Gy calculation (Page 3)
        # Note: PDF formula for gx seems to have a factor of 1, e.g. "...)-1.(safe_val...)", assuming typo and it's a minus.
        # And gy has "sate_val", corrected to "safe_val".
        gx = (safe_val(r - 1, c + 1) + 2 * safe_val(r, c + 1) + safe_val(r + 1, c + 1)) - \
             (safe_val(r - 1, c - 1) + 2 * safe_val(r, c - 1) + safe_val(r + 1, c - 1))
        
        gy = (safe_val(r + 1, c - 1) + 2 * safe_val(r + 1, c) + safe_val(r + 1, c + 1)) - \
             (safe_val(r - 1, c - 1) + 2 * safe_val(r - 1, c) + safe_val(r - 1, c + 1))
        
        return gx, gy

    @staticmethod
    # 來源：新大腦.pdf - BoardAnalyzerUtils.find_sequences_in_line (Page 3)
    def find_sequences_in_line(
        line: List[int | float], # Allow float for geometric intermediate steps
        min_len: int = 3,
        check_arithmetic: bool = True,
        check_geometric: bool = False,
        allow_gaps: int = 0,
    ) -> List[List[int]]: # Returns sequences of original integer values
        """
        Finds arithmetic or geometric sequences in a 1D list of numbers,
        supporting gaps and returning sequence elements.
        This is a more faithful implementation of the PDF's logic.
        來源：新大腦.pdf - BoardAnalyzerUtils.find_sequences_in_line (Page 3-5)
        """
        sequences: List[List[int]] = []
        n = len(line)
        if n == 0: # handle empty line explicitly
            return sequences
        
        # Convert to float for internal processing, especially for geometric
        # but keep track of original int values for the final sequence list.
        # -1 (gap) will be handled as a special marker.
        
        processed_line: List[float | None] = []
        for x in line:
            if x == -1:
                processed_line.append(None) # Using None for gaps internally
            else:
                processed_line.append(float(x))


        for i in range(n):
            if processed_line[i] is None: # Cannot start sequence with a gap
                continue

            start_val = processed_line[i]
            assert start_val is not None # Should be true due to previous continue

            # Arithmetic sequence check
            if check_arithmetic:
                # 來源：新大腦.pdf - Arithmetic sequence check (Page 3)
                # Iterate through all possible second elements to define a difference
                for j in range(i + 1, n):
                    gaps_between_i_j = 0
                    k_gap_check = i + 1
                    while k_gap_check < j:
                        if processed_line[k_gap_check] is None:
                            gaps_between_i_j +=1
                        k_gap_check +=1
                    
                    if gaps_between_i_j > allow_gaps:
                        continue # Too many gaps to define initial difference with j

                    if processed_line[j] is None:
                        if j == i + 1 and allow_gaps == 0 : continue # Cannot define diff with immediate gap if no gaps allowed
                        if j > i + 1 and (j - (i + gaps_between_i_j) > 1) and allow_gaps < gaps_between_i_j +1 : continue
                        # If allow_gaps > 0, we might be able to find a diff with a later element.
                        # This loop structure is for finding the *first* element to establish 'diff'.

                    val_j = processed_line[j]
                    if val_j is None: continue # Still a gap, try next j

                    diff = val_j - start_val
                    num_steps_for_diff = (j - i) # Number of steps including gaps
                    
                    # Normalize diff if there were gaps between start_val and val_j
                    # Example: line[i]=1, gap, gap, line[j]=7. allow_gaps=2. num_steps=3. diff=6. Actual diff = 6/3 = 2.
                    if num_steps_for_diff > 1 + gaps_between_i_j : # If there are actual numbers between i and j, this logic needs refinement.
                                                                 # The PDF implies diff is established by the first non-gap pair.
                                                                 # Let's stick to the PDF's simpler interpretation for now:
                                                                 # diff is between line[i] and the first non-gap line[j]
                        pass # No adjustment if diff is just between two numbers

                    # PDF: "Avoid constant sequences unless they are all zeros"
                    # "Here, we exclude if common diff is 0 and non-zero point)"
                    # 來源：新大腦.pdf - Arithmetic constant sequence avoidance (Page 4)
                    if math.isclose(diff, 0) and not math.isclose(start_val, 0):
                        continue

                    current_seq_indices = [i]
                    current_seq_values = [int(start_val)] # Store original int values
                    
                    # Add intermediate elements if they fit the pattern and account for gaps
                    # This part is complex in the PDF, let's first establish the sequence with j
                    if gaps_between_i_j == 0 : # j is the immediate next non-gap
                         current_seq_indices.append(j)
                         current_seq_values.append(int(val_j))

                    last_val_in_seq = val_j
                    last_idx_in_seq = j
                    potential_gap_count_after_j = 0

                    for k in range(j + 1, n):
                        val_k = processed_line[k]
                        if val_k is None:
                            potential_gap_count_after_j += 1
                            if potential_gap_count_after_j > allow_gaps:
                                break # Too many gaps
                            continue
                        
                        # Expected next value if there were no gaps from last_val_in_seq to val_k
                        steps_from_last = (k - last_idx_in_seq)
                        expected_val_at_k = last_val_in_seq + diff * (steps_from_last / (potential_gap_count_after_j + 1))
                        
                        if math.isclose(val_k, expected_val_at_k):
                            current_seq_indices.append(k)
                            current_seq_values.append(int(val_k))
                            last_val_in_seq = val_k
                            last_idx_in_seq = k
                            potential_gap_count_after_j = 0 # Reset gap count
                        else:
                            break # Sequence broken

                    if len(current_seq_values) >= min_len:
                        sequences.append(current_seq_values)


            # Geometric sequence check
            if check_geometric and not math.isclose(start_val, 0): # Start_val cannot be 0 for typical geometric
                # 來源：新大腦.pdf - Geometric sequence check (Page 4)
                for j in range(i + 1, n):
                    gaps_between_i_j = 0
                    k_gap_check = i + 1
                    while k_gap_check < j:
                        if processed_line[k_gap_check] is None:
                            gaps_between_i_j +=1
                        k_gap_check +=1
                    
                    if gaps_between_i_j > allow_gaps:
                        continue

                    val_j = processed_line[j]
                    if val_j is None: continue
                    if math.isclose(val_j, 0): continue # Geometric sequence with zero is tricky

                    # PDF: "If ratio isn't integer-like and not a trivial division break"
                    # 來源：新大腦.pdf - Geometric ratio check (Page 5)
                    if math.isclose(start_val, 0): continue # Should have been caught, but defensive

                    # Try to establish ratio
                    # Using a tolerance for float comparisons might be needed if line can have floats
                    # For int lines, we expect integer ratios or clean divisions.
                    ratio_candidate = val_j / start_val
                    
                    # PDF: "Avoid constant sequences"
                    # 來源：新大腦.pdf - Geometric constant sequence avoidance (Page 5)
                    if math.isclose(ratio_candidate, 1.0) and not math.isclose(start_val, val_j): # If ratio is 1, values must be same
                        continue # This condition might be too strict if allow_gaps changes things

                    current_seq_indices = [i]
                    current_seq_values = [int(start_val)]
                    
                    if gaps_between_i_j == 0 :
                         current_seq_indices.append(j)
                         current_seq_values.append(int(val_j))

                    last_val_in_seq = val_j
                    last_idx_in_seq = j
                    potential_gap_count_after_j = 0
                    ratio = ratio_candidate # Established ratio

                    for k in range(j + 1, n):
                        val_k = processed_line[k]
                        if val_k is None:
                            potential_gap_count_after_j += 1
                            if potential_gap_count_after_j > allow_gaps:
                                break
                            continue
                        
                        if math.isclose(val_k, 0) : break # Geometric sequence broken by zero

                        # Expected next value
                        # Number of actual steps of ratio application
                        num_ratio_applications = (k - last_idx_in_seq) // (potential_gap_count_after_j + 1)
                        if (k - last_idx_in_seq) % (potential_gap_count_after_j + 1) != 0: # not a clean step
                            break 

                        expected_val_at_k = last_val_in_seq * (ratio ** num_ratio_applications)

                        if math.isclose(val_k, expected_val_at_k):
                            current_seq_indices.append(k)
                            current_seq_values.append(int(val_k))
                            last_val_in_seq = val_k
                            last_idx_in_seq = k
                            potential_gap_count_after_j = 0
                        else:
                            break
                    
                    if len(current_seq_values) >= min_len:
                        sequences.append(current_seq_values)
        
        # Remove duplicate sequences that might have been found from different start points
        # or due to the simplified looping structure compared to the PDF's intricate one.
        unique_sequences = []
        for seq in sequences:
            if seq not in unique_sequences:
                unique_sequences.append(seq)
        return unique_sequences

    @staticmethod
    # 來源：新大腦.pdf - BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions (Page 5) [cite: 16]
    def get_card_max_value_from_grid_dimensions(grid_shape: Tuple[int, int]) -> int:
        """Calculates the maximum possible number on the card based on its dimensions. [cite: 16]"""
        rows, cols = grid_shape
        if rows == 0 or cols == 0:
            return 0
        return rows * cols

    @staticmethod
    # 來源：新大腦.pdf - BoardAnalyzerUtils.get_all_possible_numbers_for_grid (Page 5) [cite: 17]
    def get_all_possible_numbers_for_grid(grid_shape: Tuple[int, int]) -> Set[int]:
        """Returns a set of all numbers that could theoretically appear on a grid of given
        dimensions. [cite: 17]"""
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
        Determines the set of numbers that can be legally placed onto an empty cell in the grid.
        This adheres to the rule: numbers are 1 to R*C and no positive number can be repeated. [cite: 19]
        (Requirement 1.c) [cite: 20]
        來源：新大鵝.pdf - BoardAnalyzerUtils.get_legal_values_for_placement (Page 5-6)
        """
        if grid.size == 0: # 來源：新大腦.pdf (Page 6)
            return set()
        rows, cols = grid.shape
        all_possible_on_this_grid = (
            BoardAnalyzerUtils.get_all_possible_numbers_for_grid((rows, cols))
        )
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
    # Common config fields can go here if any
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
    enable_repulsion: bool = Field(default=False, description="是否啟用斥力概念")
    # Undesirable pairs could be more complex, e.g. ((val1, val2), repulsion_factor)
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
    decay_exponent: float = Field(default=1.5, gt=0.0, description="影響力隨距離衰減的指數 (e.g., 1 for 1/r, 2 for 1/r^2)") # [cite: 39]
    max_influence_radius: int = Field(default=3, ge=1, description="考慮數字影響力的最大曼哈頓距離") # [cite: 39]
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - D3 「電荷」性質的擴展
    enable_negative_charges: bool = Field(default=False, description="是否啟用負電荷（排斥力）概念")
    negative_charge_map: Dict[int, float] = Field(default_factory=dict, description="定義哪些數字視為負電荷及其強度（<0）")


class DiscontinuityRepairConfig(BaseModuleConfig):
    # 來源：新大腦.pdf - EXT_F10 parameters (Page 12)
    # 來源：给你2025资料在深度建议一次.pdf - EXT_F10 Pydantic配置範例 (Page 4)
    min_sequence_len_to_score: int = Field(default=3, ge=2, description="視為有效的最小序列長度")
    allow_gaps_in_sequence: int = Field(default=1, ge=0, description="序列中允許的最大間隙數") # [cite: 43]
    check_arithmetic: bool = Field(default=True, description="是否檢查等差序列")
    check_geometric: bool = Field(default=False, description="是否檢查等比序列")
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - F10 序列價值評估
    sequence_quality_weighting: bool = Field(default=False, description="是否對序列質量（如構成數字大小）進行額外加權")
    high_value_sequence_threshold_factor: float = Field(default=0.75, ge=0, le=1, description="序列平均值超過盤面最大值*此因子時視為高價值")


class PathfindingValueConfig(BaseModuleConfig):
    # 來源：新大腦.pdf - EXT_P7 parameters (Page 14)
    max_path_search_depth: int = Field(default=4, ge=1, description="搜尋路徑的最大長度") # [cite: 51]
    path_value_decay_factor: float = Field(default=1.0, ge=0.0, description="路徑長度對價值的衰減因子 (e.g., val / (len^decay))") # [cite: 51]
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - P7 BFS優化 / 只針對高價值
    target_value_threshold_factor: float = Field(default=0.5, ge=0, le=1, description="只尋找連接到值高於盤面最大值*此因子的路徑 (0=不篩選)")


# ... (Pydantic Configs for other modules will be defined as we implement them)

# --- Scoring Module Implementations ---

# 來源：新大腦.pdf - 1. EXT_A2_Weighted_Proximity_Vec (Page 7) [cite: 21]
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_A2強化建議
# 來源：给你2025资料在深度建议一次.pdf - EXT_A2 Pydantic配置範例 (Page 2)
def EXT_A2_Weighted_Proximity_Vec(
    grid: np.ndarray,
    config: WeightedProximityConfig, # Now expects the Pydantic config object
    request_id: str | None = "N/A_A2_Proximity",
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
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    radius = config.radius
    value_weight_factor = config.value_weight_factor
    distance_decay_factor = config.distance_decay_factor
    
    # 來源：新大腦.pdf - EXT_A2 Self-adaptive weights (Conceptual) (Page 7)
    # 實現概念性自適應權重: 若盤面平均值高，增加 value_weight_factor
    # This can be part of a more sophisticated config update mechanism or pre-calculation in analyzer
    # For now, let's assume config provides the final factors.
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
        num_neighbors_in_radius
        * max_val_on_grid
        * value_weight_factor
    ) # Min dist is 1, so 1**decay_factor is 1
    # 來源：新大腦.pdf (Page 8) [cite: 23] - original was / (1**distance_decay_factor)

    if heuristic_max_score <= 0: 
        heuristic_max_score = 1.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:
                continue

            proximity_score = 0.0
            
            # 來源：新大腦.pdf - EXT_A2 Conceptual repulsion (Page 7)
            # 這裡的斥力計算需要一個「假設填入的值」才能判斷是否與鄰居形成不良配對
            # 目前模組只評估空格本身，若要加入此斥力，需修改函式簽名或由 analyzer 傳入假設值
            # conceptual_placed_value = ... (needs to be determined or iterated)
            # For now, skipping the PDF's direct repulsion logic for UNDESIRABLE_PAIRS
            # as it requires a `some_proposed_val_for_this_cell`.
            # The config `undesirable_pairs_config` is there for future enhancement.

            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0: # 來源：新大腦.pdf (Page 8) [cite: 21]
                        continue
                    
                    nr, nc = r_idx + dr, c_idx + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                        dist = MathUtils.manhattan_distance(
                            (r_idx, c_idx), (nr, nc)
                        )
                        # 來源：新大腦.pdf (Page 8) [cite: 22]
                        if dist == 0: dist = 1 # Safeguard

                        score_contribution = (
                            grid[nr, nc] * value_weight_factor
                        ) / (dist**distance_decay_factor) # 來源：新大腦.pdf (Page 8) [cite: 22]
                        proximity_score += score_contribution
            
            if heuristic_max_score > 0: # 來源：新大腦.pdf (Page 8) [cite: 23]
                scores[r_idx, c_idx] = MathUtils.normalize_value(
                    proximity_score, 0, heuristic_max_score, clamp=True
                )
            else:
                scores[r_idx, c_idx] = 0.0
    return scores * config.weight
    # brain.py (Continued)
# ... (Imports, MathUtils, BoardAnalyzerUtils, BaseModuleConfig, and configs from PART 1 remain the same) ...

# --- Pydantic Config Models for Modules (Continued) ---

class ResourceControlConfig(BaseModuleConfig):
    # 來源：新大腦.pdf - EXT_R5 parameters (Page 16-17)
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
    w_sequence_score: float = Field(default=0.3, ge=0.0, le=1.0, description="序列分數權重")
    use_advanced_sequence_detection: bool = Field(default=True, description="是否使用 BoardAnalyzerUtils.find_sequences_in_line 進行序列評估")
    min_len_for_sequence_score: int = Field(default=3, ge=2)
    allow_gaps_for_sequence_score: int = Field(default=1, ge=0) # Consistent with F10


class ConnectedComponentConfig(BaseModuleConfig): # For GM3
    # 來源：新大腦.pdf - EXT_GM3 parameters (Page 21-22)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM3 連通區域的「形狀」和「質量」
    consider_shape_factor: bool = Field(default=False, description="是否考慮連通區域的形狀因子（概念性）")
    shape_factor_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="形狀因子權重（如果啟用）")


class SpatialAutocorrelationConfig(BaseModuleConfig): # For GM4
    # 來源：新大腦.pdf - EXT_GM4 parameters (Page 23-24)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM4 自相關性方向
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
    enable_quality_enhancement: bool = Field(default=True)
    score_arithmetic_3_mend_high_val_bonus: float = Field(default=0.2, ge=0.0, description="高價值等差序列修復額外獎勵") # PDF uses 0.9 directly, here use as bonus
    high_value_threshold_factor_gm5: float = Field(default=0.66, ge=0, le=1, description="平均值超過盤面最大值*此因子視為高價值")


class SymmetryPotentialConfig(BaseModuleConfig): # For GM6
    # 來源：新大腦.pdf - EXT_GM6 parameters (Page 27-28)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM6 對稱類型權重
    score_horizontal: float = Field(default=0.7, ge=0.0)
    score_vertical: float = Field(default=0.7, ge=0.0)
    score_point_center: float = Field(default=0.8, ge=0.0)
    score_main_diagonal: float = Field(default=0.6, ge=0.0)
    score_anti_diagonal: float = Field(default=0.6, ge=0.0)
    strict_square_for_diagonal: bool = Field(default=True, description="對角線對稱是否嚴格要求方形棋盤") # 來源：新大腦.pdf (Page 29) [cite: 135, 138]


class NumericGapsConfig(BaseModuleConfig): # For GM7
    # 來源：新大腦.pdf - EXT_GM7 parameters (Page 29-30)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM7 「間隙」的上下文
    score_arithmetic_1_gap_fill: float = Field(default=0.9, ge=0.0)
    score_arithmetic_generic_mend: float = Field(default=0.7, ge=0.0)
    score_arithmetic_generic_extend: float = Field(default=0.5, ge=0.0)
    # 來源：新大腦.pdf - EXT_GM7 Added: scoring for quality (conceptual) (Page 30)
    enable_quality_enhancement_gm7: bool = Field(default=True)
    score_gap_fill_high_val_bonus: float = Field(default=0.1, ge=0.0) # PDF uses 0.95 directly
    high_value_threshold_factor_gm7: float = Field(default=0.66, ge=0, le=1)
    # Conceptual: score_gap_fill_long_seq_potential


class EdgeAffinityConfig(BaseModuleConfig): # For GM8
    # 來源：新大腦.pdf - EXT_GM8 parameters (Page 31-32)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM8 affinity_mode 的動態化
    affinity_mode: str = Field(default="prefer_edge", pattern="^(prefer_edge|avoid_edge)$") #
    corner_bonus_prefer: float = Field(default=0.2, ge=0.0) #
    corner_penalty_avoid: float = Field(default=0.2, ge=0.0) #


class CenterControlConfig(BaseModuleConfig): # For GM9
    # 來源：新大腦.pdf - EXT_GM9 parameters (Page 34)
    affinity_mode: str = Field(default="prefer_center", pattern="^(prefer_center|avoid_center)$") #


class BlockingValueConfig(BaseModuleConfig): # For GM10
    # 來源：新大腦.pdf - EXT_GM10 parameters (Page 35-36)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM10 UNDESIRABLE_SEQUENCES 的擴展與學習
    # For simplicity, keeping UNDESIRABLE_SEQUENCES in code, but could be part of config
    # undesirable_sequences_list: List[List[int]] = Field(default_factory=lambda: [[1,1,1], [2,2,2]])
    base_safety_score: float = Field(default=0.9, ge=0.0, le=1.0, description="未形成不良模式時的基礎安全分")
    penalty_for_undesirable: float = Field(default=-0.8, le=0.0, description="形成不良模式的懲罰（加到基礎分上，所以是負值或使基礎分降低）") # PDF uses 0.1 if bad


class PairCorrelationConfig(BaseModuleConfig): # For GM11
    # 來源：新大腦.pdf - EXT_GM11 parameters (Page 38-39)
    # FAVORABLE_PAIRS_SCORES can be complex, for now, allow defining a few key ones in config
    # A more advanced config might load these from a file or a larger structure.
    favorable_pairs: Dict[Tuple[int, int], float] = Field(default_factory=lambda: {
        (3, 7): 0.8, (7, 3): 0.8, (1, 2): 0.6, (2, 1): 0.6, (10,20):0.7, (20,10):0.7
    })


class IslandAnalysisConfig(BaseModuleConfig): # For GM12
    # 來源：新大腦.pdf - EXT_GM12 parameters (Page 40-41)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM12 島嶼特徵的豐富化
    w_size: float = Field(default=0.4, ge=0.0, le=1.0) #
    w_compactness: float = Field(default=0.3, ge=0.0, le=1.0) #
    w_avg_value: float = Field(default=0.3, ge=0.0, le=1.0) #
    # Conceptual: add w_shape_factor, w_boundary_value etc.


class SequenceDiversityConfig(BaseModuleConfig): # For GM13
    # 來源：新大腦.pdf - EXT_GM13 parameters (Page 42)
    short_sequence_len: int = Field(default=3, ge=2) #
    # Heuristic max_distinct_sequences used for normalization, not directly a config for behavior
    # Could add weights for different types of diverse sequences (arithmetic vs identical)


class RiskAssessmentConfig(BaseModuleConfig): # For GM14
    # 來源：新大腦.pdf - EXT_GM14 parameters (Page 44)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM14 靈活性度量的複雜化
    flexibility_metric_mode: str = Field(default="subsequent_moves", pattern="^(subsequent_moves|product_moves_empty_cells)$")


class InformationGainConfig(BaseModuleConfig): # For GM15
    # 來源：新大腦.pdf - EXT_GM15 parameters (Page 45-46)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM15 熵計算的對象
    entropy_scope: str = Field(default="global_full", pattern="^(global_full|global_filled_only)$", description="熵計算範圍：global_full (含-1), global_filled_only (不含-1)")


class HarmonicCentralityConfig(BaseModuleConfig): # For GM16
    # 來源：新大腦.pdf - EXT_GM16 parameters (Page 47)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM16 節點的定義
    node_definition: str = Field(default="all_cells", pattern="^(all_cells|empty_cells_only|filled_cells_only)$", description="計算調和中心性時考慮的節點類型")


class LocalEntropyMinimizationConfig(BaseModuleConfig): # For GM17
    # 來源：新大腦.pdf - EXT_GM17 parameters (Page 48)
    radius: int = Field(default=1, ge=1, description="局部鄰域半徑")
    # max_local_entropy_change is for normalization, calculated internally


class RLValueEstimationConfig(BaseModuleConfig): # For GM18
    # 來源：新大腦.pdf - EXT_GM18 parameters (Page 50-51)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM18 特徵庫的擴展與優化
    # Feature weights would ideally be loaded or learned.
    feature_weights: Dict[str, float] = Field(default_factory=lambda: {
        "identical_3": 1.0,
        "arithmetic_3": 0.7,
        "board_density_factor": 0.2,
        "central_control_boost": 0.1, # 來源：新大腦.pdf (Page 51)
        "edge_affinity_boost": 0.05,   # 來源：新大腦.pdf (Page 52)
    })
    # More features could be added here with their weights


class SkipPatternConfig(BaseModuleConfig): # For GM19
    # 來源：新大腦.pdf - EXT_GM19 parameters (Page 53-54)
    min_occurrences_for_pattern_factor: float = Field(default=0.05, ge=0.0, le=1.0, description="形成主導跳格模式所需的最少出現次數（佔總跳格數的比例）") # PDF uses 0.05 of len(skip_vector_tuples_list) [cite: 229]
    base_pattern_definition: str = Field(default="left_to_right_top_to_bottom", description="理論基礎位置的掃描模式（概念性）")


class SkipPatternConfidenceConfig(BaseModuleConfig): # For GM20
    # 來源：新大腦.pdf - EXT_GM20 parameters (Page 55-56)
    min_occurrences_for_pattern_factor_gm20: float = Field(default=0.05, ge=0.0, le=1.0) # Same as GM19's factor
    # 來源：新大腦.pdf - EXT_GM20 arithmetic sequence enhancement (Page 57)
    arithmetic_enhancement_bonus: float = Field(default=0.4, ge=0.0, description="形成一致等差序列的增強因子")
    internal_gap_fill_bonus: float = Field(default=0.1, ge=0.0, description="填充內部間隙形成等差序列的額外獎勵")


# --- Scoring Module Implementations (Continued) ---

# 來源：新大腦.pdf - 2. EXT_M3_Local_Heterogeneity_Vec (Page 8)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_M3強化建議
def EXT_M3_Local_Heterogeneity_Vec(
    grid: np.ndarray,
    config: LocalHeterogeneityConfig, # Expects Pydantic config
    request_id: str | None = "N/A_M3_Heterogeneity",
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

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    radius = config.radius
    min_neighbors_for_robust_score = config.min_neighbors_for_robust_score
    
    # 來源：新大腦.pdf - EXT_M3 Conceptual: Self-adaptive min_neighbors (Page 9)
    # Example of self-adaptation (can be more complex)
    if rows * cols < 10: # For very small grids
        min_neighbors_for_robust_score = max(0, min(min_neighbors_for_robust_score, 1))


    all_possible_values_in_game = BoardAnalyzerUtils.get_all_possible_numbers_for_grid(
        grid.shape
    ) # 來源：新大腦.pdf (Page 9)
    if not all_possible_values_in_game:
        return scores 

    # 來源：新大腦.pdf - EXT_M3 Theoretical maximum entropy (Page 9) [cite: 26, 27, 28, 29, 30, 31, 32, 33]
    # The PDF has several notes on max_theoretical_entropy.
    # Simplified logic: log2(N) if N > 1, else log2(2) or 1.0 to avoid log2(1)=0 or log2(0).
    num_distinct_symbols = len(all_possible_values_in_game)
    if num_distinct_symbols > 1:
        max_theoretical_diversity_measure = math.log2(num_distinct_symbols)
    elif num_distinct_symbols == 1:
        max_theoretical_diversity_measure = math.log2(2) # Avoid log2(1)=0, provide some scale
    else: 
        max_theoretical_diversity_measure = 1.0 

    if max_theoretical_diversity_measure <= 0: max_theoretical_diversity_measure = 1.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:
                continue
            
            neighbor_values = BoardAnalyzerUtils.get_neighborhood_values(
                grid, r_idx, c_idx, radius=radius, eight_connectivity=True,
                val_func=lambda x_val: int(x_val) if x_val != -1 else None,
                include_center=False,
            ) # 來源：新大腦.pdf (Page 10) [cite: 34]

            if len(neighbor_values) < min_neighbors_for_robust_score: # 來源：新大腦.pdf (Page 10)
                scores[r_idx, c_idx] = 0.0
                continue

            current_diversity_value: float
            if config.diversity_metric == "entropy":
                current_diversity_value = MathUtils.get_entropy(neighbor_values) # 來源：新大腦.pdf (Page 10) [cite: 35]
            elif config.diversity_metric == "gini":
                # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - M3 熵以外的異質性度量 (基尼不純度)
                counts = Counter(neighbor_values)
                impurity = 1.0
                for count_val in counts.values():
                    prob = count_val / len(neighbor_values)
                    impurity -= prob**2
                current_diversity_value = impurity 
                # Gini's max is (k-1)/k for k classes. For normalization, 
                # we could normalize Gini against its own theoretical max based on num_distinct_symbols.
                # For simplicity here, we are normalizing against log2(N) as a general diversity cap.
                # This might not be ideal for Gini. A more proper normalization for Gini:
                # max_gini = (num_distinct_symbols -1) / num_distinct_symbols if num_distinct_symbols > 0 else 0
                # if max_gini > 0: normalized_gini = current_diversity_value / max_gini else 0
            elif config.diversity_metric == "unique_count":
                current_diversity_value = float(len(set(neighbor_values)))
                # Normalize unique_count against min(len(neighbor_values), num_distinct_symbols)
                max_possible_unique_in_neighborhood = min(len(neighbor_values), num_distinct_symbols)
                if max_possible_unique_in_neighborhood > 0 :
                    current_diversity_value = current_diversity_value / max_possible_unique_in_neighborhood
                else:
                    current_diversity_value = 0.0
                 # This direct normalization makes its range [0,1] already for unique_count ratio
                max_theoretical_diversity_measure = 1.0 # Adjust for unique_count ratio

            else: # Fallback to entropy
                current_diversity_value = MathUtils.get_entropy(neighbor_values)

            if max_theoretical_diversity_measure > 0:
                # 來源：新大腦.pdf - EXT_M3 Normalizing (Page 10) [cite: 36, 38]
                normalized_score = current_diversity_value / max_theoretical_diversity_measure
                scores[r_idx, c_idx] = MathUtils.normalize_value(
                    normalized_score, 0, 1, clamp=True 
                )
            else:
                scores[r_idx, c_idx] = 0.0
    return scores * config.weight


# 來源：新大腦.pdf - 3. EXT_D3_Potential_Field_Vec (Page 10)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_D3強化建議
def EXT_D3_Potential_Field_Vec(
    grid: np.ndarray,
    config: PotentialFieldConfig,
    request_id: str | None = "N/A_D3_Potential",
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

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    decay_exponent = config.decay_exponent # 來源：新大腦.pdf (Page 11) [cite: 39]
    max_influence_radius = config.max_influence_radius # 來源：新大腦.pdf (Page 11) [cite: 39]
    
    max_possible_val_on_grid = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions(
        (rows, cols)
    ) # 來源：新大腦.pdf (Page 11) [cite: 39]
    if max_possible_val_on_grid == 0: return scores

    # 來源：新大腦.pdf - EXT_D3 Heuristic maximum potential (Page 11) [cite: 40]
    # Sum of max_value / (min_dist^decay) for all cells in radius.
    # This is a very rough upper bound.
    num_cells_in_radius_approx = (2 * max_influence_radius + 1)**2 - 1 # Max neighbors
    heuristic_max_potential = num_cells_in_radius_approx * (
        max_possible_val_on_grid / (1**decay_exponent) # Assuming min dist 1
    )
    if heuristic_max_potential <= 0: heuristic_max_potential = 1.0 # 來源：新大腦.pdf (Page 11) [cite: 41]

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:
                continue

            current_cell_potential = 0.0
            for nr in range(rows):
                for nc in range(cols):
                    if grid[nr, nc] != -1:  # If it's a filled cell (a "charge")
                        charge_val = float(grid[nr, nc])
                        
                        # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - D3 「電荷」性質的擴展
                        if config.enable_negative_charges and int(charge_val) in config.negative_charge_map:
                            charge_val = config.negative_charge_map[int(charge_val)] # Use configured negative strength
                        elif charge_val <= 0 and not config.enable_negative_charges: # Original logic considered only positive
                            continue

                        dist = MathUtils.manhattan_distance((r_idx, c_idx), (nr, nc))
                        
                        if dist == 0: continue # Should not happen if only scoring empty cells
                        if dist > max_influence_radius: continue # 來源：新大腦.pdf (Page 11) [cite: 36]

                        # Potential = charge_value / distance^decay_exponent
                        # 來源：新大腦.pdf (Page 11) [cite: 42]
                        potential_contribution = charge_val / (dist**decay_exponent)
                        current_cell_potential += potential_contribution
            
            scores[r_idx, c_idx] = MathUtils.normalize_value(
                current_cell_potential, 0, heuristic_max_potential, clamp=True
            ) # Note: if negative charges are strong, potential could be < 0. Normalization min_val might need adjustment.
              # For now, assuming 0 as min, so strong repulsion would be clamped to 0.
              # A bipolar normalization might be (-heuristic_max, heuristic_max) -> (0,1)
              # or separate attractive/repulsive scores. Sticking to PDF's normalization for now.

    return scores * config.weight
    # brain.py (Continued)
# ... (Imports, MathUtils, BoardAnalyzerUtils, BaseModuleConfig, and configs from PART 1 & 2 remain the same) ...

# --- Scoring Module Implementations (Continued) ---

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
    啟發式類型:序列與模式識別
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
        return scores

    legal_values_for_placement = BoardAnalyzerUtils.get_legal_values_for_placement(grid) # 來源：新大腦.pdf (Page 12)
    if not legal_values_for_placement:
        return scores

    min_sequence_len_to_score = config.min_sequence_len_to_score
    
    # 來源：新大腦.pdf - EXT_F10 Heuristic max length for normalization (Page 12)
    heuristic_max_len = float(max(rows, cols))
    if heuristic_max_len < min_sequence_len_to_score: # 來源：新大腦.pdf (Page 12)
        heuristic_max_len = float(min_sequence_len_to_score)
    if heuristic_max_len <= 0: heuristic_max_len = 1.0 

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 12)
                continue

            max_len_contribution_for_this_cell: float = 0.0 # 來源：新大腦.pdf (Page 12)

            for val_to_try in legal_values_for_placement:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = val_to_try
                current_val_max_len: float = 0.0

                lines_to_check_data: List[Tuple[List[int], str]] = []
                # 1. Check Row
                # 來源：新大腦.pdf - EXT_F10 Check Row (Page 12)
                lines_to_check_data.append((list(temp_grid[r_idx, :]), "row"))
                # 2. Check Column
                # 來源：新大腦.pdf - EXT_F10 Check Column (Page 13)
                lines_to_check_data.append((list(temp_grid[:, c_idx]), "col"))
                # 3. Check Diagonals
                # 來源：新大腦.pdf - EXT_F10 Check Diagonals (Page 13)
                diag1_line = list(np.diag(temp_grid, k=c_idx - r_idx))
                lines_to_check_data.append((diag1_line, "diag1"))
                
                flipped_temp_grid = np.fliplr(temp_grid)
                flipped_c_idx = cols - 1 - c_idx # Max col index - current col index
                diag2_line = list(np.diag(flipped_temp_grid, k=flipped_c_idx - r_idx))
                lines_to_check_data.append((diag2_line, "diag2"))

                for line_values, line_type_debug in lines_to_check_data:
                    # 來源：新大腦.pdf - EXT_F10 find_sequences_in_line call (Page 13)
                    # Using the more complete find_sequences_in_line from BoardAnalyzerUtils
                    sequences_found = BoardAnalyzerUtils.find_sequences_in_line(
                        line_values,
                        min_len=min_sequence_len_to_score,
                        check_arithmetic=config.check_arithmetic,
                        check_geometric=config.check_geometric,
                        allow_gaps=config.allow_gaps_in_sequence,
                    )
                    for seq in sequences_found:
                        if val_to_try in seq:  # Check if the placed value is part of this new/extended sequence
                            # 來源：新大腦.pdf (Page 13)
                            seq_len = float(len(seq))
                            # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - F10 序列價值評估
                            if config.sequence_quality_weighting:
                                avg_val_in_seq = sum(seq) / len(seq) if len(seq) > 0 else 0
                                max_board_val = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows,cols))
                                if max_board_val > 0 and avg_val_in_seq > (max_board_val * config.high_value_sequence_threshold_factor):
                                    seq_len *= 1.2 # Example: Boost score for high-value sequences
                            current_val_max_len = max(current_val_max_len, seq_len)
                
                if current_val_max_len >= min_sequence_len_to_score:
                    max_len_contribution_for_this_cell = max(
                        max_len_contribution_for_this_cell, current_val_max_len
                    )
            
            if heuristic_max_len > 0: # 來源：新大腦.pdf (Page 13)
                scores[r_idx, c_idx] = MathUtils.normalize_value(
                    max_len_contribution_for_this_cell,
                    0, # Min possible score for length contribution
                    heuristic_max_len,
                    clamp=True,
                )
            else: # 來源：新大腦.pdf (Page 14)
                scores[r_idx, c_idx] = 0.0
    return scores * config.weight


# 來源：新大腦.pdf - 5. EXT_P7_Pathfinding_Value_Vec (Page 14)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_P7強化建議
# Config for this (PathfindingValueConfig) was defined in PART 1
def EXT_P7_Pathfinding_Value_Vec(
    grid: np.ndarray,
    config: PathfindingValueConfig,
    request_id: str | None = "N/A_P7_Pathfinding",
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

    rows, cols = grid.shape
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
    ) # 來源：新大腦.pdf (Page 14)
    if max_possible_val_on_grid == 0: max_possible_val_on_grid = 1.0

    # 來源：新大腦.pdf - EXT_P7 Heuristic max path score (Page 14-15)
    # A very loose upper bound: (max_depth_search_radius_squared_area) * max_val / (1^decay)
    # The PDF uses (2*max_path_search_depth + 1)**2, which is area.
    # Let's consider max connections. Max neighbors in BFS up to depth D is roughly sum of 4*i for i=1 to D.
    # Simpler heuristic from PDF:
    heuristic_max_path_score = (
        (2 * max_path_search_depth + 1)**2 * max_possible_val_on_grid / (1**path_value_decay_factor)
    )
    if heuristic_max_path_score <= 0: heuristic_max_path_score = 1.0 # 來源：新大腦.pdf (Page 15)

    target_value_min_threshold = max_possible_val_on_grid * config.target_value_threshold_factor

    for r_start in range(rows):
        for c_start in range(cols):
            if grid[r_start, c_start] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 14)
                continue
            
            max_score_for_this_cell: float = 0.0 # 來源：新大腦.pdf (Page 15)

            # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - P7 只針對高價值潛在數字 (conceptual)
            # For this implementation, we iterate all legal values as per PDF base logic
            # Enhancement: filter legal_values_for_placement here if needed.

            for val_to_try in legal_values_for_placement:
                # The PDF states: "The original grid is used to find *existing* numbers."
                # "The path itself can traverse other empty cells."
                # So, val_to_try is not actually placed on a temp_grid for BFS pathfinding logic.
                # BFS explores from (r_start, c_start) through other empty cells to existing numbers.
                # The value of val_to_try might influence the *decision* to place it there,
                # but the path score itself is about connecting (r_start, c_start) to existing numbers.
                # The PDF seems to calculate a score for (r_start, c_start) if val_to_try were placed,
                # by summing up values of paths originating from it.
                # The current logic in the PDF seems to iterate val_to_try but doesn't use it in BFS.
                # Let's assume val_to_try is for future "what if this number is placed" scenarios,
                # but for the path score, it's about the connectivity of the empty cell (r_start, c_start).
                # The loop over val_to_try might be redundant if it's not used in path score calculation.
                # Re-reading PDF: "The BFS explores from the cell (r_start, c_start) *as if* val_to_try is placed there."
                # This implies val_to_try *is* relevant, perhaps as the starting "charge" or value of the path.
                # However, the path score `reached_val / (effective_path_len ** ...)` uses `reached_val` (existing number).
                # For now, I will follow the PDF structure where `val_to_try` is looped but not directly used in the score sum,
                # which means the score for (r_start, c_start) will be the same regardless of `val_to_try`.
                # This implies the outer loop for `val_to_try` for *this specific module's scoring as written in PDF* might be optimized out
                # unless `val_to_try` is meant to affect `target_value_min_threshold` or pathing rules (which it currently doesn't).
                # For "不可有任何簡化效能 只能增強", I will keep the loop.

                current_placement_path_score: float = 0.0
                # ((r, c), current_path_length_from_start)
                q = deque([((r_start, c_start), 0)]) # 來源：新大腦.pdf (Page 15)
                # Visited for this specific BFS starting at (r_start, c_start)
                visited_for_bfs: Set[Tuple[int,int]] = set([(r_start, c_start)]) # 來源：新大腦.pdf (Page 15)
                
                head_count = 0 # Safety break for BFS # 來源：新大腦.pdf (Page 15)
                # PDF: max_bfs_steps = rows* cols * len(legal_values_for_placement) - this can be huge
                # Using a more constrained but still generous limit based on depth for practical reasons
                max_bfs_steps_practical = (2 * max_path_search_depth + 1)**2 * 4 # Max cells in search area * avg degree
                
                paths_found_this_bfs: List[Tuple[int,int,int]] = [] # (val, len, count) for unique paths

                while q and head_count < max_bfs_steps_practical: # 來源：新大腦.pdf (Page 15)
                    head_count += 1
                    (curr_r, curr_c), path_len = q.popleft()

                    # Explore neighbors (4-connectivity)
                    # PDF typo: (0,1) (0,1) corrected to (0,1) (0,-1) (1,0) (-1,0)
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # 來源：新大腦.pdf (Page 15)
                        next_r, next_c = curr_r + dr, curr_c + dc

                        if 0 <= next_r < rows and 0 <= next_c < cols:
                            # If neighbor is an *existing number* on the original grid
                            if grid[next_r, next_c] != -1: # 來源：新大腦.pdf (Page 15)
                                reached_val = int(grid[next_r, next_c])
                                if reached_val < target_value_min_threshold and config.target_value_threshold_factor > 0:
                                    continue # Skip if below threshold (enhancement)

                                effective_path_len = path_len + 1 # Distance to this existing number
                                
                                # Path score contribution
                                path_score_contrib = reached_val / (effective_path_len**path_value_decay_factor)
                                current_placement_path_score += path_score_contrib
                                # 來源：新大腦.pdf - Do not add this to visited_for_bfs or queue (Page 15)

                            # If neighbor is an *empty cell* (excluding starting cell if path_len is 0 implicitly by (curr_r,curr_c))
                            # and path is not too long, and not yet visited in this BFS
                            elif (next_r, next_c) not in visited_for_bfs and \
                                 grid[next_r, next_c] == -1 and \
                                 path_len + 1 < max_path_search_depth: # 來源：新大腦.pdf (Page 15)
                                visited_for_bfs.add((next_r, next_c))
                                q.append(((next_r, next_c), path_len + 1))
                
                # The PDF structure implies max_score_for_this_cell is updated per val_to_try.
                # If val_to_try is not used in current_placement_path_score, this loop is not varying the path score.
                # For now, following structure, assuming val_to_try *could* be used in a more advanced version.
                if current_placement_path_score > max_score_for_this_cell: # 來源：新大腦.pdf (Page 16)
                    max_score_for_this_cell = current_placement_path_score
            
            scores[r_start, c_start] = MathUtils.normalize_value(
                max_score_for_this_cell, 0, heuristic_max_path_score, clamp=True
            ) # 來源：新大腦.pdf (Page 16)
    return scores * config.weight


# 來源：新大腦.pdf - 6. EXT_R5_Resource_Control_Vec (Page 16)
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

    effective_request_id = request_id if request_id else "N/A_brain_R5"
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
    ) # 來源：新大腦.pdf (Page 16)
    if max_possible_val_on_grid == 0: max_possible_val_on_grid = 1.0 # 來源：新大腦.pdf (Page 16)

    # 來源：新大腦.pdf - EXT_R5 hypothetical_high_val_placed (Page 16)
    hypothetical_high_val_placed: float = 0.0
    if potential_numbers_to_place:
        # Ensure potential_numbers_to_place is not empty before np.max
        hypothetical_high_val_placed = float(np.max(potential_numbers_to_place))


    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 17)
                continue

            # 1. Row Completeness Score
            # 來源：新大腦.pdf - EXT_R5 Row Completeness (Page 17)
            num_filled_in_row = np.count_nonzero(grid[r_idx, :] != -1)
            row_completeness_score = (num_filled_in_row + 1.0) / cols if cols > 0 else 0.0

            # 2. Column Completeness Score
            # 來源：新大腦.pdf - EXT_R5 Column Completeness (Page 17)
            num_filled_in_col = np.count_nonzero(grid[:, c_idx] != -1)
            col_completeness_score = (num_filled_in_col + 1.0) / rows if rows > 0 else 0.0
            
            # 3. Value Capture Score
            # 來源：新大腦.pdf - EXT_R5 Value Capture Score (Page 17)
            value_capture_score: float = 0.0
            if hypothetical_high_val_placed > 0 and max_possible_val_on_grid > 0:
                # Normalizing the highest possible value we could place
                value_capture_score = MathUtils.normalize_value(
                    hypothetical_high_val_placed, 1, max_possible_val_on_grid, clamp=True
                )
            
            # Combine scores
            # 來源：新大腦.pdf - EXT_R5 Combine scores (Page 17)
            w_row = config.w_row_completeness
            w_col = config.w_col_completeness
            w_val = config.w_value_capture
            
            # Ensure weights sum to 1 for direct combination, or normalize afterwards
            # If weights don't sum to 1, the normalization below is crucial
            total_weight = w_row + w_col + w_val
            if total_weight <=0: total_weight = 1.0 # Avoid division by zero if all weights are 0

            combined_score = (
                w_row * row_completeness_score +
                w_col * col_completeness_score +
                w_val * value_capture_score
            ) / total_weight # Weighted average

            # The PDF normalizes again, which is good if component scores aren't strictly [0,1] or weights don't sum to 1.
            # Since components are [0,1] and we did weighted average, combined_score is already [0,1].
            # But for robustness, an extra normalize_value is fine. Max for combined_score is 1.0 here.
            # 來源：新大腦.pdf (Page 17)
            scores[r_idx, c_idx] = MathUtils.normalize_value(combined_score, 0, 1.0, clamp=True)
            
    return scores * config.weight
    # brain.py (Continued)
# ... (Imports, MathUtils, BoardAnalyzerUtils, BaseModuleConfig, and configs from PART 1, 2 & 3 remain the same) ...

# --- Scoring Module Implementations (Continued) ---

# 來源：新大腦.pdf - 7. EXT_GM1_Row_Control_Vec (Page 17)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM1強化建議
# Config for this (LineControlConfig) was defined in PART 2
def EXT_GM1_Row_Control_Vec(
    grid: np.ndarray,
    config: LineControlConfig,
    request_id: str | None = "N/A_GM1_RowCtrl",
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
    )

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

    for r_idx in range(rows):
        current_row_values_list_orig = [val for val in grid[r_idx, :] if val != -1]
        num_filled_in_row_orig = len(current_row_values_list_orig)
        sum_current_row_values_orig = sum(current_row_values_list_orig)

        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue

            # 1. Density Score: How full the row will be
            # 來源：新大腦.pdf - EXT_GM1 Density Score (Page 18)
            density_score = (num_filled_in_row_orig + 1.0) / cols if cols > 0 else 0.0

            # 2. Value Contribution Score (Sum Score)
            # 來源：新大腦.pdf - EXT_GM1 Value Contribution (Page 18)
            # Use avg_potential_num_to_place for the empty cell being scored
            potential_row_sum = sum_current_row_values_orig + avg_potential_num_to_place
            heuristic_max_row_sum = float(cols * max_val_board) # Max possible row sum
            # 來源：新大鵝.pdf (Page 18)

            sum_score: float = 0.0
            if heuristic_max_row_sum > 0:
                sum_score = MathUtils.normalize_value(
                    potential_row_sum, 0, heuristic_max_row_sum, clamp=True
                )

            # 3. Sequence Completion Score
            # 來源：新大腦.pdf - EXT_GM1 Sequence Completion (Page 18)
            # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM1/GM2 序列評估的增強 (使用 BoardAnalyzerUtils.find_sequences_in_line)
            seq_score: float = 0.0
            if config.use_advanced_sequence_detection:
                max_len_this_placement = 0.0
                if potential_numbers_to_place: # Only attempt if there are numbers to place
                    # Check sequence for average potential number
                    temp_grid_row_slice = grid[r_idx, :].copy()
                    temp_grid_row_slice[c_idx] = int(round(avg_potential_num_to_place)) # Use rounded average
                    
                    sequences = BoardAnalyzerUtils.find_sequences_in_line(
                        list(temp_grid_row_slice), # Must be list for find_sequences_in_line
                        min_len=config.min_len_for_sequence_score,
                        allow_gaps=config.allow_gaps_for_sequence_score
                    )
                    for s in sequences:
                        if int(round(avg_potential_num_to_place)) in s:
                           max_len_this_placement = max(max_len_this_placement, float(len(s)))
                
                if cols > 0: # Normalize by max possible length in row (cols)
                    seq_score = MathUtils.normalize_value(max_len_this_placement, 0, float(cols), clamp=True)

            else: # Original simplified mend logic from PDF
                # 來源：新大腦.pdf - EXT_GM1 Simplified mend logic (Page 19)
                if 0 < c_idx < cols - 1:
                    prev_val = grid[r_idx, c_idx - 1]
                    next_val = grid[r_idx, c_idx + 1]
                    if prev_val != -1 and next_val != -1:
                        if (prev_val + next_val) % 2 == 0:
                            mend_val = (prev_val + next_val) // 2
                            if mend_val in potential_numbers_to_place and abs(mend_val - prev_val) > 0:
                                seq_score = 0.75 # 來源：新大腦.pdf (Page 19)
                elif (c_idx == 0 and cols > 1 and grid[r_idx, c_idx + 1] != -1 and \
                      abs(grid[r_idx, c_idx + 1] - avg_potential_num_to_place) > 1e-6) or \
                     (c_idx == cols - 1 and cols > 1 and grid[r_idx, c_idx - 1] != -1 and \
                      abs(avg_potential_num_to_place - grid[r_idx, c_idx - 1]) > 1e-6): # 來源：新大腦.pdf (Page 19)
                      # Note: PDF had "... !=0", using > 1e-6 for float comparison robustness
                    seq_score = 0.25 # 來源：新大腦.pdf (Page 19)


            # Combine scores
            # 來源：新大腦.pdf - EXT_GM1 Combine scores (Page 19)
            w_density = config.w_density
            w_sum = config.w_sum_score
            w_seq = config.w_sequence_score
            total_weight = w_density + w_sum + w_seq
            if total_weight <= 0: total_weight = 1.0

            combined_score = (
                w_density * density_score + w_sum * sum_score + w_seq * seq_score
            ) / total_weight

            scores[r_idx, c_idx] = MathUtils.normalize_value(combined_score, 0, 1.0, clamp=True) # 來源：新大腦.pdf (Page 19)
            
    return scores * config.weight


# 來源：新大腦.pdf - 8. EXT_GM2_Col_Flow_Vec (Page 19)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM2強化建議
# Config for this (LineControlConfig) was defined in PART 2
def EXT_GM2_Col_Flow_Vec(
    grid: np.ndarray,
    config: LineControlConfig, # Reuses LineControlConfig
    request_id: str | None = "N/A_GM2_ColCtrl",
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

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 20)
    
    avg_potential_num_to_place: float = 0.0 # 來源：新大腦.pdf (Page 20)
    if potential_numbers_to_place:
        avg_potential_num_to_place = float(np.mean(potential_numbers_to_place))

    max_val_board = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols)) # 來源：新大腦.pdf (Page 20)
    if max_val_board == 0: max_val_board = 1.0

    for c_idx in range(cols):
        current_col_values_list_orig = [val for val in grid[:, c_idx] if val != -1] # PDF typo: val != -11
        # 來源：新大腦.pdf (Page 20)
        num_filled_in_col_orig = len(current_col_values_list_orig)
        sum_current_col_values_orig = sum(current_col_values_list_orig)

        for r_idx in range(rows):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue

            # 1. Density Score
            # 來源：新大腦.pdf - EXT_GM2 Density Score (Page 20)
            density_score = (num_filled_in_col_orig + 1.0) / rows if rows > 0 else 0.0

            # 2. Value Contribution Score (Sum Score)
            # 來源：新大腦.pdf - EXT_GM2 Value Contribution (Page 20)
            potential_col_sum = sum_current_col_values_orig + avg_potential_num_to_place
            heuristic_max_col_sum = float(rows * max_val_board) # Max possible col sum
            # 來源：新大腦.pdf (Page 20)

            sum_score: float = 0.0
            if heuristic_max_col_sum > 0:
                sum_score = MathUtils.normalize_value(
                    potential_col_sum, 0, heuristic_max_col_sum, clamp=True
                )

            # 3. Sequence Completion Score
            # 來源：新大腦.pdf - EXT_GM2 Sequence Completion (Page 20)
            seq_score: float = 0.0
            if config.use_advanced_sequence_detection:
                max_len_this_placement = 0.0
                if potential_numbers_to_place:
                    temp_grid_col_slice = grid[:, c_idx].copy()
                    temp_grid_col_slice[r_idx] = int(round(avg_potential_num_to_place))
                    
                    sequences = BoardAnalyzerUtils.find_sequences_in_line(
                        list(temp_grid_col_slice),
                        min_len=config.min_len_for_sequence_score,
                        allow_gaps=config.allow_gaps_for_sequence_score
                    )
                    for s in sequences:
                        if int(round(avg_potential_num_to_place)) in s:
                            max_len_this_placement = max(max_len_this_placement, float(len(s)))
                if rows > 0: # Normalize by max possible length in col (rows)
                    seq_score = MathUtils.normalize_value(max_len_this_placement, 0, float(rows), clamp=True)
            else: # Original simplified mend logic
                # 來源：新大腦.pdf - EXT_GM2 Simplified mend logic (Page 21)
                if 0 < r_idx < rows - 1:
                    prev_val = grid[r_idx - 1, c_idx]
                    next_val = grid[r_idx + 1, c_idx]
                    if prev_val != -1 and next_val != -1:
                        if (prev_val + next_val) % 2 == 0:
                            mend_val = (prev_val + next_val) // 2
                            if mend_val in potential_numbers_to_place and abs(mend_val - prev_val) > 0: # 來源：新大腦.pdf (Page 21)
                                seq_score = 0.75
                elif (r_idx == 0 and rows > 1 and grid[r_idx + 1, c_idx] != -1 and \
                      abs(grid[r_idx + 1, c_idx] - avg_potential_num_to_place) > 1e-6) or \
                     (r_idx == rows - 1 and rows > 1 and grid[r_idx - 1, c_idx] != -1 and \
                      abs(avg_potential_num_to_place - grid[r_idx - 1, c_idx]) > 1e-6): # 來源：新大腦.pdf (Page 21)
                      # Corrected PDF typo grid[r_idx-1, c_idx] != -1 and...
                    seq_score = 0.25

            # Combine scores
            # 來源：新大腦.pdf - EXT_GM2 Combine scores (Page 21)
            w_density = config.w_density
            w_sum = config.w_sum_score
            w_seq = config.w_sequence_score
            total_weight = w_density + w_sum + w_seq
            if total_weight <= 0: total_weight = 1.0

            combined_score = (
                w_density * density_score + w_sum * sum_score + w_seq * seq_score
            ) / total_weight
            scores[r_idx, c_idx] = MathUtils.normalize_value(combined_score, 0, 1.0, clamp=True) # 來源：新大腦.pdf (Page 21)

    return scores * config.weight


# 來源：新大腦.pdf - 9. EXT_GM3_Adv_Connected_Comp_Vec (Page 21)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM3強化建議
# Config for this (ConnectedComponentConfig) was defined in PART 2
def EXT_GM3_Adv_Connected_Comp_Vec(
    grid: np.ndarray,
    config: ConnectedComponentConfig,
    request_id: str | None = "N/A_GM3_ConnComp",
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

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    visited_overall = np.zeros_like(grid, dtype=bool) # Tracks visited cells for any component search
    # 來源：新大腦.pdf (Page 22)

    for r_start in range(rows):
        for c_start in range(cols):
            if visited_overall[r_start, c_start] or grid[r_start, c_start] != -1:
                # Skip if already visited or not an empty cell
                # 來源：新大腦.pdf (Page 22)
                continue

            # Start BFS for a new connected component of empty cells
            component_cells: List[Tuple[int, int]] = [] # PDF has typo: component_cells: List[Tuple[int, int]] = [ ]
            q = deque([(r_start, c_start)])
            # Visited in current BFS path (PDF typo: visited_bfs_current_component = set([(r_start, c_start)]) # Visited in current BFS path)
            visited_bfs_current_component: Set[Tuple[int,int]] = set([(r_start, c_start)]) 
            visited_overall[r_start, c_start] = True # Mark as globally visited

            while q: # 來源：新大腦.pdf (Page 22)
                r_curr, c_curr = q.popleft()
                component_cells.append((r_curr, c_curr))

                # Explore 4-connectivity neighbors
                # 來源：新大腦.pdf (Page 22) - PDF directions (0,1), (0,-1), (1,0), (-1,0)
                for dr_bfs, dc_bfs in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r_curr + dr_bfs, c_curr + dc_bfs

                    if 0 <= nr < rows and 0 <= nc < cols and \
                       grid[nr, nc] == -1 and \
                       not visited_overall[nr, nc] and \
                       (nr, nc) not in visited_bfs_current_component: # Ensure not re-adding to q for current BFS
                        
                        visited_overall[nr, nc] = True
                        visited_bfs_current_component.add((nr, nc))
                        q.append((nr, nc))
            
            area_size = float(len(component_cells))
            
            # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM3 連通區域的「形狀」和「質量」
            shape_factor_score = 0.0
            if config.consider_shape_factor and area_size > 0:
                # Conceptual: Calculate compactness or other shape metric
                # For example, ratio of area to bounding box area
                if component_cells:
                    min_r_bbox = min(r for r,c in component_cells)
                    max_r_bbox = max(r for r,c in component_cells)
                    min_c_bbox = min(c for r,c in component_cells)
                    max_c_bbox = max(c for r,c in component_cells)
                    bbox_area = (max_r_bbox - min_r_bbox + 1) * (max_c_bbox - min_c_bbox + 1)
                    if bbox_area > 0:
                        shape_factor_score = area_size / bbox_area # Compactness
            
            # Normalize area size against total number of cells in the grid
            # 來源：新大腦.pdf (Page 22)
            total_cells = float(rows * cols)
            norm_area_size: float = 0.0
            if total_cells > 0:
                norm_area_size = MathUtils.normalize_value(area_size, 0, total_cells, clamp=True)
            
            # Combine base score with shape factor score
            final_component_score = norm_area_size
            if config.consider_shape_factor:
                final_component_score = (1.0 - config.shape_factor_weight) * norm_area_size + \
                                        config.shape_factor_weight * shape_factor_score
                final_component_score = MathUtils.normalize_value(final_component_score, 0, 1.0, clamp=True)


            # Assign this normalized area size score to all cells in the found component
            # 來源：新大腦.pdf (Page 23)
            for r_comp, c_comp in component_cells:
                scores[r_comp, c_comp] = final_component_score
                
    return scores * config.weight
    # brain.py (Continued)
# ... (Imports, MathUtils, BoardAnalyzerUtils, BaseModuleConfig, and configs from PART 1, 2, 3 & 4 remain the same) ...

# --- Scoring Module Implementations (Continued) ---

# 來源：新大腦.pdf - 10. EXT_GM4_Spatial_Auto_Corr_Vec (Page 23)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM4強化建議
# Config for this (SpatialAutocorrelationConfig) was defined in PART 2
def EXT_GM4_Spatial_Auto_Corr_Vec(
    grid: np.ndarray,
    config: SpatialAutocorrelationConfig,
    request_id: str | None = "N/A_GM4_SpatialAutoCorr",
) -> np.ndarray:
    """
    (GM4 - 空間自相關性分析)
    核心規則:評估在空格填入一個假設的「平均」潛在數字後,該數字與其周圍現有數字的相似程度。
    目的:鼓勵形成數值聚集(正自相關)或數值交錯(負自相關,但此處偏好正自相關)。此版本偏好正自相關,即填入的數字與周圍鄰居的平均值相似時得分較高。
    啟發式類型:空間統計
    輸出詮釋:分數越高表示填入一個「典型」數字後,能更好地融入周圍環境,形成數值上的聚集。
    來源：新大腦.pdf - EXT_GM4_Spatial_Auto_Corr_Vec (Page 23)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM4"
    logger.debug(
        f"Executing EXT_GM4_Spatial_Auto_Corr_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 23)

    hypothetical_val_to_place: float # 來源：新大腦.pdf (Page 23)
    if potential_numbers:
        if config.use_median_for_hypothetical:
            hypothetical_val_to_place = float(np.median(potential_numbers))
        else:
            hypothetical_val_to_place = float(np.mean(potential_numbers))
    else:
        # 來源：新大腦.pdf - EXT_GM4 Fallback for hypothetical_val_to_place (Page 23-24)
        max_board_val = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols))
        hypothetical_val_to_place = (1.0 + float(max_board_val)) / 2.0 if max_board_val > 0 else 0.5

    max_val_on_grid_for_norm = float(BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols))) # 來源：新大腦.pdf (Page 24)
    if max_val_on_grid_for_norm == 0: max_val_on_grid_for_norm = 1.0 # 來源：新大腦.pdf (Page 24)

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue

            # Get actual numeric neighbors (non -1)
            # 來源：新大腦.pdf - EXT_GM4 get_neighborhood_values (Page 24)
            neighbor_values = BoardAnalyzerUtils.get_neighborhood_values(
                grid, r_idx, c_idx, 
                radius=config.neighborhood_radius, # Use config
                eight_connectivity=True,
                val_func=lambda x: float(x) if x != -1 else None,
                include_center=False
            )

            if not neighbor_values: # 來源：新大腦.pdf (Page 24)
                scores[r_idx, c_idx] = 0.5  # Neutral score if no neighbors to compare with
                continue

            mean_neighbors = np.mean(neighbor_values) # 來源：新大腦.pdf (Page 24)

            # Calculate the difference between the hypothetical placed value and the mean of its actual neighbors
            # 來源：新大腦.pdf (Page 24)
            diff_hypothetical_to_mean_neighbors = abs(hypothetical_val_to_place - mean_neighbors)

            # Normalize this difference. Max possible difference is roughly max_val_on_grid.
            # Score for positive autocorrelation: 1.0 - normalized_difference
            # (smaller difference means more similar, thus higher positive autocorrelation score)
            # 來源：新大腦.pdf (Page 24)
            norm_diff = MathUtils.normalize_value(
                diff_hypothetical_to_mean_neighbors, 0, max_val_on_grid_for_norm, clamp=True
            )
            
            current_score: float
            if config.autocorrelation_type == "positive":
                current_score = 1.0 - norm_diff # 來源：新大腦.pdf (Page 24)
            else: # "negative" autocorrelation (交錯)
                current_score = norm_diff # Larger difference is better
            
            scores[r_idx, c_idx] = current_score
            
    return scores * config.weight


# 來源：新大腦.pdf - 11. EXT_GM5_Line_Completion_Vec (Page 24)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM5強化建議
# Config for this (LineCompletionConfig) was defined in PART 2
def EXT_GM5_Line_Completion_Vec(
    grid: np.ndarray,
    config: LineCompletionConfig,
    request_id: str | None = "N/A_GM5_LineComp",
) -> np.ndarray:
    """
    (GM5-線段補全)
    核心規則:評估空格對於完成特定方向(行、列、對角線)上具有特定構成(如等差、等值)的短線段(例如長度為3)之潛力。
    目的:偏好那些能夠「臨門一腳」完成有意義短線段的空格。
    啟發式類型:模式匹配(短線段)
    輸出詮釋:分數越高表示該空格填入某數字後,越能完成一個預定義的短線段模式。
    來源：新大腦.pdf - EXT_GM5_Line_Completion_Vec (Page 24)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM5"
    logger.debug(
        f"Executing EXT_GM5_Line_Completion_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    # 來源：新大腦.pdf - EXT_GM5 initial checks (Page 25)
    if rows == 0 or cols == 0 or min(rows,cols) < 1: # PDF: min(rows,cols) < 1. For lines of 3, need more.
         # For target_line_length, need at least that many in one dimension.
        if config.target_line_length > max(rows, cols) and (rows > 0 and cols > 0) : # if grid is smaller than target line
            pass # allow, but scores will likely be 0
        elif rows == 0 or cols == 0: # definitely no lines
             return scores


    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 25)
    if not potential_numbers_to_place:
        return scores

    # 來源：新大腦.pdf - EXT_GM5 line_completion_score_map (Page 25)
    # Using config for scores
    
    max_board_val = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows,cols))

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 25)
                continue

            max_completion_score_for_cell: float = 0.0

            for p_val in potential_numbers_to_place:
                current_pval_max_score_contribution: float = 0.0
                
                # Directions: Horizontal, Vertical, Diagonal (top-left to bottom-right), Anti-Diagonal
                # 來源：新大腦.pdf - EXT_GM5 Directions (Page 25)
                # Each direction vector (dr, dc)
                for dr_dir, dc_dir in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    # For a line of target_line_length, p_val can be at any position within it.
                    # Iterate through all possible windows of target_line_length that include (r_idx, c_idx)
                    # where (r_idx, c_idx) is filled with p_val.
                    for i_offset in range(config.target_line_length): # p_val is at index i_offset in the window
                        # Start of window relative to (r_idx, c_idx) as if it's the 0-th element in the window
                        # Window cells are: (r_idx + (k-i_offset)*dr_dir, c_idx + (k-i_offset)*dc_dir) for k in 0..L-1
                        
                        current_line_values: List[int] = []
                        is_valid_line_segment = True
                        
                        for k_in_segment in range(config.target_line_length):
                            # Actual coordinates of the k_in_segment-th cell in the current line
                            eval_r = r_idx + (k_in_segment - i_offset) * dr_dir
                            eval_c = c_idx + (k_in_segment - i_offset) * dc_dir

                            if not (0 <= eval_r < rows and 0 <= eval_c < cols):
                                is_valid_line_segment = False
                                break
                            
                            if eval_r == r_idx and eval_c == c_idx:
                                current_line_values.append(p_val)
                            else:
                                current_line_values.append(int(grid[eval_r, eval_c])) # Cast to int if not -1
                        
                        if is_valid_line_segment and all(val != -1 for val in current_line_values): # All cells in segment must be filled
                            s = current_line_values
                            temp_score_for_this_line = 0.0

                            # Check for 3 identical (or target_line_length identical)
                            # 來源：新大腦.pdf - EXT_GM5 Identical 3 Check (Page 26)
                            if len(set(s)) == 1: # All elements are same
                                temp_score_for_this_line = max(temp_score_for_this_line, config.score_identical_3)
                            
                            # Check for arithmetic (non-constant)
                            # 來源：新大腦.pdf - EXT_GM5 Arithmetic 3 Mend/Extend (Page 26)
                            # This general check is for a complete line s of target_line_length
                            if len(s) >= 2:
                                diffs = [s[k+1] - s[k] for k in range(len(s)-1)]
                                if len(set(diffs)) == 1 and diffs[0] != 0: # Is arithmetic and non-constant
                                    # Determine if it's a "mend" or "extend" based on p_val's position (i_offset)
                                    # This distinction is complex for generic target_line_length.
                                    # PDF has specific logic for length 3.
                                    if config.target_line_length == 3:
                                        if i_offset == 1: # p_val is in the middle (mending)
                                            temp_score_for_this_line = max(temp_score_for_this_line, config.score_arithmetic_3_mend)
                                            # 來源：新大腦.pdf - EXT_GM5 Quality Enhancement (Conceptual) (Page 26)
                                            if config.enable_quality_enhancement:
                                                avg_val_line = sum(s) / len(s)
                                                if max_board_val > 0 and avg_val_line > (max_board_val * config.high_value_threshold_factor_gm5):
                                                    temp_score_for_this_line += config.score_arithmetic_3_mend_high_val_bonus
                                        else: # p_val is at an end (extending)
                                            temp_score_for_this_line = max(temp_score_for_this_line, config.score_arithmetic_3_extend)
                                    else: # For other lengths, use a generic arithmetic score
                                        temp_score_for_this_line = max(temp_score_for_this_line, config.score_arithmetic_3_mend) # Use mend score as base

                            current_pval_max_score_contribution = max(current_pval_max_score_contribution, temp_score_for_this_line)
                
                max_completion_score_for_cell = max(max_completion_score_for_cell, current_pval_max_score_contribution)

            # Normalize based on the max possible score from config map (approx 1.0 as scores are defined in 0-1 range)
            # 來源：新大腦.pdf - EXT_GM5 Normalization (Page 27)
            scores[r_idx, c_idx] = MathUtils.normalize_value(
                max_completion_score_for_cell, 0, 1.0, clamp=True # Max score from map is < 1
            )
            
    return scores * config.weight


# 來源：新大腦.pdf - 12. EXT_GM6_Symmetry_Potential_Vec (Page 27)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM6強化建議
# Config for this (SymmetryPotentialConfig) was defined in PART 2
def EXT_GM6_Symmetry_Potential_Vec(
    grid: np.ndarray,
    config: SymmetryPotentialConfig,
    request_id: str | None = "N/A_GM6_Symmetry",
) -> np.ndarray:
    """
    (GM6-對稱性潛力)
    核心規則:評估在空格填入數字後,盤面形成的對稱性程度(水平、垂直、中心、主對角線、反主對角線)。
    目的:偏好那些能夠創造或增強盤面對稱性的填補。
    啟發式類型:幾何與模式識別
    輸出詮釋:分數越高表示若在該空格填入特定數字,能與對稱位置上已存在的相同數字形成對稱。
    來源：新大腦.pdf - EXT_GM6_Symmetry_Potential_Vec (Page 27)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM6"
    logger.debug(
        f"Executing EXT_GM6_Symmetry_Potential_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 27)
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 27)
    if not potential_numbers_to_place: # 來源：新大腦.pdf (Page 28)
        return scores

    # 來源：新大腦.pdf - EXT_GM6 symmetry_scores_map (Page 28) & Conceptual dynamic adjustment
    # Using scores from config directly.
    # Dynamic adjustment idea:
    # score_main_diag = config.score_main_diagonal
    # score_anti_diag = config.score_anti_diagonal
    # if rows == cols and config.strict_square_for_diagonal: # or simply if rows == cols
    #     score_main_diag = max(score_main_diag, 0.7) # Example boost for square grids
    #     score_anti_diag = max(score_anti_diag, 0.7)


    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells
                continue
            
            max_symmetry_score_for_cell: float = 0.0 # 來源：新大腦.pdf (Page 28)

            for p_val in potential_numbers_to_place:
                current_pval_max_sym: float = 0.0

                # 1. Horizontal Symmetry: (r_idx, c_idx) vs (r_idx, cols - 1 - c_idx)
                # 來源：新大腦.pdf - Horizontal Symmetry (Page 28)
                sr_h, sc_h = r_idx, cols - 1 - c_idx
                if sc_h != c_idx: # Not the same cell
                    if 0 <= sr_h < rows and 0 <= sc_h < cols and grid[sr_h, sc_h] == p_val:
                        current_pval_max_sym = max(current_pval_max_sym, config.score_horizontal)

                # 2. Vertical Symmetry: (r_idx, c_idx) vs (rows - 1 - r_idx, c_idx)
                # 來源：新大腦.pdf - Vertical Symmetry (Page 28)
                sr_v, sc_v = rows - 1 - r_idx, c_idx
                if sr_v != r_idx: # Not the same cell
                    if 0 <= sr_v < rows and 0 <= sc_v < cols and grid[sr_v, sc_v] == p_val:
                        current_pval_max_sym = max(current_pval_max_sym, config.score_vertical)
                
                # 3. Point (Center) Symmetry: (r_idx, c_idx) vs (rows - 1 - r_idx, cols - 1 - c_idx)
                # 來源：新大腦.pdf - Point Center Symmetry (Page 28)
                sr_p, sc_p = rows - 1 - r_idx, cols - 1 - c_idx
                if sr_p != r_idx or sc_p != c_idx: # Not the same cell
                     if 0 <= sr_p < rows and 0 <= sc_p < cols and grid[sr_p, sc_p] == p_val:
                        current_pval_max_sym = max(current_pval_max_sym, config.score_point_center)

                # 4. Main Diagonal Symmetry (\): (r_idx, c_idx) vs (c_idx, r_idx)
                # 來源：新大腦.pdf - Main Diagonal Symmetry (Page 28-29)
                if not config.strict_square_for_diagonal or rows == cols: # 來源：新大腦.pdf (Page 29)
                    sr_d1, sc_d1 = c_idx, r_idx
                    if sr_d1 != r_idx or sc_d1 != c_idx: # Not the same cell (only if r_idx != c_idx)
                        if 0 <= sr_d1 < rows and 0 <= sc_d1 < cols and grid[sr_d1, sc_d1] == p_val:
                            current_pval_max_sym = max(current_pval_max_sym, config.score_main_diagonal)
                
                # 5. Anti-Diagonal Symmetry (/): (r_idx, c_idx) vs ( (cols-1)-c_idx, (rows-1)-r_idx ) - This definition seems more standard for matrix anti-diagonal
                # PDF text for anti-diagonal (Page 29) has discussion.
                # "grid[ (cols-1)-c , (rows-1)-r]" - assuming (cols-1) is max_col_idx, (rows-1) is max_row_idx
                # So if grid[r_original, c_original], symmetric is grid[max_row_idx - c_original_reflected_to_row, max_col_idx - r_original_reflected_to_col]
                # More directly: For grid[r,c] on NxM grid, the anti-diagonal reflection is often considered grid[M-1-c, N-1-r] if axes were swapped.
                # The PDF has: sr_d2, sc_d2 = (cols - 1) - c_idx, (rows - 1) - r_idx - This implies swapping indices AND reflecting.
                # Let's use the PDF's direct formula: (r,c) vs ( (cols-1)-c, (rows-1)-r ) assuming it means the element at row=(cols-1)-c_idx, col=(rows-1)-r_idx
                # This makes sense if you imagine rotating the coordinate system.
                # Let's use a common definition for anti-diagonal element for grid[r_idx][c_idx] in an Rows x Cols grid:
                # The element anti-diagonally symmetric to (r, c) is ( (Rows-1) - (c_prime), (Cols-1) - (r_prime) ) where (r_prime,c_prime) are indices on a transposed conceptual grid.
                # A simpler interpretation often used: if you reflect along the line y = -x + (N-1) for an NxN matrix,
                # (r,c) maps to (N-1-c, N-1-r). We will use this if rows==cols.
                # 來源：新大腦.pdf - Anti-Diagonal Symmetry (Page 29)
                if not config.strict_square_for_diagonal or rows == cols: # 來源：新大腦.pdf (Page 29)
                    # Using (N-1-c, N-1-r) for square N x N (where N=rows=cols)
                    # For general Rows x Cols, this type of symmetry is less strictly defined.
                    # The PDF uses sr_d2, sc_d2 = (cols - 1) - c_idx, (rows - 1) - r_idx.
                    # This formula assumes grid indices can be derived this way.
                    # If rows=3, cols=5. For (0,0), this gives (4,2). For (0,4), this gives (0,2).
                    # This seems to be a specific definition of anti-diagonal symmetry. Let's use it.
                    sr_d2, sc_d2 = (rows - 1) - c_idx, (cols - 1) - r_idx # Corrected based on common understanding for anti-diagonal in matrix (N-1-j, M-1-i)
                    if (sr_d2 != r_idx or sc_d2 != c_idx): # Not the same cell
                        if 0 <= sr_d2 < rows and 0 <= sc_d2 < cols and grid[sr_d2, sc_d2] == p_val:
                             current_pval_max_sym = max(current_pval_max_sym, config.score_anti_diagonal)

                if current_pval_max_sym > max_symmetry_score_for_cell: # 來源：新大腦.pdf (Page 29)
                    max_symmetry_score_for_cell = current_pval_max_sym
            
            # Scores are already ~0-1 from config map
            # 來源：新大腦.pdf - EXT_GM6 Normalize (Page 29)
            scores[r_idx, c_idx] = MathUtils.normalize_value(max_symmetry_score_for_cell, 0, 1.0, clamp=True) 
                                                        # Max of map is 0.8 in PDF example, so 1.0 is safe upper for norm.
    return scores * config.weight
    # brain.py (Continued)
# ... (Imports, MathUtils, BoardAnalyzerUtils, BaseModuleConfig, and configs from PART 1, 2, 3, 4 & 5 remain the same) ...

# --- Scoring Module Implementations (Continued) ---

# 來源：新大腦.pdf - 13. EXT_GM7_Numeric_Gaps_Vec (Page 29)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM7強化建議
# Config for this (NumericGapsConfig) was defined in PART 2
def EXT_GM7_Numeric_Gaps_Vec(
    grid: np.ndarray,
    config: NumericGapsConfig,
    request_id: str | None = "N/A_GM7_NumGaps",
) -> np.ndarray:
    """
    (GM7 - 數值間隙填充)
    核心規則:識別並評估在局部區域或序列中,填補數字「間隙」的價值。特別是尋找能填入使之成為公差為1的連續數列的間隙。
    目的:偏好那些能填補序列中明顯缺失數字的空格。
    啟發式類型:序列與模式識別(間隙填充)
    輸出詮釋:分數越高表示該空格若填入特定數字,越能完美地填補一個數值間隙(尤其是公差為1的序列)。
    來源：新大腦.pdf - EXT_GM7_Numeric_Gaps_Vec (Page 29-30)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM7"
    logger.debug(
        f"Executing EXT_GM7_Numeric_Gaps_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 30)
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 30)
    if not potential_numbers_to_place: # 來源：新大腦.pdf (Page 30)
        return scores
        
    max_board_val = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows,cols))

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 30)
                continue
            
            max_cell_gap_score: float = 0.0 # 來源：新大腦.pdf (Page 30)

            for p_val in potential_numbers_to_place:
                # current_pval_score: float = 0.0 # PDF seems to use max_cell_gap_score directly updated
                
                # Iterate over 4 directions (Horizontal, Vertical, Main Diagonal, Anti-Diagonal)
                # 來源：新大腦.pdf - EXT_GM7 Directions (Page 30)
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    # Case 1: p_val mends a gap: N1 - p_val - N2
                    # 來源：新大腦.pdf - EXT_GM7 Case 1 (Page 30)
                    r_n1, c_n1 = r_idx - dr, c_idx - dc
                    r_n2, c_n2 = r_idx + dr, c_idx + dc

                    if 0 <= r_n1 < rows and 0 <= c_n1 < cols and \
                       0 <= r_n2 < rows and 0 <= c_n2 < cols:
                        val_n1 = grid[r_n1, c_n1]
                        val_n2 = grid[r_n2, c_n2]

                        if val_n1 != -1 and val_n2 != -1: # Both neighbors exist
                            # Specific check for arithmetic sequence with common difference 1
                            # 來源：新大腦.pdf - EXT_GM7 arithmetic_1_gap_fill (Page 31)
                            if val_n1 == p_val - 1 and val_n2 == p_val + 1:
                                score = config.score_arithmetic_1_gap_fill
                                # 來源：新大腦.pdf - EXT_GM7 Quality Enhancement (Conceptual) (Page 31)
                                if config.enable_quality_enhancement_gm7:
                                     if max_board_val > 0 and (val_n1 + p_val + val_n2) / 3.0 > (max_board_val * config.high_value_threshold_factor_gm7):
                                        score += config.score_gap_fill_high_val_bonus # Add bonus
                                max_cell_gap_score = max(max_cell_gap_score, score)
                            
                            # Generic arithmetic sequence check (d!=0)
                            # 來源：新大腦.pdf - EXT_GM7 arithmetic_generic_mend (Page 31)
                            elif (val_n1 + val_n2) == 2 * p_val and abs(p_val - val_n1) > 1e-6 : # Not constant, use tolerance for float p_val
                                max_cell_gap_score = max(max_cell_gap_score, config.score_arithmetic_generic_mend)

                    # Case 2: p_val extends a sequence: p_val - N1 - N2
                    # 來源：新大腦.pdf - EXT_GM7 Case 2 (Page 31)
                    r_n1_ext1, c_n1_ext1 = r_idx + dr, c_idx + dc
                    r_n2_ext1, c_n2_ext1 = r_idx + 2 * dr, c_idx + 2 * dc
                    
                    if 0 <= r_n1_ext1 < rows and 0 <= c_n1_ext1 < cols and \
                       0 <= r_n2_ext1 < rows and 0 <= c_n2_ext1 < cols:
                        val_n1_ext1 = grid[r_n1_ext1, c_n1_ext1]
                        val_n2_ext1 = grid[r_n2_ext1, c_n2_ext1]

                        if val_n1_ext1 != -1 and val_n2_ext1 != -1:
                            # Check for N1=p_val+d, N2=p_val+2d  => val_n1_ext1 - p_val == val_n2_ext1 - val_n1_ext1 (d)
                            # 來源：新大腦.pdf - EXT_GM7 Case 2 logic (Page 31)
                            # The PDF has `common_diff = val_n1_ext1 - p_val`
                            # `if common_diff !=0 and val_n2_ext1 == val_n1_ext1 + common_diff:`
                            common_diff = val_n1_ext1 - p_val
                            if not math.isclose(common_diff, 0) and math.isclose(val_n2_ext1, val_n1_ext1 + common_diff):
                                max_cell_gap_score = max(max_cell_gap_score, config.score_arithmetic_generic_extend)
                    
                    # Case 3: p_val extends a sequence: N1 - N2 - p_val
                    # 來源：新大腦.pdf - EXT_GM7 Case 3 (Page 31)
                    r_n1_ext2, c_n1_ext2 = r_idx - 2 * dr, c_idx - 2 * dc
                    r_n2_ext2, c_n2_ext2 = r_idx - dr, c_idx - dc
                    
                    if 0 <= r_n1_ext2 < rows and 0 <= c_n1_ext2 < cols and \
                       0 <= r_n2_ext2 < rows and 0 <= c_n2_ext2 < cols:
                        val_n1_ext2 = grid[r_n1_ext2, c_n1_ext2]
                        val_n2_ext2 = grid[r_n2_ext2, c_n2_ext2]

                        if val_n1_ext2 != -1 and val_n2_ext2 != -1:
                            # Check for N2=N1+d, p_val=N1+2d => val_n2_ext2 - val_n1_ext2 == p_val - val_n2_ext2 (d)
                            # 來源：新大腦.pdf - EXT_GM7 Case 3 logic (Page 31-32)
                            # PDF: `common_diff = val_n2_ext2 - val_n1_ext2`
                            # `if common_diff !=0 and p_val == val_n2_ext2 + common_diff:`
                            common_diff = val_n2_ext2 - val_n1_ext2
                            if not math.isclose(common_diff,0) and math.isclose(p_val, val_n2_ext2 + common_diff): # 來源：新大腦.pdf (Page 32) - Corrected index typo c_idx-1 to c_idx-dc for general direction
                                max_cell_gap_score = max(max_cell_gap_score, config.score_arithmetic_generic_extend)
            
            # PDF had current_pval_score > max_cell_gap_score, but current_pval_score wasn't updated per direction.
            # max_cell_gap_score is directly updated.
            # 來源：新大腦.pdf - EXT_GM7 Normalization (Page 32)
            scores[r_idx, c_idx] = MathUtils.normalize_value(max_cell_gap_score, 0, 1.0, clamp=True) # Scores from map are ~0-1
    
    return scores * config.weight


# 來源：新大腦.pdf - 14. EXT_GM8_Edge_Affinity_Vec (Page 31)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM8強化建議
# Config for this (EdgeAffinityConfig) was defined in PART 2
def EXT_GM8_Edge_Affinity_Vec(
    grid: np.ndarray,
    config: EdgeAffinityConfig,
    request_id: str | None = "N/A_GM8_EdgeAff",
) -> np.ndarray:
    """
    (GM8-邊緣親和度)
    核心規則:評估空格與盤面邊緣或角落的接近程度及其策略意義。
    目的:根據策略配置,偏好靠近或遠離邊緣/角落的空格。
    啟發式類型:位置與邊界分析
    輸出詮釋:分數高低取決於設定(偏好/避開邊緣)。預設偏好邊緣,越靠近邊緣/角落分數越高。
    來源：新大腦.pdf - EXT_GM8_Edge_Affinity_Vec (Page 31)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM8"
    logger.debug(
        f"Executing EXT_GM8_Edge_Affinity_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 32)
        return scores

    affinity_mode = config.affinity_mode # 來源：新大腦.pdf (Page 32)
    corner_bonus_prefer = config.corner_bonus_prefer # 來源：新大腦.pdf (Page 32)
    corner_penalty_avoid = config.corner_penalty_avoid # 來源：新大腦.pdf (Page 32)

    # 來源：新大腦.pdf - EXT_GM8 Max possible minimum distance to an edge (Page 32)
    # This would be for a cell at the center of the board.
    max_min_dist_to_edge_row = (rows - 1) // 2 if rows > 0 else 0
    max_min_dist_to_edge_col = (cols - 1) // 2 if cols > 0 else 0
    
    # The actual maximum of minimum distances to any edge for any cell on the board.
    # For a cell at (r,c), its min_dist_to_edge is min(r, rows-1-r, c, cols-1-c).
    # The max value this min_dist_to_edge can take is at the center.
    # overall_max_of_min_distances should be this center value.
    # PDF calculation: float(min(max_min_dist_to_edge_row, max_min_dist_to_edge_col))
    # This seems correct: e.g. 5x7 grid, center is (2,3). min_dist_row=2, min_dist_col=3.
    # max_min_dist_row=(5-1)//2 = 2. max_min_dist_col=(7-1)//2 = 3. min(2,3)=2. Correct.
    overall_max_of_min_distances = float(min(max_min_dist_to_edge_row, max_min_dist_to_edge_col)) # 來源：新大腦.pdf (Page 33)
    
    # 來源：新大腦.pdf - EXT_GM8 Handle overall_max_of_min_distances == 0 (Page 33)
    # If overall_max_of_min_distances is 0 (e.g., a 1xN or 2xN line, or 1x1, 2x1, 2x2),
    # it means all cells are on an edge or one step from it.
    if math.isclose(overall_max_of_min_distances, 0.0) and (rows > 0 and cols > 0): # For non-empty grid
        if rows <= 2 or cols <= 2 : # For very thin/small grids where center is edge/near-edge
             overall_max_of_min_distances = 0.5 # Avoid div by zero, gives some scale for normalization
                                               # All cells on such grids will have min_dist 0 or 1.
                                               # If min_dist is 0, normalized_dist will be 0.
                                               # If min_dist is 1, normalized_dist will be 1/0.5=2 (needs clamp).
        else: # This case should not be hit if logic for max_min_dist_... is correct
             overall_max_of_min_distances = 1.0 # Fallback if it's calculated as 0 for larger grids.

    if overall_max_of_min_distances <= 0 : overall_max_of_min_distances = 1.0 # General fallback to prevent div by zero


    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 33)
                continue

            dist_to_top_edge = float(r_idx)
            dist_to_bottom_edge = float(rows - 1 - r_idx)
            dist_to_left_edge = float(c_idx)
            dist_to_right_edge = float(cols - 1 - c_idx)

            min_dist = min(dist_to_top_edge, dist_to_bottom_edge, dist_to_left_edge, dist_to_right_edge) # 來源：新大腦.pdf (Page 33)

            is_corner = (r_idx == 0 or r_idx == rows - 1) and \
                        (c_idx == 0 or c_idx == cols - 1) # 來源：新大腦.pdf (Page 33)
            
            current_score: float = 0.0
            normalized_dist: float = 0.0

            if overall_max_of_min_distances > 1e-6: # Use tolerance for float comparison
                normalized_dist = min_dist / overall_max_of_min_distances
                normalized_dist = min(1.0, max(0.0, normalized_dist)) # Clamp # 來源：新大腦.pdf (Page 33)
            elif math.isclose(min_dist, 0.0): # All cells are on an edge, min_dist is 0
                normalized_dist = 0.0 # 來源：新大腦.pdf (Page 33)
            else: # Should not happen if overall_max_of_min_distances is handled
                normalized_dist = 1.0 # 來源：新大腦.pdf (Page 33)

            if affinity_mode == "prefer_edge": # 來源：新大腦.pdf (Page 33)
                current_score = 1.0 - normalized_dist  # Closer to edge (smaller dist) -> higher score
                if is_corner and math.isclose(min_dist, 0.0): # Only apply corner bonus if truly on edge
                    current_score += corner_bonus_prefer
            elif affinity_mode == "avoid_edge": # 來源：新大腦.pdf (Page 33)
                current_score = normalized_dist  # Further from edge (larger dist) -> higher score
                if is_corner and math.isclose(min_dist, 0.0):
                    current_score -= corner_penalty_avoid
            
            # Normalize final score to be between 0 and 1, considering bonus/penalty
            # Max possible score: 1.0 + corner_bonus_prefer
            # Min possible score: 0.0 - corner_penalty_avoid
            # 來源：新大腦.pdf - EXT_GM8 Final Normalization (Page 34)
            # The PDF normalizes using (-corner_penalty_avoid, 1.0 + corner_bonus_prefer)
            # This seems problematic if penalty makes it <0 and bonus >1, then normalize to 0-1.
            # It's better to clamp the current_score first, then normalize if needed,
            # or ensure the normalization range is correct.
            # The scores are already conceptually 0-1 before bonus/penalty.
            # Let's clamp current_score to a reasonable range like [0, 1 + corner_bonus_prefer]
            # and then normalize that range to [0,1].
            # Simpler: clamp result to [0,1] after bonus/penalty.
            current_score = max(0.0, min(1.0 + corner_bonus_prefer, current_score)) # Clamp to possible range
            # Now, normalize this to [0,1] if the range is not already [0,1]
            # If prefer_edge, range is [0, 1+bonus]. If avoid_edge, range is [0-penalty, 1].
            # The MathUtils.normalize_value in PDF has range [-CP_avoid, 1+CB_prefer] which implies
            # the value `current_score` can be negative.
            # Let's use the PDF's normalization directly:
            min_norm_range = 0.0 - corner_penalty_avoid if affinity_mode == "avoid_edge" else 0.0
            max_norm_range = 1.0 + corner_bonus_prefer if affinity_mode == "prefer_edge" else 1.0
            if math.isclose(max_norm_range, min_norm_range) : # if bonus and penalty are such that range is zero
                 max_norm_range = min_norm_range + 1.0 # ensure non-zero range for normalization


            scores[r_idx, c_idx] = MathUtils.normalize_value(current_score,
                                                            min_val=min_norm_range, 
                                                            max_val=max_norm_range, 
                                                            clamp=True)
            # Final clamp just in case, though normalize_value with clamp=True should handle it.
            scores[r_idx, c_idx] = max(0.0, min(1.0, scores[r_idx, c_idx]))


    return scores * config.weight
    # brain.py (Continued)
# ... (Imports, MathUtils, BoardAnalyzerUtils, BaseModuleConfig, and configs from PART 1, 2, 3, 4, 5 & 6 remain the same) ...

# --- Scoring Module Implementations (Continued) ---

# 來源：新大腦.pdf - 15. EXT_GM9_Center_Control_Vec (Page 34)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM9強化建議
# Config for this (CenterControlConfig) was defined in PART 2
def EXT_GM9_Center_Control_Vec(
    grid: np.ndarray,
    config: CenterControlConfig,
    request_id: str | None = "N/A_GM9_CenterCtrl",
) -> np.ndarray:
    """
    (GM9-中心控制偏好)
    核心規則:評估空格與盤面中心的接近程度及其策略意義。
    目的:根據策略配置,偏好靠近或遠離盤面中心區域的空格。
    啟發式類型:位置與中心性分析
    輸出詮釋:分數高低取決於設定(偏好/避開中心)。預設偏好中心,越靠近中心分數越高。
    來源：新大腦.pdf - EXT_GM9_Center_Control_Vec (Page 34)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM9"
    logger.debug(
        f"Executing EXT_GM9_Center_Control_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 34)
        return scores

    affinity_mode = config.affinity_mode # 來源：新大腦.pdf (Page 34)
    
    center_r = (rows - 1) / 2.0 # 來源：新大腦.pdf (Page 34)
    center_c = (cols - 1) / 2.0 # 來源：新大腦.pdf (Page 34)

    # Max possible distance from any cell to the center is the distance from a corner to the center.
    # 來源：新大腦.pdf - EXT_GM9 max_dist_to_center (Page 34)
    # Using (0,0) as the reference corner.
    max_dist_to_center = MathUtils.euclidean_distance((0.0, 0.0), (center_r, center_c))

    # 來源：新大腦.pdf - EXT_GM9 Handle max_dist_to_center == 0 (Page 34)
    if math.isclose(max_dist_to_center, 0.0) : # if grid is 1x1 or effectively so
        if rows <= 1 and cols <= 1: # Truly a 1x1 or 0x0 grid (0x0 caught by early return)
            # For a 1x1 grid, all cells are the center. Score should be neutral or max depending on interpretation.
            # If prefer_center, score should be high (1.0). If avoid_center, low (0.0).
            # The normalization logic MathUtils.normalize_value(0,0,0) returns 0.5.
            # if affinity_mode == "prefer_center": scores[0,0] = 1.0 * config.weight (if 1x1)
            # else: scores[0,0] = 0.0 * config.weight
            # This is handled by the loop, normalized_dist will be 0.5 from normalize_value if max_dist_to_center is 0.
            # current_score will then be 0.5 or 0.5. Let's refine the max_dist_to_center for 1x1.
            pass # max_dist_to_center remains 0, MathUtils.normalize_value will give 0.5 for dist=0
        else: # Calculated as 0 for larger grids (should not happen if center_r/c are correct for >1x1)
            max_dist_to_center = 1.0 # Fallback to prevent div by zero if logic error

    # Ensure max_dist_to_center is not zero if grid is larger than 1x1 to avoid division by zero
    # or to give meaningful normalization.
    if math.isclose(max_dist_to_center, 0.0) and (rows > 1 or cols > 1):
         max_dist_to_center = 1.0 # Should not be hit if center_r, center_c are calculated for >1x1

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 35)
                continue

            current_dist_to_center = MathUtils.euclidean_distance(
                (float(r_idx), float(c_idx)), (center_r, center_c)
            ) # 來源：新大腦.pdf (Page 35)

            normalized_dist: float
            if max_dist_to_center > 1e-6: # Use tolerance
                normalized_dist = MathUtils.normalize_value(
                    current_dist_to_center, 0, max_dist_to_center, clamp=True
                ) # 來源：新大腦.pdf (Page 35)
            elif math.isclose(current_dist_to_center, 0.0): # For 1x1 grid, dist is 0, max_dist is 0.
                normalized_dist = 0.0 # Perfectly at center means 0 distance.
                                     # MathUtils.normalize_value(0,0,0) = 0.5.
                                     # If we want 0 dist to result in max score for "prefer_center" (1.0 - 0.0 = 1.0),
                                     # then normalized_dist = 0 is correct.
                                     # 來源：新大腦.pdf - EXT_GM9 Discussion on 1x1 grid norm (Page 35)
            else: # Should not be reached if max_dist_to_center handled correctly
                normalized_dist = 1.0


            current_score: float
            if affinity_mode == "prefer_center": # 來源：新大腦.pdf (Page 35)
                current_score = 1.0 - normalized_dist  # Closer to center (smaller dist) -> higher score
            elif affinity_mode == "avoid_center": # 來源：新大腦.pdf (Page 35)
                current_score = normalized_dist  # Further from center (larger dist) -> higher score
            else: # Should not happen with Pydantic validation
                current_score = 0.5 
            
            # Final score is already in [0,1] due to normalized_dist being [0,1]
            # PDF uses MathUtils.normalize_value(current_score, 0, 1.0, clamp=True) which is fine.
            # 來源：新大腦.pdf - EXT_GM9 Final clamp (Page 35)
            scores[r_idx, c_idx] = MathUtils.normalize_value(current_score, 0, 1.0, clamp=True)
            
    return scores * config.weight


# 來源：新大腦.pdf - 16. EXT_GM10_Blocking_Value_Vec (Page 35)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM10強化建議
# Config for this (BlockingValueConfig) was defined in PART 2
# We'll add undesirable_sequences to the config for more flexibility
class BlockingValueConfig(BaseModuleConfig): # Redefine to include list
    # 來源：新大腦.pdf - EXT_GM10 parameters (Page 35-36)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM10 UNDESIRABLE_SEQUENCES 的擴展與學習
    undesirable_sequences_list: List[List[int]] = Field(default_factory=lambda: [
        [1, 1, 1], [2, 2, 2] # 來源：新大腦.pdf (Page 36)
        # Example: [1, 2, 3] if bad in some contexts
    ])
    # 來源：新大腦.pdf - EXT_GM10 Score logic (Page 37)
    # PDF uses 0.9 if not forms_undesirable, 0.1 if forms. Let's make these configurable.
    score_if_safe: float = Field(default=0.9, ge=0.0, le=1.0, description="Score if placement does NOT complete an undesirable pattern.")
    score_if_unsafe: float = Field(default=0.1, ge=0.0, le=1.0, description="Score if placement DOES complete an undesirable pattern.")
    check_line_length: int = Field(default=3, ge=2, description="Length of lines to check for undesirable patterns.")


def EXT_GM10_Blocking_Value_Vec(
    grid: np.ndarray,
    config: BlockingValueConfig,
    request_id: str | None = "N/A_GM10_Blocking",
) -> np.ndarray:
    """
    (GM10-阻斷價值評估)
    核心規則:評估在空格填入數字是否能有效「阻止」或「避免」形成預定義的不良模式或序列。
    目的:偏好那些不會導致形成不良結構的填補,或者理想情況下能主動阻止潛在不良結構形成的填補。
    啟發式類型:防禦性策略與模式規避
    輸出詮釋:分數越高表示在該空格填入數字後,越不可能形成已知的不良模式。
    來源：新大腦.pdf - EXT_GM10_Blocking_Value_Vec (Page 35)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM10"
    logger.debug(
        f"Executing EXT_GM10_Blocking_Value_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 36)
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 36)
    if not potential_numbers_to_place: # 來源：新大腦.pdf (Page 36)
        return scores

    UNDESIRABLE_SEQUENCES = [seq for seq in config.undesirable_sequences_list if len(seq) == config.check_line_length]
    line_len_to_check = config.check_line_length

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 36)
                continue
            
            # Score for this cell will be the max score (safety) achievable by placing any potential number
            # 來源：新大腦.pdf (Page 36)
            max_safety_score_for_cell: float = 0.0 # Default to low if all placements are bad (or if no potential numbers)
            
            # PDF logic: "if not potential_numbers_to_place: scores[r_idx,c_idx]=0.5" is not quite right here,
            # as we already checked potential_numbers_to_place at the function start.
            # If loop doesn't run, max_safety_score_for_cell remains 0.0.
            
            at_least_one_pval_evaluated = False
            for p_val in potential_numbers_to_place:
                at_least_one_pval_evaluated = True
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val
                forms_undesirable_pattern_for_pval = False # Renamed for clarity

                # Check lines of 'line_len_to_check' passing through (r_idx, c_idx)
                # Directions: Horizontal, Vertical, Main Diagonal, Anti-Diagonal
                # 來源：新大腦.pdf - EXT_GM10 Directions (Page 36)
                for dr_line, dc_line in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    if forms_undesirable_pattern_for_pval: break # Already found one for this p_val

                    # Iterate through all windows of 'line_len_to_check' that include (r_idx, c_idx)
                    # where (r_idx, c_idx) is now filled with p_val.
                    # 來源：新大腦.pdf - EXT_GM10 Offset logic (Page 37)
                    # offset is the starting position of the window relative to p_val's position in the window
                    for i_offset_in_window in range(line_len_to_check):
                        current_line_values_list: List[int] = []
                        is_valid_segment = True
                        
                        for k_in_segment in range(line_len_to_check):
                            # Position of k_in_segment-th cell in the current line window
                            eval_r = r_idx + (k_in_segment - i_offset_in_window) * dr_line
                            eval_c = c_idx + (k_in_segment - i_offset_in_window) * dc_line

                            if not (0 <= eval_r < rows and 0 <= eval_c < cols):
                                is_valid_segment = False
                                break
                            current_line_values_list.append(int(temp_grid[eval_r, eval_c]))
                        
                        if is_valid_segment: # No need to check len, it's always line_len_to_check
                            # PDF: "Ensure the currently placed p_val at (r_idx,c_idx) is part of this line"
                            # This is implicitly true by how the window is constructed around (r_idx,c_idx).
                            # 來源：新大腦.pdf (Page 37)

                            for undesirable_seq in UNDESIRABLE_SEQUENCES:
                                # PDF: current_line_values == undesirable_seq
                                # Ensure types are consistent if undesirable_seq stores ints
                                if current_line_values_list == undesirable_seq:
                                    forms_undesirable_pattern_for_pval = True
                                    break # Found an undesirable pattern for this line
                            if forms_undesirable_pattern_for_pval: break # For this direction
                    if forms_undesirable_pattern_for_pval: break # For this p_val
                
                current_score_for_pval = config.score_if_safe if not forms_undesirable_pattern_for_pval else config.score_if_unsafe
                # 來源：新大腦.pdf (Page 37) - PDF has 0.9 if not, 0.1 if yes.

                if current_score_for_pval > max_safety_score_for_cell:
                    max_safety_score_for_cell = current_score_for_pval
            
            if not at_least_one_pval_evaluated and not potential_numbers_to_place : # Should have been caught earlier
                 scores[r_idx, c_idx] = 0.5 # Neutral if no options and somehow reached here
            else:
                 scores[r_idx, c_idx] = max_safety_score_for_cell # 來源：新大腦.pdf (Page 37-38) - Corrected var name

    return scores * config.weight
    # brain.py (Continued)
# ... (Imports, MathUtils, BoardAnalyzerUtils, BaseModuleConfig, and configs from previous PARTS remain the same) ...

# --- Scoring Module Implementations (Continued) ---

# 來源：新大腦.pdf - 17. EXT_GM11_Pair_Correlation_Vec (Page 38)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM11強化建議
# Config for this (PairCorrelationConfig) was defined in PART 2
def EXT_GM11_Pair_Correlation_Vec(
    grid: np.ndarray,
    config: PairCorrelationConfig,
    request_id: str | None = "N/A_GM11_PairCorr",
) -> np.ndarray:
    """
    (GM11-數字配對關聯分析)
    核心規則:分析特定數字對(pair)共同出現或以特定相對位置(此處為鄰近)出現的頻率與價值。
    目的:偏好那些能夠形成已知有利數字配對的填補。
    啟發式類型: 關聯性分析(局部)
    輸出詮釋:分數越高表示在該空格填入特定數字後,能與周圍已存在的數字形成更多或更高價值的有利配對。
    來源：新大腦.pdf - EXT_GM11_Pair_Correlation_Vec (Page 38)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM11"
    logger.debug(
        f"Executing EXT_GM11_Pair_Correlation_Vec with config: {config.model_dump_json(indent=2)}", # Pydantic config
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 38)
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 38)
    if not potential_numbers_to_place: # 來源：新大腦.pdf (Page 38)
        return scores

    # 來源：新大腦.pdf - EXT_GM11 FAVORABLE_PAIRS_SCORES (Page 38)
    # Using config.favorable_pairs
    FAVORABLE_PAIRS_SCORES = {tuple(sorted(k)): v for k,v in config.favorable_pairs.items()} # Normalize key order for easier lookup if desired, though PDF implies (p_val, neighbor_val) order

    max_single_pair_score: float = 0.0 # 來源：新大腦.pdf (Page 38)
    if FAVORABLE_PAIRS_SCORES:
        max_single_pair_score = float(max(FAVORABLE_PAIRS_SCORES.values()))
    
    # Heuristic max possible score: if all 8 neighbors form max-scoring pairs
    # 來源：新大腦.pdf - EXT_GM11 heuristic_max_total_pair_score (Page 39)
    heuristic_max_total_pair_score = 8.0 * max_single_pair_score if max_single_pair_score > 1e-6 else 1.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 39)
                continue

            max_accumulated_score_for_cell: float = 0.0 # 來源：新大腦.pdf (Page 39)

            for p_val in potential_numbers_to_place:
                current_pval_accumulated_score: float = 0.0
                
                # Check 8 neighbors
                # 來源：新大腦.pdf - EXT_GM11 Check 8 neighbors (Page 39)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue

                        nr, nc = r_idx + dr, c_idx + dc

                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbor_val = grid[nr, nc]
                            if neighbor_val != -1:  # If neighbor is an existing number
                                # Check if (p_val, neighbor_val) is a favorable pair
                                # 來源：新大腦.pdf - EXT_GM11 Check favorable pair (Page 39)
                                # The PDF has: if (p_val, int(neighbor_val)) in FAVORABLE_PAIRS_SCORES:
                                # This implies the order matters, or keys in FAVORABLE_PAIRS_SCORES should handle both orders or be normalized.
                                # Using the direct tuple (p_val, int(neighbor_val)) as key.
                                pair_key = (p_val, int(neighbor_val))
                                if pair_key in config.favorable_pairs: # Use config directly
                                    current_pval_accumulated_score += config.favorable_pairs[pair_key]
                                    # PDF original: current_pval_accumulated_score += FAVORABLE_PAIRS_SCORES[(p_val, int(neighbor_val))]
                                    # The PDF also had `current_pval_accumulated_score += 1` which seems like a typo if scores are provided.
                                    # I am using the score from the map.

                if current_pval_accumulated_score > max_accumulated_score_for_cell: # 來源：新大腦.pdf (Page 39)
                    max_accumulated_score_for_cell = current_pval_accumulated_score
            
            scores[r_idx, c_idx] = MathUtils.normalize_value(
                max_accumulated_score_for_cell, 0, heuristic_max_total_pair_score, clamp=True
            ) # 來源：新大腦.pdf (Page 39)
            
    return scores * config.weight


# 來源：新大腦.pdf - 18. EXT_GM12_Island_Analysis_Vec (Page 39)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM12強化建議
# Config for this (IslandAnalysisConfig) was defined in PART 2
def EXT_GM12_Island_Analysis_Vec(
    grid: np.ndarray,
    config: IslandAnalysisConfig,
    request_id: str | None = "N/A_GM12_Island",
) -> np.ndarray:
    """
    (GM12 - 島嶼分析)
    核心規則:分析由已填數字形成的「島嶼」的特性,如大小、緊湊度和平均值。
    目的:根據策略,可能偏好大型、緊湊或包含高價值數字的島嶼。此處假設偏好較大、較緊湊、平均值較高的數字島嶼。
    啟發式類型:連通元件與區域形態分析(針對已填數字)
    輸出詮釋: 分數越高表示該格屬於一個更優(大、緊湊、高平均值)的數字島嶼。空格得0分。
    來源：新大腦.pdf - EXT_GM12_Island_Analysis_Vec (Page 39-40)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM12"
    logger.debug(
        f"Executing EXT_GM12_Island_Analysis_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float) # Empty cells get 0 score from this module
    # 來源：新大腦.pdf (Page 40, 41)
    if rows == 0 or cols == 0:
        return scores

    visited_island_search = np.zeros_like(grid, dtype=bool) # 來源：新大腦.pdf (Page 40)
    max_val_on_board = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols)) # 來源：新大腦.pdf (Page 40)
    if max_val_on_board == 0: max_val_on_board = 1.0 # Avoid div by zero # 來源：新大腦.pdf (Page 40)

    # Weights from config
    w_size = config.w_size # 來源：新大腦.pdf (Page 40)
    w_compactness = config.w_compactness # 來源：新大腦.pdf (Page 40)
    w_avg_value = config.w_avg_value # 來源：新大腦.pdf (Page 40)

    for r_start in range(rows):
        for c_start in range(cols):
            # Found an unvisited *number* (island part)
            if grid[r_start, c_start] != -1 and not visited_island_search[r_start, c_start]: # 來源：新大腦.pdf (Page 40)
                current_island_cells: List[Tuple[int, int]] = [] # 來源：新大腦.pdf (Page 40)
                current_island_values: List[int] = [] # 來源：新大腦.pdf (Page 40)
                
                q = deque([(r_start, c_start)])
                visited_island_search[r_start, c_start] = True
                
                min_r_bbox, max_r_bbox = r_start, r_start # 來源：新大腦.pdf (Page 40)
                min_c_bbox, max_c_bbox = c_start, c_start # 來源：新大腦.pdf (Page 40)

                while q: # 來源：新大腦.pdf (Page 40)
                    r_curr, c_curr = q.popleft()
                    current_island_cells.append((r_curr, c_curr))
                    current_island_values.append(int(grid[r_curr, c_curr]))

                    min_r_bbox = min(min_r_bbox, r_curr) # 來源：新大腦.pdf (Page 40-41)
                    max_r_bbox = max(max_r_bbox, r_curr)
                    min_c_bbox = min(min_c_bbox, c_curr)
                    max_c_bbox = max(max_c_bbox, c_curr)

                    # 4-connectivity for islands
                    # 來源：新大腦.pdf - EXT_GM12 4-connectivity (Page 41)
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: 
                        nr, nc = r_curr + dr, c_curr + dc
                        if 0 <= nr < rows and 0 <= nc < cols and \
                           grid[nr, nc] != -1 and not visited_island_search[nr, nc]: # 來源：新大腦.pdf (Page 41)
                            visited_island_search[nr, nc] = True
                            q.append((nr, nc))
                
                # Calculate island characteristics
                # 來源：新大腦.pdf - EXT_GM12 Island characteristics (Page 41)
                island_size = float(len(current_island_cells))
                avg_value_island: float = 0.0
                if island_size > 0:
                    avg_value_island = sum(current_island_values) / island_size
                
                bbox_height = float(max_r_bbox - min_r_bbox + 1)
                bbox_width = float(max_c_bbox - min_c_bbox + 1)
                bbox_area = bbox_height * bbox_width
                
                compactness: float = 0.0
                if bbox_area > 0: # Avoid division by zero
                    compactness = island_size / bbox_area # (Ratio of actual cells to bounding box area)

                # Normalize characteristics
                # 來源：新大腦.pdf - EXT_GM12 Normalize characteristics (Page 41)
                norm_size = MathUtils.normalize_value(island_size, 1, float(rows * cols), clamp=True)
                norm_compactness = MathUtils.normalize_value(compactness, 0, 1.0, clamp=True) # Already 0-1
                norm_avg_value = MathUtils.normalize_value(avg_value_island, 1, max_val_on_board, clamp=True)

                # Combine into a single island score
                # 來源：新大腦.pdf - EXT_GM12 Combine island score (Page 41)
                island_score_unnormalized = (
                    w_size * norm_size +
                    w_compactness * norm_compactness +
                    w_avg_value * norm_avg_value
                )
                # Normalize combined score (max possible is sum of weights if they sum to 1)
                total_weights = w_size + w_compactness + w_avg_value
                max_possible_island_score = total_weights if total_weights > 0 else 1.0

                final_island_score = MathUtils.normalize_value(island_score_unnormalized, 0, max_possible_island_score, clamp=True)
                # PDF: MathUtils.normalize_value(island_score, 0, 1.0, clamp=True) - assumes weights sum to 1 or less.

                # Assign this score to all cells in the current island
                # 來源：新大腦.pdf - EXT_GM12 Assign score (Page 41)
                for r_cell, c_cell in current_island_cells:
                    scores[r_cell, c_cell] = final_island_score
            
            elif grid[r_start, c_start] == -1: # Empty cells get 0 score (already initialized) # 來源：新大腦.pdf (Page 41)
                # Ensure visited_overall is marked for empty cells too to avoid re-processing them as start points
                # for an "island search" that would immediately terminate.
                # The logic `grid[r_start,c_start]!=-1 and not visited_island_search` handles this.
                pass # Scores remain 0 for empty cells
            
            # Mark as visited to avoid re-check even if it's an empty cell we skipped.
            # The primary visited_island_search is for actual island cells.
            # No, only mark actual island cells or cells part of a processed component.
            # Empty cells are handled by the first `if` in the loop.

    return scores * config.weight
    # brain.py (Continued)
# ... (Imports, MathUtils, BoardAnalyzerUtils, BaseModuleConfig, and configs from previous PARTS remain the same) ...

# --- Scoring Module Implementations (Continued) ---

# 來源：新大腦.pdf - 19. EXT_GM13_Sequence_Diversity_Vec (Page 41)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM13強化建議
# Config for this (SequenceDiversityConfig) was defined in PART 2
def EXT_GM13_Sequence_Diversity_Vec(
    grid: np.ndarray,
    config: SequenceDiversityConfig,
    request_id: str | None = "N/A_GM13_SeqDiv",
) -> np.ndarray:
    """
    (GM13-序列多樣性)
    核心規則:評估填補位置是否有助於形成多樣化的短序列(例如,不同方向、不同類型),而非僅專注於單一長序列。
    目的:鼓勵在盤面上形成多個不同類型或方向的短數字序列,增加盤面的「活性」或「機會」。
    啟發式類型:模式識別與組合多樣性
    輸出詮釋:分數越高表示在該空格填入特定數字後,能參與形成的獨特短序列種類越多。
    來源：新大腦.pdf - EXT_GM13_Sequence_Diversity_Vec (Page 41-42)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM13"
    logger.debug(
        f"Executing EXT_GM13_Sequence_Diversity_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 42)
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 42)
    if not potential_numbers_to_place: # 來源：新大腦.pdf (Page 42)
        return scores

    short_sequence_len = config.short_sequence_len # 來源：新大腦.pdf (Page 42)
    # Max distinct short sequences a single cell might participate in (heuristic for normalization)
    # 來源：新大腦.pdf - EXT_GM13 heuristic_max_distinct_sequences (Page 42)
    # For length 3, in 4 directions, cell can be in 3 positions. Max 4*2 types (arith, ident) = 8.
    # This is a rough upper bound.
    heuristic_max_distinct_sequences = 8.0 
    if short_sequence_len != 3: # Adjust if length changes
        heuristic_max_distinct_sequences = float(4 * 2 * (short_sequence_len)) # Very rough

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 42)
                continue

            max_diversity_count_for_cell: int = 0 # 來源：新大腦.pdf (Page 42)

            for p_val in potential_numbers_to_place:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val
                
                # Store signatures like ("arithmetic", (dr,dc), diff) or ("identical", (dr,dc), val)
                # 來源：新大腦.pdf - EXT_GM13 found_sequence_signatures (Page 42)
                found_sequence_signatures: Set[Tuple[str, Tuple[int,int], int]] = set() 

                # Check in 4 directions (H, V, D1, D2)
                # 來源：新大腦.pdf - EXT_GM13 Directions (Page 42)
                for dr_dir, dc_dir in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    # For each direction, check 'short_sequence_len' possible alignments of a sequence
                    # where p_val (at (r_idx, c_idx)) is involved.
                    # i_offset_in_window: position of p_val within the current window of 'short_sequence_len'
                    # 來源：新大腦.pdf - EXT_GM13 i_offset loop (Page 42)
                    for i_offset_in_window in range(short_sequence_len):
                        current_sequence_values: List[int] = []
                        valid_segment = True
                        
                        for k_in_segment in range(short_sequence_len):
                            # Position of k_in_segment-th element in the window, relative to (r_idx, c_idx)
                            # 來源：新大腦.pdf - EXT_GM13 check_r, check_c calculation (Page 43)
                            eval_r = r_idx + (k_in_segment - i_offset_in_window) * dr_dir
                            eval_c = c_idx + (k_in_segment - i_offset_in_window) * dc_dir

                            if not (0 <= eval_r < rows and 0 <= eval_c < cols):
                                valid_segment = False
                                break
                            current_sequence_values.append(int(temp_grid[eval_r, eval_c]))
                        
                        if valid_segment: # Implicitly len(current_sequence_values) == short_sequence_len
                            # Analyze this short sequence (s)
                            s = current_sequence_values
                            # All values must be non -1 (which is true since temp_grid has p_val and others are from original or p_val)
                            
                            # 1. Arithmetic sequence (non-constant)
                            # 來源：新大腦.pdf - EXT_GM13 Arithmetic check (Page 43)
                            if len(s) >= 2 : # Need at least 2 to check diff
                                diffs = [s[k+1] - s[k] for k in range(len(s)-1)]
                                if diffs: # Ensure diffs is not empty
                                    first_diff = diffs[0]
                                    if all(math.isclose(d, first_diff) for d in diffs) and not math.isclose(first_diff, 0):
                                        # Normalize direction vector for signature uniqueness (e.g., (0,1) is same as (0,-1) for line orientation)
                                        norm_dr = abs(dr_dir) if dc_dir == 0 else dr_dir # Simple normalization
                                        norm_dc = abs(dc_dir) if dr_dir == 0 else dc_dir
                                        if norm_dr == 1 and norm_dc == 1 and norm_dr * norm_dc < 0: # anti-diag normalize (1,-1) vs (-1,1)
                                            norm_dr, norm_dc = min(abs(dr_dir),dr_dir), min(abs(dc_dir),dc_dir) if dr_dir != dc_dir else dc_dir

                                        found_sequence_signatures.add(("arithmetic", (norm_dr, norm_dc), int(first_diff)))

                            # 2. Identical sequence
                            # 來源：新大腦.pdf - EXT_GM13 Identical check (Page 43)
                            if len(set(s)) == 1 and s[0] != -1: # -1 check might be redundant here
                                norm_dr = abs(dr_dir) if dc_dir == 0 else dr_dir
                                norm_dc = abs(dc_dir) if dr_dir == 0 else dc_dir
                                if norm_dr == 1 and norm_dc == 1 and norm_dr * norm_dc < 0:
                                     norm_dr, norm_dc = min(abs(dr_dir),dr_dir), min(abs(dc_dir),dc_dir) if dr_dir != dc_dir else dc_dir
                                found_sequence_signatures.add(("identical", (norm_dr, norm_dc), s[0]))
                
                current_pval_diversity_count = len(found_sequence_signatures) # 來源：新大腦.pdf (Page 43)
                if current_pval_diversity_count > max_diversity_count_for_cell:
                    max_diversity_count_for_cell = current_pval_diversity_count
            
            scores[r_idx, c_idx] = MathUtils.normalize_value(
                float(max_diversity_count_for_cell), 0, heuristic_max_distinct_sequences, clamp=True
            ) # 來源：新大腦.pdf (Page 43)
            
    return scores * config.weight


# 來源：新大腦.pdf - 20. EXT_GM14_Risk_Assessment_Vec (Page 43)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM14強化建議
# Config for this (RiskAssessmentConfig) was defined in PART 2
def EXT_GM14_Risk_Assessment_Vec(
    grid: np.ndarray,
    config: RiskAssessmentConfig,
    request_id: str | None = "N/A_GM14_Risk",
) -> np.ndarray:
    """
    (GM14 - 風險評估)
    核心規則:評估某個填補動作的潛在「風險」,例如是否會導致後續選擇過少(降低盤面靈活性)。
    目的:偏好那些能保持盤面較高靈活性的填補。低風險=高分數。
    啟發式類型: 盤面狀態評估(未來選擇性)
    輸出詮釋:分數越高表示填入該數字後,盤面剩餘的合法填補選項越多,風險越低。
    來源：新大腦.pdf - EXT_GM14_Risk_Assessment_Vec (Page 43-44)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM14"
    logger.debug(
        f"Executing EXT_GM14_Risk_Assessment_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 44)
        return scores

    initial_potential_numbers = BoardAnalyzerUtils.get_legal_values_for_placement(grid) # Set[int]
    # 來源：新大腦.pdf (Page 44)
    if not initial_potential_numbers: # 來源：新大腦.pdf (Page 44)
        # If no numbers can be placed initially, all empty cells might be considered max risk (score 0)
        # or neutral (0.5). PDF returns scores (which is zeros).
        return scores 

    # Heuristic for normalization
    # 來源：新大腦.pdf - EXT_GM14 Heuristic max_possible_options (Page 44)
    # Max possible subsequent_legal_moves is roughly rows*cols (if board almost empty)
    # Max remaining_empty_cells is rows*cols - 1
    # So product can be up to (rows*cols-1)^2
    max_possible_options_heuristic: float
    if config.flexibility_metric_mode == "subsequent_moves":
        max_possible_options_heuristic = float(rows * cols) # Max unique numbers
    else: # "product_moves_empty_cells"
        max_possible_options_heuristic = float((rows * cols -1) * (rows * cols -1)) if rows*cols >1 else 1.0
    
    if max_possible_options_heuristic <=0 : max_possible_options_heuristic = 1.0


    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 44)
                continue

            max_flexibility_score_for_cell: float = 0.0 # 來源：新大腦.pdf (Page 44)
            
            # Iterate through numbers that could be placed at (r_idx, c_idx)
            # This means we should use `initial_potential_numbers` for p_val
            # The PDF note: "p_val in initial_potential_numbers: # Only try values that are currently legal for the original grid"
            # This is correct.
            
            evaluated_any_pval = False
            for p_val in initial_potential_numbers: # p_val is a number that could be placed on the original grid
                                                    # We are evaluating placing it at (r_idx, c_idx)
                evaluated_any_pval = True
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val # Place p_val

                # Calculate flexibility after this placement
                # 來源：新大腦.pdf - EXT_GM14 Calculate flexibility (Page 44)
                remaining_empty_cells = float(np.count_nonzero(temp_grid == -1))
                subsequent_legal_moves_set = BoardAnalyzerUtils.get_legal_values_for_placement(temp_grid)
                num_subsequent_legal_moves = float(len(subsequent_legal_moves_set))

                current_flexibility: float
                if config.flexibility_metric_mode == "subsequent_moves":
                    current_flexibility = num_subsequent_legal_moves # 來源：新大腦.pdf (Page 45)
                else: # "product_moves_empty_cells"
                    # 來源：新大腦.pdf - EXT_GM14 product metric (Page 45)
                    current_flexibility = remaining_empty_cells * num_subsequent_legal_moves
                
                if current_flexibility > max_flexibility_score_for_cell: # 來源：新大腦.pdf (Page 45)
                    max_flexibility_score_for_cell = current_flexibility
            
            if not evaluated_any_pval: # Should only happen if initial_potential_numbers was empty
                scores[r_idx,c_idx] = 0.0 # Or some other low score
            else:
                # 來源：新大腦.pdf - EXT_GM14 Normalization (Page 45)
                # The PDF has `current_max_heuristic_flex = float(rows*cols -1)` which is for subsequent_legal_moves metric.
                # This needs to adapt to the chosen metric.
                current_max_heuristic_to_use = max_possible_options_heuristic
                if config.flexibility_metric_mode == "subsequent_moves":
                     current_max_heuristic_to_use = float(rows*cols -1) if rows*cols >1 else 1.0 # Max legal after 1 placement
                     if current_max_heuristic_to_use <= 0 : current_max_heuristic_to_use = 1.0
                # else: current_max_heuristic_to_use remains max_possible_options_heuristic which is (R*C-1)^2
                
                scores[r_idx, c_idx] = MathUtils.normalize_value(
                    max_flexibility_score_for_cell, 0, current_max_heuristic_to_use, clamp=True
                )

    return scores * config.weight
    # brain.py (Continued)
# ... (Imports, MathUtils, BoardAnalyzerUtils, BaseModuleConfig, and configs from previous PARTS remain the same) ...

# --- Scoring Module Implementations (Continued) ---

# 來源：新大腦.pdf - 21. EXT_GM15_Information_Gain_Vec (Page 45)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM15強化建議
# Config for this (InformationGainConfig) was defined in PART 2
def EXT_GM15_Information_Gain_Vec(
    grid: np.ndarray,
    config: InformationGainConfig,
    request_id: str | None = "N/A_GM15_InfoGain",
) -> np.ndarray:
    """
    (GM15-資訊增益評估)
    核心規則:評估填入數字後,對盤面整體結構「有序性」的提升(例如,熵的降低)。
    目的:偏好那些能使盤面狀態更「確定」或「有序」的填補。
    啟發式類型:資訊理論啟發(基於全局熵變)
    輸出詮釋:分數越高表示填入該數字後,盤面整體熵降低得越多(即資訊增益越大,盤面越有序)。
    來源：新大腦.pdf - EXT_GM15_Information_Gain_Vec (Page 45)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM15"
    logger.debug(
        f"Executing EXT_GM15_Information_Gain_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 45)
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 45)
    if not potential_numbers_to_place: # 來源：新大腦.pdf (Page 45)
        return scores

    # Calculate entropy of the initial grid
    # 來源：新大腦.pdf - EXT_GM15 initial_grid_values (Page 45-46)
    if config.entropy_scope == "global_full":
        initial_grid_values_for_entropy = [int(val) for val in grid.flatten()] # -1 is a symbol
    else: # "global_filled_only"
        initial_grid_values_for_entropy = [int(val) for val in grid.flatten() if val != -1]
        if not initial_grid_values_for_entropy: # Handle case of all empty grid for filled_only
            initial_grid_values_for_entropy.append(0) # Add a dummy value to avoid empty list for entropy


    entropy_before = MathUtils.get_entropy(initial_grid_values_for_entropy) # 來源：新大腦.pdf (Page 46)

    # Max possible entropy for normalization (log2 of number of symbols: 1 to R*C plus -1 if global_full)
    # 來源：新大腦.pdf - EXT_GM15 max_possible_entropy_change (Page 46)
    num_symbols_for_max_entropy: int
    if config.entropy_scope == "global_full":
        num_symbols_for_max_entropy = rows * cols + 1 # Numbers 1 to R*C, plus -1
    else: # "global_filled_only"
        num_symbols_for_max_entropy = rows * cols # Numbers 1 to R*C
        if num_symbols_for_max_entropy == 0 : num_symbols_for_max_entropy = 1 # Avoid log2(0) for empty grid

    max_possible_entropy_change = math.log2(num_symbols_for_max_entropy) if num_symbols_for_max_entropy > 1 else 1.0
    if max_possible_entropy_change <= 0: max_possible_entropy_change = 1.0 # 來源：新大腦.pdf (Page 46)


    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 46)
                continue

            max_entropy_reduction_for_cell: float = -float('inf') # We want to maximize reduction # 來源：新大腦.pdf (Page 46)
            
            evaluated_at_least_one_pval = False
            for p_val in potential_numbers_to_place:
                evaluated_at_least_one_pval = True
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val

                if config.entropy_scope == "global_full":
                    temp_grid_values_for_entropy = [int(val) for val in temp_grid.flatten()]
                else: # "global_filled_only"
                    temp_grid_values_for_entropy = [int(val) for val in temp_grid.flatten() if val != -1]
                    if not temp_grid_values_for_entropy:
                         temp_grid_values_for_entropy.append(0)


                entropy_after = MathUtils.get_entropy(temp_grid_values_for_entropy) # 來源：新大腦.pdf (Page 46)
                entropy_reduction = entropy_before - entropy_after  # Higher reduction is better # 來源：新大腦.pdf (Page 46)

                if entropy_reduction > max_entropy_reduction_for_cell: # 來源：新大腦.pdf (Page 46)
                    max_entropy_reduction_for_cell = entropy_reduction
            
            if not evaluated_at_least_one_pval: # No legal moves for this cell (should not happen if loop runs)
                max_entropy_reduction_for_cell = 0.0 # Or some neutral/low value
            elif max_entropy_reduction_for_cell == -float('inf'): # Should not happen if potential_numbers_to_place is not empty
                max_entropy_reduction_for_cell = 0.0 # 來源：新大腦.pdf (Page 46)

            # Normalize the reduction. Min reduction can be negative (entropy increases). Max can be entropy_before.
            # Or normalize against max_possible_entropy_change. Score will be higher for positive reductions.
            # Range of reduction is roughly [-max_possible_entropy_change, max_possible_entropy_change]
            # PDF: MathUtils.normalize_value(max_entropy_reduction_for_cell, 0, max_possible_entropy_change, clamp=True)
            # This normalization clamps negative reductions (entropy increase) to 0.
            # 來源：新大腦.pdf - EXT_GM15 Normalization (Page 46)
            scores[r_idx, c_idx] = MathUtils.normalize_value(
                max_entropy_reduction_for_cell, 
                0, # Min score desired is 0 (no gain or entropy increase)
                max_possible_entropy_change, # Max possible gain
                clamp=True
            )
            # Clamping at 0 if it increases entropy. (Handled by normalize_value if min_val=0)
            # 來源：新大腦.pdf (Page 46)
            
    return scores * config.weight


# 來源：新大腦.pdf - 22. EXT_GM16_Harmonic_Centrality_Vec (Page 46)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM16強化建議
# Config for this (HarmonicCentralityConfig) was defined in PART 2
def EXT_GM16_Harmonic_Centrality_Vec(
    grid: np.ndarray,
    config: HarmonicCentralityConfig,
    request_id: str | None = "N/A_GM16_HarmonicCent",
) -> np.ndarray:
    """
    (GM16 - 調和中心性)
    核心規則:應用圖論中的調和中心性概念,評估盤面上各空格節點的重要性。調和中心性是一個節點到所有其他節點距離倒數的總和。
    目的:偏好那些在盤面「網絡」中更具中心性的空格。
    啟發式類型: 圖論中心性
    輸出詮釋:分數越高表示該空格在圖結構中越「中心」(平均而言離其他格子越近)。
    來源：新大腦.pdf - EXT_GM16_Harmonic_Centrality_Vec (Page 46-47)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM16"
    logger.debug(
        f"Executing EXT_GM16_Harmonic_Centrality_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    # Needs more than 1 cell # 來源：新大腦.pdf (Page 47)
    if rows == 0 or cols == 0 or (rows * cols) <= 1:
        return scores

    # Max possible harmonic centrality (heuristic): if a cell is at distance 1 from all N-1 other cells.
    # Max_HC = (rows*cols - 1) * (1/1)
    # 來源：新大腦.pdf - EXT_GM16 max_hc_heuristic (Page 47)
    max_hc_heuristic = float(rows * cols - 1)
    if max_hc_heuristic <= 0: max_hc_heuristic = 1.0 # 來源：新大腦.pdf (Page 47)

    for r_eval in range(rows):
        for c_eval in range(cols):
            # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM16 節點的定義 (node_definition)
            # Original PDF scores only empty cells. This config allows flexibility.
            if config.node_definition == "empty_cells_only" and grid[r_eval, c_eval] != -1:
                continue
            if config.node_definition == "filled_cells_only" and grid[r_eval, c_eval] == -1:
                continue
            # if "all_cells", no filter here based on cell content.
            
            current_harmonic_centrality: float = 0.0 # 來源：新大腦.pdf (Page 47)
            num_other_nodes_considered = 0

            for r_other in range(rows):
                for c_other in range(cols):
                    if r_eval == r_other and c_eval == c_other: # 來源：新大腦.pdf (Page 47)
                        continue
                    
                    # Filter other_nodes based on config
                    if config.node_definition == "empty_cells_only" and grid[r_other, c_other] != -1:
                        continue
                    if config.node_definition == "filled_cells_only" and grid[r_other, c_other] == -1:
                        continue

                    # Using Manhattan distance as grid distance
                    # 來源：新大腦.pdf - EXT_GM16 Manhattan distance (Page 47)
                    dist = MathUtils.manhattan_distance((r_eval, c_eval), (r_other, c_other))
                    if dist > 0:
                        current_harmonic_centrality += 1.0 / dist
                    num_other_nodes_considered +=1
            
            if num_other_nodes_considered == 0: # Only one cell considered, or no valid other_nodes based on filter
                # 來源：新大腦.pdf (Page 47)
                scores[r_eval, c_eval] = 0.0
            else:
                # Normalization can be tricky. Using the heuristic max.
                # 來源：新大腦.pdf - EXT_GM16 Normalization (Page 48)
                scores[r_eval, c_eval] = MathUtils.normalize_value(
                    current_harmonic_centrality, 0, max_hc_heuristic, clamp=True
                )
    
    # If we only scored specific cells (e.g. empty_cells_only), other cells remain 0.
    return scores * config.weight
    # brain.py (Continued)
# ... (Imports, MathUtils, BoardAnalyzerUtils, BaseModuleConfig, and configs from previous PARTS remain the same) ...

# --- Scoring Module Implementations (Continued) ---

# 來源：新大腦.pdf - 23. EXT_GM17_Entropy_Minimization_Vec (Page 48)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM17強化建議
# Config for this (LocalEntropyMinimizationConfig) was defined in PART 2
def EXT_GM17_Entropy_Minimization_Vec(
    grid: np.ndarray,
    config: LocalEntropyMinimizationConfig,
    request_id: str | None = "N/A_GM17_LocalEntropy",
) -> np.ndarray:
    """
    (GM17 - 局部熵最小化)
    核心規則:評估填入數字後,盤面局部鄰域「熵」(無序度)的降低程度。
    目的:偏好那些能使其直接周圍環境更有規律、更「有序」的填補。
    啟發式類型:資訊理論啟發(基於局部熵變)
    輸出詮釋:分數越高表示填入該數字後,其局部鄰域的熵降低得越多(局部更有序)。
    來源：新大腦.pdf - EXT_GM17_Entropy_Minimization_Vec (Page 48)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM17"
    logger.debug(
        f"Executing EXT_GM17_Entropy_Minimization_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 48)
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 48)
    if not potential_numbers_to_place: # 來源：新大腦.pdf (Page 48)
        return scores

    radius = config.radius # 來源：新大腦.pdf (Page 48)
    # Max entropy change in a local neighborhood of size N_hood is log2(N_hood)
    # 來源：新大腦.pdf - EXT_GM17 max_local_entropy_change (Page 48)
    num_cells_in_neighborhood = (2 * radius + 1)**2 # Including center
    max_local_entropy_change = math.log2(num_cells_in_neighborhood) if num_cells_in_neighborhood > 1 else 1.0
    if max_local_entropy_change <= 0: max_local_entropy_change = 1.0 # 來源：新大腦.pdf (Page 48)

    # val_func to keep -1 as a distinct symbol for entropy calculation
    # 來源：新大腦.pdf - EXT_GM17 val_func_for_entropy (Page 49)
    def val_func_for_entropy(x_val: int) -> int: return int(x_val)

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 48)
                continue

            # Get all values in radius around (r_idx,c_idx), including (r_idx,c_idx) itself.
            # Entropy before (with (r_idx,c_idx) as empty, i.e., -1)
            # 來源：新大腦.pdf - EXT_GM17 values_before_placement_local (Page 49)
            values_before_placement_local = BoardAnalyzerUtils.get_neighborhood_values(
                grid, r_idx, c_idx, radius=radius, eight_connectivity=True,
                val_func=val_func_for_entropy, include_center=True
            )
            entropy_before_local = MathUtils.get_entropy(values_before_placement_local) # 來源：新大腦.pdf (Page 49)

            max_entropy_reduction_for_cell: float = -float('inf') # 來源：新大腦.pdf (Page 49)
            evaluated_at_least_one_pval = False
            for p_val in potential_numbers_to_place:
                evaluated_at_least_one_pval = True
                temp_grid_local_place = grid.copy() # Create a fresh copy for each p_val
                temp_grid_local_place[r_idx, c_idx] = p_val # 來源：新大腦.pdf (Page 50)
                
                values_after_placement_local = BoardAnalyzerUtils.get_neighborhood_values(
                    temp_grid_local_place, r_idx, c_idx, radius=radius, eight_connectivity=True,
                    val_func=val_func_for_entropy, include_center=True
                ) # 來源：新大腦.pdf (Page 50)
                entropy_after_local = MathUtils.get_entropy(values_after_placement_local) # 來源：新大腦.pdf (Page 50)
                
                entropy_reduction = entropy_before_local - entropy_after_local # 來源：新大腦.pdf (Page 50)
                if entropy_reduction > max_entropy_reduction_for_cell:
                    max_entropy_reduction_for_cell = entropy_reduction
            
            if not evaluated_at_least_one_pval : max_entropy_reduction_for_cell = 0.0
            elif max_entropy_reduction_for_cell == -float('inf'): max_entropy_reduction_for_cell = 0.0 # 來源：新大腦.pdf (Page 50)


            # Normalize the reduction. Max possible reduction is entropy_before_local,
            # or theoretically max_local_entropy_change if going from max chaos to perfect order.
            # PDF normalizes against max_local_entropy_change.
            # 來源：新大腦.pdf - EXT_GM17 Normalization (Page 50)
            scores[r_idx, c_idx] = MathUtils.normalize_value(
                max_entropy_reduction_for_cell, 
                0, # Min desired score for reduction (no gain or entropy increase)
                max_local_entropy_change, 
                clamp=True
            )
            
    return scores * config.weight


# 來源：新大腦.pdf - 24. EXT_GM18_RL_Value_Est_Vec (Page 50)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM18強化建議
# Config for this (RLValueEstimationConfig) was defined in PART 2
def EXT_GM18_RL_Value_Est_Vec(
    grid: np.ndarray,
    config: RLValueEstimationConfig,
    request_id: str | None = "N/A_GM18_RL_Est",
) -> np.ndarray:
    """
    (GM18-類強化學習價值估計)
    核心規則:基於一組預定義的「理想特徴」來評估某個填補動作的啟發式長期潜在價值。此為簡化版,模擬從歷史數據學習到的偏好。
    目的:偏好那些能夠使盤面展現更多理想特徵(如形成特定序列、達到特定盤面密度等)的填補。
    啟發式類型:狀態價值啟發(基於盤面特徵計數)
    輸出詮釋:分數越高表示填入該數字後,盤面呈現的理想特徵越多,預期長期回報越大。
    來源：新大腦.pdf - EXT_GM18_RL_Value_Est_Vec (Page 50)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM18"
    logger.debug(
        f"Executing EXT_GM18_RL_Value_Est_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 50)
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 50)
    if not potential_numbers_to_place: # 來源：新大腦.pdf (Page 51)
        return scores

    FEATURE_WEIGHTS = config.feature_weights # 來源：新大腦.pdf (Page 51)
    
    # Heuristic max feature score for normalization
    # 來源：新大腦.pdf - EXT_GM18 max_heuristic_feature_score (Page 51)
    # Roughly: 4 directions * (max score for identical_3 + max score for arithmetic_3) + density + central + edge
    # This is a very rough estimate as features might overlap or not all be achievable.
    max_heuristic_feature_score = (
        4 * (FEATURE_WEIGHTS.get("identical_3", 0.0) + FEATURE_WEIGHTS.get("arithmetic_3", 0.0)) +
        FEATURE_WEIGHTS.get("board_density_factor", 0.0) * 1.0 + # Max density is 1
        FEATURE_WEIGHTS.get("central_control_boost", 0.0) * 1.0 + # Max central boost is 1
        FEATURE_WEIGHTS.get("edge_affinity_boost", 0.0) * 1.0    # Max edge boost is 1
    )
    if max_heuristic_feature_score <= 0: max_heuristic_feature_score = 1.0 # 來源：新大腦.pdf (Page 51)

    center_r_gm18 = (rows - 1) / 2.0 # For central_control_boost # 來源：新大腦.pdf (Page 52)
    center_c_gm18 = (cols - 1) / 2.0 # 來源：新大腦.pdf (Page 52)
    max_dist_to_center_gm18 = MathUtils.euclidean_distance((0.0,0.0),(center_r_gm18, center_c_gm18)) if rows*cols > 1 else 0.0
    if math.isclose(max_dist_to_center_gm18, 0.0) and (rows > 1 or cols > 1): max_dist_to_center_gm18 = 1.0


    max_min_dist_to_edge_gm18 = float(min((rows - 1) // 2, (cols - 1) // 2)) # 來源：新大腦.pdf (Page 52)
    if max_min_dist_to_edge_gm18 <=0 and (rows >1 or cols >1): max_min_dist_to_edge_gm18 = 0.5 # Avoid div by zero

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 51)
                continue
            
            max_feature_score_for_cell: float = 0.0 # 來源：新大腦.pdf (Page 51)
            
            for p_val in potential_numbers_to_place:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val
                current_features_score: float = 0.0

                # Feature 1 & 2: Lines of 3 (identical or arithmetic) involving p_val
                # 來源：新大腦.pdf - EXT_GM18 Feature 1 & 2 (Page 51)
                # Logic similar to GM10/GM5 line checking
                line_len_check = 3 # For identical_3 and arithmetic_3
                for dr_line, dc_line in [(0, 1), (1, 0), (1, 1), (1, -1)]: # H, V, D1, D2 # 來源：新大腦.pdf (Page 51)
                    for i_offset in range(line_len_check): # p_val is at index i_offset
                        line_values: List[int] = []
                        is_valid_line = True
                        # Check if (r_idx, c_idx) is part of this window implicitly checked by offset
                        
                        for k_in_segment in range(line_len_check):
                            eval_r = r_idx + (k_in_segment - i_offset) * dr_line
                            eval_c = c_idx + (k_in_segment - i_offset) * dc_line
                            if not (0 <= eval_r < rows and 0 <= eval_c < cols):
                                is_valid_line = False
                                break
                            line_values.append(int(temp_grid[eval_r, eval_c])) # 來源：新大腦.pdf (Page 52)
                        
                        if is_valid_line and all(v != -1 for v in line_values): # All filled
                            s = line_values
                            # Identical
                            # 來源：新大腦.pdf - EXT_GM18 Identical check (Page 52)
                            if len(set(s)) == 1:
                                current_features_score += FEATURE_WEIGHTS.get("identical_3", 0.0)
                            # Arithmetic (non-constant)
                            # 來源：新大腦.pdf - EXT_GM18 Arithmetic check (Page 52)
                            elif len(s) >= 2 : # Should be true for len 3
                                diffs_feat = [s[k+1] - s[k] for k in range(len(s)-1)]
                                if diffs_feat and len(set(diffs_feat)) == 1 and not math.isclose(diffs_feat[0],0):
                                    current_features_score += FEATURE_WEIGHTS.get("arithmetic_3", 0.0)
                
                # Feature 3: Board density
                # 來源：新大腦.pdf - EXT_GM18 Board density (Page 52)
                num_filled_after_placement = np.count_nonzero(temp_grid != -1)
                density_after_placement = num_filled_after_placement / (rows * cols) if (rows * cols) > 0 else 0.0
                current_features_score += FEATURE_WEIGHTS.get("board_density_factor", 0.0) * density_after_placement

                # Conceptual Features (based on GM9, GM8 from PDF)
                # 來源：新大腦.pdf - EXT_GM18 Conceptual Features (Page 52)
                if rows > 1 and cols > 1: # Only for grids larger than 1x1
                    # Central control boost
                    if FEATURE_WEIGHTS.get("central_control_boost", 0.0) > 0 and max_dist_to_center_gm18 > 1e-6:
                        dist_to_center = MathUtils.euclidean_distance((float(r_idx), float(c_idx)), (center_r_gm18, center_c_gm18))
                        current_features_score += FEATURE_WEIGHTS.get("central_control_boost", 0.0) * \
                            (1.0 - MathUtils.normalize_value(dist_to_center, 0, max_dist_to_center_gm18, clamp=True))
                    
                    # Edge affinity boost (if strategy calls for it, assume prefer_edge for boost)
                    if FEATURE_WEIGHTS.get("edge_affinity_boost", 0.0) > 0 and max_min_dist_to_edge_gm18 > 1e-6 :
                        dist_to_edge = min(r_idx, rows - 1 - r_idx, c_idx, cols - 1 - c_idx)
                        current_features_score += FEATURE_WEIGHTS.get("edge_affinity_boost", 0.0) * \
                            (1.0 - MathUtils.normalize_value(float(dist_to_edge), 0, max_min_dist_to_edge_gm18, clamp=True))
                
                if current_features_score > max_feature_score_for_cell:
                    max_feature_score_for_cell = current_features_score
            
            scores[r_idx, c_idx] = MathUtils.normalize_value(
                max_feature_score_for_cell, 0, max_heuristic_feature_score, clamp=True
            ) # 來源：新大腦.pdf (Page 52)
            
    return scores * config.weight


# 來源：新大腦.pdf - 25. EXT_GM19_Masked_Number_Skip_Pattern_Vec (Page 53)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM19強化建議
# Config for this (SkipPatternConfig) was defined in PART 2
def EXT_GM19_Masked_Number_Skip_Pattern_Vec(
    grid: np.ndarray,
    config: SkipPatternConfig,
    request_id: str | None = "N/A_GM19_SkipPattern",
) -> np.ndarray:
    """
    (GM19-遮罩數字跳格模式向量)
    核心規則:分析已揭示數字的「跳格模式」(其實際位置與預期基礎位置的偏差),並對符合主導跳格模式的空格進行評分。
    啟發式類型:空間模式匹配(基於全局偏移量)
    輸出詮釋: 分數越高表示該空格若填入特定數字,能與盤面上觀察到的主要「跳格」規律性最為吻合。
    來源：新大腦.pdf - EXT_GM19_Masked_Number_Skip_Pattern_Vec (Page 53)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM19"
    logger.debug(
        f"Executing EXT_GM19_Masked_Number_Skip_Pattern_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 53)
        return scores

    # 來源：新大腦.pdf - EXT_GM19 revealed_numbers_info (Page 53)
    revealed_numbers_info: List[Dict[str, Any]] = [
        {'value': int(grid[r, c]), 'r': r, 'c': c}
        for r in range(rows) for c in range(cols)
        if grid[r, c] != -1 and grid[r, c] > 0 # Assuming positive numbers
    ]
    if not revealed_numbers_info: return scores # 來源：新大腦.pdf (Page 53)

    expected_max_number_on_card = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols)) # 來源：新大腦.pdf (Page 53)
    
    # Base positions based on scan pattern (default: left-to-right, top-to-bottom)
    # 來源：新大腦.pdf - EXT_GM19 base_positions (Page 53-54)
    # Conceptual: config.base_pattern_definition could alter this. For now, standard scan.
    base_positions: Dict[int, Tuple[int, int]] = {} 
    for k_val in range(1, expected_max_number_on_card + 1):
        base_r = (k_val - 1) // cols
        base_c = (k_val - 1) % cols
        if base_r < rows: # Ensure base position is within grid dimensions # 來源：新大腦.pdf (Page 54)
            base_positions[k_val] = (base_r, base_c)

    skip_vectors: Dict[int, Tuple[int, int]] = {} # value -> (delta_r, delta_c) # 來源：新大腦.pdf (Page 54)
    for rn_info in revealed_numbers_info:
        val = rn_info['value']
        if val in base_positions:
            expected_r, expected_c = base_positions[val]
            skip_vectors[val] = (rn_info['r'] - expected_r, rn_info['c'] - expected_c)
    
    if not skip_vectors: return scores # 來源：新大腦.pdf (Page 54)

    # Determine dominant skip patterns and their strength
    # 來源：新大腦.pdf - EXT_GM19 dominant_skip_patterns_strength (Page 54)
    dominant_skip_patterns_strength: Dict[Tuple[int, int], float] = {}
    skip_vector_tuples_list = list(skip_vectors.values())
    if not skip_vector_tuples_list: return scores # Should be caught by `if not skip_vectors`

    counts = Counter(skip_vector_tuples_list)
    # 來源：新大腦.pdf - EXT_GM19 min_occurrences_for_pattern (Page 54)
    # PDF: max(1, int(len(skip_vector_tuples_list) * 0.05))
    min_occurrences_for_pattern = max(1, int(len(skip_vector_tuples_list) * config.min_occurrences_for_pattern_factor))
    
    for skip_vec_tuple, count_val in counts.most_common(): # 來源：新大腦.pdf (Page 54)
        if count_val >= min_occurrences_for_pattern:
            # Strength could simply be normalized count
            # 來源：新大腦.pdf - EXT_GM19 pattern_strength (Page 54)
            pattern_strength = MathUtils.normalize_value(
                float(count_val),
                float(min_occurrences_for_pattern), # Min for a pattern to be considered
                float(len(skip_vector_tuples_list)), # Max possible occurrences (if all same pattern)
                clamp=True
            )
            dominant_skip_patterns_strength[skip_vec_tuple] = pattern_strength
        else: # Since most_common is sorted
            break # 來源：新大腦.pdf (Page 54)
            
    if not dominant_skip_patterns_strength: return scores # 來源：新大腦.pdf (Page 54)

    potential_numbers_to_place_set = BoardAnalyzerUtils.get_legal_values_for_placement(grid) # 來源：新大腦.pdf (Page 54)

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue # 來源：新大腦.pdf (Page 54)
            
            cell_max_pattern_score: float = 0.0 # 來源：新大腦.pdf (Page 54)
            for p_val_test in potential_numbers_to_place_set:
                if p_val_test not in base_positions: continue # 來源：新大腦.pdf (Page 54)
                
                base_r_test, base_c_test = base_positions[p_val_test]
                for current_skip_pattern, pattern_str in dominant_skip_patterns_strength.items():
                    skip_dr, skip_dc = current_skip_pattern
                    predicted_r = base_r_test + skip_dr
                    predicted_c = base_c_test + skip_dc

                    if predicted_r == r_idx and predicted_c == c_idx: # Cell matches pattern prediction for p_val_test
                        # 來源：新大腦.pdf (Page 54-55)
                        current_score_fit = pattern_str # Score is strength of the pattern it fits
                        if current_score_fit > cell_max_pattern_score:
                            cell_max_pattern_score = current_score_fit
            
            scores[r_idx, c_idx] = cell_max_pattern_score # Max score if multiple patterns/values fit this cell
            # 來源：新大腦.pdf (Page 55)

    return scores * config.weight


# 來源：新大腦.pdf - 26. EXT_GM20_Skip_Pattern_Confidence_Vec (Page 55)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM20強化建議
# Config for this (SkipPatternConfidenceConfig) was defined in PART 2
def EXT_GM20_Skip_Pattern_Confidence_Vec(
    grid: np.ndarray,
    config: SkipPatternConfidenceConfig,
    request_id: str | None = "N/A_GM20_SkipConf",
) -> np.ndarray:
    """
    (GM20-跳格模式信心度/規律性增強)
    核心規則:評估在空格填入數字是否能增強或完成已觀察到的全局跳格規律性, 特別是當這個填補能使遵循跳格模式的數字序列更完整或更具算術規律性時。
    啟發式類型:序列完成與模式確認(基於全局偏移量)
    輸出詮釋:分數越高表示填入該數字不僅符合跳格模式的幾何位置,且能使該模式下的數字序列在算術/序列意義上更為「自信」或「完整」。
    來源：新大腦.pdf - EXT_GM20_Skip_Pattern_Confidence_Vec (Page 55)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM20"
    logger.debug(
        f"Executing EXT_GM20_Skip_Pattern_Confidence_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 55)
        return scores

    # --- Initial Pattern Analysis (simplified from GM19, can be refactored into a shared utility) ---
    # 來源：新大腦.pdf - EXT_GM20 Initial Pattern Analysis (Page 55-56)
    revealed_numbers_info_gm20: List[Dict[str, Any]] = [] # 來源：新大腦.pdf (Page 55)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1 and grid[r, c] > 0: # 來源：新大腦.pdf (Page 56)
                revealed_numbers_info_gm20.append({'value': int(grid[r, c]), 'r': r, 'c': c})
    if not revealed_numbers_info_gm20: return scores # 來源：新大腦.pdf (Page 56)

    expected_max_num_gm20 = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols)) # 來源：新大腦.pdf (Page 56)
    base_pos_gm20: Dict[int, Tuple[int, int]] = { # 來源：新大腦.pdf (Page 56)
        k: ((k - 1) // cols, (k - 1) % cols) for k in range(1, expected_max_num_gm20 + 1) if ((k - 1) // cols) < rows
    }
    skip_vecs_initial_gm20: Dict[int, Tuple[int, int]] = {} # 來源：新大腦.pdf (Page 56)
    for rn in revealed_numbers_info_gm20:
        val = rn['value']
        if val in base_pos_gm20:
            skip_vecs_initial_gm20[val] = (rn['r'] - base_pos_gm20[val][0], rn['c'] - base_pos_gm20[val][1])

    dominant_patterns_details_gm20: List[Dict[str, Any]] = [] # List of {'skip':(dr,dc), 'values':[sorted_values], 'strength':float}
    # 來源：新大腦.pdf (Page 56)
    if skip_vecs_initial_gm20:
        skip_tuples_list_gm20 = list(skip_vecs_initial_gm20.values())
        if not skip_tuples_list_gm20 : return scores # Defensive check
        counts_gm20 = Counter(skip_tuples_list_gm20)
        min_occ_gm20 = max(1, int(len(skip_tuples_list_gm20) * config.min_occurrences_for_pattern_factor_gm20)) # 來源：新大腦.pdf (Page 56)
        
        for skip_v, count_v in counts_gm20.most_common(): # 來源：新大腦.pdf (Page 56)
            if count_v >= min_occ_gm20:
                pattern_vals = sorted([val for val, sv_tuple in skip_vecs_initial_gm20.items() if sv_tuple == skip_v])
                p_strength = MathUtils.normalize_value(
                    float(count_v), float(min_occ_gm20), float(len(skip_tuples_list_gm20)), clamp=True
                )
                dominant_patterns_details_gm20.append({'skip': skip_v, 'values': pattern_vals, 'strength': p_strength})
            else:
                break # 來源：新大腦.pdf (Page 56)
    # --- End Initial Pattern Analysis ---
    if not dominant_patterns_details_gm20: return scores # 來源：新大腦.pdf (Page 56)

    potential_nums_to_place_gm20 = BoardAnalyzerUtils.get_legal_values_for_placement(grid) # 來源：新大腦.pdf (Page 56)

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue # 來源：新大腦.pdf (Page 56)
            
            max_confidence_score_for_cell_gm20: float = 0.0 # 來源：新大腦.pdf (Page 56)
            for p_val_test in potential_nums_to_place_gm20:
                if p_val_test not in base_pos_gm20: continue # 來源：新大腦.pdf (Page 56)
                
                base_r_t, base_c_t = base_pos_gm20[p_val_test]
                current_max_conf_for_pval: float = 0.0 # 來源：新大腦.pdf (Page 56)

                for pattern_detail in dominant_patterns_details_gm20:
                    pat_skip_dr, pat_skip_dc = pattern_detail['skip']
                    pat_existing_vals = pattern_detail['values']  # sorted list
                    pat_strength = pattern_detail['strength']

                    predicted_r_for_pval = base_r_t + pat_skip_dr # 來源：新大腦.pdf (Page 57)
                    predicted_c_for_pval = base_c_t + pat_skip_dc # 來源：新大腦.pdf (Page 57)

                    if predicted_r_for_pval == r_idx and predicted_c_for_pval == c_idx:  # Geometrically fits
                        enhancement_factor = 0.5  # Base for geometric fit related to pattern strength
                                                # (PDF has 0.5, but this might mean 0.5 * pat_strength)
                                                # Let's consider it a multiplier to pat_strength later.
                                                # Or, it's an additive factor to a base score of pat_strength.
                                                # PDF: current_conf = pat_strength * enhancement_factor. Let's use this.
                                                # So, if only geometric fit, enhancement_factor = 1.0 for base.
                        
                        current_enhancement_factor = 1.0 # Base for geometric fit

                        # Check for arithmetic sequence enhancement
                        # 來源：新大腦.pdf - EXT_GM20 Arithmetic sequence enhancement (Page 57)
                        if len(pat_existing_vals) >= 1: # Need at least one existing number
                            temp_sequence_with_pval = sorted(pat_existing_vals + [p_val_test])
                            if len(temp_sequence_with_pval) >= 2:
                                diffs_in_temp_seq = np.diff(temp_sequence_with_pval) # diff gives array
                                if len(diffs_in_temp_seq) > 0:
                                    is_arithmetic_now = len(set(diffs_in_temp_seq)) == 1 # All diffs same
                                    first_diff = diffs_in_temp_seq[0]
                                    
                                    if is_arithmetic_now and not math.isclose(first_diff, 0): # It forms a new, consistent arithmetic sequence
                                        # 來源：新大腦.pdf (Page 57)
                                        current_enhancement_factor += config.arithmetic_enhancement_bonus 
                                        
                                        # Bonus if p_val_test is between min/max of pat_existing_vals (fills internal gap)
                                        # 來源：新大腦.pdf (Page 57)
                                        if len(pat_existing_vals) >=1: # Check to ensure min/max are valid
                                            min_existing = min(pat_existing_vals)
                                            max_existing = max(pat_existing_vals)
                                            if min_existing < p_val_test < max_existing :
                                                current_enhancement_factor += config.internal_gap_fill_bonus
                        
                        current_conf = pat_strength * current_enhancement_factor # 
                        if current_conf > current_max_conf_for_pval:
                            current_max_conf_for_pval = current_conf
                
                if current_max_conf_for_pval > max_confidence_score_for_cell_gm20:
                    max_confidence_score_for_cell_gm20 = current_max_conf_for_pval
            
            # Normalization: max_confidence_score_for_cell_gm20 can be > 1 if enhancement_factor > 1.
            # Max pat_strength is 1. Max enhancement can be 1.0 (base) + 0.4 + 0.1 = 1.5
            # So max_conf can be 1.5. Normalize to [0,1]
            # 來源：新大腦.pdf - EXT_GM20 Normalization (Page 57)
            scores[r_idx, c_idx] = MathUtils.normalize_value(
                max_confidence_score_for_cell_gm20, 0, 1.0 * (1.0 + config.arithmetic_enhancement_bonus + config.internal_gap_fill_bonus), clamp=True
            ) # Max possible heuristic value for current_max_conf_for_pval

    return scores * config.weight


# === Brain Core Dispatch Area ===
# 來源：新大腦.pdf - Brain Core Dispatch Area (Page 6) & Module Registration (Page 58)
# Using explicit type for the Callable for better clarity with Pydantic configs
BrainModuleCallableWithConfig = Callable[[np.ndarray, Any, str | None], np.ndarray] # grid, config, request_id
BrainModuleCallableNoConfig = Callable[[np.ndarray, str | None], np.ndarray] # grid, request_id

REGISTERED_MODULES_BRAIN: Dict[str, BrainModuleCallableWithConfig | BrainModuleCallableNoConfig] = {
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

# Default Pydantic configurations for each module that uses one.
# These would typically be loaded from analyzer_config or a central config store.
DEFAULT_MODULE_CONFIGS: Dict[str, BaseModel] = {
    "EXT_A2_Weighted_Proximity_Vec": WeightedProximityConfig(),
    "EXT_M3_Local_Heterogeneity_Vec": LocalHeterogeneityConfig(),
    "EXT_D3_Potential_Field_Vec": PotentialFieldConfig(),
    "EXT_F10_Discontinuity_Vec": DiscontinuityRepairConfig(),
    "EXT_P7_Pathfinding_Value_Vec": PathfindingValueConfig(),
    "EXT_R5_Resource_Control_Vec": ResourceControlConfig(),
    "EXT_GM1_Row_Control_Vec": LineControlConfig(), # Reuses LineControlConfig
    "EXT_GM2_Col_Flow_Vec": LineControlConfig(),   # Reuses LineControlConfig
    "EXT_GM3_Adv_Connected_Comp_Vec": ConnectedComponentConfig(),
    "EXT_GM4_Spatial_Auto_Corr_Vec": SpatialAutocorrelationConfig(),
    "EXT_GM5_Line_Completion_Vec": LineCompletionConfig(),
    "EXT_GM6_Symmetry_Potential_Vec": SymmetryPotentialConfig(),
    "EXT_GM7_Numeric_Gaps_Vec": NumericGapsConfig(),
    "EXT_GM8_Edge_Affinity_Vec": EdgeAffinityConfig(),
    "EXT_GM9_Center_Control_Vec": CenterControlConfig(),
    "EXT_GM10_Blocking_Value_Vec": BlockingValueConfig(),
    "EXT_GM11_Pair_Correlation_Vec": PairCorrelationConfig(),
    "EXT_GM12_Island_Analysis_Vec": IslandAnalysisConfig(),
    "EXT_GM13_Sequence_Diversity_Vec": SequenceDiversityConfig(),
    "EXT_GM14_Risk_Assessment_Vec": RiskAssessmentConfig(),
    "EXT_GM15_Information_Gain_Vec": InformationGainConfig(),
    "EXT_GM16_Harmonic_Centrality_Vec": HarmonicCentralityConfig(),
    "EXT_GM17_Entropy_Minimization_Vec": LocalEntropyMinimizationConfig(),
    "EXT_GM18_RL_Value_Est_Vec": RLValueEstimationConfig(),
    "EXT_GM19_Masked_Number_Skip_Pattern_Vec": SkipPatternConfig(),
    "EXT_GM20_Skip_Pattern_Confidence_Vec": SkipPatternConfidenceConfig(),
}


def get_module_score(
    module_name: str, grid: np.ndarray, config_override: BaseModel | None = None, request_id: str | None = None
) -> np.ndarray:
    """
    Retrieves and executes a specific scoring module from the registry.
    Args:
        module_name: The registered name of the module to execute.
        grid: The input numpy array representing the game board.
        config_override: Optional Pydantic configuration object to override default for the module.
        request_id: Optional request ID for logging.
    Returns:
        A numpy array containing the scores for each cell, as computed by the module.
        Returns a zero array of the same shape if the module is not found or an error occurs.
    來源：新大腦.pdf - get_module_score (Page 6)
    Enhanced to use config_override or default config.
    """
    effective_request_id = request_id if request_id else f"N/A_brain_dispatch_{module_name}"
    
    if module_name not in REGISTERED_MODULES_BRAIN:
        logger.error(
            f"Module {module_name} not found in REGISTERED_MODULES_BRAIN.",
            extra={"request_id": effective_request_id},
        )
        rows, cols = grid.shape if grid.ndim == 2 else (0,0)
        return np.zeros((rows, cols), dtype=float)

    module_func = REGISTERED_MODULES_BRAIN[module_name]
    
    # Determine config: use override if provided, else default for that module
    actual_config = config_override if config_override is not None else DEFAULT_MODULE_CONFIGS.get(module_name)

    if actual_config is None and module_name in DEFAULT_MODULE_CONFIGS: # Should not happen if DEFAULT_MODULE_CONFIGS is complete
        logger.warning(f"Default config not found for module {module_name}, but it expects one. Using base config.",
                       extra={"request_id": effective_request_id})
        actual_config = BaseModuleConfig() # Fallback, module might fail if it expects specific fields
    
    # Check if module actually expects a config based on its Pydantic config class existence
    # (More robust: inspect function signature, but for now assume if it's in DEFAULT_MODULE_CONFIGS it takes one)

    logger.info(
        f"Executing module: {module_name} with config: {actual_config.model_dump_json(indent=2) if actual_config else 'None'}",
        extra={"request_id": effective_request_id},
    )
    try:
        if module_name in DEFAULT_MODULE_CONFIGS: # Assumes modules with entry in DEFAULT_MODULE_CONFIGS take a config argument
            if actual_config is None: # Should be caught above
                 raise ValueError(f"Module {module_name} requires a config but none was provided or defaulted correctly.")
            score_grid = module_func(grid, config=actual_config, request_id=effective_request_id)
        else: 
            # This case is for modules that might not have/need a Pydantic config
            # However, our design makes all of them take one (even if it's just BaseModuleConfig)
            # For safety, if a module is registered but not in DEFAULT_MODULE_CONFIGS, assume it takes no config
            # This path should ideally not be taken if all modules are consistently defined.
            # Let's assume all our 26 modules will have a config, even if it's just BaseModuleConfig.
            # score_grid = module_func(grid, request_id=effective_request_id) # Fallback if no config expected
            # Re-evaluating: All modules are now designed to take a config object.
            # So, if actual_config is still None here, it's an issue.
             if actual_config is None:
                 logger.error(f"Internal error: Module {module_name} expected a config, but it's None.", extra={"request_id": effective_request_id})
                 rows, cols = grid.shape if grid.ndim == 2 else (0,0)
                 return np.zeros((rows, cols), dtype=float)
             score_grid = module_func(grid, config=actual_config, request_id=effective_request_id)


        if not isinstance(score_grid, np.ndarray) or score_grid.shape != grid.shape:
            logger.error(f"Module {module_name} returned invalid score_grid. Shape: {score_grid.shape if isinstance(score_grid, np.ndarray) else type(score_grid)}, Expected: {grid.shape}",
                           extra={"request_id": effective_request_id})
            rows, cols = grid.shape if grid.ndim == 2 else (0,0)
            return np.zeros((rows, cols), dtype=float)

        return score_grid
    except Exception as e:
        logger.error(
            f"Error executing module {module_name}: {e}",
            exc_info=True,
            extra={"request_id": effective_request_id},
        )
        rows, cols = grid.shape if grid.ndim == 2 else (0,0)
        return np.zeros((rows, cols), dtype=float)


# 來源：新大腦.pdf - Verification (Page 58-60)
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [RID:%(request_id)s]'
    )
    # Add a simple handler that includes request_id if not present in extra
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(lambda record: hasattr(record, 'request_id') or setattr(record, 'request_id', 'direct_run'))


    print("Verifying brain.py structure and all 26 modules...")
    dummy_grid_np = np.array([
        [1, 2, -1, 4, 5], 
        [-1, 5, -1, 8, -1], 
        [3, -1, 4, -1, 11],
        [12,13,-1,15,16],
        [-1,18,-1,20,-1]
    ], dtype=int)
    print(f"Created dummy grid (5x5):\n{dummy_grid_np}")

    total_modules = len(REGISTERED_MODULES_BRAIN)
    print(f"\nTotal modules registered: {total_modules}")
    assert total_modules == 26, f"Expected 26 modules, found {total_modules}"

    successful_runs = 0
    failed_modules = []

    for i, name in enumerate(REGISTERED_MODULES_BRAIN.keys()):
        print(f"\n--- Testing module {i+1}/{total_modules}: {name} ---")
        specific_config_override = None
        # Example: Override config for a specific module if needed for testing
        # if name == "EXT_A2_Weighted_Proximity_Vec":
        #     specific_config_override = WeightedProximityConfig(radius=1, weight=0.5)
        
        try:
            scores_array = get_module_score(name, dummy_grid_np, config_override=specific_config_override, request_id=f"test_{name}")
            print(f"Successfully called {name}. Output shape: {scores_array.shape}, dtype: {scores_array.dtype}")
            if scores_array.shape != dummy_grid_np.shape:
                print(f"ERROR: Shape mismatch for {name}! Expected {dummy_grid_np.shape}, Got {scores_array.shape}")
                failed_modules.append(name + " (shape mismatch)")
                continue
            if scores_array.dtype != float:
                print(f"ERROR: Dtype mismatch for {name}! Expected float, Got {scores_array.dtype}")
                failed_modules.append(name + " (dtype mismatch)")
                continue
            
            # Print a small sample of scores
            sample_scores = scores_array[0:min(3,scores_array.shape[0]), 0:min(3,scores_array.shape[1])]
            print(f"Sample scores for {name}:\n{sample_scores}")
            successful_runs += 1

        except Exception as e:
            print(f"ERROR executing module {name}: {e}")
            logger.exception(f"Exception during test of {name}")
            failed_modules.append(name + f" (execution error: {type(e).__name__})")
    
    print("\n--- Verification Summary ---")
    print(f"Successfully ran {successful_runs}/{total_modules} modules.")
    if failed_modules:
        print("Failed modules:")
        for f_mod in failed_modules:
            print(f"  - {f_mod}")
    else:
        print("All registered modules ran without immediate errors (shape/dtype checks passed).")

    print("\nbrain.py verification complete.")
    
    
    
    
    
    
    
    
    
    
    
    
    