# brain.py

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional, Any
import math
from collections import Counter, deque
import logging
import random

# --- Logging Configuration ---
logger = logging.getLogger(__name__)

# === Helper Utilities ===

class MathUtils:
    """
    提供通用數學工具，所有模組統一計算風格
    """
    
    def sigmoid(x: float, k: float = 1.0) -> float:
        """安全型 sigmoid，避免 overflow"""
        try:
            clamped_x = max(-700.0, min(700.0, -k * x))
            return 1 / (1 + math.exp(clamped_x))
        except OverflowError:
            return 0.0 if -k * x > 0 else 1.0

# === 大腦統一調度區 ===

# 所有自動註冊的 scoring modules 都會進這 dict
REGISTERED_MODULES_BRAIN: Dict[str, Callable] = {}

def get_module_score(module_name: str, grid: np.ndarray, **kwargs) -> np.ndarray:
    """
    大腦統一呼叫介面
    :param module_name: 註冊名稱（如 LIMIT_A6_FIXEDPOSITION_FN）
    :param grid: 評分盤面（np.ndarray）
    :param kwargs: 其它模組自定參數
    :return: np.ndarray 評分結果
    """
    mod_func = REGISTERED_MODULES_BRAIN.get(module_name)
    if mod_func is None:
        raise ValueError(f"Module '{module_name}' not found in brain.")
    return mod_func(grid, **kwargs)

# === 記憶體 / 特徵 / 數據層可在此擴充 ===
# EXAMPLE:
# MEMORY_DB: Dict[str, Any] = {}
# def remember(...): ...
# def recall(...): ...
# ...自訂記憶工具...

# --- 模組註冊示範（通常 auto_register.py 會自動註冊，不須手動加）---
# def sample_module(grid: np.ndarray, **kwargs):
#     return np.zeros_like(grid, dtype=float)
# REGISTERED_MODULES_BRAIN['LIMIT_SAMPLE_MODULE_FN'] = sample_module
 
   
   def normalize_value(value: float, min_val: float, max_val: float, clamp: bool = True) -> float:
       """
       Normalizes a value to the [0, 1] range.
       Handles cases where min_val equals max_val to prevent division by zero.
       Addresses Requirement 2.c (reasonable score distribution).
       """
       if math.isclose(max_val, min_val): #
           if math.isclose(value, min_val): #
               return 0.5 #
           elif value < min_val: #
               return 0.0 #
           else: # value > max_val (which is min_val)
               return 1.0 #
               
       normalized = (value - min_val) / (max_val - min_val) #
       if clamp: #
           return max(0.0, min(1.0, normalized)) #
       return normalized #
 
   
   def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
       """ Calculates Manhattan distance between two points (r, c). """ #
       return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) # Corrected from source 42
 
   
   def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
       """ Calculates Euclidean distance between two points (r, c). """ #
       return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) # Corrected from source 43
 
   
   def get_entropy(values: List[Any]) -> float:
       """Calculates Shannon entropy for a list of values.""" #
       if not values: #
           return 0.0 #
       counts = Counter(values) #
       total_count = len(values) #
       entropy = 0.0 #
       for count in counts.values(): #
           probability = count / total_count #
           entropy -= probability * math.log2(probability) #
       return entropy #
 
class BoardAnalyzerUtils:
   """
   Provides common board analysis utility functions.
   Used by modules to inspect grid neighborhoods, gradients, etc.
   """
   
   def get_neighborhood_values(grid: np.ndarray, r: int, c: int, radius: int = 1,
                               eight_connectivity: bool = True,
                               val_func: Callable[[int], Optional[float]] = lambda x_val: float(x_val) if x_val != -1 else None, #
                               include_center: bool = False) -> List[float]:
       """
       Retrieves values from the neighborhood of a cell.
       Supports configurable radius, connectivity, and value processing.
       """
       neighbors: List[float] = [] # Corrected from source 47
       rows, cols = grid.shape #
       for dr in range(-radius, radius + 1): #
           for dc in range(-radius, radius + 1): #
               if not include_center and dr == 0 and dc == 0: #
                   continue #
               if not eight_connectivity: #
                   if radius == 1 and abs(dr) + abs(dc) != 1: # More precise for 4-conn with radius 1
                       continue #
                   elif radius > 1 and abs(dr) + abs(dc) > radius: # For larger radius 4-conn (diamond shape)
                       continue #
 
               nr, nc = r + dr, c + dc #
           
               if 0 <= nr < rows and 0 <= nc < cols: #
                   processed_val = val_func(grid[nr, nc]) #
                   if processed_val is not None: #
                       neighbors.append(processed_val) #
       return neighbors #
 
   
   def get_value_gradient_at_cell(grid: np.ndarray, r: int, c: int,
                                  val_func: Callable[[int], float] = lambda x_val: float(x_val) if x_val != -1 else 0.0) -> Tuple[float, float]: #
       """
       Calculates an approximate gradient (Sobel-like) at a cell.
       Useful for modules analyzing value changes.
       """
       rows, cols = grid.shape #
       def safe_val(r_in, c_in): #
           if 0 <= r_in < rows and 0 <= c_in < cols: #
               return val_func(grid[r_in, c_in]) #
           return 0.0 #
       
       # Sobel-like operators for Gx and Gy
       gx = (safe_val(r-1, c+1) + 2*safe_val(r, c+1) + safe_val(r+1, c+1)) - \
            (safe_val(r-1, c-1) + 2*safe_val(r, c-1) + safe_val(r+1, c-1)) #
       gy = (safe_val(r+1, c-1) + 2*safe_val(r+1, c) + safe_val(r+1, c+1)) - \
            (safe_val(r-1, c-1) + 2*safe_val(r-1, c) + safe_val(r-1, c+1)) #
       return gx, gy #
 
   
   def find_sequences_in_line(line: List[int], min_len: int = 3,
                              check_arithmetic: bool = True, check_geometric: bool = False) -> List[List[int]]: #
       """
       Finds arithmetic or geometric sequences in a 1D list of numbers.
       -1 is treated as a potential gap or end of sequence.
       """
       sequences: List[List[int]] = [] #
       n = len(line) #
       if n < min_len: #
           return sequences #
 
       for i in range(n - min_len + 1): #
           if line[i] == -1: continue #
 
           # Arithmetic sequence check
           if check_arithmetic: #
               for j in range(i + 1, n): #
                   if line[j] == -1: break # Gap or end
                   diff = line[j] - line[i] #
                   if diff == 0 and line[i] != 0 : continue # Avoid constant sequences unless they are all zeros
                   
                   current_seq = [line[i], line[j]] #
                   expected_next = line[j] + diff #
                   for k in range(j + 1, n): #
                       if line[k] == expected_next: #
                           current_seq.append(line[k]) #
                           expected_next += diff #
                       elif line[k] != -1: # Sequence broken by a different number
                           break #
                       # If line[k] == -1, it's a gap, could potentially be part of a gapped sequence (more complex)
                       # For now, -1 breaks the simple sequence.
                   if len(current_seq) >= min_len: #
                       sequences.append(current_seq) #
           
           # Geometric sequence check (simplified, careful with division by zero and floating point)
           if check_geometric and line[i] != 0: # Cannot start geometric sequence with 0 unless all are 0
               for j in range(i + 1, n): #
                   if line[j] == -1 or line[j] == 0: break  #
                   if line[i] == 0 and line[j] != 0: break # 0, non-zero cannot be start of geom seq
                   # Basic check for integer ratio
                   if line[i] != 0 and line[j] % line[i] != 0 and line[i] % line[j] != 0 :  #
                       if not math.isclose(line[j]/line[i], round(line[j]/line[i])): # If ratio isn't integer like
                            continue #
 
                   ratio = line[j] / line[i] if line[i] != 0 else (1.0 if line[j] == 0 else float('inf')) # Handle division by zero
                   if math.isclose(ratio, 1.0) and line[i] != line[j]: continue # Avoid constant sequences
                   if ratio == float('inf'): continue #
 
                   current_seq = [line[i], line[j]] #
                   expected_next_float = float(line[j]) * ratio #
                   for k in range(j + 1, n): #
                       if math.isclose(float(line[k]), expected_next_float): #
                           current_seq.append(line[k]) #
                           expected_next_float *= ratio #
                       elif line[k] != -1: #
                           break #
                   if len(current_seq) >= min_len: #
                       sequences.append(current_seq) #
       return sequences #
 
def get_card_max_value_from_grid_dimensions(grid_shape: Tuple[int, int]) -> int:
   """ Calculates the maximum possible number on the card based on its dimensions. """
   rows, cols = grid_shape #
   if rows == 0 or cols == 0: return 0 #
   return rows * cols #
 
def get_all_possible_numbers_for_grid(grid_shape: Tuple[int, int]) -> set[int]:
   """ Returns a set of all numbers that could theoretically appear on a grid of given dimensions. """
   max_val = get_card_max_value_from_grid_dimensions(grid_shape) #
   if max_val == 0: #
       return set() #
   return set(range(1, max_val + 1)) #
 
def get_legal_values_for_placement(grid: np.ndarray) -> set[int]:
   """
   Determines the set of numbers that can be legally placed onto an empty cell in the grid.
   This adheres to the rule: numbers are 1 to R*C, and no positive number can be repeated.
   (Requirement 1.c)
   """
   if grid.size == 0: #
       return set() #
       
   rows, cols = grid.shape #
   all_possible_on_this_grid = get_all_possible_numbers_for_grid((rows, cols)) #
   
   used_positive_values_on_board = set(int(v) for v in grid.flatten() if v != -1 and v > 0) #
   
   legal_placements = all_possible_on_this_grid - used_positive_values_on_board #
   return legal_placements #
 
# --- Scoring Module Definitions ---
 
def EXT_A2_Weighted_Proximity_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (A2 - 加權鄰近性)
   核心規則：評估空格周圍已填數字的接近程度及其值的影響。
   目的：偏好靠近高價值數字或數字密集區域的空格。
   啟發式類型: 空間鄰近性
   輸出詮釋: 分數越高表示鄰近效應越強 (受周圍數字的值與密度影響)
   """
   # Using a default request_id if None for logging consistency
   effective_request_id = request_id or "N/A_brain_A2"
   logger.debug("Executing EXT_A2_Weighted_Proximity_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   radius = 2  # Consider a neighborhood radius
   value_weight_factor = 0.1  # Weight factor for the value of neighboring numbers
   distance_decay_factor = 1.5 # Higher value means faster decay with distance
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           proximity_score = 0.0 #
           # Iterate through neighborhood to get values and distances
           for dr in range(-radius, radius + 1): #
               for dc in range(-radius, radius + 1): #
                   if dr == 0 and dc == 0: continue # Skip center cell
 
                   nr, nc = r_idx + dr, c_idx + dc #
                   if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1: #
                       dist = MathUtils.manhattan_distance((r_idx, c_idx), (nr, nc)) #
                       if dist == 0: dist = 1 # Should not happen due to skip center cell, but as safeguard
                       
                       # Score contribution: value of neighbor * value_weight, decayed by distance
                       # Inverse distance decay: 1 / dist^decay_factor
                       score_contribution = (grid[nr, nc] * value_weight_factor) / (dist ** distance_decay_factor) #
                       proximity_score += score_contribution #
           
           # Normalization based on a heuristic max score
           num_neighbors_in_radius = (2*radius+1)**2 - 1 #
           # A rough upper bound for proximity_score, assuming max value (rows*cols) at distance 1 for all neighbors
           heuristic_max_score = num_neighbors_in_radius * (rows*cols) * value_weight_factor #
           if heuristic_max_score > 0: #
               scores[r_idx, c_idx] = MathUtils.normalize_value(proximity_score, 0, heuristic_max_score, clamp=True) #
           else: #
               scores[r_idx, c_idx] = 0.0 #
               
   return scores #
 
# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
   "EXT_A2_Weighted_Proximity_Vec": EXT_A2_Weighted_Proximity_Vec,
   # Other modules will be added here
}
 
# --- Main Brain Interface Function ---
def get_module_score(module_name: str, grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   Retrieves and executes a specific scoring module from the registry.
   Args:
       module_name: The registered name of the module to execute.
       grid: The input numpy array representing the game board.
       request_id: Optional request ID for logging and traceability.
   Returns:
       A numpy array containing the scores for each cell, as computed by the module.
       Returns a zero array of the same shape if the module is not found or an error occurs.
   """
   effective_request_id = request_id or "N/A_brain_dispatch"
   if module_name not in REGISTERED_MODULES_BRAIN:
       logger.error(f"Module {module_name} not found in REGISTERED_MODULES_BRAIN.", extra={'request_id': effective_request_id})
       rows, cols = grid.shape
       return np.zeros((rows, cols), dtype=float)
       
   module_func = REGISTERED_MODULES_BRAIN[module_name]
   logger.info(f"Executing module: {module_name}", extra={'request_id': effective_request_id})
   try:
       score_grid = module_func(grid, effective_request_id) # Pass grid and request_id
       return score_grid
   except Exception as e:
       logger.error(f"Error executing module {module_name}: {e}", exc_info=True, extra={'request_id': effective_request_id})
       rows, cols = grid.shape
       return np.zeros((rows, cols), dtype=float) # Return zero grid on error
 
def EXT_M3_Local_Heterogeneity_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (M3 - 局部異質性)
   核心規則：評估空格周圍數字的多樣性。
   目的：偏好周圍數字分佈更隨機、更少重複的空格。
   啟發式類型: 分佈統計 (基於熵)
   輸出詮釋: 分數越高表示周圍環境的數字異質性越高 (熵越大)
   """
   effective_request_id = request_id or "N/A_brain_M3"
   logger.debug("Executing EXT_M3_Local_Heterogeneity_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   radius = 1  # Neighborhood radius for heterogeneity calculation
   min_neighbors_for_robust_score = 2 # Minimum neighbors to calculate meaningful entropy
 
   # Determine the set of all possible values that *could* appear in any cell
   # This is used for calculating theoretical maximum entropy for normalization
   all_possible_values_in_game = get_all_possible_numbers_for_grid(grid.shape) #
   if not all_possible_values_in_game: #
       # If no possible values (e.g., 0x0 grid), max entropy is ill-defined, scores remain 0
       return scores #
   
   # Theoretical maximum entropy for normalization: log2 of the number of unique possible values.
   # Add 1 to avoid log2(0) or log2(1)=0 if only one value is possible (though rare for this game)
   # If len(all_possible_values_in_game) is 1, log2(1) = 0. If 0, it's handled.
   # Max entropy is log2(N) where N is the number of distinct symbols.
   if len(all_possible_values_in_game) > 1: #
       max_theoretical_entropy = math.log2(len(all_possible_values_in_game)) #
   elif len(all_possible_values_in_game) == 1: #
       max_theoretical_entropy = math.log2(2) # Avoid log2(1)=0, give some scale, or just 0 if N=1 means no diversity
                                              # Let's consider log2(1) as 0 meaning no diversity is possible.
                                              # Or, if only one value can exist, any neighborhood with it has 0 entropy.
                                              # For normalization, a non-zero max_entropy is better. Let's use log2(count) or 1.0 if count < 2.
       max_theoretical_entropy = 1.0 # if only one possible value, entropy is 0. Max must be > 0 for normalization.
                                      # This means if only one number can be on board, this score is less meaningful.
   else: # No possible values (empty set)
       max_theoretical_entropy = 1.0 # Fallback, though handled by early exit
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           # Get values in the neighborhood (excluding -1s and the center cell itself)
           neighbor_values = BoardAnalyzerUtils.get_neighborhood_values(
               grid, r_idx, c_idx, radius=radius, eight_connectivity=True,
               val_func=lambda x_val: int(x_val) if x_val != -1 else None, # Process as ints, filter -1
               include_center=False
           ) #
 
           if len(neighbor_values) < min_neighbors_for_robust_score: #
               scores[r_idx, c_idx] = 0.0  # Not enough info for a robust entropy score
               continue #
 
           # Calculate Shannon entropy for the neighborhood
           current_entropy = MathUtils.get_entropy(neighbor_values) #
 
           # Normalize the entropy score
           # The actual max entropy for a small neighborhood is log2(k) where k is number of distinct items.
           # Normalizing by max_theoretical_entropy based on all game values gives a global perspective.
           if max_theoretical_entropy > 0: #
               normalized_score = current_entropy / max_theoretical_entropy #
               scores[r_idx, c_idx] = MathUtils.normalize_value(normalized_score, 0, 1, clamp=True) # Already 0-1 conceptually
           else: #
               scores[r_idx, c_idx] = 0.0 # Should not happen if max_theoretical_entropy is handled as >= 1.0
               
   return scores #
 
# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
   "EXT_A2_Weighted_Proximity_Vec": EXT_A2_Weighted_Proximity_Vec,
   "EXT_M3_Local_Heterogeneity_Vec": EXT_M3_Local_Heterogeneity_Vec,
   # Other modules will be added here
}
 
def EXT_D3_Potential_Field_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (D3 - 位勢場分析)
   核心規則：將盤面上的數字視為「電荷」，空格則根據其位置的「綜合位勢」來評分。
   目的：偏好位於受高價值數字「吸引」或低價值數字「排斥」(如果設計如此)區域的空格。
          此處簡化為僅正向吸引。
   啟發式類型: 物理類比 (類似靜電場或重力場)
   輸出詮釋: 分數越高表示該空格受到周圍數字的正向「位勢影響」越大
   """
   effective_request_id = request_id or "N/A_brain_D3"
   logger.debug("Executing EXT_D3_Potential_Field_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   decay_exponent = 1.5  # How quickly influence decays with distance (e.g., 1 for 1/r, 2 for 1/r^2)
   max_influence_radius = 3 # Consider numbers within this Manhattan distance
 
   # Max possible value on this specific grid for normalization scaling
   max_possible_val_on_grid = get_card_max_value_from_grid_dimensions((rows, cols)) #
   if max_possible_val_on_grid == 0: return scores # Fallback
 
   # Heuristic maximum potential for normalization:
   # Sum of max_value / (min_dist^decay) for all cells in radius.
   # This is a very rough upper bound.
   num_cells_in_radius_approx = (2 * max_influence_radius + 1)**2 -1
   heuristic_max_potential = num_cells_in_radius_approx * (max_possible_val_on_grid / (1**decay_exponent)) # Assuming min dist 1
   if heuristic_max_potential == 0: heuristic_max_potential = 1.0 # Avoid division by zero if somehow calculated as 0
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           current_cell_potential = 0.0 #
           for nr in range(rows): #
               for nc in range(cols): #
                   if grid[nr, nc] != -1:  # If it's a filled cell (a "charge")
                       num_val = grid[nr, nc] #
                       if num_val <= 0: continue # Consider only positive charges for attraction
 
                       dist = MathUtils.manhattan_distance((r_idx, c_idx), (nr, nc)) #
                       
                       if dist == 0: continue # Should not happen if only scoring empty cells
                       if dist > max_influence_radius: continue # Limit influence range
 
                       # Potential = charge_value / distance^decay_exponent
                       potential_contribution = num_val / (dist ** decay_exponent) #
                       current_cell_potential += potential_contribution #
           
           scores[r_idx, c_idx] = MathUtils.normalize_value(current_cell_potential, 0, heuristic_max_potential, clamp=True) #
           
   return scores #
 
# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
   "EXT_A2_Weighted_Proximity_Vec": EXT_A2_Weighted_Proximity_Vec,
   "EXT_M3_Local_Heterogeneity_Vec": EXT_M3_Local_Heterogeneity_Vec,
   "EXT_D3_Potential_Field_Vec": EXT_D3_Potential_Field_Vec,
   # Other modules will be added here
}
 
def EXT_F10_Discontinuity_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (F10 - 不連續性修復/序列完成度)
   核心規則：評估在空格填入數字後，是否能修復或完成某個方向上的數字序列 (例如等差)。
   目的：偏好那些能夠「承先啟後」，使斷裂的序列得以延續或形成的空格。
   啟發式類型: 序列與模式識別
   輸出詮釋: 分數越高表示該空格填入某個合法數字後，能形成或延長的序列越長/越重要
   """
   effective_request_id = request_id or "N/A_brain_F10"
   logger.debug("Executing EXT_F10_Discontinuity_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   legal_values_for_placement = get_legal_values_for_placement(grid) #
   if not legal_values_for_placement: #
       return scores # No legal values to try
 
   min_sequence_len_to_score = 3 # Minimum length of a sequence to be considered significant
 
   # Heuristic max length for normalization (length of the longest possible line)
   heuristic_max_len = float(max(rows, cols)) #
   if heuristic_max_len < min_sequence_len_to_score : #
       heuristic_max_len = float(min_sequence_len_to_score) # Ensure max_len is at least min_len for normalization
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           max_len_contribution_for_this_cell = 0.0 # Max sequence length this cell can contribute to
 
           for val_to_try in legal_values_for_placement: #
               # Temporarily place the value
               temp_grid = grid.copy() #
               temp_grid[r_idx, c_idx] = val_to_try #
 
               current_val_max_len = 0.0 # Max sequence length with val_to_try
 
               # 1. Check Row
               row_line = list(temp_grid[r_idx, :]) #
               sequences_in_row = BoardAnalyzerUtils.find_sequences_in_line(row_line, min_len=min_sequence_len_to_score) #
               for seq in sequences_in_row: #
                   if val_to_try in seq: # Check if the placed value is part of this new/extended sequence
                       # Ensure the sequence actually passes through the current cell (r_idx, c_idx)
                       # by checking if its original position (now filled with val_to_try) is part of the sequence logic
                       try: #
                           # find_sequences_in_line doesn't directly give index, but if val_to_try is in seq, it contributed
                           current_val_max_len = max(current_val_max_len, len(seq)) #
                       except ValueError: # Should not happen if val_to_try in seq
                           pass #
 
               # 2. Check Column
               col_line = list(temp_grid[:, c_idx]) #
               sequences_in_col = BoardAnalyzerUtils.find_sequences_in_line(col_line, min_len=min_sequence_len_to_score) #
               for seq in sequences_in_col: #
                   if val_to_try in seq: #
                        current_val_max_len = max(current_val_max_len, len(seq)) #
               
               # 3. Check Diagonals
               # Main diagonal (top-left to bottom-right)
               diag1_line = list(np.diag(temp_grid, k=c_idx-r_idx)) #
               sequences_in_diag1 = BoardAnalyzerUtils.find_sequences_in_line(diag1_line, min_len=min_sequence_len_to_score) #
               for seq in sequences_in_diag1: #
                   if val_to_try in seq: #
                       current_val_max_len = max(current_val_max_len, len(seq)) #
 
               # Anti-diagonal (top-right to bottom-left)
               # For anti_diag, we can flip the grid and take the main diagonal
               flipped_temp_grid = np.fliplr(temp_grid) #
               # New column index for the flipped grid
               flipped_c_idx = cols - 1 - c_idx #
               diag2_line = list(np.diag(flipped_temp_grid, k=flipped_c_idx-r_idx)) #
               sequences_in_diag2 = BoardAnalyzerUtils.find_sequences_in_line(diag2_line, min_len=min_sequence_len_to_score) #
               for seq in sequences_in_diag2: #
                   if val_to_try in seq: #
                       current_val_max_len = max(current_val_max_len, len(seq)) #
               
               if current_val_max_len >= min_sequence_len_to_score: #
                   max_len_contribution_for_this_cell = max(max_len_contribution_for_this_cell, current_val_max_len) #
 
           # Normalize the max length contribution for this cell
           if heuristic_max_len > 0: #
                scores[r_idx, c_idx] = MathUtils.normalize_value(max_len_contribution_for_this_cell, 0, heuristic_max_len, clamp=True) #
           else: #
                scores[r_idx, c_idx] = 0.0 #
               
   return scores #
 
# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
   "EXT_A2_Weighted_Proximity_Vec": EXT_A2_Weighted_Proximity_Vec,
   "EXT_M3_Local_Heterogeneity_Vec": EXT_M3_Local_Heterogeneity_Vec,
   "EXT_D3_Potential_Field_Vec": EXT_D3_Potential_Field_Vec,
   "EXT_F10_Discontinuity_Vec": EXT_F10_Discontinuity_Vec,
   # Other modules will be added here
}
 
def EXT_P7_Pathfinding_Value_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (P7 - 路徑尋找價值)
   核心規則：評估在空格填入數字後，形成連接到其他現有數字的路徑的價值。
   目的：偏好那些能夠「橋接」盤面區域，或連接到高價值目標的空格。
   啟發式類型: 連通性與圖論
   輸出詮釋: 分數越高表示該空格填入某數字後，能形成更有價值的路徑 (考慮路徑長度與連接到的數字大小)
   """
   effective_request_id = request_id or "N/A_brain_P7"
   logger.debug("Executing EXT_P7_Pathfinding_Value_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   legal_values_for_placement = get_legal_values_for_placement(grid) #
   if not legal_values_for_placement: #
       return scores #
 
   max_path_search_depth = 4  # Max length of path to search
   path_value_decay_factor = 1.0  # Decay for path length, e.g., val / (len^decay)
 
   max_possible_val_on_grid = get_card_max_value_from_grid_dimensions((rows, cols)) #
   if max_possible_val_on_grid == 0: max_possible_val_on_grid = 1.0 # Avoid zero
 
   # Heuristic max score for normalization:
   # Assume connecting to max_val at dist 1 from multiple directions up to depth
   # A very loose upper bound: (max_depth_search_radius_squared_area) * max_val / (1^decay)
   heuristic_max_path_score = ( (2*max_path_search_depth+1)**2 * max_possible_val_on_grid / (1**path_value_decay_factor) )
   if heuristic_max_path_score == 0: heuristic_max_path_score = 1.0 #
 
 
   for r_start in range(rows): #
       for c_start in range(cols): #
           if grid[r_start, c_start] != -1:  # Only score empty cells (where we might place a number)
               continue #
 
           max_score_for_this_cell = 0.0 #
 
           for val_to_try in legal_values_for_placement: #
               # Simulate placing val_to_try. The original grid is used to find *existing* numbers.
               # The path itself can traverse other empty cells.
               
               current_placement_path_score = 0.0 #
               
               q = deque([((r_start, c_start), 0)])  # ((r, c), current_path_length_from_start)
               visited_for_bfs = set([(r_start, c_start)]) # Visited for this specific BFS starting at (r_start, c_start)
 
               # The BFS explores from the cell (r_start, c_start) *as if* val_to_try is placed there.
               # It seeks paths through *other currently empty cells* to reach *existing numbers on the original grid*.
               
               head_count = 0 # Safety break for BFS
               max_bfs_steps = rows * cols * len(legal_values_for_placement) # Generous safety break
 
               while q and head_count < max_bfs_steps : #
                   head_count += 1 #
                   (curr_r, curr_c), path_len = q.popleft() #
 
                   # Explore neighbors (4-connectivity)
                   for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: #
                       next_r, next_c = curr_r + dr, curr_c + dc #
 
                       if 0 <= next_r < rows and 0 <= next_c < cols: #
                           # If neighbor is an *existing number* on the original grid
                           if grid[next_r, next_c] != -1:
                               reached_val = grid[next_r, next_c] #
                               effective_path_len = path_len + 1 #
                               current_placement_path_score += reached_val / (effective_path_len ** path_value_decay_factor) #
                               # Do not add this to visited_for_bfs or queue, as it's a terminal number for this path segment.
                           
                           # If neighbor is an *empty cell* (excluding starting cell if path_len is 0)
                           # and path is not too long, and not yet visited in this BFS
                           elif (next_r, next_c) not in visited_for_bfs and \
                                grid[next_r, next_c] == -1 and \
                                path_len + 1 < max_path_search_depth :
                               
                               visited_for_bfs.add((next_r, next_c)) #
                               q.append(((next_r, next_c), path_len + 1)) #
               
               if current_placement_path_score > max_score_for_this_cell: #
                   max_score_for_this_cell = current_placement_path_score #
           
           scores[r_start, c_start] = MathUtils.normalize_value(max_score_for_this_cell, 0, heuristic_max_path_score, clamp=True) #
           
   return scores #
 
# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
   "EXT_A2_Weighted_Proximity_Vec": EXT_A2_Weighted_Proximity_Vec,
   "EXT_M3_Local_Heterogeneity_Vec": EXT_M3_Local_Heterogeneity_Vec,
   "EXT_D3_Potential_Field_Vec": EXT_D3_Potential_Field_Vec,
   "EXT_F10_Discontinuity_Vec": EXT_F10_Discontinuity_Vec,
   "EXT_P7_Pathfinding_Value_Vec": EXT_P7_Pathfinding_Value_Vec,
   # Other modules will be added here
}
 
def EXT_R5_Resource_Control_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (R5 - 資源控制)
   核心規則：從資源控制角度評估填補位置的策略價值。資源可包括行/列的完成度、
              對高價值數字的獲取潛力等。
   目的：偏好那些能夠鞏固盤面控制權，或獲取潛在高價值數字的空格。
   啟發式類型: 策略與控制
   輸出詮釋: 分數越高表示該空格在填入數字後，對資源的控制（如行列完成度、高價值數字佔據）越強
   """
   effective_request_id = request_id or "N/A_brain_R5"
   logger.debug("Executing EXT_R5_Resource_Control_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   potential_numbers_to_place = list(get_legal_values_for_placement(grid)) #
   # If no numbers can be placed, this module might score based on structural control only.
   # However, value_capture_score would be low or zero.
 
   max_possible_val_on_grid = get_card_max_value_from_grid_dimensions((rows, cols)) #
   if max_possible_val_on_grid == 0 : max_possible_val_on_grid = 1.0 # Avoid issues with 0 max val
 
   # Determine a representative high value that could be placed for value_capture_score
   hypothetical_high_val_placed = 0.0 #
   if potential_numbers_to_place: #
       hypothetical_high_val_placed = np.max(potential_numbers_to_place) #
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           # 1. Row Completeness Score
           num_filled_in_row = np.count_nonzero(grid[r_idx, :] != -1) #
           row_completeness_score = (num_filled_in_row + 1) / cols if cols > 0 else 0 #
 
           # 2. Column Completeness Score
           num_filled_in_col = np.count_nonzero(grid[:, c_idx] != -1) #
           col_completeness_score = (num_filled_in_col + 1) / rows if rows > 0 else 0 #
           
           # 3. Value Capture Score (potential to place a high value)
           # This score reflects the value of placing the best possible *available* number here.
           value_capture_score = 0.0 #
           if hypothetical_high_val_placed > 0 and max_possible_val_on_grid > 0: #
               value_capture_score = MathUtils.normalize_value(hypothetical_high_val_placed, 1, max_possible_val_on_grid, clamp=True) #
           
           # Combine scores: Example weights, can be tuned
           # Weights for row control, column control, value capture
           w_row = 0.3 #
           w_col = 0.3 #
           w_val = 0.4 #
           
           combined_score = (w_row * row_completeness_score +
                             w_col * col_completeness_score +
                             w_val * value_capture_score) #
           
           # The combined score should already be in [0,1] if weights sum to 1 and components are [0,1]
           # Normalizing again ensures it, or handles cases where weights don't sum to 1.
           scores[r_idx, c_idx] = MathUtils.normalize_value(combined_score, 0, 1.0, clamp=True) #
 
   return scores #
 
# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
   "EXT_A2_Weighted_Proximity_Vec": EXT_A2_Weighted_Proximity_Vec,
   "EXT_M3_Local_Heterogeneity_Vec": EXT_M3_Local_Heterogeneity_Vec,
   "EXT_D3_Potential_Field_Vec": EXT_D3_Potential_Field_Vec,
   "EXT_F10_Discontinuity_Vec": EXT_F10_Discontinuity_Vec,
   "EXT_P7_Pathfinding_Value_Vec": EXT_P7_Pathfinding_Value_Vec,
   "EXT_R5_Resource_Control_Vec": EXT_R5_Resource_Control_Vec,
   # Other modules will be added here
}
 
def EXT_GM1_Row_Control_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM1 - 行控制力)
   核心規則：評估在特定空格填入數字後，對該行的完成度、數值總和或序列形成的貢獻。
   目的：偏好那些能增強單行控制力或形成有價值行模式的填補。
   啟發式類型: 線性結構控制 (行)
   輸出詮釋: 分數越高表示對該行的潛在控制力或完成度越強
   """
   effective_request_id = request_id or "N/A_brain_GM1"
   logger.debug("Executing EXT_GM1_Row_Control_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   potential_numbers_to_place = list(get_legal_values_for_placement(grid)) #
   # If no numbers can be placed, sum_score and seq_score contributions will be minimal or zero.
   
   avg_potential_num_to_place = 0.0 #
   if potential_numbers_to_place: #
       avg_potential_num_to_place = np.mean(potential_numbers_to_place) #
 
   max_val_board = get_card_max_value_from_grid_dimensions((rows, cols)) #
   if max_val_board == 0: max_val_board = 1.0 # Avoid issues if grid is tiny
 
   for r_idx in range(rows): #
       current_row_values_list = [val for val in grid[r_idx, :] if val != -1] #
       num_filled_in_row = len(current_row_values_list) #
       sum_current_row_values = sum(current_row_values_list) #
 
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           # 1. Density Score: How full the row will be
           density_score = (num_filled_in_row + 1.0) / cols if cols > 0 else 0.0 #
 
           # 2. Value Contribution Score (Sum Score)
           potential_row_sum = sum_current_row_values + avg_potential_num_to_place #
           # Max possible row sum: cols * max_val_board (simplistic heuristic)
           heuristic_max_row_sum = float(cols * max_val_board) #
           sum_score = 0.0 #
           if heuristic_max_row_sum > 0 : #
               sum_score = MathUtils.normalize_value(potential_row_sum, 0, heuristic_max_row_sum, clamp=True) #
           
           # 3. Sequence Completion Score (Simplified: mending a 3-part arithmetic sequence)
           seq_score = 0.0 #
           if 0 < c_idx < cols - 1:  # Cell must have two horizontal neighbors to mend a sequence
               prev_val = grid[r_idx, c_idx - 1] #
               next_val = grid[r_idx, c_idx + 1] #
               if prev_val != -1 and next_val != -1: # Both neighbors must be filled
                   if (prev_val + next_val) % 2 == 0: # If their sum is even, an integer average exists
                       mend_val = (prev_val + next_val) // 2 #
                       # Check if this mend_val is a legal placement AND forms a non-constant sequence
                       if mend_val in potential_numbers_to_place and abs(mend_val - prev_val) > 0: #
                           seq_score = 0.75 # Give a significant score for mending a sequence
           elif (c_idx == 0 and cols > 1 and grid[r_idx, c_idx+1] != -1 and grid[r_idx, c_idx+1] - avg_potential_num_to_place != 0) or \
                (c_idx == cols -1 and cols > 1 and grid[r_idx, c_idx-1] != -1 and avg_potential_num_to_place - grid[r_idx, c_idx-1] != 0):
                # Potential to start/end a 2-number sequence (less score than mending)
                # This is a very rough heuristic, could be expanded
                seq_score = 0.25 #
 
           # Combine scores (weights can be tuned)
           w_density = 0.4 #
           w_sum = 0.3 #
           w_seq = 0.3 #
           combined_score = (w_density * density_score +
                             w_sum * sum_score +
                             w_seq * seq_score) #
           
           scores[r_idx, c_idx] = MathUtils.normalize_value(combined_score, 0, 1.0, clamp=True) # Ensure final is 0-1
 
   return scores #
 
# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
   "EXT_A2_Weighted_Proximity_Vec": EXT_A2_Weighted_Proximity_Vec,
   "EXT_M3_Local_Heterogeneity_Vec": EXT_M3_Local_Heterogeneity_Vec,
   "EXT_D3_Potential_Field_Vec": EXT_D3_Potential_Field_Vec,
   "EXT_F10_Discontinuity_Vec": EXT_F10_Discontinuity_Vec,
   "EXT_P7_Pathfinding_Value_Vec": EXT_P7_Pathfinding_Value_Vec,
   "EXT_R5_Resource_Control_Vec": EXT_R5_Resource_Control_Vec,
   "EXT_GM1_Row_Control_Vec": EXT_GM1_Row_Control_Vec,
   # Other modules will be added here
}
 
def EXT_GM2_Col_Flow_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM2 - 列流動性/列控制力)
   核心規則：評估在特定空格填入數字後，對該列的完成度、數值總和或序列形成的貢獻。
   目的：偏好那些能增強單列控制力或形成有價值列模式的填補。
   啟發式類型: 線性結構控制 (列)
   輸出詮釋: 分數越高表示對該列的潛在控制力或完成度越強
   """
   effective_request_id = request_id or "N/A_brain_GM2"
   logger.debug("Executing EXT_GM2_Col_Flow_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   potential_numbers_to_place = list(get_legal_values_for_placement(grid)) #
   
   avg_potential_num_to_place = 0.0 #
   if potential_numbers_to_place: #
       avg_potential_num_to_place = np.mean(potential_numbers_to_place) #
 
   max_val_board = get_card_max_value_from_grid_dimensions((rows, cols)) #
   if max_val_board == 0: max_val_board = 1.0 #
 
   for c_idx in range(cols): #
       current_col_values_list = [val for val in grid[:, c_idx] if val != -1] #
       num_filled_in_col = len(current_col_values_list) #
       sum_current_col_values = sum(current_col_values_list) #
 
       for r_idx in range(rows): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           # 1. Density Score: How full the column will be
           density_score = (num_filled_in_col + 1.0) / rows if rows > 0 else 0.0 #
 
           # 2. Value Contribution Score (Sum Score)
           potential_col_sum = sum_current_col_values + avg_potential_num_to_place #
           heuristic_max_col_sum = float(rows * max_val_board) #
           sum_score = 0.0 #
           if heuristic_max_col_sum > 0: #
               sum_score = MathUtils.normalize_value(potential_col_sum, 0, heuristic_max_col_sum, clamp=True) #
           
           # 3. Sequence Completion Score (Simplified: mending a 3-part arithmetic sequence)
           seq_score = 0.0 #
           if 0 < r_idx < rows - 1:  # Cell must have two vertical neighbors
               prev_val = grid[r_idx - 1, c_idx] #
               next_val = grid[r_idx + 1, c_idx] #
               if prev_val != -1 and next_val != -1: #
                   if (prev_val + next_val) % 2 == 0: #
                       mend_val = (prev_val + next_val) // 2 #
                       if mend_val in potential_numbers_to_place and abs(mend_val - prev_val) > 0: #
                           seq_score = 0.75 #
           elif (r_idx == 0 and rows > 1 and grid[r_idx+1, c_idx] != -1 and grid[r_idx+1, c_idx] - avg_potential_num_to_place != 0) or \
                (r_idx == rows -1 and rows > 1 and grid[r_idx-1, c_idx] != -1 and avg_potential_num_to_place - grid[r_idx-1, c_idx] != 0): #
                seq_score = 0.25 #
 
           # Combine scores
           w_density = 0.4 #
           w_sum = 0.3 #
           w_seq = 0.3 #
           combined_score = (w_density * density_score +
                             w_sum * sum_score +
                             w_seq * seq_score) #
           
           scores[r_idx, c_idx] = MathUtils.normalize_value(combined_score, 0, 1.0, clamp=True) #
 
   return scores #
 
# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
   "EXT_A2_Weighted_Proximity_Vec": EXT_A2_Weighted_Proximity_Vec,
   "EXT_M3_Local_Heterogeneity_Vec": EXT_M3_Local_Heterogeneity_Vec,
   "EXT_D3_Potential_Field_Vec": EXT_D3_Potential_Field_Vec,
   "EXT_F10_Discontinuity_Vec": EXT_F10_Discontinuity_Vec,
   "EXT_P7_Pathfinding_Value_Vec": EXT_P7_Pathfinding_Value_Vec,
   "EXT_R5_Resource_Control_Vec": EXT_R5_Resource_Control_Vec,
   "EXT_GM1_Row_Control_Vec": EXT_GM1_Row_Control_Vec,
   "EXT_GM2_Col_Flow_Vec": EXT_GM2_Col_Flow_Vec,
   # Other modules will be added here
}
 
def EXT_GM3_Adv_Connected_Comp_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM3 - 高級連通元件分析 - 空格區域)
   核心規則：分析空格所屬的連續空格區域的大小。
   目的：偏好那些屬於較大連續空格區域的空格，這些區域可能提供更大的填補潛力或形成大型結構的機會。
   啟發式類型: 連通元件分析 (針對空格)
   輸出詮釋: 分數越高表示該空格屬於一個面積越大的連續空格區域 (分數經盤面總大小正規化)
   """
   effective_request_id = request_id or "N/A_brain_GM3"
   logger.debug("Executing EXT_GM3_Adv_Connected_Comp_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   visited_overall = np.zeros_like(grid, dtype=bool) # Tracks visited cells for any component search
 
   for r_start in range(rows): #
       for c_start in range(cols): #
           if visited_overall[r_start, c_start] or grid[r_start, c_start] != -1:
               # Skip if already visited or not an empty cell
               continue #
           
           # Start BFS for a new connected component of empty cells
           component_cells: List[Tuple[int,int]] = [] #
           q = deque([(r_start, c_start)]) #
           visited_bfs_current_component = set([(r_start, c_start)]) # Visited in current BFS path
           visited_overall[r_start, c_start] = True # Mark as globally visited
 
           while q: #
               r_curr, c_curr = q.popleft() #
               component_cells.append((r_curr, c_curr)) #
 
               # Explore 4-connectivity neighbors
               for dr_bfs, dc_bfs in [(0, 1), (0, -1), (1, 0), (-1, 0)]: #
                   nr, nc = r_curr + dr_bfs, c_curr + dc_bfs #
 
                   if 0 <= nr < rows and 0 <= nc < cols and \
                      grid[nr, nc] == -1 and \
                      not visited_overall[nr, nc] and \
                      (nr, nc) not in visited_bfs_current_component:
                       
                       visited_overall[nr, nc] = True #
                       visited_bfs_current_component.add((nr,nc)) #
                       q.append((nr, nc)) #
           
           area_size = float(len(component_cells)) #
           
           # Normalize area size against total number of cells in the grid
           total_cells = float(rows * cols) #
           norm_area_size = 0.0 #
           if total_cells > 0: #
               norm_area_size = MathUtils.normalize_value(area_size, 0, total_cells, clamp=True) #
           
           # Assign this normalized area size score to all cells in the found component
           for r_comp, c_comp in component_cells: #
               scores[r_comp, c_comp] = norm_area_size #
           
   return scores #
 
# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
   "EXT_A2_Weighted_Proximity_Vec": EXT_A2_Weighted_Proximity_Vec,
   "EXT_M3_Local_Heterogeneity_Vec": EXT_M3_Local_Heterogeneity_Vec,
   "EXT_D3_Potential_Field_Vec": EXT_D3_Potential_Field_Vec,
   "EXT_F10_Discontinuity_Vec": EXT_F10_Discontinuity_Vec,
   "EXT_P7_Pathfinding_Value_Vec": EXT_P7_Pathfinding_Value_Vec,
   "EXT_R5_Resource_Control_Vec": EXT_R5_Resource_Control_Vec,
   "EXT_GM1_Row_Control_Vec": EXT_GM1_Row_Control_Vec,
   "EXT_GM2_Col_Flow_Vec": EXT_GM2_Col_Flow_Vec,
   "EXT_GM3_Adv_Connected_Comp_Vec": EXT_GM3_Adv_Connected_Comp_Vec,
   # Other modules will be added here
}
 
def EXT_GM4_Spatial_Auto_Corr_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM4 - 空間自相關性分析)
   核心規則：評估在空格填入一個假設的「平均」潛在數字後，該數字與其周圍現有數字的相似程度。
   目的：鼓勵形成數值聚集（正自相關）或數值交錯（負自相關，但此處偏好正自相關）。
          此版本偏好正自相關，即填入的數字與周圍鄰居的平均值相似時得分較高。
   啟發式類型: 空間統計
   輸出詮釋: 分數越高表示填入一個「典型」數字後，能更好地融入周圍環境，形成數值上的聚集。
   """
   effective_request_id = request_id or "N/A_brain_GM4"
   logger.debug("Executing EXT_GM4_Spatial_Auto_Corr_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   potential_numbers = list(get_legal_values_for_placement(grid)) #
   
   # Determine a hypothetical value to test placement with
   hypothetical_val_to_place: float #
   if potential_numbers: #
       hypothetical_val_to_place = float(np.median(potential_numbers)) #
   else: #
       # If no numbers can be legally placed, use a generic mid-value for the board
       max_board_val = get_card_max_value_from_grid_dimensions((rows,cols)) #
       hypothetical_val_to_place = (1.0 + float(max_board_val)) / 2.0 if max_board_val > 0 else 0.5 #
 
   max_val_on_grid_for_norm = float(get_card_max_value_from_grid_dimensions((rows,cols))) #
   if max_val_on_grid_for_norm == 0: max_val_on_grid_for_norm = 1.0 # Avoid div by zero if grid is 0x0 or 1x0 etc.
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           # Get actual numeric neighbors (non -1)
           neighbor_values = BoardAnalyzerUtils.get_neighborhood_values(
               grid, r_idx, c_idx, radius=1, eight_connectivity=True,
               val_func=lambda x: float(x) if x != -1 else None,
               include_center=False
           ) #
 
           if not neighbor_values: #
               scores[r_idx, c_idx] = 0.5  # Neutral score if no neighbors to compare with
               continue #
 
           mean_neighbors = np.mean(neighbor_values) #
           
           # Calculate the difference between the hypothetical placed value and the mean of its actual neighbors
           diff_hypothetical_to_mean_neighbors = abs(hypothetical_val_to_place - mean_neighbors) #
           
           # Normalize this difference. Max possible difference is roughly max_val_on_grid.
           # Score for positive autocorrelation: 1.0 - normalized_difference
           # (smaller difference means more similar, thus higher positive autocorrelation score)
           norm_diff = MathUtils.normalize_value(diff_hypothetical_to_mean_neighbors, 0, max_val_on_grid_for_norm, clamp=True) #
           
           positive_autocorr_score = 1.0 - norm_diff #
           scores[r_idx, c_idx] = positive_autocorr_score #
           
   return scores #
 
# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
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
   # Other modules will be added here
}
 
def EXT_GM5_Line_Completion_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM5 - 線段補全)
   核心規則：評估空格對於完成特定方向（行、列、對角線）上具有特定構成（如等差、等值）的短線段（例如長度為3）之潛力。
   目的：偏好那些能夠「臨門一腳」完成有意義短線段的空格。
   啟發式類型: 模式匹配 (短線段)
   輸出詮釋: 分數越高表示該空格填入某數字後，越能完成一個預定義的短線段模式。
   """
   effective_request_id = request_id or "N/A_brain_GM5"
   logger.debug("Executing EXT_GM5_Line_Completion_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0 or min(rows, cols) < 1: return scores # Need at least 1 cell. For lines of 3, need more.
 
   potential_numbers_to_place = list(get_legal_values_for_placement(grid)) #
   if not potential_numbers_to_place: #
       return scores #
 
   line_completion_score_map = { # Define scores for different types of completions
       "identical_3": 0.6, # Completing a line of 3 identical numbers
       "arithmetic_3_mend": 0.7, # Mending an arithmetic sequence X, val, Y
       "arithmetic_3_extend": 0.5, # Starting or ending an arithmetic sequence val, X, Y or X, Y, val
   }
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           max_completion_score_for_cell = 0.0 #
 
           for p_val in potential_numbers_to_place: #
               # Check all 8 directions + center for line completion checks around (r_idx, c_idx)
               # Directions: Horizontal, Vertical, Diagonal (top-left to bottom-right), Anti-Diagonal (top-right to bottom-left)
               
               # For a line of 3, centered at (r_idx, c_idx) with p_val: [N1, p_val, N2]
               # Or, p_val is at one end: [p_val, N1, N2] or [N1, N2, p_val]
 
               for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:  # Horizontal, Vertical, Diag, Anti-Diag
                   if dr == 0 and dc == 0: continue # Should not happen with this set of dr, dc
 
                   # Case 1: Mending a line -> N1 - p_val - N2
                   # N1 is at (r_idx - dr, c_idx - dc), N2 is at (r_idx + dr, c_idx + dc)
                   r_n1, c_n1 = r_idx - dr, c_idx - dc #
                   r_n2, c_n2 = r_idx + dr, c_idx + dc #
 
                   if 0 <= r_n1 < rows and 0 <= c_n1 < cols and \
                      0 <= r_n2 < rows and 0 <= c_n2 < cols:
                       val_n1 = grid[r_n1, c_n1] #
                       val_n2 = grid[r_n2, c_n2] #
 
                       if val_n1 != -1 and val_n2 != -1: # Both neighbors exist
                           # Check for 3 identical
                           if val_n1 == p_val and val_n2 == p_val: #
                               max_completion_score_for_cell = max(max_completion_score_for_cell, line_completion_score_map["identical_3"]) #
                           # Check for 3 arithmetic
                           if (val_n1 + val_n2) == 2 * p_val and abs(p_val - val_n1) > 0 : # Arithmetic and not constant
                               max_completion_score_for_cell = max(max_completion_score_for_cell, line_completion_score_map["arithmetic_3_mend"]) #
               
                   # Case 2: Extending a line -> p_val - N1 - N2
                   # N1 is at (r_idx + dr, c_idx + dc), N2 is at (r_idx + 2*dr, c_idx + 2*dc)
                   r_n1_ext1, c_n1_ext1 = r_idx + dr, c_idx + dc #
                   r_n2_ext1, c_n2_ext1 = r_idx + 2*dr, c_idx + 2*dc #
                   if 0 <= r_n1_ext1 < rows and 0 <= c_n1_ext1 < cols and \
                      0 <= r_n2_ext1 < rows and 0 <= c_n2_ext1 < cols:
                       val_n1_ext1 = grid[r_n1_ext1, c_n1_ext1] #
                       val_n2_ext1 = grid[r_n2_ext1, c_n2_ext1] #
                       if val_n1_ext1 != -1 and val_n2_ext1 != -1: #
                           if p_val == val_n1_ext1 and p_val == val_n2_ext1: #
                               max_completion_score_for_cell = max(max_completion_score_for_cell, line_completion_score_map["identical_3"]) #
                           if (p_val + val_n2_ext1) == 2 * val_n1_ext1 and abs(val_n1_ext1 - p_val) > 0: #
                               max_completion_score_for_cell = max(max_completion_score_for_cell, line_completion_score_map["arithmetic_3_extend"]) #
 
                   # Case 3: Extending a line (other end) -> N1 - N2 - p_val
                   # N1 is at (r_idx - 2*dr, c_idx - 2*dc), N2 is at (r_idx - dr, c_idx - dc)
                   r_n1_ext2, c_n1_ext2 = r_idx - 2*dr, c_idx - 2*dc #
                   r_n2_ext2, c_n2_ext2 = r_idx - dr, c_idx - dc #
                   if 0 <= r_n1_ext2 < rows and 0 <= c_n1_ext2 < cols and \
                      0 <= r_n2_ext2 < rows and 0 <= c_n2_ext2 < cols:
                       val_n1_ext2 = grid[r_n1_ext2, c_n1_ext2] #
                       val_n2_ext2 = grid[r_n2_ext2, c_n2_ext2] #
                       if val_n1_ext2 != -1 and val_n2_ext2 != -1: #
                           if val_n1_ext2 == val_n2_ext2 and val_n1_ext2 == p_val: #
                               max_completion_score_for_cell = max(max_completion_score_for_cell, line_completion_score_map["identical_3"]) #
                           if (val_n1_ext2 + p_val) == 2 * val_n2_ext2 and abs(val_n2_ext2 - val_n1_ext2) > 0: #
                               max_completion_score_for_cell = max(max_completion_score_for_cell, line_completion_score_map["arithmetic_3_extend"]) #
 
           scores[r_idx, c_idx] = MathUtils.normalize_value(max_completion_score_for_cell, 0, 1.0, clamp=True) # Scores are already ~0-1
           
   return scores #
 
import numpy as np

# -----------------------------------------------------------------------------
# Module Implementations
# -----------------------------------------------------------------------------

# --- Template for new modules ---
# def EXT_XXX_ModuleName(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
# """
# Docstring explaining the module's purpose, parameters, and return value.
# Parameters:
# grid (np.ndarray): The input grid.
# args: Additional positional arguments.
# kwargs: Additional keyword arguments.
# Returns:
# np.ndarray: A numpy array representing the scores or transformed grid.
# """
# # Module logic here
# return np.zeros_like(grid, dtype=float) # Placeholder return

def EXT_A2_Weighted_Proximity_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Calculates scores based on weighted proximity to certain features.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_M3_Local_Heterogeneity_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Measures the local heterogeneity or diversity of elements.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_D3_Potential_Field_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Applies a potential field concept for scoring.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_F10_Discontinuity_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Detects and scores discontinuities in the grid.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_P7_Pathfinding_Value_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Evaluates grid cells based on pathfinding values or costs.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_R5_Resource_Control_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Assesses control over resources or key areas.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM1_Row_Control_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Evaluates control or patterns within rows.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM2_Col_Flow_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Analyzes flow or sequences within columns.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM3_Adv_Connected_Comp_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Advanced analysis of connected components.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM4_Spatial_Auto_Corr_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Calculates spatial auto-correlation.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM5_Line_Completion_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Scores based on potential for line completion.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM6_Symmetry_Potential_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Evaluates symmetry potential within the grid.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM7_Numeric_Gaps_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Identifies and scores numeric gaps or sequences.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM8_Edge_Affinity_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Measures affinity or attraction to edges/borders.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM9_Center_Control_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Assesses control or influence over the center of the grid.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM10_Blocking_Value_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Evaluates the strategic value of blocking opponent moves.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM11_Pair_Correlation_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Analyzes correlations between pairs of cells or values.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM12_Island_Analysis_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Identifies and analyzes isolated groups or 'islands' of cells.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM13_Sequence_Diversity_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Measures the diversity of sequences or patterns.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM14_Risk_Assessment_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Assesses risks associated with certain configurations or areas.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM15_Information_Gain_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Evaluates potential moves based on information gain.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM16_Harmonic_Centrality_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Calculates harmonic centrality for grid cells.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM17_Entropy_Minimization_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Scores based on the principle of entropy minimization.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM18_RL_Value_Est_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Estimates value using a reinforcement learning based approach.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM19_Masked_Number_Skip_Pattern_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Identifies skip patterns in masked numbers.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)

def EXT_GM20_Skip_Pattern_Confidence_Vec(grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Assesses the confidence of identified skip patterns.
    Parameters:
        grid (np.ndarray): The input grid.
    Returns:
        np.ndarray: Score array.
    """
    # Placeholder logic
    return np.zeros_like(grid, dtype=float)


# -----------------------------------------------------------------------------
# Module Registration
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Module Query Function
# -----------------------------------------------------------------------------

def get_module_score(module_name: str, grid: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Retrieves and executes a registered module by its name.

    Parameters:
        module_name (str): The name of the module to execute.
        grid (np.ndarray): The input grid to pass to the module.
        *args: Additional positional arguments for the module.
        **kwargs: Additional keyword arguments for the module.

    Returns:
        np.ndarray: The output from the executed module.

    Raises:
        ValueError: If the module_name is not found in REGISTERED_MODULES_BRAIN.
    """
    func = REGISTERED_MODULES_BRAIN.get(module_name)
    if not func:
        raise ValueError(f"Module {module_name} not found in REGISTERED_MODULES_BRAIN.")
    return func(grid, *args, **kwargs)

# -----------------------------------------------------------------------------
# Verification (Optional - for testing brain.py directly)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Example usage for verification:
    print("Verifying brain.py structure...")

    # Create a dummy grid
    dummy_grid = np.array([[1, 2, 0], [0, 1, 5], [3, 0, 4]])
    print(f"Created dummy grid:\n{dummy_grid}")

    # Test retrieving a module
    module_to_test = "EXT_A2_Weighted_Proximity_Vec"
    print(f"\nTesting get_module_score with '{module_to_test}'...")
    try:
        scores = get_module_score(module_to_test, dummy_grid)
        print(f"Successfully called {module_to_test}. Output (should be zeros_like grid):\n{scores}")
        assert isinstance(scores, np.ndarray), "Return type is not np.ndarray"
        assert scores.shape == dummy_grid.shape, "Return shape does not match grid shape"
        assert scores.dtype == float, "Return dtype is not float"
    except ValueError as e:
        print(f"Error: {e}")

    # Test retrieving a non-existent module
    non_existent_module = "EXT_XXX_NonExistentModule"
    print(f"\nTesting get_module_score with non-existent module '{non_existent_module}'...")
    try:
        scores = get_module_score(non_existent_module, dummy_grid)
        print(f"Output for non-existent module (should not happen):\n{scores}")
    except ValueError as e:
        print(f"Successfully caught error for non-existent module: {e}")

    print("\nListing all registered modules:")
    for i, name in enumerate(REGISTERED_MODULES_BRAIN.keys()):
        print(f"{i+1}. {name}")

    print(f"\nTotal modules registered: {len(REGISTERED_MODULES_BRAIN)}")
    print("\nbrain.py verification complete.")

# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
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
   # Other modules will be added here
}
 
def EXT_GM6_Symmetry_Potential_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM6 - 對稱性潛力)
   核心規則：評估在空格填入數字後，盤面形成的對稱性程度（水平、垂直、中心、主對角線、反主對角線）。
   目的：偏好那些能夠創造或增強盤面對稱性的填補。
   啟發式類型: 幾何與模式識別
   輸出詮釋: 分數越高表示若在該空格填入特定數字，能與對稱位置上已存在的相同數字形成對稱。
   """
   effective_request_id = request_id or "N/A_brain_GM6"
   logger.debug("Executing EXT_GM6_Symmetry_Potential_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   potential_numbers_to_place = list(get_legal_values_for_placement(grid)) #
   if not potential_numbers_to_place: #
       return scores #
 
   symmetry_scores_map = {
       "horizontal": 0.7,
       "vertical": 0.7,
       "point_center": 0.8, # Center symmetry might be rarer/stronger
       "main_diagonal": 0.6, # (r,c) vs (c,r) - only if rows==cols for full meaning
       "anti_diagonal": 0.6, # (r,c) vs (rows-1-c, cols-1-r) - only if rows==cols
   }
 
   center_r_float, center_c_float = (rows - 1) / 2.0, (cols - 1) / 2.0 #
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           max_symmetry_score_for_cell = 0.0 #
 
           for p_val in potential_numbers_to_place: #
               current_pval_max_sym = 0.0 #
 
               # 1. Horizontal Symmetry: (r_idx, c_idx) vs (r_idx, cols - 1 - c_idx)
               sr_h, sc_h = r_idx, cols - 1 - c_idx #
               if sc_h != c_idx and 0 <= sr_h < rows and 0 <= sc_h < cols and grid[sr_h, sc_h] == p_val: #
                   current_pval_max_sym = max(current_pval_max_sym, symmetry_scores_map["horizontal"]) #
 
               # 2. Vertical Symmetry: (r_idx, c_idx) vs (rows - 1 - r_idx, c_idx)
               sr_v, sc_v = rows - 1 - r_idx, c_idx #
               if sr_v != r_idx and 0 <= sr_v < rows and 0 <= sc_v < cols and grid[sr_v, sc_v] == p_val: #
                   current_pval_max_sym = max(current_pval_max_sym, symmetry_scores_map["vertical"]) #
               
               # 3. Point (Center) Symmetry: (r_idx, c_idx) vs (rows - 1 - r_idx, cols - 1 - c_idx)
               sr_p, sc_p = rows - 1 - r_idx, cols - 1 - c_idx #
               if (sr_p != r_idx or sc_p != c_idx) and \
                  0 <= sr_p < rows and 0 <= sc_p < cols and grid[sr_p, sc_p] == p_val: #
                   current_pval_max_sym = max(current_pval_max_sym, symmetry_scores_map["point_center"]) #
 
               # 4. Main Diagonal Symmetry (\): (r_idx, c_idx) vs (c_idx, r_idx)
               # Meaningful especially for square or near-square grids.
               if rows == cols: # More strictly for square grids, can be relaxed
                   sr_d1, sc_d1 = c_idx, r_idx #
                   if (sr_d1 != r_idx or sc_d1 != c_idx) and \
                      0 <= sr_d1 < rows and 0 <= sc_d1 < cols and grid[sr_d1, sc_d1] == p_val: #
                       current_pval_max_sym = max(current_pval_max_sym, symmetry_scores_map["main_diagonal"]) #
 
               # 5. Anti-Diagonal Symmetry (/): (r_idx, c_idx) vs (cols - 1 - c_idx, rows - 1 - r_idx) (for matrix indices)
               # This reflects across the anti-diagonal for square matrices.
               if rows == cols: # More strictly for square grids
                   # The symmetric element to grid[r][c] w.r.t anti-diagonal is grid[cols-1-c][rows-1-r]
                   # So, (r,c) maps to ((rows-1)-(cols-1-c_orig), (cols-1)-(rows-1-r_orig))? No this is confusing.
                   # Simpler: for grid[r][c], the anti-diagonal symmetric element is grid[N-1-c][N-1-r] for an NxN matrix.
                   # Here, (r_idx, c_idx) becomes ( (rows-1)-c_idx, (cols-1)-r_idx ) NO, it's ( (N-1)-col_of_original, (N-1)-row_of_original)
                   # For grid[r_idx, c_idx], symmetric position is (rows-1-c_idx, cols-1-r_idx) only if we assume r is first index, c is second.
                   # [cite_start]Let's use the definition from source text [cite:13] - point symmetry description implies (rows-1-r, cols-1-c)
                   # The equivalent of anti-diagonal reflection for value at grid[r,c] is grid[ (cols-1)-c , (rows-1)-r ] if (rows-1) and (cols-1) are the max indices
                   # If grid[r,c] reflects to grid[r',c'] then r' = max_col_idx - c and c' = max_row_idx - r
                   sr_d2, sc_d2 = (cols - 1) - c_idx, (rows - 1) - r_idx # This assumes indices are swapped and flipped from max.
                                                                   # Example: (0,0) -> (cols-1, rows-1). (0,1) -> (cols-2, rows-1).
                   if (sr_d2 != r_idx or sc_d2 != c_idx) and \
                      0 <= sr_d2 < rows and 0 <= sc_d2 < cols and grid[sr_d2, sc_d2] == p_val: #
                       current_pval_max_sym = max(current_pval_max_sym, symmetry_scores_map["anti_diagonal"]) #
               
               if current_pval_max_sym > max_symmetry_score_for_cell: #
                   max_symmetry_score_for_cell = current_pval_max_sym #
           
           scores[r_idx, c_idx] = MathUtils.normalize_value(max_symmetry_score_for_cell, 0, 1.0, clamp=True) # Max of map is 0.8
 
   return scores #
 
# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
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
   # Other modules will be added here
}
 
def EXT_GM7_Numeric_Gaps_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM7 - 數值間隙填充)
   核心規則：識別並評估在局部區域或序列中，填補數字「間隙」的價值。
              特別是尋找能填入使之成為公差為1的連續數列的間隙。
   目的：偏好那些能填補序列中明顯缺失數字的空格。
   啟發式類型: 序列與模式識別 (間隙填充)
   輸出詮釋: 分數越高表示該空格若填入特定數字，越能完美地填補一個數值間隙（尤其是公差為1的序列）。
   """
   effective_request_id = request_id or "N/A_brain_GM7"
   logger.debug("Executing EXT_GM7_Numeric_Gaps_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   potential_numbers_to_place = list(get_legal_values_for_placement(grid)) #
   if not potential_numbers_to_place: #
       return scores #
 
   gap_fill_scores_map = {
       "arithmetic_1_gap_fill": 0.9,  # Fills X, p_val, X+2 (i.e. p_val = X+1)
       "arithmetic_generic_mend": 0.7, # Fills X, p_val, Y where X, p_val, Y is arithmetic
       "arithmetic_generic_extend": 0.5, # p_val, X, Y or X, Y, p_val is arithmetic
   }
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           max_cell_gap_score = 0.0 #
 
           for p_val in potential_numbers_to_place: #
               current_pval_score = 0.0 #
 
               # Iterate over 4 directions (Horizontal, Vertical, Main Diagonal, Anti-Diagonal)
               for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]: #
                   # Case 1: p_val mends a gap: N1 - p_val - N2
                   r_n1, c_n1 = r_idx - dr, c_idx - dc #
                   r_n2, c_n2 = r_idx + dr, c_idx + dc #
                   if 0 <= r_n1 < rows and 0 <= c_n1 < cols and \
                      0 <= r_n2 < rows and 0 <= c_n2 < cols:
                       val_n1 = grid[r_n1, c_n1] #
                       val_n2 = grid[r_n2, c_n2] #
                       if val_n1 != -1 and val_n2 != -1: #
                           # Specific check for arithmetic sequence with common difference 1
                           if val_n1 == p_val - 1 and val_n2 == p_val + 1: #
                               current_pval_score = max(current_pval_score, gap_fill_scores_map["arithmetic_1_gap_fill"]) #
                           # Generic arithmetic sequence check (d != 0)
                           elif (val_n1 + val_n2) == 2 * p_val and abs(p_val - val_n1) > 0 : #
                               current_pval_score = max(current_pval_score, gap_fill_scores_map["arithmetic_generic_mend"]) #
                   
                   # Case 2: p_val extends a sequence: p_val - N1 - N2
                   r_n1_ext1, c_n1_ext1 = r_idx + dr, c_idx + dc #
                   r_n2_ext1, c_n2_ext1 = r_idx + 2 * dr, c_idx + 2 * dc #
                   if 0 <= r_n1_ext1 < rows and 0 <= c_n1_ext1 < cols and \
                      0 <= r_n2_ext1 < rows and 0 <= c_n2_ext1 < cols:
                       val_n1_ext1 = grid[r_n1_ext1, c_n1_ext1] #
                       val_n2_ext1 = grid[r_n2_ext1, c_n2_ext1] #
                       if val_n1_ext1 != -1 and val_n2_ext1 != -1: #
                            # Check for N1=p_val+d, N2=p_val+2d -> val_n1_ext1 - p_val == val_n2_ext1 - val_n1_ext1 (d)
                           common_diff = val_n1_ext1 - p_val #
                           if common_diff != 0 and val_n2_ext1 == val_n1_ext1 + common_diff: #
                               current_pval_score = max(current_pval_score, gap_fill_scores_map["arithmetic_generic_extend"]) #
 
                   # Case 3: p_val extends a sequence: N1 - N2 - p_val
                   r_n1_ext2, c_n1_ext2 = r_idx - 2 * dr, c_idx - 2 * dc #
                   r_n2_ext2, c_n2_ext2 = r_idx - dr, c_idx - dc #
                   if 0 <= r_n1_ext2 < rows and 0 <= c_n1_ext2 < cols and \
                      0 <= r_n2_ext2 < rows and 0 <= c_n2_ext2 < cols:
                       val_n1_ext2 = grid[r_n1_ext2, c_n1_ext2] #
                       val_n2_ext2 = grid[r_n2_ext2, c_n2_ext2] #
                       if val_n1_ext2 != -1 and val_n2_ext2 != -1: #
                           # Check for N2=N1+d, p_val=N1+2d -> val_n2_ext2 - val_n1_ext2 == p_val - val_n2_ext2 (d)
                           common_diff = val_n2_ext2 - val_n1_ext2 #
                           if common_diff != 0 and p_val == val_n2_ext2 + common_diff: #
                               current_pval_score = max(current_pval_score, gap_fill_scores_map["arithmetic_generic_extend"]) #
               
               if current_pval_score > max_cell_gap_score: #
                   max_cell_gap_score = current_pval_score #
           
           scores[r_idx, c_idx] = MathUtils.normalize_value(max_cell_gap_score, 0, 1.0, clamp=True) # Scores are already ~0-1
 
   return scores #
 
# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
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
   # Other modules will be added here
}
 
def EXT_GM8_Edge_Affinity_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM8 - 邊緣親和度)
   核心規則：評估空格與盤面邊緣或角落的接近程度及其策略意義。
   目的：根據策略配置，偏好靠近或遠離邊緣/角落的空格。
   啟發式類型: 位置與邊界分析
   輸出詮釋: 分數高低取決於設定（偏好/避開邊緣）。預設偏好邊緣，越靠近邊緣/角落分數越高。
   """
   effective_request_id = request_id or "N/A_brain_GM8"
   logger.debug("Executing EXT_GM8_Edge_Affinity_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   # Internal parameter: "prefer_edge" or "avoid_edge"
   affinity_mode = "prefer_edge"
   corner_bonus_prefer = 0.2 # Bonus if preferring edges and it's a corner
   corner_penalty_avoid = 0.2 # Penalty if avoiding edges and it's a corner
 
   # Max possible minimum distance to an edge (for normalization)
   # This would be for a cell at the center of the board.
   # For a 1D line of length L, center is (L-1)/2. Min dist is 0. Max min_dist is floor((L-1)/2).
   max_min_dist_to_edge_row = (rows - 1) // 2 if rows > 0 else 0 #
   max_min_dist_to_edge_col = (cols - 1) // 2 if cols > 0 else 0 #
   # The actual maximum of minimum distances to any edge.
   # e.g. 5x5 grid, center (2,2), min_dist=2. max_min_dist = 2.
   # e.g. 5x3 grid, center-ish (2,1), min_dist_r=2, min_dist_c=1. min_dist=1. max_min_dist for row is 2, for col is 1.
   # We need the max value that min_dist can take.
   overall_max_of_min_distances = float(min(max_min_dist_to_edge_row, max_min_dist_to_edge_col)) #
   if overall_max_of_min_distances == 0 and (rows > 1 or cols > 1) : # e.g. a 1xN or Nx1 line, max_min_dist is 0.
       overall_max_of_min_distances = 0.5 # Avoid div by zero for normalization if all cells are on edge
                                          # This implies if all cells are edges, normalized_dist will be 0.
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           dist_to_top_edge = r_idx #
           dist_to_bottom_edge = rows - 1 - r_idx #
           dist_to_left_edge = c_idx #
           dist_to_right_edge = cols - 1 - c_idx #
 
           min_dist = float(min(dist_to_top_edge, dist_to_bottom_edge, dist_to_left_edge, dist_to_right_edge)) #
           
           is_corner = (r_idx == 0 or r_idx == rows - 1) and \
                       (c_idx == 0 or c_idx == cols - 1) #
 
           current_score = 0.0 #
 
           if overall_max_of_min_distances > 0: #
               normalized_dist = min_dist / overall_max_of_min_distances #
               normalized_dist = min(1.0, normalized_dist) # Clamp if min_dist somehow exceeds expected max
           elif min_dist == 0 : # all cells are on an edge, min_dist is 0
               normalized_dist = 0.0 #
           else: # Should not happen if overall_max_of_min_distances is handled
               normalized_dist = 1.0 #
 
           if affinity_mode == "prefer_edge": #
               current_score = 1.0 - normalized_dist # Closer to edge (smaller dist) -> higher score
               if is_corner and min_dist == 0: #
                   current_score += corner_bonus_prefer #
           elif affinity_mode == "avoid_edge": #
               current_score = normalized_dist # Further from edge (larger dist) -> higher score
               if is_corner and min_dist == 0: #
                   current_score -= corner_penalty_avoid #
           
           scores[r_idx, c_idx] = MathUtils.normalize_value(current_score, -corner_penalty_avoid, 1.0 + corner_bonus_prefer, clamp=True) #
 
   return scores #
 
# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
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
   # Other modules will be added here
}
 
def EXT_GM9_Center_Control_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM9 - 中心控制偏好)
   核心規則：評估空格與盤面中心的接近程度及其策略意義。
   目的：根據策略配置，偏好靠近或遠離盤面中心區域的空格。
   啟發式類型: 位置與中心性分析
   輸出詮釋: 分數高低取決於設定（偏好/避開中心）。預設偏好中心，越靠近中心分數越高。
   """
   effective_request_id = request_id or "N/A_brain_GM9"
   logger.debug("Executing EXT_GM9_Center_Control_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   # Internal parameter: "prefer_center" or "avoid_center"
   affinity_mode = "prefer_center"
 
   center_r = (rows - 1) / 2.0 #
   center_c = (cols - 1) / 2.0 #
 
   # Max possible distance from any cell to the center is the distance from a corner to the center.
   # Using (0,0) as the reference corner.
   max_dist_to_center = MathUtils.euclidean_distance((0.0, 0.0), (center_r, center_c)) #
   if max_dist_to_center == 0 and (rows > 1 or cols > 1): # e.g. a 2x2 grid, center is (0.5,0.5), max_dist > 0
       # This case is for 1x1 grid where center is (0,0) and max_dist is 0.
       # Or for any grid, if calculation is 0, set a small positive to avoid div by zero for normalization.
       max_dist_to_center = 1.0 # Avoid division by zero in normalization if grid is 1x1
   elif max_dist_to_center == 0 and rows <=1 and cols <=1: # Truly a 1x1 or 0x0 grid
        pass # max_dist_to_center remains 0, normalization will handle.
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           current_dist_to_center = MathUtils.euclidean_distance((float(r_idx), float(c_idx)), (center_r, center_c)) #
           
           normalized_dist = 0.0 #
           if max_dist_to_center > 0: #
               normalized_dist = MathUtils.normalize_value(current_dist_to_center, 0, max_dist_to_center, clamp=True) #
           elif current_dist_to_center == 0 : # For 1x1 grid, dist is 0, max_dist is 0. Should be 0.5 normalized.
               normalized_dist = 0.0 # MathUtils.normalize_value will return 0.5 if val=min=max
                                   # but if we want 0 dist to mean "perfectly at center", then 0 is fine here
                                   # The score logic below will invert this.
 
           current_score = 0.0 #
           if affinity_mode == "prefer_center": #
               current_score = 1.0 - normalized_dist # Closer to center (smaller dist) -> higher score
           elif affinity_mode == "avoid_center": #
               current_score = normalized_dist # Further from center (larger dist) -> higher score
           
           scores[r_idx, c_idx] = MathUtils.normalize_value(current_score, 0, 1.0, clamp=True) # Final clamp
 
   return scores #
 
# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
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
   # Other modules will be added here
}
 
def EXT_GM10_Blocking_Value_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM10 - 阻斷價值評估)
   核心規則：評估在空格填入數字是否能有效「阻止」或「避免」形成預定義的不良模式或序列。
   目的：偏好那些不會導致形成不良結構的填補，或者理想情況下能主動阻止潛在不良結構形成的填補。
   啟發式類型: 防禦性策略與模式規避
   輸出詮釋: 分數越高表示在該空格填入數字後，越不可能形成已知的不良模式。
   """
   effective_request_id = request_id or "N/A_brain_GM10"
   logger.debug("Executing EXT_GM10_Blocking_Value_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   potential_numbers_to_place = list(get_legal_values_for_placement(grid)) #
   if not potential_numbers_to_place: #
       return scores #
 
   # Define some simple undesirable sequences (length 3 for this example)
   # In a real system, these could be learned or more complex.
   UNDESIRABLE_SEQUENCES = [
       [1, 1, 1],  # Three 1s in a row
       [2, 2, 2],  # Three 2s in a row
       # [1, 2, 3], # Example: if a short ascending sequence is bad in some contexts
       # [5, 5, -1] # A partial bad pattern that could be completed (more complex to check "blocking")
   ]
   # For simplicity, this implementation checks if placing a number *completes* an undesirable sequence.
   # A high score means the placement *avoids* completing such sequences.
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           # Score for this cell will be the max score achievable by placing any potential number
           # where "score" means "does not complete an undesirable pattern".
           max_safety_score_for_cell = 0.0 # Default to low score if all placements are bad
 
           if not potential_numbers_to_place: # If somehow list is empty now
               scores[r_idx, c_idx] = 0.5 # Neutral if no options
               continue
 
           for p_val in potential_numbers_to_place: #
               temp_grid = grid.copy() #
               temp_grid[r_idx, c_idx] = p_val #
               
               forms_undesirable_pattern = False #
 
               # Check lines of length 3 passing through (r_idx, c_idx)
               # Directions: Horizontal, Vertical, Main Diagonal, Anti-Diagonal
               for dr_line, dc_line in [(0, 1), (1, 0), (1, 1), (1, -1)]: #
                   # Check 3 positions for each line: p_val in pos 0, 1, or 2
                   for offset in range(-2, 1): # Start of a 3-cell window
                       line_coords = [] #
                       current_line_values = [] #
                       valid_line = True #
                       for i in range(3): #
                           check_r, check_c = r_idx + (offset + i) * dr_line, c_idx + (offset + i) * dc_line #
                           # Ensure the p_val cell is part of this 3-cell line
                           if not ( (r_idx == check_r and c_idx == check_c) or \
                                    (r_idx == r_idx + (offset + i-1) * dr_line and c_idx == c_idx + (offset + i-1) * dc_line and i==1) or \
                                    (r_idx == r_idx + (offset + i-2) * dr_line and c_idx == c_idx + (offset + i-2) * dc_line and i==2) ):
                               # This specific 3-cell window check is slightly complex.
                               # A simpler way: extract line segment of length 3 centered at (r,c) for each component,
                               # and also check lines where (r,c) is an endpoint.
                               # The current way will form 3 segments around (r,c) for each direction.
                               # e.g., for horizontal (dr=0, dc=1):
                               # offset -2: (r,c-2), (r,c-1), (r,c) <-- p_val is at end
                               # offset -1: (r,c-1), (r,c),   (r,c+1) <-- p_val is in middle
                               # offset  0: (r,c),   (r,c+1), (r,c+2) <-- p_val is at start
                               pass # This complex condition for p_val check is removed; simpler below
 
                           if 0 <= check_r < rows and 0 <= check_c < cols: #
                               line_coords.append((check_r, check_c)) #
                               current_line_values.append(temp_grid[check_r, check_c]) #
                           else: #
                               valid_line = False #
                               break #
                       
                       if valid_line and len(current_line_values) == 3: #
                           # Ensure the currently placed p_val at (r_idx, c_idx) is part of this line
                           if (r_idx, c_idx) not in line_coords: #
                               continue #
 
                           for undesirable_seq in UNDESIRABLE_SEQUENCES: #
                               if len(undesirable_seq) == 3 and current_line_values == undesirable_seq: #
                                   forms_undesirable_pattern = True #
                                   break #
                   if forms_undesirable_pattern: #
                       break #
               
               current_score_for_pval = 0.9 if not forms_undesirable_pattern else 0.1 #
               if current_score_for_pval > max_safety_score_for_cell: #
                   max_safety_score_for_cell = current_score_for_pval #
           
           scores[r_idx, c_idx] = max_safety_score_for_cell #
 
   return scores #
 
# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
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
   # Other modules will be added here
}
 
def EXT_GM11_Pair_Correlation_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM11 - 數字配對關聯分析)
   核心規則：分析特定數字對（pair）共同出現或以特定相對位置（此處為鄰近）出現的頻率與價值。
   目的：偏好那些能夠形成已知有利數字配對的填補。
   啟發式類型: 關聯性分析 (局部)
   輸出詮釋: 分數越高表示在該空格填入特定數字後，能與周圍已存在的數字形成更多或更高價值的有利配對。
   """
   effective_request_id = request_id or "N/A_brain_GM11"
   logger.debug("Executing EXT_GM11_Pair_Correlation_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   potential_numbers_to_place = list(get_legal_values_for_placement(grid)) #
   if not potential_numbers_to_place: #
       return scores #
 
   # Define some favorable pairs and their scores.
   # (val1, val2) means val1 placed next to existing val2.
   # These could be learned or be part of a more complex configuration.
   FAVORABLE_PAIRS_SCORES = {
       (3, 7): 0.8, (7, 3): 0.8,  # 3 and 7 like to be together
       (1, 2): 0.6, (2, 1): 0.6,  # Sequential small numbers
       (10, 20): 0.7, (20, 10): 0.7, # Example of decade pairing
       (5, 10): 0.5, (10, 5): 0.5,
       (max(1,get_card_max_value_from_grid_dimensions((rows,cols))//2), max(1,get_card_max_value_from_grid_dimensions((rows,cols))//2)+1) : 0.4 # mid range consecutive
   }
   if not FAVORABLE_PAIRS_SCORES: #
        max_single_pair_score = 0.0 #
   else: #
       max_single_pair_score = max(FAVORABLE_PAIRS_SCORES.values()) if FAVORABLE_PAIRS_SCORES else 0.0 #
 
   # Heuristic max possible score: if all 8 neighbors form max-scoring pairs
   heuristic_max_total_pair_score = 8.0 * max_single_pair_score if max_single_pair_score > 0 else 1.0 #
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           max_accumulated_score_for_cell = 0.0 #
 
           for p_val in potential_numbers_to_place: #
               current_pval_accumulated_score = 0.0 #
 
               # Check 8 neighbors
               for dr in [-1, 0, 1]: #
                   for dc in [-1, 0, 1]: #
                       if dr == 0 and dc == 0: #
                           continue #
 
                       nr, nc = r_idx + dr, c_idx + dc #
 
                       if 0 <= nr < rows and 0 <= nc < cols: #
                           neighbor_val = grid[nr, nc] #
                           if neighbor_val != -1: # If neighbor is an existing number
                               # Check if (p_val, neighbor_val) is a favorable pair
                               if (p_val, int(neighbor_val)) in FAVORABLE_PAIRS_SCORES: #
                                   current_pval_accumulated_score += FAVORABLE_PAIRS_SCORES[(p_val, int(neighbor_val))] #
               
               if current_pval_accumulated_score > max_accumulated_score_for_cell: #
                   max_accumulated_score_for_cell = current_pval_accumulated_score #
           
           scores[r_idx, c_idx] = MathUtils.normalize_value(max_accumulated_score_for_cell, 0, heuristic_max_total_pair_score, clamp=True) #
 
   return scores #
 
# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
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
   # Other modules will be added here
}
 
def EXT_GM12_Island_Analysis_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM12 - 島嶼分析)
   核心規則：分析由已填數字形成的「島嶼」的特性，如大小、緊湊度和平均值。
   目的：根據策略，可能偏好大型、緊湊或包含高價值數字的島嶼。
          此處假設偏好較大、較緊湊、平均值較高的數字島嶼。
   啟發式類型: 連通元件與區域形態分析 (針對已填數字)
   輸出詮釋: 分數越高表示該格屬於一個更優（大、緊湊、高平均值）的數字島嶼。空格得0分。
   """
   effective_request_id = request_id or "N/A_brain_GM12"
   logger.debug("Executing EXT_GM12_Island_Analysis_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   visited_island_search = np.zeros_like(grid, dtype=bool) #
   max_val_on_board = get_card_max_value_from_grid_dimensions((rows, cols)) #
   if max_val_on_board == 0: max_val_on_board = 1.0 # Avoid div by zero
 
   # Weights for combining island characteristics
   w_size = 0.4 #
   w_compactness = 0.3 #
   w_avg_value = 0.3 #
 
   for r_start in range(rows): #
       for c_start in range(cols): #
           if grid[r_start, c_start] != -1 and not visited_island_search[r_start, c_start]: # Found an unvisited number
               current_island_cells: List[Tuple[int, int]] = [] #
               current_island_values: List[int] = [] #
               
               q = deque([(r_start, c_start)]) #
               visited_island_search[r_start, c_start] = True #
               
               min_r_bbox, max_r_bbox = r_start, r_start #
               min_c_bbox, max_c_bbox = c_start, c_start #
 
               while q: #
                   r_curr, c_curr = q.popleft() #
                   current_island_cells.append((r_curr, c_curr)) #
                   current_island_values.append(int(grid[r_curr, c_curr])) #
 
                   min_r_bbox = min(min_r_bbox, r_curr) #
                   max_r_bbox = max(max_r_bbox, r_curr) #
                   min_c_bbox = min(min_c_bbox, c_curr) #
                   max_c_bbox = max(max_c_bbox, c_curr) #
 
                   for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # 4-connectivity for islands
                       nr, nc = r_curr + dr, c_curr + dc #
                       if 0 <= nr < rows and 0 <= nc < cols and \
                          grid[nr, nc] != -1 and not visited_island_search[nr, nc]:
                           visited_island_search[nr, nc] = True #
                           q.append((nr, nc)) #
               
               # Calculate island characteristics
               island_size = float(len(current_island_cells)) #
               avg_value = 0.0 #
               if island_size > 0: #
                   avg_value = sum(current_island_values) / island_size #
 
               bbox_height = float(max_r_bbox - min_r_bbox + 1) #
               bbox_width = float(max_c_bbox - min_c_bbox + 1) #
               bbox_area = bbox_height * bbox_width #
               compactness = 0.0 #
               if bbox_area > 0: #
                   compactness = island_size / bbox_area # (Ratio of actual cells to bounding box area)
 
               # Normalize characteristics
               norm_size = MathUtils.normalize_value(island_size, 1, rows * cols, clamp=True) #
               norm_compactness = MathUtils.normalize_value(compactness, 0, 1.0, clamp=True) # Already 0-1
               norm_avg_value = MathUtils.normalize_value(avg_value, 1, max_val_on_board, clamp=True) #
 
               # Combine into a single island score
               island_score = (w_size * norm_size +
                               w_compactness * norm_compactness +
                               w_avg_value * norm_avg_value) #
               
               final_island_score = MathUtils.normalize_value(island_score, 0, 1.0, clamp=True) #
 
               # Assign this score to all cells in the current island
               for r_cell, c_cell in current_island_cells: #
                   scores[r_cell, c_cell] = final_island_score #
           
           elif grid[r_start, c_start] == -1 : # Empty cells get 0 score from this module
               scores[r_start, c_start] = 0.0 #
               visited_island_search[r_start,c_start] = True # Mark as visited to avoid re-check logic
 
   return scores #
 
# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
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
   # Other modules will be added here
}
 
def EXT_GM13_Sequence_Diversity_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM13 - 序列多樣性)
   核心規則：評估填補位置是否有助於形成多樣化的短序列（例如，不同方向、不同類型），而非僅專注於單一長序列。
   目的：鼓勵在盤面上形成多個不同類型或方向的短數字序列，增加盤面的「活性」或「機會」。
   啟發式類型: 模式識別與組合多樣性
   輸出詮釋: 分數越高表示在該空格填入特定數字後，能參與形成的獨特短序列種類越多。
   """
   effective_request_id = request_id or "N/A_brain_GM13"
   logger.debug("Executing EXT_GM13_Sequence_Diversity_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   potential_numbers_to_place = list(get_legal_values_for_placement(grid)) #
   if not potential_numbers_to_place: #
       return scores #
 
   short_sequence_len = 3 # Define length of "short sequences"
   heuristic_max_distinct_sequences = 8.0 # Max distinct short sequences a single cell might participate in (heuristic for normalization)
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           max_diversity_count_for_cell = 0 #
 
           for p_val in potential_numbers_to_place: #
               temp_grid = grid.copy() #
               temp_grid[r_idx, c_idx] = p_val #
               
               found_sequence_signatures = set() # Store signatures like ("arithmetic", (dr,dc), diff) or ("identical", (dr,dc), val)
 
               # Check in 4 directions (H, V, D1, D2)
               # Each direction vector also defines the line orientation
               for dr_dir, dc_dir in [(0, 1), (1, 0), (1, 1), (1, -1)]: #
                   # For each direction, check 3 possible alignments of a length-3 sequence where p_val is involved
                   for i_offset in range(short_sequence_len): # p_val is at index i_offset in the 3-number window
                       # Start of window: (r_idx - i_offset*dr_dir, c_idx - i_offset*dc_dir)
                       # End of window:   (r_idx + (short_sequence_len-1-i_offset)*dr_dir, c_idx + (short_sequence_len-1-i_offset)*dc_dir)
                       
                       current_sequence_values = [] #
                       valid_segment = True #
                       for k_seq in range(short_sequence_len): #
                           # Position of k_seq-th element in the window
                           # Relative to (r_idx, c_idx), this element is at (k_seq - i_offset) * (dr_dir, dc_dir)
                           check_r = r_idx + (k_seq - i_offset) * dr_dir #
                           check_c = c_idx + (k_seq - i_offset) * dc_dir #
 
                           if 0 <= check_r < rows and 0 <= check_c < cols: #
                               current_sequence_values.append(temp_grid[check_r, check_c]) #
                           else: #
                               valid_segment = False #
                               break #
                       
                       if valid_segment and len(current_sequence_values) == short_sequence_len: #
                           # Analyze this short sequence
                           s = current_sequence_values #
                           # 1. Arithmetic sequence (non-constant)
                           if s[0] != -1 and s[1] != -1 and s[2] != -1: # Ensure all are numbers
                               diff1 = s[1] - s[0] #
                               diff2 = s[2] - s[1] #
                               if diff1 == diff2 and diff1 != 0: #
                                   found_sequence_signatures.add(("arithmetic", (dr_dir, dc_dir), diff1)) #
                               
                               # 2. Identical sequence
                               if s[0] == s[1] and s[1] == s[2] and s[0] != -1 : #
                                   found_sequence_signatures.add(("identical", (dr_dir, dc_dir), s[0])) #
               
               current_pval_diversity_count = len(found_sequence_signatures) #
               if current_pval_diversity_count > max_diversity_count_for_cell: #
                   max_diversity_count_for_cell = current_pval_diversity_count #
           
           scores[r_idx, c_idx] = MathUtils.normalize_value(float(max_diversity_count_for_cell), 0, heuristic_max_distinct_sequences, clamp=True) #
 
   return scores #
 
# --- Module Registration (will be populated as modules are added) ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
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
   # Other modules will be added here
}
 
def EXT_GM14_Risk_Assessment_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM14 - 風險評估)
   核心規則：評估某個填補動作的潛在「風險」，例如是否會導致後續選擇過少（降低盤面靈活性）。
   目的：偏好那些能保持盤面較高靈活性的填補。低風險=高分數。
   啟發式類型: 盤面狀態評估 (未來選擇性)
   輸出詮釋: 分數越高表示填入該數字後，盤面剩餘的合法填補選項越多，風險越低。
   """
   effective_request_id = request_id or "N/A_brain_GM14"
   logger.debug("Executing EXT_GM14_Risk_Assessment_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   initial_potential_numbers = list(get_legal_values_for_placement(grid)) #
   if not initial_potential_numbers: #
       return scores # No numbers to place initially, risk is not well-defined by this logic
 
   max_possible_options = (rows * cols) * (rows * cols) # Heuristic: num_empty_cells * num_potential_values
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           max_flexibility_score_for_cell = 0.0 #
 
           # If this cell cannot be filled by any initial potential number, its score remains 0 or low.
           # This check isn't strictly needed as p_val loop won't run.
 
           for p_val in initial_potential_numbers: # Only try values that are currently legal for the original grid
               # Check if p_val can be placed at (r_idx, c_idx) without creating duplicates with existing grid numbers
               # This is implicitly handled as p_val comes from legal_values_for_placement
               # and we are only evaluating empty cells.
               
               temp_grid = grid.copy() #
               temp_grid[r_idx, c_idx] = p_val #
               
               # Calculate flexibility after this placement
               remaining_empty_cells = np.count_nonzero(temp_grid == -1) #
               subsequent_legal_moves = len(get_legal_values_for_placement(temp_grid)) #
               
               # Flexibility metric: product of remaining empty cells and number of unique values that can be placed
               # A simpler metric: just the number of subsequent legal moves
               current_flexibility = float(subsequent_legal_moves) #
               # Or: current_flexibility = float(remaining_empty_cells * subsequent_legal_moves)
 
 
               if current_flexibility > max_flexibility_score_for_cell: #
                   max_flexibility_score_for_cell = current_flexibility #
           
           # Normalize: Max possible subsequent_legal_moves is roughly rows*cols
           # Max for (remaining_empty_cells * subsequent_legal_moves) is much larger.
           # Using max_possible_options based on subsequent_legal_moves.
           current_max_heuristic_flex = float(rows * cols -1) # Max legal values after 1 placement
           if current_max_heuristic_flex == 0 : current_max_heuristic_flex = 1.0
 
           scores[r_idx, c_idx] = MathUtils.normalize_value(max_flexibility_score_for_cell, 0, current_max_heuristic_flex, clamp=True) #
 
   return scores #
 
def EXT_GM15_Information_Gain_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM15 - 資訊增益評估)
   核心規則：評估填入數字後，對盤面整體結構「有序性」的提升（例如，熵的降低）。
   目的：偏好那些能使盤面狀態更「確定」或「有序」的填補。
   啟發式類型: 資訊理論啟發 (基於全局熵變)
   輸出詮釋: 分數越高表示填入該數字後，盤面整體熵降低得越多（即資訊增益越大，盤面越有序）。
   """
   effective_request_id = request_id or "N/A_brain_GM15"
   logger.debug("Executing EXT_GM15_Information_Gain_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   potential_numbers_to_place = list(get_legal_values_for_placement(grid)) #
   if not potential_numbers_to_place: #
       return scores #
 
   # Calculate entropy of the initial grid (all cells, -1 is a symbol)
   initial_grid_values = [int(val) for val in grid.flatten()] #
   entropy_before = MathUtils.get_entropy(initial_grid_values) #
 
   # Max possible entropy for normalization (log2 of number of symbols: 1 to R*C plus -1)
   num_symbols = rows * cols + 1 #
   max_possible_entropy_change = math.log2(num_symbols) if num_symbols > 1 else 1.0 # Max possible entropy itself
   if max_possible_entropy_change == 0 : max_possible_entropy_change = 1.0
 
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           max_entropy_reduction_for_cell = -float('inf') # We want to maximize reduction
 
           for p_val in potential_numbers_to_place: #
               temp_grid = grid.copy() #
               temp_grid[r_idx, c_idx] = p_val #
               
               temp_grid_values = [int(val) for val in temp_grid.flatten()] #
               entropy_after = MathUtils.get_entropy(temp_grid_values) #
               
               entropy_reduction = entropy_before - entropy_after # Higher reduction is better
               
               if entropy_reduction > max_entropy_reduction_for_cell: #
                   max_entropy_reduction_for_cell = entropy_reduction #
           
           if max_entropy_reduction_for_cell == -float('inf'): # Should not happen if potential_numbers_to_place is not empty
               max_entropy_reduction_for_cell = 0.0
 
           # Normalize the reduction. Min reduction can be negative (entropy increases). Max can be entropy_before.
           # Or normalize against max_possible_entropy_change.
           # Score will be higher for positive reductions.
           # Range of reduction is roughly [-max_possible_entropy_change, max_possible_entropy_change]
           scores[r_idx, c_idx] = MathUtils.normalize_value(max_entropy_reduction_for_cell, 0, max_possible_entropy_change, clamp=True) #
           # Clamping at 0 if it increases entropy.
 
   return scores #
 
def EXT_GM16_Harmonic_Centrality_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM16 - 調和中心性)
   核心規則：應用圖論中的調和中心性概念，評估盤面上各空格節點的重要性。
              調和中心性是一個節點到所有其他節點距離倒數的總和。
   目的：偏好那些在盤面「網絡」中更具中心性的空格。
   啟發式類型: 圖論中心性
   輸出詮釋: 分數越高表示該空格在圖結構中越「中心」（平均而言離其他格子越近）。
   """
   effective_request_id = request_id or "N/A_brain_GM16"
   logger.debug("Executing EXT_GM16_Harmonic_Centrality_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0 or (rows * cols) <= 1: return scores # Needs more than 1 cell
 
   # Max possible harmonic centrality (heuristic): if a cell is at distance 1 from all N-1 other cells.
   # Max_HC = (rows * cols - 1) * (1/1)
   max_hc_heuristic = float(rows * cols -1) #
   if max_hc_heuristic == 0: max_hc_heuristic = 1.0 #
 
   for r_eval in range(rows): #
       for c_eval in range(cols): #
           if grid[r_eval, c_eval] != -1:  # Only score empty cells
               continue #
 
           current_harmonic_centrality = 0.0 #
           num_other_nodes = 0 #
 
           for r_other in range(rows): #
               for c_other in range(cols): #
                   if r_eval == r_other and c_eval == c_other: #
                       continue # Skip self
 
                   # Using Manhattan distance as grid distance
                   dist = MathUtils.manhattan_distance((r_eval, c_eval), (r_other, c_other)) #
                   
                   if dist > 0: #
                       current_harmonic_centrality += 1.0 / dist #
                   num_other_nodes +=1 #
 
           if num_other_nodes == 0 : # Only one cell in grid, should have been caught earlier
               scores[r_eval, c_eval] = 0.0 #
           else: #
               # Normalization can be tricky. The sum of reciprocals can vary.
               # Using the heuristic max.
               scores[r_eval, c_eval] = MathUtils.normalize_value(current_harmonic_centrality, 0, max_hc_heuristic, clamp=True) #
   
   return scores #
 
def EXT_GM17_Entropy_Minimization_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM17 - 局部熵最小化)
   核心規則：評估填入數字後，盤面局部鄰域「熵」（無序度）的降低程度。
   目的：偏好那些能使其直接周圍環境更有規律、更「有序」的填補。
   啟發式類型: 資訊理論啟發 (基於局部熵變)
   輸出詮釋: 分數越高表示填入該數字後，其局部鄰域的熵降低得越多（局部更有序）。
   """
   effective_request_id = request_id or "N/A_brain_GM17"
   logger.debug("Executing EXT_GM17_Entropy_Minimization_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   potential_numbers_to_place = list(get_legal_values_for_placement(grid)) #
   if not potential_numbers_to_place: #
       return scores #
 
   radius = 1 # Define local neighborhood radius
   num_cells_in_neighborhood = (2 * radius + 1)**2 # Including center
   max_local_entropy_change = math.log2(num_cells_in_neighborhood) if num_cells_in_neighborhood > 1 else 1.0 #
   if max_local_entropy_change == 0 : max_local_entropy_change = 1.0 #
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           # Get neighbors *excluding* the current cell (r_idx, c_idx)
           # This will be used to calculate entropy_before_local
           # For entropy_after_local, we'll add p_val to these.
           base_neighbor_values = BoardAnalyzerUtils.get_neighborhood_values(
               grid, r_idx, c_idx, radius=radius, eight_connectivity=True,
               val_func=lambda x: int(x) if x != -1 else None, # Use None for -1, to filter out
               include_center=False # We are interested in context around the cell to be filled
           ) #
           # if not base_neighbor_values: continue # No context, no score change or neutral
 
           max_entropy_reduction_for_cell = -float('inf') #
 
           for p_val in potential_numbers_to_place: #
               # Calculate entropy of local neighborhood *including* p_val
               # The neighborhood is defined around (r_idx, c_idx).
               # One way: create a temp small grid of the neighborhood.
               # Another way: use base_neighbor_values and add p_val to it for calculation.
               
               # Entropy of the neighborhood if p_val is placed
               # The neighborhood values for entropy calculation should include the p_val itself
               # as it becomes part of that local configuration.
               # To be consistent, let's make a temp grid for the neighborhood
               
               # Simpler: consider the list of numbers if p_val was there.
               # This means the conceptual neighborhood includes the cell (r_idx,c_idx)
               # For "entropy_before", (r_idx,c_idx) is empty (-1 symbol)
               # For "entropy_after", (r_idx,c_idx) has p_val
               
               # For comparing just the effect of p_val on its context:
               # Entropy of context (base_neighbor_values) - should be constant for this (r_idx,c_idx)
               # Entropy of context + p_val
               
               # Let's redefine entropy calculation for local area including the evaluated cell:
               # Get all values in radius around (r_idx,c_idx), including (r_idx,c_idx) itself.
               
               # Entropy before (with (r_idx,c_idx) as empty, i.e., -1)
               # This requires a version of get_neighborhood_values that can include center and -1
               def val_func_for_entropy(x_val: int) -> int : return int(x_val) # Keep -1 as a symbol
 
               values_before_placement_local = BoardAnalyzerUtils.get_neighborhood_values(
                    grid, r_idx, c_idx, radius=radius, eight_connectivity=True,
                    val_func=val_func_for_entropy, include_center=True
               ) #
               entropy_before_local = MathUtils.get_entropy(values_before_placement_local) #
 
 
               temp_grid_local_place = grid.copy() #
               temp_grid_local_place[r_idx,c_idx] = p_val #
               values_after_placement_local = BoardAnalyzerUtils.get_neighborhood_values(
                    temp_grid_local_place, r_idx, c_idx, radius=radius, eight_connectivity=True,
                    val_func=val_func_for_entropy, include_center=True
               ) #
               entropy_after_local = MathUtils.get_entropy(values_after_placement_local) #
               
               entropy_reduction = entropy_before_local - entropy_after_local #
               
               if entropy_reduction > max_entropy_reduction_for_cell: #
                   max_entropy_reduction_for_cell = entropy_reduction #
           
           if max_entropy_reduction_for_cell == -float('inf'): max_entropy_reduction_for_cell=0.0 #
 
           scores[r_idx, c_idx] = MathUtils.normalize_value(max_entropy_reduction_for_cell, 0, max_local_entropy_change, clamp=True) #
 
   return scores #
 
def EXT_GM18_RL_Value_Est_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM18 - 類強化學習價值估計)
   核心規則：基於一組預定義的「理想特徵」來評估某個填補動作的啟發式長期潛在價值。
              此為簡化版，模擬從歷史數據學習到的偏好。
   目的：偏好那些能夠使盤面展現更多理想特徵（如形成特定序列、達到特定盤面密度等）的填補。
   啟發式類型: 狀態價值啟發 (基於盤面特徵計數)
   輸出詮釋: 分數越高表示填入該數字後，盤面呈現的理想特徵越多，預期長期回報越大。
   """
   effective_request_id = request_id or "N/A_brain_GM18"
   logger.debug("Executing EXT_GM18_RL_Value_Est_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   potential_numbers_to_place = list(get_legal_values_for_placement(grid)) #
   if not potential_numbers_to_place: #
       return scores #
 
   # Define desirable features and their heuristic scores/weights
   # These simulate a learned value function.
   FEATURE_WEIGHTS = {
       "identical_3": 1.0,  # Completing a line of 3 identical numbers
       "arithmetic_3": 0.7, # Completing an arithmetic sequence of 3
       "board_density_factor": 0.2 # General preference for denser boards (more numbers)
   }
   # Max possible feature score for normalization (heuristic)
   # Roughly: 4 directions * (1 per type) + density
   max_heuristic_feature_score = (4 * (FEATURE_WEIGHTS["identical_3"] + FEATURE_WEIGHTS["arithmetic_3"])) + FEATURE_WEIGHTS["board_density_factor"]
   if max_heuristic_feature_score == 0 : max_heuristic_feature_score = 1.0 #
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1:  # Only score empty cells
               continue #
 
           max_feature_score_for_cell = 0.0 #
 
           for p_val in potential_numbers_to_place: #
               temp_grid = grid.copy() #
               temp_grid[r_idx, c_idx] = p_val #
               
               current_features_score = 0.0 #
 
               # Feature 1 & 2: Lines of 3 (identical or arithmetic) involving p_val
               for dr_line, dc_line in [(0, 1), (1, 0), (1, 1), (1, -1)]: # H, V, D1, D2
                   for offset in range(-2, 1): # Window start relative to (r_idx,c_idx)
                       line_values = [] #
                       is_valid_line = True #
                       involved_pval = False #
                       for i in range(3): #
                           check_r, check_c = r_idx + (offset + i) * dr_line, c_idx + (offset + i) * dc_line #
                           if r_idx == check_r and c_idx == check_c : involved_pval = True #
                           if 0 <= check_r < rows and 0 <= check_c < cols: #
                               line_values.append(temp_grid[check_r, check_c]) #
                           else: #
                               is_valid_line = False #
                               break #
                       
                       if is_valid_line and involved_pval and len(line_values) == 3 and all(v != -1 for v in line_values): #
                           s = line_values #
                           # Identical
                           if s[0] == s[1] and s[1] == s[2]: #
                               current_features_score += FEATURE_WEIGHTS["identical_3"] #
                           # Arithmetic (non-constant)
                           elif (s[1] - s[0]) == (s[2] - s[1]) and (s[1] - s[0]) != 0 : #
                               current_features_score += FEATURE_WEIGHTS["arithmetic_3"] #
               
               # Feature 3: Board density
               num_filled_after_placement = np.count_nonzero(temp_grid != -1) #
               density_after_placement = num_filled_after_placement / (rows * cols) if (rows*cols) > 0 else 0 #
               current_features_score += FEATURE_WEIGHTS["board_density_factor"] * density_after_placement #
 
               if current_features_score > max_feature_score_for_cell: #
                   max_feature_score_for_cell = current_features_score #
           
           scores[r_idx, c_idx] = MathUtils.normalize_value(max_feature_score_for_cell, 0, max_heuristic_feature_score, clamp=True) #
           
   return scores #
 
def EXT_GM19_Masked_Number_Skip_Pattern_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM19 - 遮罩數字跳格模式向量)
   核心規則: 分析已揭示數字的「跳格模式」（其實際位置與預期基礎位置的偏差），
              並對符合主導跳格模式的空格進行評分。
   啟發式類型: 空間模式匹配 (基於全局偏移量)
   輸出詮釋: 分數越高表示該空格若填入特定數字，能與盤面上觀察到的主要「跳格」規律性最為吻合。
   """
   effective_request_id = request_id or "N/A_brain_GM19"
   logger.debug("Executing EXT_GM19_Masked_Number_Skip_Pattern_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   revealed_numbers_info = [] # List of {'value': val, 'r': r, 'c': c}
   for r in range(rows): #
       for c in range(cols): #
           if grid[r, c] != -1 and grid[r,c] > 0: # Assuming positive numbers are actual game numbers
               revealed_numbers_info.append({'value': int(grid[r, c]), 'r': r, 'c': c}) #
   
   if not revealed_numbers_info: return scores # No numbers to analyze pattern from
 
   expected_max_number_on_card = get_card_max_value_from_grid_dimensions((rows,cols)) #
   
   base_positions: Dict[int, Tuple[int, int]] = {} # value -> (expected_r, expected_c)
   for k_val in range(1, expected_max_number_on_card + 1): #
       base_r = (k_val - 1) // cols #
       base_c = (k_val - 1) % cols #
       if base_r < rows: # Ensure base position is within grid dimensions
           base_positions[k_val] = (base_r, base_c) #
 
   skip_vectors: Dict[int, Tuple[int, int]] = {} # value -> (delta_r, delta_col)
   for rn_info in revealed_numbers_info: #
       val = rn_info['value'] #
       if val in base_positions: #
           expected_r, expected_c = base_positions[val] #
           skip_vectors[val] = (rn_info['r'] - expected_r, rn_info['c'] - expected_c) #
 
   if not skip_vectors: return scores #
 
   dominant_skip_patterns_strength: Dict[Tuple[int,int], float] = {} # (dr,dc) -> strength
   if skip_vectors: #
       skip_vector_tuples_list = list(skip_vectors.values()) #
       if not skip_vector_tuples_list: return scores #
       
       counts = Counter(skip_vector_tuples_list) #
       min_occurrences_for_pattern = max(1, int(len(skip_vector_tuples_list) * 0.05)) # Pattern needs some support
                                                                                   # Adjusted from 2, to allow even single unique skips if they are the only ones.
       
       for skip_vec_tuple, count_val in counts.most_common(): #
           if count_val >= min_occurrences_for_pattern: #
               # Strength could simply be normalized count
               pattern_strength = MathUtils.normalize_value(float(count_val), float(min_occurrences_for_pattern), float(len(skip_vector_tuples_list)), clamp=True) #
               dominant_skip_patterns_strength[skip_vec_tuple] = pattern_strength #
           else: # Since most_common is sorted
               break #
 
   if not dominant_skip_patterns_strength: return scores #
 
   potential_numbers_to_place_set = get_legal_values_for_placement(grid) #
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1: continue #
 
           cell_max_pattern_score = 0.0 #
           for p_val_test in potential_numbers_to_place_set: #
               if p_val_test not in base_positions: continue #
               
               base_r_test, base_c_test = base_positions[p_val_test] #
               
               for current_skip_pattern, pattern_str in dominant_skip_patterns_strength.items(): #
                   skip_dr, skip_dc = current_skip_pattern #
                   predicted_r = base_r_test + skip_dr #
                   predicted_c = base_c_test + skip_dc #
                   
                   if predicted_r == r_idx and predicted_c == c_idx: #
                       # This empty cell (r_idx, c_idx) is where p_val_test would land if it followed this skip pattern
                       current_score_fit = pattern_str # Score is strength of the pattern it fits
                       if current_score_fit > cell_max_pattern_score: #
                           cell_max_pattern_score = current_score_fit #
           
           scores[r_idx, c_idx] = cell_max_pattern_score # Max score if multiple patterns/values fit this cell
 
   return scores #
 
def EXT_GM20_Skip_Pattern_Confidence_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
   """
   (GM20 - 跳格模式信心度/規律性增強)
   核心規則：評估在空格填入數字是否能增強或完成已觀察到的全局跳格規律性，
              特別是當這個填補能使遵循跳格模式的數字序列更完整或更具算術規律性時。
   啟發式類型: 序列完成與模式確認 (基於全局偏移量)
   輸出詮釋: 分數越高表示填入該數字不僅符合跳格模式的幾何位置，
              且能使該模式下的數字序列在算術/序列意義上更為「自信」或「完整」。
   """
   effective_request_id = request_id or "N/A_brain_GM20"
   logger.debug("Executing EXT_GM20_Skip_Pattern_Confidence_Vec", extra={'request_id': effective_request_id}) #
   rows, cols = grid.shape #
   scores = np.zeros((rows, cols), dtype=float) #
   if rows == 0 or cols == 0: return scores #
 
   # --- Initial Pattern Analysis (simplified from GM19, can be refactored) ---
   revealed_numbers_info_gm20 = [] #
   for r in range(rows): #
       for c in range(cols): #
           if grid[r, c] != -1 and grid[r,c] > 0 : #
               revealed_numbers_info_gm20.append({'value': int(grid[r, c]), 'r': r, 'c': c}) #
   
   if not revealed_numbers_info_gm20: return scores #
 
   expected_max_num_gm20 = get_card_max_value_from_grid_dimensions((rows,cols)) #
   base_pos_gm20: Dict[int, Tuple[int, int]] = { #
       k: ((k - 1) // cols, (k - 1) % cols) for k in range(1, expected_max_num_gm20 + 1) #
       if ((k - 1) // cols) < rows #
   }
 
   skip_vecs_initial_gm20: Dict[int, Tuple[int, int]] = {} #
   for rn in revealed_numbers_info_gm20: #
       val = rn['value'] #
       if val in base_pos_gm20: #
           skip_vecs_initial_gm20[val] = (rn['r'] - base_pos_gm20[val][0], rn['c'] - base_pos_gm20[val][1]) #
   
   dominant_patterns_details_gm20: List[Dict[str, Any]] = [] # List of {'skip': (dr,dc), 'values': [sorted_values_in_pattern], 'strength': float}
   if skip_vecs_initial_gm20: #
       skip_tuples_list_gm20 = list(skip_vecs_initial_gm20.values()) #
       counts_gm20 = Counter(skip_tuples_list_gm20) #
       min_occ_gm20 = max(1, int(len(skip_tuples_list_gm20) * 0.05)) #
 
       for skip_v, count_v in counts_gm20.most_common(): #
           if count_v >= min_occ_gm20: #
               pattern_vals = sorted([val for val, sv_tuple in skip_vecs_initial_gm20.items() if sv_tuple == skip_v]) #
               p_strength = MathUtils.normalize_value(float(count_v), float(min_occ_gm20), float(len(skip_tuples_list_gm20)), clamp=True) #
               dominant_patterns_details_gm20.append({'skip': skip_v, 'values': pattern_vals, 'strength': p_strength}) #
           else: #
               break #
   # --- End Initial Pattern Analysis ---
 
   if not dominant_patterns_details_gm20: return scores #
 
   potential_nums_to_place_gm20 = get_legal_values_for_placement(grid) #
 
   for r_idx in range(rows): #
       for c_idx in range(cols): #
           if grid[r_idx, c_idx] != -1: continue #
 
           max_confidence_score_for_cell_gm20 = 0.0 #
 
           for p_val_test in potential_nums_to_place_gm20: #
               if p_val_test not in base_pos_gm20: continue #
               
               base_r_t, base_c_t = base_pos_gm20[p_val_test] #
               current_max_conf_for_pval = 0.0 #
 
               for pattern_detail in dominant_patterns_details_gm20: #
                   pat_skip_dr, pat_skip_dc = pattern_detail['skip'] #
                   pat_existing_vals = pattern_detail['values'] # sorted list
                   pat_strength = pattern_detail['strength'] #
 
                   predicted_r_for_pval = base_r_t + pat_skip_dr #
                   predicted_c_for_pval = base_c_t + pat_skip_dc #
 
                   if predicted_r_for_pval == r_idx and predicted_c_for_pval == c_idx: # Geometrically fits
                       enhancement_factor = 0.5 # Base for geometric fit related to pattern strength
                       
                       # Check for arithmetic sequence enhancement
                       if len(pat_existing_vals) >= 1: # Need at least one existing number to extend/mend
                           temp_sequence_with_pval = sorted(pat_existing_vals + [p_val_test]) #
                           
                           if len(temp_sequence_with_pval) >= 2 : # Need at least 2 for a diff
                               diffs_in_temp_seq = np.diff(temp_sequence_with_pval) #
                               if len(diffs_in_temp_seq) > 0: #
                                   # Check if it becomes more consistently arithmetic
                                   is_arithmetic_now = len(set(diffs_in_temp_seq)) == 1 # All diffs are the same
                                   first_diff = diffs_in_temp_seq[0]
 
                                   if is_arithmetic_now and first_diff != 0 : # It forms a new, consistent arithmetic sequence
                                       enhancement_factor += 0.4 # Strong enhancement
                                   elif len(temp_sequence_with_pval) >=3 : # Check if it mends a specific gap
                                       # E.g. pat_existing_vals = [2,6], p_val_test = 4. temp = [2,4,6]
                                       # This is covered if the whole sequence is_arithmetic_now.
                                       # Could add bonus if p_val_test is between min/max of pat_existing_vals
                                       if min(pat_existing_vals) < p_val_test < max(pat_existing_vals) and is_arithmetic_now and first_diff !=0:
                                           enhancement_factor += 0.1 # Bonus for filling internal gap
 
                       current_conf = pat_strength * enhancement_factor # Confidence is base strength * enhancement
                       if current_conf > current_max_conf_for_pval: #
                           current_max_conf_for_pval = current_conf #
               
               if current_max_conf_for_pval > max_confidence_score_for_cell_gm20: #
                   max_confidence_score_for_cell_gm20 = current_max_conf_for_pval #
           
           scores[r_idx, c_idx] = MathUtils.normalize_value(max_confidence_score_for_cell_gm20, 0, 1.0, clamp=True) #
 
   return scores #
 
# --- Module Registration ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
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
 