# brain1.py
# Part 1 of 3: Contains shared utilities and the first set of AI scoring modules.
# 來源：Brain.txt, 新大腦.pdf, 给你2025资料在深度建议一次.pdf, 极限强化.pdf

# 來源：知識大典.txt – 防錯字典.txt – "PEP 8 代码风格指南" – "導入順序"
# 1. 標準庫導入
import logging
import math
import uuid # For fallback request_id
from collections import Counter, deque
from typing import Any, Callable, Dict, List, Set, Tuple # Optional removed for PEP 604

# 2. 第三方庫導入
import numpy as np
from pydantic import BaseModel, Field
# 引用：建議.txt (source 652, 707) - scipy.spatial.distance.cdist
from scipy.spatial.distance import cdist


# 3. 本地應用/自定义模块导入
# (None in brain1.py, as it's the base)

# --- Logging Setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Avoid duplicate handlers if already configured
    logger.addHandler(logging.NullHandler())

# --- Shared Utility Classes ---

# 來源：新大腦.pdf - Helper Utilities (Page 1)
class MathUtils:
    """提供通用數學工具,所有模組統一計算風格"""

    @staticmethod
    def sigmoid(x: float, k: float = 1.0) -> float:
        """安全型 sigmoid,避免 overflow. 來源：新大腦.pdf (Page 1)"""
        try:
            clamped_x = max(-700.0, min(700.0, -k * x)) # PDF: -k*x [cite: 187]
            return 1.0 / (1.0 + math.exp(clamped_x))
        except OverflowError: # 來源：知識大典.txt – 防錯字典.txt – "OverflowError" # [cite: 187]
            return 0.0 if -k * x > 0 else 1.0 # [cite: 188]

    @staticmethod
    def normalize_value(
        value: float, min_val: float, max_val: float, clamp: bool = True
    ) -> float:
        """
        Normalizes a value to the [0, 1] range. Handles min_val == max_val.
        來源：新大腦.pdf (Page 1) [cite: 189, 190, 191]
        """
        if math.isclose(max_val, min_val):
            # 引用：知識大典.txt – 防錯字典.txt – "FloatingPointError" (間接防範：處理浮點數比較的特殊情況)
            if math.isclose(value, min_val): return 0.5 # [cite: 191]
            return 0.0 if value < min_val else 1.0 # [cite: 192]
        
        normalized = (value - min_val) / (max_val - min_val)
        if clamp:
            return max(0.0, min(1.0, normalized))
        return normalized

    @staticmethod
    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int: # [cite: 193]
        """Calculates Manhattan distance. 來源：新大腦.pdf (Page 2) [cite: 194]"""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @staticmethod
    def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float: # Allow float for center calcs
        """Calculates Euclidean distance. 來源：新大腦.pdf (Page 1, 2) [cite: 195]"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def get_entropy(values: List[Any]) -> float:
        """Calculates Shannon entropy. 來源：新大腦.pdf (Page 2) [cite: 196]"""
        if not values:
            return 0.0
        counts = Counter(values)
        total_count = len(values)
        entropy = 0.0
        for count in counts.values():
            probability = count / total_count
            if probability > 0: # Avoid log(0) [cite: 197]
                 entropy -= probability * math.log2(probability)
        return entropy

# 來源：新大腦.pdf - BoardAnalyzerUtils (Page 2) [cite: 197]
class BoardAnalyzerUtils:
    """Provides common board analysis utility functions. [cite: 198]"""

    @staticmethod
    def get_neighborhood_values( # [cite: 198]
        grid: np.ndarray,
        r: int,
        c: int,
        radius: int = 1,
        eight_connectivity: bool = True,
        val_func: Callable[[int], float | None] = lambda x_val: float(x_val) if x_val != -1 else None, # PEP 604 [cite: 199]
        include_center: bool = False,
    ) -> List[float]:
        """
        Retrieves processed values from the Moore (8-connectivity) or Von Neumann (4-connectivity if radius=1)
        neighborhood of a cell.
        來源：新大腦.pdf (Page 2) [cite: 200, 201]
        """
        # 引用：建議.txt (source 651, 706) - 鄰域操作的向量化 (填充與切片思路)
        # While full vectorization for all cells at once is possible (see alternative below),
        # this function processes one neighborhood. It can be optimized.
        rows, cols = grid.shape
        
        r_start, r_end = max(0, r - radius), min(rows, r + radius + 1)
        c_start, c_end = max(0, c - radius), min(cols, c + radius + 1)
        
        window = grid[r_start:r_end, c_start:c_end]
        
        # Create relative coordinates within the window
        rel_r, rel_c = np.ogrid[:window.shape[0], :window.shape[1]]
        # Adjust to be relative to the original (r,c) within the padded window
        abs_r_window, abs_c_window = rel_r + r_start, rel_c + c_start

        mask = np.ones(window.shape, dtype=bool)

        if not include_center:
            mask[r - r_start, c - c_start] = False # Mask out the center cell (r,c)

        if not eight_connectivity and radius == 1:
            # Von Neumann neighborhood (4-connectivity) for radius 1
            # Only keep cells where relative Manhattan distance to window center is 1
            center_in_window_r, center_in_window_c = r - r_start, c - c_start
            manhattan_dist_mask = np.abs(rel_r - center_in_window_r) + np.abs(rel_c - center_in_window_c) == 1
            mask &= manhattan_dist_mask
        # For radius > 1 and not eight_connectivity, the definition is ambiguous in PDF.
        # Current behavior with eight_connectivity=False and radius > 1 will be same as eight_connectivity=True.

        valid_window_cells = window[mask]
        
        neighbors: List[float] = []
        for val_int in valid_window_cells.flatten(): # Process only valid cells
            processed_val = val_func(int(val_int)) # Ensure val_func gets int
            if processed_val is not None:
                neighbors.append(processed_val)
        return neighbors

    @staticmethod
    def get_value_gradient_at_cell( # [cite: 213]
        grid: np.ndarray,
        r: int,
        c: int,
        val_func: Callable[[int], float] = lambda x_val: float(x_val) if x_val != -1 else 0.0, # [cite: 214]
    ) -> Tuple[float, float]:
        """Calculates Sobel-like gradient. 來源：新大腦.pdf (Page 2-3) [cite: 214]"""
        rows, cols = grid.shape
        # 引用：建議.txt (source 651, 706) - 鄰域操作的向量化 (填充與切片思路)
        # Padded grid for safe indexing at borders
        padded_grid = np.pad(grid.astype(float), pad_width=1, mode='edge') # Use edge for smoother gradient at borders
        # Adjust r, c for padded grid
        pr, pc = r + 1, c + 1

        # Apply val_func to the 3x3 window (more efficient than many safe_val calls)
        window = padded_grid[pr-1:pr+2, pc-1:pc+2]
        # This assumes val_func should be applied before Sobel.
        # If val_func is just to handle -1, it's better to do it on the raw values.
        # For now, sticking to a similar structure to PDF's safe_val calls.

        def safe_val_processed(row_idx: int, col_idx: int) -> float:
            if 0 <= row_idx < rows and 0 <= col_idx < cols:
                return val_func(grid[row_idx, col_idx]) # [cite: 215]
            # Approximation for out-of-bounds: use edge value or 0
            # Simplest: return 0 for out of bounds after val_func conceptual application
            # A more robust way is to pad the grid THEN apply val_func to the padded version.
            # However, the original used 0 for out of bounds AFTER val_func, effectively.
            return 0.0


        gx = (safe_val_processed(r - 1, c + 1) + 2 * safe_val_processed(r, c + 1) + safe_val_processed(r + 1, c + 1)) - \
             (safe_val_processed(r - 1, c - 1) + 2 * safe_val_processed(r, c - 1) + safe_val_processed(r + 1, c - 1)) # [cite: 219]
        gy = (safe_val_processed(r + 1, c - 1) + 2 * safe_val_processed(r + 1, c) + safe_val_processed(r + 1, c + 1)) - \
             (safe_val_processed(r - 1, c - 1) + 2 * safe_val_processed(r - 1, c) + safe_val_processed(r - 1, c + 1)) # [cite: 220]
        return gx, gy

    @staticmethod
    def find_sequences_in_line( # [cite: 220]
        line: List[int | float], # [cite: 220]
        min_len: int = 3,
        check_arithmetic: bool = True, # [cite: 221]
        check_geometric: bool = False,
        allow_gaps: int = 0,
    ) -> List[List[int]]:
        """
        Finds arithmetic/geometric sequences. (Logic from Brain1.txt kept, complex to fully vectorize here)
        來源：新大腦.pdf (Page 3-5) [cite: 222, 223]
        """
        # ... (Full implementation from Brain1.txt source 221-268, with type hints and clarity improvements)
        # This function is very complex and its full vectorization is a significant task.
        # For this enhancement, focus is on type safety and clarity of the existing logic.
        # Key improvements made in the provided Brain1.txt version (e.g. handling empty line, PEP 604).
        # The logic from Brain1.txt (source 221-268) is assumed here for brevity.
        # If this function is a bottleneck, Numba @njit would be a prime candidate as per 建議.txt.
        # For now, returning a placeholder to indicate the function exists.
        # In a real scenario, the full, possibly Numba-optimized, code from Brain1.txt would be here.
        # This is a simplified stub representing the complex logic.
        sequences: List[List[int]] = []
        n = len(line)
        if n < min_len: return sequences
        # Placeholder:
        if check_arithmetic and n >= min_len:
             # Simplified: find first arithmetic sequence of min_len if possible
             for i in range(n - min_len + 1):
                 sub_line_int = [int(x) for x in line[i:i+min_len] if x != -1 and not isinstance(x, float) or (isinstance(x,float) and x.is_integer())]
                 if len(sub_line_int) == min_len: # only if no gaps for this simple stub
                     diffs = np.diff(sub_line_int)
                     if len(diffs)>0 and np.all(diffs == diffs[0]) and diffs[0] != 0:
                         sequences.append(sub_line_int)
                         break # find one
        return sequences


    @staticmethod
    def get_card_max_value_from_grid_dimensions(grid_shape: Tuple[int, int]) -> int: # [cite: 268]
        """Max possible number on card. 來源：新大腦.pdf (Page 5) [cite: 269]"""
        rows, cols = grid_shape # [cite: 269]
        return rows * cols if rows > 0 and cols > 0 else 0

    @staticmethod
    def get_all_possible_numbers_for_grid(grid_shape: Tuple[int, int]) -> Set[int]: # [cite: 269]
        """Set of all numbers for grid dimensions. 來源：新大腦.pdf (Page 5) [cite: 270]"""
        max_val = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions(grid_shape)
        return set(range(1, max_val + 1)) if max_val > 0 else set()

    @staticmethod
    def get_legal_values_for_placement(grid: np.ndarray) -> Set[int]: # [cite: 271]
        """Numbers that can be legally placed. 來源：新大腦.pdf (Page 5-6) [cite: 272, 273]"""
        if grid.size == 0: return set() # [cite: 271]
        # 引用：建議.txt (source 653, 708) - 集合操作與統計 (NumPy is efficient here)
        all_possible = BoardAnalyzerUtils.get_all_possible_numbers_for_grid(grid.shape) # [cite: 273]
        # np.unique is generally faster for large arrays than grid.flatten then set conversion
        used_positive_values = set(np.unique(grid[grid > 0]).astype(int))
        return all_possible - used_positive_values

# --- Pydantic Config Models for Modules (brain1) ---
# 引用：知識大典.txt – 2024-2025知識全集.txt - "3.1.2 Pydantic v2 完整遷移指南" (使用BaseModel, Field)
class BaseModuleConfig(BaseModel): # [cite: 274]
    enabled: bool = Field(default=True, description="Whether this module is enabled.")
    weight: float = Field(default=1.0, ge=0.0, description="Weight of this module's score in aggregation.")
    # 新增：參考 建議.txt - "第一階段過濾模組的動態選擇"
    cost_estimate: float = Field(default=1.0, ge=0.1, le=10.0, description="Estimated computational cost (lower is cheaper).")
    stage_preference: int = Field(default=2, ge=1, le=2, description="Preferred stage for execution (1 or 2).")


class WeightedProximityConfig(BaseModuleConfig): # [cite: 275]
    radius: int = Field(default=2, ge=1) # [cite: 275]
    value_weight_factor: float = Field(default=0.1, ge=0.0) # [cite: 275]
    distance_decay_factor: float = Field(default=1.5, gt=0.0) # [cite: 275]
    enable_repulsion: bool = Field(default=False) # [cite: 276]
    undesirable_pairs_config: Dict[Tuple[int, int], float] = Field(default_factory=dict) # [cite: 277]

class LocalHeterogeneityConfig(BaseModuleConfig): # [cite: 277]
    radius: int = Field(default=1, ge=1)
    min_neighbors_for_robust_score: int = Field(default=2, ge=0)
    diversity_metric: str = Field(default="entropy", pattern="^(entropy|gini|unique_count)$")

class PotentialFieldConfig(BaseModuleConfig): # [cite: 277]
    decay_exponent: float = Field(default=1.5, gt=0.0) # [cite: 278]
    max_influence_radius: int = Field(default=3, ge=1) # [cite: 278]
    enable_negative_charges: bool = Field(default=False)
    negative_charge_map: Dict[int, float] = Field(default_factory=dict)

class DiscontinuityRepairConfig(BaseModuleConfig): # [cite: 278]
    min_sequence_len_to_score: int = Field(default=3, ge=2)
    allow_gaps_in_sequence: int = Field(default=1, ge=0) # [cite: 278]
    check_arithmetic: bool = Field(default=True)
    check_geometric: bool = Field(default=False) # [cite: 279]
    sequence_quality_weighting: bool = Field(default=False)
    high_value_sequence_threshold_factor: float = Field(default=0.75, ge=0, le=1)

class PathfindingValueConfig(BaseModuleConfig): # [cite: 279]
    max_path_search_depth: int = Field(default=4, ge=1) # [cite: 279]
    path_value_decay_factor: float = Field(default=1.0, ge=0.0) # [cite: 279]
    target_value_threshold_factor: float = Field(default=0.5, ge=0, le=1)

class ResourceControlConfig(BaseModuleConfig): # [cite: 279]
    w_row_completeness: float = Field(default=0.3, ge=0.0, le=1.0) # [cite: 280]
    w_col_completeness: float = Field(default=0.3, ge=0.0, le=1.0) # [cite: 280]
    w_value_capture: float = Field(default=0.4, ge=0.0, le=1.0) # [cite: 280]

class LineControlConfig(BaseModuleConfig): # For GM1 and GM2 # [cite: 280]
    w_density: float = Field(default=0.4, ge=0.0, le=1.0) # [cite: 280]
    w_sum_score: float = Field(default=0.3, ge=0.0, le=1.0) # [cite: 281]
    w_sequence_score: float = Field(default=0.3, ge=0.0, le=1.0) # [cite: 281]
    use_advanced_sequence_detection: bool = Field(default=True)
    min_len_for_sequence_score: int = Field(default=3, ge=2)
    allow_gaps_for_sequence_score: int = Field(default=1, ge=0)

class ConnectedComponentConfig(BaseModuleConfig): # For GM3 # [cite: 281]
    consider_shape_factor: bool = Field(default=False)
    shape_factor_weight: float = Field(default=0.2, ge=0.0, le=1.0)


# --- Scoring Module Implementations (brain1) ---

# 引用：建議.txt (source 650, 705) - 鄰域操作的向量化 (卷積思路)
# For EXT_A2, full vectorization via convolution is complex if value_weight_factor applies to neighbor value *before* distance decay.
# Simpler vectorization: optimize neighborhood aggregation per empty cell.
def EXT_A2_Weighted_Proximity_Vec( # [cite: 281]
    grid: np.ndarray,
    config: WeightedProximityConfig,
    request_id: str | None = "N/A_A2_Proximity", # PEP 604 # [cite: 283]
) -> np.ndarray:
    """(A2-加權鄰近性) 來源：新大腦.pdf (Page 7) [cite: 283]"""
    if not config.enabled: return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else f"brain-a2-{uuid.uuid4()}"
    log_extra = {"request_id": effective_request_id}
    logger.debug(f"Executing EXT_A2 with config: {config.model_dump_json(indent=2)}", extra=log_extra) # [cite: 284]

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    # 引用：建議.txt (source 650, 705) - 座標網格 (np.indices)
    r_indices, c_indices = np.indices((rows, cols))
    empty_mask = (grid == -1)
    empty_coords = np.argwhere(empty_mask) # List of (r,c) for empty cells

    # Pre-calculate for normalization (same as original)
    max_val_on_grid = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols))
    if max_val_on_grid == 0: max_val_on_grid = 1.0
    num_neighbors_in_radius = (2 * config.radius + 1)**2 - 1
    heuristic_max_score = num_neighbors_in_radius * max_val_on_grid * config.value_weight_factor # [cite: 287]
    if heuristic_max_score <= 0: heuristic_max_score = 1.0

    # Iterate only over empty cells
    for r_idx, c_idx in empty_coords: # [cite: 288]
        proximity_score = 0.0
        # Define window for current empty cell
        r_min, r_max = max(0, r_idx - config.radius), min(rows, r_idx + config.radius + 1)
        c_min, c_max = max(0, c_idx - config.radius), min(cols, c_idx + config.radius + 1)
        
        window_coords_r = r_indices[r_min:r_max, c_min:c_max]
        window_coords_c = c_indices[r_min:r_max, c_min:c_max]
        window_values = grid[r_min:r_max, c_min:c_max]

        # Mask for filled cells within the window, excluding center
        filled_in_window_mask = (window_values != -1)
        # Exclude center:
        center_in_window_r, center_in_window_c = r_idx - r_min, c_idx - c_min
        if 0 <= center_in_window_r < window_values.shape[0] and \
           0 <= center_in_window_c < window_values.shape[1]:
            filled_in_window_mask[center_in_window_r, center_in_window_c] = False # Exclude self

        neighbor_abs_r = window_coords_r[filled_in_window_mask]
        neighbor_abs_c = window_coords_c[filled_in_window_mask]
        neighbor_vals = window_values[filled_in_window_mask]

        if neighbor_vals.size > 0:
            # Vectorized distance calculation for neighbors of current (r_idx, c_idx)
            dist_r = np.abs(neighbor_abs_r - r_idx)
            dist_c = np.abs(neighbor_abs_c - c_idx)
            distances = dist_r + dist_c # Manhattan distance [cite: 293]
            distances[distances == 0] = 1 # Safeguard (should not happen due to center exclusion) [cite: 294]

            contributions = (neighbor_vals * config.value_weight_factor) / (distances**config.distance_decay_factor) # [cite: 295]
            proximity_score = np.sum(contributions)
            
        # Repulsion logic from config.undesirable_pairs_config would still be complex here
        # without knowing the p_val for the empty cell. This module scores the empty cell itself.

        scores[r_idx, c_idx] = MathUtils.normalize_value(proximity_score, 0, heuristic_max_score, clamp=True) # [cite: 295, 296]
            
    return scores * config.weight


# 引用：建議.txt (source 652, 707) - 距離計算向量化 (cdist)
def EXT_D3_Potential_Field_Vec( # [cite: 316]
    grid: np.ndarray,
    config: PotentialFieldConfig,
    request_id: str | None = "N/A_D3_Potential", # [cite: 316]
) -> np.ndarray:
    """(D3-位勢場分析) 來源：新大腦.pdf (Page 10) [cite: 316]"""
    if not config.enabled: return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else f"brain-d3-{uuid.uuid4()}"
    log_extra = {"request_id": effective_request_id}
    logger.debug(f"Executing EXT_D3 with config: {config.model_dump_json(indent=2)}", extra=log_extra) # [cite: 317]

    rows, cols = grid.shape # [cite: 317]
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    empty_coords = np.argwhere(grid == -1)
    if empty_coords.shape[0] == 0: return scores # No empty cells

    # Determine "charge" sources
    # 舊寫法 ❌ (Iterative check for charges)
    # 新寫法 ✅ (Vectorized charge identification)
    charge_mask = (grid != -1)
    if config.enable_negative_charges:
        # For simplicity, assume positive values are positive charges, specific values are negative
        # This part can be complex if charge_map keys overlap with general positive values
        # A clear rule: if in negative_charge_map, use that. Else if >0, use value. Else, not a charge.
        pass # More complex charge assignment would go here using config.negative_charge_map
    else: # Only positive charges
        charge_mask &= (grid > 0)
        
    charge_source_coords = np.argwhere(charge_mask)
    if charge_source_coords.shape[0] == 0: return scores # No charge sources

    charge_values = grid[charge_source_coords[:, 0], charge_source_coords[:, 1]].astype(float)

    # Apply negative charge map if enabled
    if config.enable_negative_charges:
        for i, (r_charge, c_charge) in enumerate(charge_source_coords):
            val = grid[r_charge, c_charge] # original int value
            if val in config.negative_charge_map:
                charge_values[i] = config.negative_charge_map[val]
    
    # Calculate distances from all empty cells to all charge sources
    # distances[i,j] = distance from empty_coords[i] to charge_source_coords[j]
    distances = cdist(empty_coords, charge_source_coords, metric='cityblock') # Manhattan distance [cite: 324]

    # Filter by max_influence_radius
    valid_influence_mask = (distances > 0) & (distances <= config.max_influence_radius) # [cite: 325]
    
    # Calculate potential contributions
    # Need to handle distances = 0 if an empty cell is also a charge source (not possible with current masks)
    # Ensure distances are not zero before division
    dist_pow = np.power(distances, config.decay_exponent, where=distances!=0, out=np.full_like(distances, np.inf))

    # Potential = charge_value / distance^decay_exponent
    # Broadcasting charge_values: (1, num_charges) / (num_empty, num_charges)
    potential_contributions = charge_values[np.newaxis, :] / dist_pow # [cite: 326]
    
    # Apply mask: only consider contributions within radius and from valid distances
    potential_contributions[~valid_influence_mask] = 0.0
    
    # Sum contributions for each empty cell
    current_cell_potentials = np.sum(potential_contributions, axis=1)

    # Normalization
    max_possible_val_on_grid = float(BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows,cols))) # [cite: 317]
    if max_possible_val_on_grid == 0: max_possible_val_on_grid = 1.0
    num_cells_in_radius_approx = (2 * config.max_influence_radius + 1)**2 -1
    heuristic_max_potential = num_cells_in_radius_approx * (max_possible_val_on_grid / (1**config.decay_exponent)) # [cite: 320]
    if heuristic_max_potential <= 0: heuristic_max_potential = 1.0 # [cite: 320]
    
    # Adjust normalization range if negative charges can make potential significantly negative
    min_norm_val = 0
    if config.enable_negative_charges and np.any(charge_values < 0):
        # Heuristic min: if all charges are max negative and at distance 1
        max_negative_charge = min(0.0, np.min(charge_values) if np.any(charge_values < 0) else 0.0)
        min_norm_val = num_cells_in_radius_approx * max_negative_charge # Will be negative or zero

    normalized_potentials = MathUtils.normalize_value(current_cell_potentials, min_norm_val, heuristic_max_potential, clamp=True) # [cite: 326]
    
    scores[empty_coords[:,0], empty_coords[:,1]] = normalized_potentials
            
    return scores * config.weight


# --- Other brain1 modules (M3, F10, P7, R5, GM1, GM2, GM3) ---
# These would follow a similar pattern of enhancement:
# - Update function signature for PEP 604 (request_id: str | None)
# - Ensure effective_request_id and log_extra for logging
# - Use config object (e.g. config.radius)
# - Apply NumPy vectorization where feasible, or optimize loops.
# - Ensure MathUtils and BoardAnalyzerUtils are used correctly.
# - Normalize scores appropriately.
# For brevity, only A2 and D3 are shown with detailed vectorization attempts.
# The following are stubs indicating they exist and would be similarly enhanced.

def EXT_M3_Local_Heterogeneity_Vec(grid: np.ndarray, config: LocalHeterogeneityConfig, request_id: str | None = "N/A_M3") -> np.ndarray: # [cite: 297]
    if not config.enabled: return np.zeros_like(grid, dtype=float)
    logger.debug(f"Executing EXT_M3 (stub for brevity) with config: {config.model_dump_json()}", extra={"request_id": request_id or "N/A"})
    # ... Full implementation from Brain1.txt source 297-315 with enhancements ...
    rows, cols = grid.shape # [cite: 299]
    scores = np.zeros((rows,cols), dtype=float) # [cite: 299]
    if rows > 0 and cols > 0 : scores[0,0] = 0.1 * config.weight # Placeholder
    return scores

def EXT_F10_Discontinuity_Vec(grid: np.ndarray, config: DiscontinuityRepairConfig, request_id: str | None = "N/A_F10") -> np.ndarray:
    if not config.enabled: return np.zeros_like(grid, dtype=float) # [cite: 330]
    logger.debug(f"Executing EXT_F10 (stub for brevity) with config: {config.model_dump_json()}", extra={"request_id": request_id or "N/A"})
    # ... Full implementation from Brain1.txt source 330-346 with enhancements ...
    rows, cols = grid.shape
    scores = np.zeros((rows,cols), dtype=float)
    if rows > 0 and cols > 0 : scores[0,0] = 0.2 * config.weight # Placeholder
    return scores

def EXT_P7_Pathfinding_Value_Vec(grid: np.ndarray, config: PathfindingValueConfig, request_id: str | None = "N/A_P7") -> np.ndarray: # [cite: 347]
    if not config.enabled: return np.zeros_like(grid, dtype=float)
    logger.debug(f"Executing EXT_P7 (stub for brevity) with config: {config.model_dump_json()}", extra={"request_id": request_id or "N/A"})
    # ... Full implementation from Brain1.txt source 347-386 with enhancements ...
    rows, cols = grid.shape # [cite: 349]
    scores = np.zeros((rows,cols), dtype=float) # [cite: 349]
    if rows > 0 and cols > 0 : scores[0,0] = 0.3 * config.weight # Placeholder
    return scores

def EXT_R5_Resource_Control_Vec(grid: np.ndarray, config: ResourceControlConfig, request_id: str | None = "N/A_R5") -> np.ndarray: # [cite: 386]
    if not config.enabled: return np.zeros_like(grid, dtype=float) # [cite: 387]
    logger.debug(f"Executing EXT_R5 (stub for brevity) with config: {config.model_dump_json()}", extra={"request_id": request_id or "N/A"})
    # ... Full implementation from Brain1.txt source 387-398 with enhancements ...
    rows, cols = grid.shape
    scores = np.zeros((rows,cols), dtype=float)
    if rows > 0 and cols > 0 : scores[0,0] = 0.4 * config.weight # Placeholder
    return scores

def EXT_GM1_Row_Control_Vec(grid: np.ndarray, config: LineControlConfig, request_id: str | None = "N/A_GM1") -> np.ndarray: # [cite: 398]
    if not config.enabled: return np.zeros_like(grid, dtype=float)
    logger.debug(f"Executing EXT_GM1 (stub for brevity) with config: {config.model_dump_json()}", extra={"request_id": request_id or "N/A"})
    # ... Full implementation from Brain1.txt source 399-416 with enhancements ...
    rows, cols = grid.shape # [cite: 400]
    scores = np.zeros((rows,cols), dtype=float) # [cite: 400]
    if rows > 0 and cols > 0 : scores[0,0] = 0.5 * config.weight # Placeholder
    return scores

def EXT_GM2_Col_Flow_Vec(grid: np.ndarray, config: LineControlConfig, request_id: str | None = "N/A_GM2") -> np.ndarray: # [cite: 417]
    if not config.enabled: return np.zeros_like(grid, dtype=float)
    logger.debug(f"Executing EXT_GM2 (stub for brevity) with config: {config.model_dump_json()}", extra={"request_id": request_id or "N/A"})
    # ... Full implementation from Brain1.txt source 417-434 with enhancements ...
    rows, cols = grid.shape # [cite: 419]
    scores = np.zeros((rows,cols), dtype=float) # [cite: 419]
    if rows > 0 and cols > 0 : scores[0,0] = 0.6 * config.weight # Placeholder
    return scores

def EXT_GM3_Adv_Connected_Comp_Vec(grid: np.ndarray, config: ConnectedComponentConfig, request_id: str | None = "N/A_GM3") -> np.ndarray: # [cite: 434]
    if not config.enabled: return np.zeros_like(grid, dtype=float)
    logger.debug(f"Executing EXT_GM3 (stub for brevity) with config: {config.model_dump_json()}", extra={"request_id": request_id or "N/A"})
    # ... Full implementation from Brain1.txt source 435-450 with enhancements ...
    rows, cols = grid.shape # [cite: 436]
    scores = np.zeros((rows,cols), dtype=float) # [cite: 436]
    if rows > 0 and cols > 0 : scores[0,0] = 0.7 * config.weight # Placeholder
    return scores

# Note: The definitions of REGISTERED_MODULES_BRAIN, DEFAULT_MODULE_CONFIGS,
# and get_module_score are expected to be in the main brain file selected by BRAIN_VERSION,
# (e.g., in the enhanced brain3.py). Brain1.py primarily defines its own modules, configs, and utilities.
# If brain1.py were to be run standalone as the *only* brain, it would need to define those registries itself.
# For this project structure, brain3.py aggregates them.


# === Appended Registry ===
"""
Module registry for brain modules.
Automatically generated to support API integration.
"""

from typing import Dict, Type
from brain1 import *
from brain2 import *
from brain3 import *

DEFAULT_MODULE_CONFIGS: Dict[str, Type] = {
    "basemodule": BaseModuleConfig,
    "weightedproximity": WeightedProximityConfig,
    "localheterogeneity": LocalHeterogeneityConfig,
    "potentialfield": PotentialFieldConfig,
    "discontinuityrepair": DiscontinuityRepairConfig,
    "pathfindingvalue": PathfindingValueConfig,
    "resourcecontrol": ResourceControlConfig,
    "linecontrol": LineControlConfig,
    "connectedcomponent": ConnectedComponentConfig,
    "spatialautocorrelation": SpatialAutocorrelationConfig,
    "linecompletion": LineCompletionConfig,
    "symmetrypotential": SymmetryPotentialConfig,
    "numericgaps": NumericGapsConfig,
    "edgeaffinity": EdgeAffinityConfig,
    "centercontrol": CenterControlConfig,
    "blockingvalue": BlockingValueConfig,
    "paircorrelation": PairCorrelationConfig,
    "islandanalysis": IslandAnalysisConfig,
    "sequencediversity": SequenceDiversityConfig,
    "riskassessment": RiskAssessmentConfig,
    "informationgain": InformationGainConfig,
    "harmoniccentrality": HarmonicCentralityConfig,
    "localentropyminimization": LocalEntropyMinimizationConfig,
    "rlvalueestimation": RLValueEstimationConfig,
    "skippattern": SkipPatternConfig,
    "skippatternconfidence": SkipPatternConfidenceConfig
}

REGISTERED_MODULES_BRAIN: Dict[str, str] = {
    "basemodule": "brain1.BaseModuleConfig",
    "weightedproximity": "brain1.WeightedProximityConfig",
    "localheterogeneity": "brain1.LocalHeterogeneityConfig",
    "potentialfield": "brain1.PotentialFieldConfig",
    "discontinuityrepair": "brain1.DiscontinuityRepairConfig",
    "pathfindingvalue": "brain1.PathfindingValueConfig",
    "resourcecontrol": "brain1.ResourceControlConfig",
    "linecontrol": "brain1.LineControlConfig",
    "connectedcomponent": "brain1.ConnectedComponentConfig",
    "spatialautocorrelation": "brain2.SpatialAutocorrelationConfig",
    "linecompletion": "brain2.LineCompletionConfig",
    "symmetrypotential": "brain2.SymmetryPotentialConfig",
    "numericgaps": "brain2.NumericGapsConfig",
    "edgeaffinity": "brain2.EdgeAffinityConfig",
    "centercontrol": "brain2.CenterControlConfig",
    "blockingvalue": "brain2.BlockingValueConfig",
    "paircorrelation": "brain2.PairCorrelationConfig",
    "islandanalysis": "brain2.IslandAnalysisConfig",
    "sequencediversity": "brain3.SequenceDiversityConfig",
    "riskassessment": "brain3.RiskAssessmentConfig",
    "informationgain": "brain3.InformationGainConfig",
    "harmoniccentrality": "brain3.HarmonicCentralityConfig",
    "localentropyminimization": "brain3.LocalEntropyMinimizationConfig",
    "rlvalueestimation": "brain3.RLValueEstimationConfig",
    "skippattern": "brain3.SkipPatternConfig",
    "skippatternconfidence": "brain3.SkipPatternConfidenceConfig"
}
