import os
import json
import math
import time
import uuid
import logging
import numpy as np
from typing import Any, List, Dict, Tuple, Optional, Callable
from collections import deque, Counter
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, validator
from ortools.sat.python import cp_model
import uvicorn

# main.py (FastAPI with "Industry Extreme" Analyzer)

import json
import os
import uuid # For request IDs
import time # For process time
import logging # For enhanced logging
from datetime import datetime # For health check timestamp
import math # For advanced math in modules
from collections import Counter, deque # For advanced logic in modules

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, validator, Field
from typing import List, Dict, Tuple, Callable, Optional, Any
import numpy as np
from ortools.sat.python import cp_model # Assuming Google OR-Tools is used
from celery.result import AsyncResult # Kept as per original, though not used in /analyze
from celery_worker import solve_task  # Kept as per original

# --- Logging Configuration ---
# (User's original logging config - kept as is)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - RequestID: %(request_id)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# --- Application Setup ---
# (User's original app setup - kept as is)
app = FastAPI(
    title="Plug-inæ¬é + å¼µéæµ + èªåæ¸å­ç¯å - AI Manager Enabled (Extreme Analyzer v1.0)",
    version="3.3", # Incremented version for extreme analyzer
    description="Enhanced analysis API with AI Manager capabilities and 'Industry Extreme' analyzer modules."
)

# --- Middleware for Request ID and Logging ---
# (User's original middleware - kept as is)
class RequestContextLogMiddleware:
    async def __call__(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        logger.info(f"Request processed: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s", extra={'request_id': request_id})
        return response

app.middleware("http")(RequestContextLogMiddleware())

# -----------------------------------------------------------------------------
# 0. Helper Utilities for Advanced Modules
# (These are from our previous "standalone extreme" version)
# -----------------------------------------------------------------------------

class MathUtils:
    @staticmethod
    def sigmoid(x: float, k: float = 1.0) -> float: # Added k for steepness control
        """Standard sigmoid function, maps any real number to (0,1). k controls steepness."""
        try:
            return 1 / (1 + math.exp(-k * x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    @staticmethod
    def normalize_value(value: float, min_val: float, max_val: float, clamp: bool = True) -> float:
        """Normalizes value to [0,1]. If clamp is True, clamps result to [0,1]."""
        if max_val == min_val:
            if value == min_val: return 0.0 # Or 0.5 if it's the only value
            # Ambiguous case: if all reference values are same, what's the normalized score?
            # Could return 0.0, 0.5, or 1.0 based on context. For now, 0.0 if below, 1.0 if above, 0.5 if equal.
            return 0.5 if math.isclose(value, min_val) else (0.0 if value < min_val else 1.0)

        normalized = (value - min_val) / (max_val - min_val)
        if clamp:
            return max(0.0, min(1.0, normalized))
        return normalized

    @staticmethod
    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

class BoardAnalyzerUtils: # Renamed to avoid conflict if user has similar named class
    @staticmethod
    def get_neighborhood_values(grid: np.ndarray, r: int, c: int, radius: int = 1,
                                eight_connectivity: bool = True,
                                val_func: Callable[[int], Optional[float]] = lambda x: float(x) if x != -1 else None,
                                include_center: bool = False) -> List[float]:
        """Gets numerical values from the neighborhood, excluding invalid/None from val_func."""
        neighbors = []
        rows, cols = grid.shape
        
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if not include_center and dr == 0 and dc == 0:
                    continue
                if not eight_connectivity and abs(dr) + abs(dc) > radius:
                    continue
                
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    processed_val = val_func(grid[nr, nc])
                    if processed_val is not None:
                        neighbors.append(processed_val)
        return neighbors

    @staticmethod
    def get_value_gradient_at_cell(grid: np.ndarray, r: int, c: int,
                                   val_func: Callable[[int], float] = lambda x: float(x) if x != -1 else 0.0) -> Tuple[float, float]:
        """Calculates simple numerical gradient (Sobel-like) at a cell."""
        rows, cols = grid.shape
        
        # Access grid safely, apply val_func, default to 0.0 if out of bounds or val_func fails
        def safe_val(r_in, c_in):
            if 0 <= r_in < rows and 0 <= c_in < cols:
                return val_func(grid[r_in, c_in])
            return 0.0 # Assume 0 for out-of-bounds for gradient calculation

        gx = (safe_val(r-1, c+1) + 2*safe_val(r, c+1) + safe_val(r+1, c+1)) - \
             (safe_val(r-1, c-1) + 2*safe_val(r, c-1) + safe_val(r+1, c-1))
        gy = (safe_val(r+1, c-1) + 2*safe_val(r+1, c) + safe_val(r+1, c+1)) - \
             (safe_val(r-1, c-1) + 2*safe_val(r-1, c) + safe_val(r-1, c-1)) # Typo in original, should be c and c+1 for last row
        # Corrected Gy:
        gy_corrected = (safe_val(r+1, c-1) + 2*safe_val(r+1, c) + safe_val(r+1, c+1)) - \
                       (safe_val(r-1, c-1) + 2*safe_val(r-1, c) + safe_val(r-1, c+1))

        return gx, gy_corrected

# --- Pydantic Models for API Request and Response ---
# (User's original Pydantic models - slight modifications for clarity if needed)
class ProposedValue(BaseModel):
    pos: List[int] = Field(..., min_items=2, max_items=2, description="Position [row, col]")
    value: int = Field(..., description="Proposed value for the cell (can be any int, -1 means proposing to clear, positive for placing)") # Allow positive or -1 for clearing. Original was gt=0

class AnalyzeRequest(BaseModel):
    new_card: List[List[int]] = Field(..., description="The game grid, -1 for empty cells")
    proposed_values: List[ProposedValue] = Field(..., description="List of proposed values to analyze")

    @validator("new_card")
    def check_rectangular_and_numeric(cls, g): # Kept as is
        if not g: raise ValueError("new_card cannot be empty")
        if any(not isinstance(row, list) for row in g): raise ValueError("new_card must be a list of lists")
        if any(len(row) != len(g[0]) for row in g): raise ValueError("new_card must be a rectangular grid")
        if any(not isinstance(val, int) for row in g for val in row): raise ValueError("All values in new_card must be integers")
        return g

    @validator("proposed_values", each_item=True)
    def check_pv_bounds_and_value(cls, pv, values): # Kept as is, pv.value > 0 check removed to align with ProposedValue model
        grid = values.get("new_card")
        if grid:
            rows, cols = len(grid), len(grid[0])
            r, c = pv.pos
            if not (0 <= r < rows and 0 <= c < cols):
                raise ValueError(f"Proposed position {pv.pos} is out of bounds for grid size {rows}x{cols}")
            # Allow proposing on filled cell IF the value is different (implies changing it)
            # OR if value is -1 (implies clearing it)
            # Original: if grid[r][c] != -1: raise ValueError -> only allow proposing on empty cells
            # This depends on game rules. For now, assume we can only propose on empty cells (-1)
            # for placing a positive value. If pv.value is -1, it could be to clear an existing cell.
            if pv.value != -1 and grid[r][c] != -1 : # Trying to place a positive value on an already filled cell
                 raise ValueError(f"Cell at {pv.pos} is already filled with {grid[r][c]}. To change, propose value=-1 first or new rules needed.")
            if pv.value == -1 and grid[r][c] == -1: # Trying to clear an already empty cell
                 raise ValueError(f"Cell at {pv.pos} is already empty. Cannot propose to clear it further.")
        return pv

class TensorRuleContribution(BaseModel):
    rule_name: str
    score_if_applied: float # CHANGED: This is now the raw float score from the module for the cell (before weighting)
    weight: float
    weighted_score: float # score_if_applied * weight

class CandidateDetail(BaseModel): # Kept as is
    pos: List[int]
    value: int
    is_valid_proposal: bool = True
    tensor_flow_contributions: List[TensorRuleContribution] = Field(default_factory=list)
    raw_tensor_flow_score: float
    mem_score_value: float
    final_objective_score: float
    is_selected_by_cp: bool = False
    cp_solver_notes: Optional[str] = None

class AnalyzeResultDetail(CandidateDetail): pass # Kept as is

class AnalyzeSuccessResponse(BaseModel): # Kept as is
    request_id: str
    status: str = "success"
    main_module_version: str = Field(default=app.version)
    analysis_engine_version: str # Will be defined for the "extreme" engine
    message: Optional[str] = None
    result: Optional[AnalyzeResultDetail] = None
    all_candidates_evaluated: List[CandidateDetail]

class AnalyzeErrorResponse(BaseModel): # Kept as is
    request_id: str
    status: str
    main_module_version: str = Field(default=app.version)
    analysis_engine_version: str
    message: str
    error_type: Optional[str] = None
    details: Optional[Any] = None

# --- Memory Handling ---
# (User's original memory handling - kept as is)
MEM_PATH = os.path.join(os.path.dirname(__file__), "memory_cards.json")
_memory_freq: Dict[Tuple[int, int, int], int] = {}
_total_samples_in_memory = 0

def load_memory_data(req_id: str = "startup"):
    global _memory_freq, _total_samples_in_memory
    _memory_freq = {}
    _total_samples_in_memory = 0
    try:
        if os.path.exists(MEM_PATH):
            with open(MEM_PATH, "r", encoding="utf-8") as f: data = json.load(f)
            for card in data.get("memory_cards", []):
                for r, row_data in enumerate(card):
                    for c, v_mem in enumerate(row_data): # Renamed v to v_mem
                        if v_mem != -1:
                            _memory_freq[(r, c, v_mem)] = _memory_freq.get((r, c, v_mem), 0) + 1
                            _total_samples_in_memory += 1
            logger.info(f"Memory data loaded: {_total_samples_in_memory} samples from {len(data.get('memory_cards', []))} cards.", extra={'request_id': req_id})
        else: logger.warning(f"Memory file not found: {MEM_PATH}. Mem score will be 0.", extra={'request_id': req_id})
    except Exception as e: logger.error(f"Error loading memory data from {MEM_PATH}: {e}", exc_info=True, extra={'request_id': req_id})

load_memory_data()

def mem_score(r: int, c: int, v_mem_proposed: int, legal_values_for_position: set) -> float: # Renamed v to v_mem_proposed
    # Note: legal_values_for_position might not be relevant here if mem_score is purely historical frequency.
    # User's original code uses it.
    if v_mem_proposed not in legal_values_for_position and v_mem_proposed != -1 : # allow -1 for clearing even if not in "placeable" legal_values
        return 0.0
    if _total_samples_in_memory == 0: return 0.0
    count = _memory_freq.get((r, c, v_mem_proposed), 0)
    return float(count) / float(_total_samples_in_memory)


# --- "Industry Extreme" Analyzer Modules (Vectorized for Grid Input) ---
# These will replace the original MODULE_FUNCS_VEC
# Each function will return a np.ndarray of float scores (R, C)

def ext_a2_weighted_proximity_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """
    Core Rule: Evaluates proximity to high-value elements and their influence using distance decay.
    Purpose: Identifies cells near valuable resources or strategic points, considering influence spread.
    ---
    Design Philosophy: Based on "Weighted Proximity & Influence". Considers self-value and diminishing influence of nearby high-value cells.
    Use Case: Finding optimal placement near resources, or areas under strong positive influence.
    Scoring Formula Principle: Score_cell = sigmoid( w1 * self_value_score + w2 * sum(influence_from_neighbor_i) ). Influence decays with distance.
    Compatibility: Handles integer grid values. -1 is empty/no value. Assumes positive values are generally better or have "value".
    Extensibility: Configurable target values, influence radius, decay functions, value mapping for different integers.
    Optimization & Extension Directions: Cache distances, use spatial indexing for large boards, dynamic target values.
    Possible Multi-version Logic: A2.v1 (simple proximity), A2.v2 (this version), A2.v3 (influence diffusion model).
    """
    logger.debug(f"Executing ext_a2_weighted_proximity_vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    # Parameters (could be class members if these were classes)
    target_value_threshold = 0.8 # For normalized 0-1 values; for raw ints, this needs to be dynamic or higher
    influence_radius = 3
    self_value_weight = 0.4
    influence_weight = 0.6
    
    def val_func_normalized(x_val: int) -> Optional[float]: # Example normalization for A2
        if x_val == -1: return 0.0 # Empty cells have no positive value
        # Normalize positive values, e.g. if max expected value is 10, then val/10
        # For simplicity, let's assume positive values are already somewhat scaled or use a fixed scale.
        # This part is crucial and game-specific. Let's say values 1-10 are common.
        if x_val > 0: return MathUtils.normalize_value(float(x_val), 1, 10, clamp=True) # Normalize from 1-10 to 0-1
        return 0.0 # Other non-positive values

    # This is a cell-by-cell calculation, which is less "vectorized" in numpy style
    # but necessary for complex per-cell logic.
    for r_idx in range(rows):
        for c_idx in range(cols):
            current_cell_val_norm = val_func_normalized(grid[r_idx, c_idx])
            
            self_value_score = 0.0
            if current_cell_val_norm >= target_value_threshold:
                self_value_score = 1.0
            elif current_cell_val_norm > 0:
                self_value_score = current_cell_val_norm * 0.5 # Partial score if positive but below threshold

            neighbor_influence_score_sum = 0.0
            total_possible_influence_normalization_factor = 0.0 # For averaging influence

            for r_offset in range(-influence_radius, influence_radius + 1):
                for c_offset in range(-influence_radius, influence_radius + 1):
                    if r_offset == 0 and c_offset == 0: continue

                    nr, nc = r_idx + r_offset, c_idx + c_offset
                    if 0 <= nr < rows and 0 <= nc < cols:
                        dist = MathUtils.manhattan_distance((r_idx, c_idx), (nr, nc))
                        if dist == 0 or dist > influence_radius: continue

                        neighbor_val_norm = val_func_normalized(grid[nr, nc])
                        
                        # Inverse square distance decay (or other: 1/dist, exp(-dist))
                        # Add epsilon to avoid division by zero if dist can be < 1 (not for Manhattan > 0)
                        weight_dist_decay = 1.0 / (dist ** 2 + 1e-6) 
                        total_possible_influence_normalization_factor += weight_dist_decay

                        if neighbor_val_norm >= target_value_threshold:
                            neighbor_influence_score_sum += weight_dist_decay * 1.0
                        elif neighbor_val_norm > 0:
                            neighbor_influence_score_sum += weight_dist_decay * neighbor_val_norm * 0.5
            
            normalized_influence_avg = 0.0
            if total_possible_influence_normalization_factor > 0:
                normalized_influence_avg = neighbor_influence_score_sum / total_possible_influence_normalization_factor
            
            combined_score_raw = (self_value_weight * self_value_score +
                                  influence_weight * normalized_influence_avg)
            
            # Sigmoid to map to 0-1 and add non-linearity. Adjust center and steepness.
            # combined_score_raw is likely 0-1. (X - 0.5) * k makes it centered around 0 for sigmoid input.
            scores[r_idx, c_idx] = MathUtils.sigmoid((combined_score_raw - 0.5) * 5.0) 
            
    return scores

def ext_m3_local_heterogeneity_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """
    Core Rule: Analyzes local region's "complexity" or "heterogeneity" using statistical measures like entropy or variance.
    Purpose: Identifies areas of high change/diversity or high consistency.
    ---
    Design Philosophy: Shift from simple count (original M3) to statistical distribution analysis of values in a neighborhood.
    Use Case: Finding "opportunity points" (high change) or "stable zones" (high consistency). This version rewards heterogeneity.
    Scoring Formula Principle: Score_cell = normalized_entropy (or normalized_std_dev) of 3x3 neighborhood values.
    Compatibility: Handles integer grid values. -1 is empty. Assumes positive values contribute to heterogeneity.
    Extensibility: Different neighborhood sizes, weighted entropy/variance, other statistical measures (e.g., Gini impurity).
    Optimization & Extension Directions: Pre-calculate value bins for entropy, use optimized statistical functions.
    Possible Multi-version Logic: M3.v1 (count), M3.v2 (entropy), M3.v3 (variance/std dev - this version), M3.v4 (Gini).
    """
    logger.debug(f"Executing ext_m3_local_heterogeneity_vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    # Parameters
    metric = "stddev" # "entropy" or "stddev"
    
    def val_func_for_stats(x_val: int) -> Optional[float]:
        if x_val == -1: return None # Exclude empty cells from statistical calculation
        return float(x_val)

    for r_idx in range(rows):
        for c_idx in range(cols):
            # Get 3x3 neighborhood values (including center)
            neighborhood_vals = BoardAnalyzerUtils.get_neighborhood_values(
                grid, r_idx, c_idx, radius=1, eight_connectivity=True,
                val_func=val_func_for_stats, include_center=True
            )

            if not neighborhood_vals or len(neighborhood_vals) < 2 : # Need at least 2 values for stddev, or for meaningful entropy
                scores[r_idx, c_idx] = 0.0 # Or a neutral 0.5 if lack of data means unknown
                continue

            if metric == "stddev":
                std_dev = np.std(neighborhood_vals)
                # Normalize std_dev. Max std_dev for values 1-10 is roughly (10-1)/2 = 4.5
                # This needs careful thought based on expected value range.
                # If values are 1,2,3...N, max std dev is approx N/2.
                # Assuming a common max value on board around 10-15 for this normalization example.
                # A more robust way is to normalize based on actual min/max in neighborhood or globally.
                # Here, if grid has max 10, std_dev could be ~5. If grid has max 50, std_dev ~25.
                # Let's normalize against a dynamic range if possible or a reasonable heuristic.
                max_possible_val_in_neighborhood = max(neighborhood_vals) if neighborhood_vals else 1
                min_possible_val_in_neighborhood = min(neighborhood_vals) if neighborhood_vals else 0
                heuristic_max_std_dev = (max_possible_val_in_neighborhood - min_possible_val_in_neighborhood) / 2.0
                if heuristic_max_std_dev < 1e-6 : heuristic_max_std_dev = 1.0 # Avoid div by zero if all neigh vals are same
                
                current_score = MathUtils.normalize_value(std_dev, 0, heuristic_max_std_dev, clamp=True)
                scores[r_idx, c_idx] = current_score

            elif metric == "entropy":
                # Simplified entropy for discrete integer values
                value_counts = Counter(neighborhood_vals)
                num_total_values = len(neighborhood_vals)
                entropy = 0.0
                for count in value_counts.values():
                    probability = count / num_total_values
                    entropy -= probability * math.log2(probability)
                
                # Normalize entropy. Max entropy is log2(num_distinct_values) or log2(num_total_values in worst case).
                # Max possible distinct values in a 3x3 neighborhood is 9. Max entropy = log2(9) approx 3.17
                max_possible_entropy = math.log2(min(num_total_values, len(set(neighborhood_vals)))) if num_total_values > 0 and len(set(neighborhood_vals)) > 0 else 1.0
                if max_possible_entropy < 1e-6 : max_possible_entropy = math.log2(9) # fallback
                
                current_score = MathUtils.normalize_value(entropy, 0, max_possible_entropy, clamp=True)
                scores[r_idx, c_idx] = current_score
            else:
                scores[r_idx, c_idx] = 0.0 # Unknown metric
                
    return scores

# More "extreme" modules will follow in the next parts...
# For now, let's define these two and update the main structures.
# In a full implementation, all 22 would be here.
# ... (æ¥çºç¬¬ä¸é¨åçç¨å¼ç¢¼: includes FastAPI setup, Pydantic models, helpers, ext_a2, ext_m3) ...

def ext_d3_potential_field_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """
    Core Rule: Evaluates cell potential based on an attractive/repulsive field generated by other significant elements.
    Purpose: Identifies strategically advantageous positions considering global force balance.
    ---
    Design Philosophy: Models the board as a field where certain cell values exert attractive or repulsive forces, diminishing with distance.
    Use Case: Finding optimal placement that maximizes attraction to "friendly" high-value points and minimizes exposure to "hostile" or "negative" points.
    Scoring Formula Principle: Score_cell = sigmoid( Sum_i [ Attraction_i / dist^p ] - Sum_j [ Repulsion_j / dist^q ] ).
    Compatibility: Handles integer grid values. -1 is empty. Positive values can be attractive, specific negative values (if any) or certain positive values could be repulsive based on rules.
    Extensibility: Configurable attraction/repulsion thresholds, strength factors, decay powers, max elements to consider for performance.
    Optimization & Extension Directions: Use k-d trees or spatial hashing to find influential neighbors faster on large boards, more complex field functions (e.g., Gaussian).
    Possible Multi-version Logic: D3.v1 (center proximity), D3.v2 (this version), D3.v3 (dynamic field based on game phase).
    """
    logger.debug(f"Executing ext_d3_potential_field_vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    # Parameters
    attractive_value_threshold = 7 # Example: Values >= 7 are attractive (assuming 1-10 range for values)
    repulsive_value_threshold = 3  # Example: Values <= 3 (but not -1) are repulsive
    attraction_strength_factor = 1.0
    repulsion_strength_factor = 1.2 # Repulsion might be stronger
    distance_decay_power_att = 1.5
    distance_decay_power_rep = 2.0
    max_relevant_elements = 50 # Performance: consider only top N influential elements
    sigmoid_scaling_factor = 0.1 # To scale total_potential before sigmoid

    # Identify all sources of potential
    sources = [] # List of (r, c, value, type_is_attractive)
    for r_s in range(rows):
        for c_s in range(cols):
            val = grid[r_s, c_s]
            if val == -1: continue
            if val >= attractive_value_threshold:
                sources.append({'r': r_s, 'c': c_s, 'val': val, 'type': 'attr'})
            elif val <= repulsive_value_threshold:
                sources.append({'r': r_s, 'c': c_s, 'val': val, 'type': 'rep'})
    
    if not sources: # No significant sources on the board
        return np.full((rows, cols), 0.5, dtype=float) # Neutral score for all

    for r_idx in range(rows):
        for c_idx in range(cols):
            total_potential = 0.0
            
            # Consider only N most relevant sources for performance if too many
            # (Relevance can be value/distance^2) - simplified here, uses all
            # A more optimized version would pre-sort or use spatial ds.

            for src in sources:
                if src['r'] == r_idx and src['c'] == c_idx: continue # Don't consider self for field generation

                dist = MathUtils.manhattan_distance((r_idx, c_idx), (src['r'], src['c']))
                if dist == 0: dist = 0.5 # Avoid division by zero if, somehow, a source is at the same loc but not self

                if src['type'] == 'attr':
                    # Attraction: strength is proportional to how much it exceeds threshold
                    charge = (src['val'] - attractive_value_threshold + 1.0) # Ensure positive charge
                    total_potential += (attraction_strength_factor * charge) / (dist ** distance_decay_power_att)
                elif src['type'] == 'rep':
                    # Repulsion: strength is proportional to how much it is below threshold
                    penalty = (repulsive_value_threshold - src['val'] + 1.0) # Ensure positive penalty
                    total_potential -= (repulsion_strength_factor * penalty) / (dist ** distance_decay_power_rep)
            
            scores[r_idx, c_idx] = MathUtils.sigmoid(total_potential * sigmoid_scaling_factor)
            
    return scores

def ext_f10_structural_discontinuity_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """
    Core Rule: Detects structural breaks or sharp value changes using a Laplacian-like operator.
    Purpose: Identifies "edges", "boundaries", or "fault lines" in the grid's value landscape.
    ---
    Design Philosophy: Applies a simplified Laplacian filter to approximate second-order derivatives, highlighting regions of rapid change.
    Use Case: Finding strategically important borders, areas of instability, or conversely, very smooth regions (low response).
    Scoring Formula Principle: Score_cell = sigmoid( k * |Laplacian_response(cell)| ). High absolute response means high discontinuity.
    Compatibility: Handles integer grid. -1 treated as a neutral value (e.g., 0) for filter or interpolated.
    Extensibility: Different Laplacian kernels, LoG (Laplacian of Gaussian) for noise reduction, directional derivatives.
    Optimization & Extension Directions: Use optimized convolution functions (e.g., from scipy.signal), link edge points into contours.
    Possible Multi-version Logic: F10.v1 (edge/corner binary), F10.v2 (this Laplacian version), F10.v3 (Canny-like edge detection principle).
    """
    logger.debug(f"Executing ext_f10_structural_discontinuity_vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    # Parameters
    laplacian_kernel_type = 1 # 1: 4-connectivity, 2: 8-connectivity
    response_scaling_factor = 2.0 # For sigmoid input scaling
    
    def val_func_for_filter(x_val: int) -> float:
        if x_val == -1: return 0.0 # Treat empty as 0 for filter, or use mean of neighbors
        # Normalize values if they have a large range. Example: if values 1-50
        # return MathUtils.normalize_value(float(x_val), 1, 50)
        return float(x_val) # Assuming values are already in a somewhat comparable range or small integers

    for r_idx in range(rows):
        for c_idx in range(cols):
            center_val = val_func_for_filter(grid[r_idx, c_idx])
            laplacian_response = 0.0

            if laplacian_kernel_type == 1: # [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
                kernel_sum = 0
                num_valid_neighbors = 0
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r_idx + dr, c_idx + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        kernel_sum += val_func_for_filter(grid[nr, nc])
                        num_valid_neighbors += 1
                # Adjust center weight based on actual neighbors to handle edges/corners gracefully
                if num_valid_neighbors > 0:
                    laplacian_response = kernel_sum - num_valid_neighbors * center_val
            
            elif laplacian_kernel_type == 2: # [[1,1,1], [1,-8,1], [1,1,1]]
                kernel_sum = 0
                num_valid_neighbors = 0
                for dr_lap in [-1,0,1]: # Renamed dr to dr_lap
                    for dc_lap in [-1,0,1]: # Renamed dc to dc_lap
                        if dr_lap == 0 and dc_lap == 0: continue
                        nr, nc = r_idx + dr_lap, c_idx + dc_lap
                        if 0 <= nr < rows and 0 <= nc < cols:
                            kernel_sum += val_func_for_filter(grid[nr, nc])
                            num_valid_neighbors += 1
                if num_valid_neighbors > 0:
                    laplacian_response = kernel_sum - num_valid_neighbors * center_val
            
            # Normalize response before sigmoid. Max possible |response| depends on value range and kernel.
            # E.g., for kernel 1, if values 0-10, center 0, neighbors 10 -> sum=40, response=40. Center 10, neighbors 0 -> response=-40.
            # A rough max absolute response could be num_kernel_positive_weights * max_val_range. For kernel 1, it's 4 * (max_val-min_val).
            # Let's assume max practical val range is 10 for this example, so max_abs_resp ~40.
            max_abs_response_heuristic = 4.0 * 10.0 # Heuristic for val_func outputting up to 10
            if laplacian_kernel_type == 2: max_abs_response_heuristic = 8.0 * 10.0
            if max_abs_response_heuristic < 1e-6 : max_abs_response_heuristic = 1.0

            normalized_abs_response = MathUtils.normalize_value(abs(laplacian_response), 0, max_abs_response_heuristic, clamp=True)
            
            # We want high score for high discontinuity (large |laplacian_response|)
            # sigmoid((X - center)*k). If X is 0-1, (X-0.2)*k gives low score for low X.
            scores[r_idx, c_idx] = MathUtils.sigmoid((normalized_abs_response - 0.1) * response_scaling_factor * 5.0, k=3.0) # k increases steepness
            
    return scores

def ext_gm1_adv_row_control_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """
    Core Rule: Evaluates weighted value concentration and high-value connectivity in rows, compared to a baseline.
    Purpose: Identifies rows with significant resource accumulation or potential barriers/channels.
    ---
    Design Philosophy: Extends simple row occupancy by considering weighted sum of values (e.g., giving more importance to cells towards one end) and explicit detection of continuous segments of high-value cells.
    Use Case: Strategic row control, forming horizontal lines of defense or attack.
    Scoring Formula Principle: Score_cell_in_row = sigmoid( factor1 * (S_row_weighted / Baseline_S_row - 1) + factor2 * has_connection_bonus ).
    Compatibility: Handles integer grid. -1 treated as low value/empty.
    Extensibility: Configurable weighting functions, connection thresholds/lengths, dynamic baseline.
    Optimization & Extension Directions: Pre-calculate row sums if board is static for many calls, use run-length encoding for connection detection.
    Possible Multi-version Logic: GM1.v1 (simple occupancy), GM1.v2 (this version), GM1.v3 (row value sequence FFT analysis).
    """
    logger.debug(f"Executing ext_gm1_adv_row_control_vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    if cols == 0: return scores

    # Parameters
    min_connection_length = 3 
    connection_value_threshold = 7 # Example: if values are 1-10, 7+ is high
    row_value_comparison_factor = 1.5
    connection_bonus_factor = 1.0
    
    def val_func_gm1(x_val: int) -> float:
        if x_val == -1: return 0.0
        return float(x_val) # Use raw values for weighting and connection

    # Calculate properties for each row
    row_properties = [] # List of dicts: {'weighted_sum': float, 'has_connection': bool}
    all_row_weighted_sums = []

    for r_idx in range(rows):
        row_data = grid[r_idx, :]
        
        # Weighted sum for the row
        current_weighted_sum = 0.0
        for c_idx in range(cols):
            val = val_func_gm1(row_data[c_idx])
            # Example weight: increases towards right end of row
            weight = 1.0 + 0.5 * (c_idx / (cols -1 if cols > 1 else 1) )
            current_weighted_sum += val * weight
        all_row_weighted_sums.append(current_weighted_sum)

        # Connection detection in the row
        current_streak = 0
        has_connection_in_row = False
        for c_idx in range(cols):
            val = val_func_gm1(row_data[c_idx])
            if val >= connection_value_threshold:
                current_streak += 1
                if current_streak >= min_connection_length:
                    has_connection_in_row = True
                    break
            else:
                current_streak = 0
        row_properties.append({'weighted_sum': current_weighted_sum, 'has_connection': has_connection_in_row})

    # Baseline for row weighted sum (e.g., average or median)
    # Using median as a more robust baseline against extreme rows
    baseline_row_w_sum = np.median(all_row_weighted_sums) if all_row_weighted_sums else 0.0
    if baseline_row_w_sum < 1e-6 : baseline_row_w_sum = 1.0 # Avoid division by zero if all sums are tiny or zero

    # Assign scores to cells based on their row's properties
    for r_idx in range(rows):
        props = row_properties[r_idx]
        
        # Normalized comparison to baseline
        row_value_score_comp = (props['weighted_sum'] / baseline_row_w_sum - 1.0)
        
        connection_bonus_val = 1.0 if props['has_connection'] else 0.0
        
        raw_score_for_row = (row_value_comparison_factor * row_value_score_comp +
                             connection_bonus_factor * connection_bonus_val)
        
        # All cells in the same row get the same base score from this module
        # (Can be modified if cell's own position within row should further modulate it)
        row_score = MathUtils.sigmoid(raw_score_for_row) 
        scores[r_idx, :] = row_score
            
    return scores

def ext_gm2_adv_col_flow_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """
    Core Rule: Analyzes value gradients and consistency within columns to assess vertical "flow" or "control".
    Purpose: Identifies columns with smooth resource flow, strong vertical control, or potential blockages.
    ---
    Design Philosophy: Examines 1st order differences (gradients) in column value sequences. Smooth gradients in high-value columns are favored.
    Use Case: Assessing vertical attack/defense lines, supply chain integrity down a column.
    Scoring Formula Principle: Score_cell_in_col = w1 * (1 - norm_gradient_variance) + w2 * norm_avg_col_value + w3 * favorable_sharp_gradient_bonus_at_cell.
    Compatibility: Handles integer grid. -1 treated as low value/empty.
    Extensibility: Higher-order differences, spectral analysis of column sequence, specific gradient patterns.
    Optimization & Extension Directions: Vectorize gradient calculation further if possible, adaptive thresholds for "sharp" gradients.
    Possible Multi-version Logic: GM2.v1 (simple occupancy), GM2.v2 (this version), GM2.v3 (run-length encoding of column values).
    """
    logger.debug(f"Executing ext_gm2_adv_col_flow_vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    if rows == 0: return scores

    # Parameters
    gradient_variance_weight = 0.4
    avg_col_value_weight = 0.4
    sharp_gradient_bonus_weight = 0.2 # Bonus if this cell is part of a favorable sharp gradient
    sharp_gradient_threshold = 3.0 # Absolute difference threshold (depends on value range, e.g. for 1-10)
    
    def val_func_gm2(x_val: int) -> float:
        if x_val == -1: return 0.0 # Treat as 0 for flow analysis
        return float(x_val)

    for c_idx in range(cols):
        column_data = grid[:, c_idx]
        column_vals_processed = np.array([val_func_gm2(x) for x in column_data])

        if len(column_vals_processed) < 2: # Single row grid
            for r_idx in range(rows):
                 # Score based on its own normalized value
                scores[r_idx, c_idx] = MathUtils.normalize_value(column_vals_processed[0],0,10) # Assuming max 10
            continue

        gradients = np.diff(column_vals_processed)
        
        # Gradient variance for the whole column
        gradient_variance = np.var(gradients) if len(gradients) > 0 else 0.0
        # Max possible variance of diffs. If vals 0-10, diffs -10 to 10. Var could be around (10^2)/4 = 25.
        # Heuristic max variance. If values are [0,10,0,10...], diffs are [10,-10,10...], var is high.
        heuristic_max_grad_var = ((np.max(column_vals_processed) - np.min(column_vals_processed) )**2)/4.0 if len(column_vals_processed)>0 else 1.0
        if heuristic_max_grad_var < 1e-6: heuristic_max_grad_var=1.0
        smoothness_score_col = 1.0 - MathUtils.normalize_value(gradient_variance, 0, heuristic_max_grad_var, clamp=True)

        # Average column value
        avg_col_value = np.mean(column_vals_processed)
        # Normalize avg_col_value (e.g. if values 0-10)
        normalized_avg_col_value_col = MathUtils.normalize_value(avg_col_value, 0, 10.0, clamp=True) # Assuming max 10

        # Assign scores to cells in this column
        for r_idx in range(rows):
            favorable_sharp_gradient_bonus_cell = 0.0
            # Check gradient AT this cell.
            # Gradient before this cell (if r_idx > 0): column_vals_processed[r_idx] - column_vals_processed[r_idx-1] = gradients[r_idx-1]
            # Gradient after this cell (if r_idx < rows-1): column_vals_processed[r_idx+1] - column_vals_processed[r_idx] = gradients[r_idx]
            
            # Example: Bonus if current cell is start of strong upward trend (cell_val < cell_below_val significantly)
            if r_idx < rows - 1: # If there's a cell below
                grad_after = gradients[r_idx] # val[r+1] - val[r]
                if grad_after > sharp_gradient_threshold: # Value increases sharply below this cell
                    favorable_sharp_gradient_bonus_cell = MathUtils.normalize_value(grad_after, sharp_gradient_threshold, 10.0, clamp=True) # Max diff 10
            
            # Or, if current cell is end of strong upward trend (cell_val > cell_above_val significantly)
            if r_idx > 0:
                grad_before = gradients[r_idx-1] # val[r] - val[r-1]
                if grad_before > sharp_gradient_threshold: # Value increased sharply to this cell
                    favorable_sharp_gradient_bonus_cell = max(favorable_sharp_gradient_bonus_cell, MathUtils.normalize_value(grad_before, sharp_gradient_threshold, 10.0, clamp=True))


            cell_score = (gradient_variance_weight * smoothness_score_col +
                          avg_col_value_weight * normalized_avg_col_value_col +
                          sharp_gradient_bonus_weight * favorable_sharp_gradient_bonus_cell)
            
            scores[r_idx, c_idx] = max(0.0, min(1.0, cell_score)) # Ensure in 0-1
            
    return scores

# More modules to follow...
# ... (æ¥çºåå analyzer.py çç¨å¼ç¢¼: MathUtils, BoardAnalyzerUtils, ext_a2 å° ext_gm14 çå½å¼å®ç¾©) ...

def ext_gm15_secure_territory_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """
    Core Rule: Evaluates if a cell (typically low-value/empty) can form a secure "eye" surrounded by friendly high-value cells.
    Purpose: Identifies potential safe havens or resource accumulation points protected by a strong perimeter.
    ---
    Design Philosophy: Inspired by "eye formation" in Go. Checks for a low-value center surrounded by a sufficiently complete and strong "wall" of friendly high-value cells.
    Use Case: Finding defensible positions, secure storage locations, or stable zones in territorial games.
    Scoring Formula Principle: Score = w1 * wall_completeness_ratio + w2 * avg_wall_strength - w3 * penalty_if_center_not_empty_enough.
    Compatibility: Handles integer grid. -1 for empty. Customizable thresholds for "eye center", "friendly wall", "opponent breach".
    Extensibility: More sophisticated "eye life and death" algorithms, variable eye radius, consideration of internal eye space properties.
    Optimization & Extension Directions: Pre-calculate influence maps for "friendly" and "opponent" values, use graph algorithms for precise encirclement detection.
    Possible Multi-version Logic: GM15.v1 (simple distance to nearest '1'), GM15.v2 (this version), GM15.v3 (dynamic eye shape analysis).
    """
    logger.debug(f"Executing ext_gm15_secure_territory_vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    # Parameters
    eye_center_max_value = 1       # Max value for a cell to be considered an eye center (e.g., -1 or low positive)
    friendly_wall_min_value = 7    # Min value for a cell to be part of the "friendly wall" (e.g., for values 1-10)
    # opponent_breach_max_value = 3 # If a wall cell is <= this, it's a breach (not explicitly used to penalize here, but affects completeness)
    eye_radius = 1                 # Checks the immediate 3x3 perimeter around the potential eye center
    min_wall_elements_ratio = 0.75 # Minimum ratio of friendly cells on the perimeter to be considered a good wall
    wall_strength_weight = 0.6
    wall_completeness_weight = 0.4
    
    def val_func_gm15(x_val: int) -> float: # No optional, always returns a float for easier comparison
        if x_val == -1: return 0.0 # Treat empty as 0 for value checks
        return float(x_val)

    for r_idx in range(rows):
        for c_idx in range(cols):
            cell_val = val_func_gm15(grid[r_idx, c_idx])

            if cell_val > eye_center_max_value:
                scores[r_idx, c_idx] = 0.0 # Not a valid eye center candidate
                continue

            potential_wall_positions_count = 0
            actual_friendly_wall_elements_count = 0
            sum_friendly_wall_strength = 0.0

            # Check perimeter defined by eye_radius
            for dr in range(-eye_radius, eye_radius + 1):
                for dc in range(-eye_radius, eye_radius + 1):
                    # Consider only the cells on the perimeter, not inside or the center itself
                    if abs(dr) != eye_radius and abs(dc) != eye_radius:
                         if not (abs(dr) == eye_radius or abs(dc) == eye_radius):
                            continue # Skip if not on the perimeter

                    nr, nc = r_idx + dr, c_idx + dc
                    if 0 <= nr < rows and 0 <= nc < cols: # If on board
                        if nr == r_idx and nc == c_idx: continue # Skip center itself, already checked

                        potential_wall_positions_count += 1
                        wall_candidate_val = val_func_gm15(grid[nr, nc])
                        
                        if wall_candidate_val >= friendly_wall_min_value:
                            actual_friendly_wall_elements_count += 1
                            sum_friendly_wall_strength += wall_candidate_val
            
            if potential_wall_positions_count == 0: # e.g. 1x1 grid, no perimeter
                scores[r_idx, c_idx] = 0.1 if cell_val <= eye_center_max_value else 0.0
                continue

            wall_completeness_ratio = actual_friendly_wall_elements_count / potential_wall_positions_count
            
            avg_friendly_wall_strength = 0.0
            if actual_friendly_wall_elements_count > 0:
                avg_friendly_wall_strength = sum_friendly_wall_strength / actual_friendly_wall_elements_count
            
            # Normalize avg_friendly_wall_strength (assuming friendly_wall_min_value to a max possible e.g. 10)
            norm_avg_wall_strength = MathUtils.normalize_value(avg_friendly_wall_strength, friendly_wall_min_value, 10.0, clamp=True)

            current_score = 0.0
            if wall_completeness_ratio >= min_wall_elements_ratio:
                current_score = (wall_completeness_weight * wall_completeness_ratio +
                                 wall_strength_weight * norm_avg_wall_strength)
                # Bonus if the eye center is truly empty (-1 in original grid)
                if grid[r_idx, c_idx] == -1 :
                    current_score += 0.15
            else: # Wall not complete enough
                current_score = 0.1 * wall_completeness_ratio # Small score for partial walls

            scores[r_idx, c_idx] = max(0.0, min(1.0, current_score))
            
    return scores

def ext_gm16_bottleneck_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """
    Core Rule: Identifies cells acting as critical "bottlenecks" or "choke points" between pre-defined major zones.
    Purpose: Highlights strategically vital cells whose control dictates connectivity between larger board areas.
    ---
    Design Philosophy: Based on "Critical Path Analysis" or "Articulation Point" concepts from graph theory. It assesses how removing a cell (making it impassable) affects the shortest path cost between two fixed zones.
    Use Case: Finding key defensive positions, or points to attack to sever opponent's supply lines or connections.
    Scoring Formula Principle: Score_cell = sigmoid( (Cost_after_removal - Cost_before_removal) / Cost_before_removal ). High relative cost increase means high bottleneck score.
    Compatibility: Handles integer grid. Cell values translated to " traversal costs". -1 can be high cost or normal.
    Extensibility: Dynamic zone definition, multiple source-sink pairs, use of actual graph algorithms for articulation points.
    Optimization & Extension Directions: Pre-calculate all-pairs shortest paths (Floyd-Warshall) if grid is static, or use A* for faster pathfinding.
    Possible Multi-version Logic: GM16.v1 (simple symmetry), GM16.v2 (this version with fixed zones), GM16.v3 (min-cut based bottleneck analysis).
    """
    logger.debug(f"Executing ext_gm16_bottleneck_vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    if rows < 2 or cols < 2: # Pathfinding not meaningful on trivial grids
        return np.full((rows, cols), 0.1, dtype=float)

    # Parameters
    # Define zones by representative points, e.g., corners
    # Ensure start_node and end_node are within current grid dimensions
    start_node = (0, 0)
    end_node = (rows - 1, cols - 1)
    
    def cost_func_gm16(cell_val: int) -> float:
        if cell_val == -1: return 1.0 # Empty cells are easy to traverse
        # Higher positive values mean higher cost (more difficult to traverse)
        # Normalize cost, e.g. if values 1-10. Cost = val / 2. Max cost 5.
        # This needs to be game specific. Let's assume lower values are easier.
        cost = 1.0 + MathUtils.normalize_value(float(cell_val), 1, 10, clamp=False) * 5.0 # Cost 1 to 6 for values 1-10
        return max(1.0, cost) # Minimum cost of 1

    removed_cell_cost_penalty = 1000.0 # Effective cost when a cell is "removed"

    # Dijkstra's implementation (can be a static method or defined outside if used by multiple modules)
    # For brevity, it's similar to the one shown in the standalone extreme main.py's GM16
    # (Assuming _dijkstra_shortest_path from that context is available or re-implemented here)
    # Since it's not available, I'll sketch a simplified BFS for unweighted path length as a proxy.
    # A full Dijkstra is needed for weighted costs.
    # For now, I'll use a placeholder BFS for path *existence* and length as rough cost.
    # This simplification means `cost_func_gm16` isn't fully utilized by BFS.

    def bfs_shortest_path_length(current_grid_state: np.ndarray, r_start: int, c_start: int, r_end: int, c_end: int,
                                 rows_bfs: int, cols_bfs: int,
                                 removed_cell_bfs: Optional[Tuple[int,int]] = None) -> float:
        q_bfs = deque([((r_start, c_start), 0)]) # ((r,c), length)
        visited_bfs = set([(r_start, c_start)])
        if removed_cell_bfs: visited_bfs.add(removed_cell_bfs) # Treat removed cell as already visited/impassable

        max_path_len_heuristic = rows_bfs * cols_bfs # A loose upper bound for normalization

        while q_bfs:
            (r,c), length = q_bfs.popleft()
            if r == r_end and c == c_end:
                return float(length)
            if length > max_path_len_heuristic / 2: # Optimization: stop if path too long
                continue

            for dr_bfs, dc_bfs in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr_bfs, c + dc_bfs
                if 0 <= nr < rows_bfs and 0 <= nc < cols_bfs and (nr,nc) not in visited_bfs:
                    if grid[nr,nc] == -1000: continue # Hardcoded impassable for this BFS example
                    
                    visited_bfs.add((nr,nc))
                    q_bfs.append(((nr,nc), length + 1))
        return float('inf') # Path not found

    # Path cost before removing any cell
    # For a proper implementation, use Dijkstra with cost_func_gm16
    # Here, using BFS length as a proxy for "cost"
    cost_before_removal = bfs_shortest_path_length(grid, start_node[0], start_node[1], end_node[0], end_node[1], rows, cols)

    for r_idx in range(rows):
        for c_idx in range(cols):
            if (r_idx, c_idx) == start_node or (r_idx, c_idx) == end_node:
                scores[r_idx, c_idx] = 0.05 # Start/end points are not bottlenecks *between* these zones
                continue

            # Path cost after "removing" (r_idx, c_idx)
            cost_after_removal = bfs_shortest_path_length(grid, start_node[0], start_node[1], end_node[0], end_node[1], 
                                                          rows, cols, removed_cell_bfs=(r_idx, c_idx))
            
            current_score = 0.0
            if math.isinf(cost_before_removal) and math.isinf(cost_after_removal):
                current_score = 0.0 # Still no path
            elif not math.isinf(cost_before_removal) and math.isinf(cost_after_removal):
                current_score = 1.0 # Removing this cell broke the only path(s) - strong bottleneck
            elif not math.isinf(cost_before_removal) and not math.isinf(cost_after_removal):
                if cost_before_removal < 1e-6: cost_before_removal = 1e-6 # Avoid division by zero
                increase_ratio = (cost_after_removal - cost_before_removal) / cost_before_removal
                # Sigmoid factor should make meaningful ratios (e.g. 0.5 to 5) map well to 0-1
                current_score = MathUtils.sigmoid(increase_ratio * 1.0, k=2.0) # k makes it more responsive
            else: # cost_before is inf, cost_after is not -> should not happen
                current_score = 0.0

            scores[r_idx, c_idx] = current_score
            
    return scores

def ext_gm17_network_flow_hub_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """
    Core Rule: Evaluates a cell's potential as a "hub" in a flow network between sources and sinks, considering cell capacity.
    Purpose: Identifies cells crucial for maintaining high "throughput" or flow capacity between key board regions.
    ---
    Design Philosophy: Simplified network flow concept. A cell is a good hub if it has high capacity itself and lies on plausible high-capacity paths between predefined source/sink regions.
    Use Case: Analyzing supply chain choke points, communication network hubs, or critical infrastructure nodes.
    Scoring Formula Principle: Score_cell = cell_capacity_factor * path_centrality_factor. Path centrality could be proximity to a geodesic path between S-T pairs, and how many such paths it serves.
    Compatibility: Integer grid, where cell values can be interpreted as "flow capacities". -1 is zero capacity.
    Extensibility: Use actual max-flow algorithms (Edmonds-Karp), consider multi-commodity flows, dynamic source/sink definition.
    Optimization & Extension Directions: Pre-calculate path capacities, approximate max flow using simpler heuristics for speed.
    Possible Multi-version Logic: GM17.v1 (simple symmetry), GM17.v2 (this simplified hub potential), GM17.v3 (min-cut impact analysis if cell removed).
    """
    logger.debug(f"Executing ext_gm17_network_flow_hub_vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    if rows < 2 or cols < 2: return np.full((rows,cols), 0.1, dtype=float)

    # Parameters
    # For simplicity, one source (top-left) and one sink (bottom-right)
    source_nodes = [(0,0)]
    sink_nodes = [(rows-1, cols-1)]
    
    def capacity_func_gm17(x_val: int) -> float:
        if x_val == -1: return 0.0 # No capacity
        # Assuming values 1-10, map to capacity 0.1-1.0
        return MathUtils.normalize_value(float(x_val), 1, 10, clamp=True) if x_val > 0 else 0.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            cell_capacity_norm = capacity_func_gm17(grid[r_idx, c_idx])
            if cell_capacity_norm < 1e-6: # No capacity, cannot be a hub
                scores[r_idx, c_idx] = 0.0
                continue

            # Simplified Path Centrality: How "between" sources and sinks is this cell?
            # Average of (1 - normalized distance from optimal S-T path)
            # Optimal S-T path length (Manhattan)
            
            path_centrality_scores_for_cell = []
            for s_node in source_nodes:
                for t_node in sink_nodes:
                    if s_node == t_node: continue
                    
                    dist_s_t = MathUtils.manhattan_distance(s_node, t_node)
                    dist_s_cell = MathUtils.manhattan_distance(s_node, (r_idx, c_idx))
                    dist_cell_t = MathUtils.manhattan_distance((r_idx, c_idx), t_node)

                    if dist_s_t == 0: # Should be caught by s_node == t_node
                        path_centrality_scores_for_cell.append(0.0)
                        continue
                    
                    # If sum of sub-paths equals total path, it's on A shortest path
                    # Deviation = (dist_s_cell + dist_cell_t - dist_s_t)
                    # Normalized deviation: deviation / dist_s_t
                    # Perfect path: deviation = 0. Score = 1.
                    # Far off path: deviation high. Score low.
                    deviation = (dist_s_cell + dist_cell_t) - dist_s_t
                    # Max possible deviation (cell is very far off, forming a long V shape) could be up to dist_s_t * 2 or more
                    # Heuristic: normalize deviation against dist_s_t.
                    # If deviation is 0, centrality = 1. If deviation = dist_s_t, centrality = 0.5. If deviation = 2*dist_s_t, centrality = 0.
                    centrality_for_pair = 1.0 - MathUtils.normalize_value(deviation, 0, dist_s_t + rows + cols, clamp=True) # Max deviation roughly sum of board dims
                    path_centrality_scores_for_cell.append(centrality_for_pair)
            
            avg_path_centrality = np.mean(path_centrality_scores_for_cell) if path_centrality_scores_for_cell else 0.0
            
            # Score combines cell's own capacity and its centrality
            scores[r_idx, c_idx] = cell_capacity_norm * avg_path_centrality
            
    return scores

def ext_gm18_rl_value_estimator_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """
    Core Rule: Estimates cell's strategic value using a predefined set of features and a weighted evaluation function (RL-inspired).
    Purpose: Simulates a simplified AI agent's assessment of a cell's long-term potential based on a feature-rich local state.
    ---
    Design Philosophy: Mimics a State-Value function V(s) from Reinforcement Learning. Extracts multiple local and semi-local features from the cell's perspective and combines them linearly (or via simple rules) to produce a value estimate.
    Use Case: General strategic assessment, identifying cells that are good according to a pre-defined "expert" or learned policy.
    Scoring Formula Principle: V_cell = sigmoid( bias + Sum_i (weight_i * feature_i(cell_state)) ).
    Compatibility: Handles integer grid. Features must be designed to process these values. -1 is empty.
    Extensibility: More sophisticated features, non-linear feature combination (e.g., small neural net, decision tree), dynamically adjusted weights.
    Optimization & Extension Directions: Train weights using ML methods if game outcomes are available, feature selection, use convolutional features.
    Possible Multi-version Logic: GM18.v1 (simple subgrid sum), GM18.v2 (this linear feature-based version), GM18.v3 (hardcoded decision tree).
    """
    logger.debug(f"Executing ext_gm18_rl_value_estimator_vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    # Define value function for features
    def feat_val_func(x_val: int) -> Optional[float]:
        if x_val == -1: return 0.0 # Treat empty as 0 for feature calculations
        # Normalize positive values for features, e.g., for a 1-10 game value range
        return MathUtils.normalize_value(float(x_val), 1, 10, clamp=True) if x_val > 0 else 0.0
    
    # Define features and their weights (example set)
    # Each feature function takes (grid, r, c, feat_val_func, rows, cols)
    features_and_weights = [
        {"name": "f_self_value", 
         "func": lambda g, r, c, vf, R, C: vf(g[r,c]), "weight": 0.3},
        {"name": "f_avg_neighbor_3x3_val", 
         "func": lambda g, r, c, vf, R, C: np.mean(BoardAnalyzerUtils.get_neighborhood_values(g,r,c,1,True,vf,False) or [0.0]),
         "weight": 0.2},
        {"name": "f_is_edge_penalty", # Higher score is worse if it's a penalty
         "func": lambda g, r, c, vf, R, C: 1.0 if (r==0 or r==R-1 or c==0 or c==C-1) else 0.0,
         "weight": -0.1}, # Negative weight for penalty
        {"name": "f_local_gradient_mag", 
         "func": lambda g, r, c, vf, R, C: MathUtils.normalize_value(math.hypot(*BoardAnalyzerUtils.get_value_gradient_at_cell(g,r,c,vf)),0,6.0), # Assuming max grad mag ~6
         "weight": 0.15},
        {"name": "f_num_high_value_neighbors", # e.g. value > 0.7 after normalization by feat_val_func
         "func": lambda g, r, c, vf, R, C: MathUtils.normalize_value(sum(1 for val_n in BoardAnalyzerUtils.get_neighborhood_values(g,r,c,1,True,vf,False) if val_n >= 0.7), 0, 8), # Max 8 neighbors
         "weight": 0.25}
    ]
    bias_w0 = 0.05 # Small positive bias

    for r_idx in range(rows):
        for c_idx in range(cols):
            current_value_estimate = bias_w0
            for feat_def in features_and_weights:
                try:
                    feat_val = feat_def["func"](grid, r_idx, c_idx, feat_val_func, rows, cols)
                    current_value_estimate += feat_def["weight"] * feat_val
                except Exception as e_feat:
                    logger.warning(f"Error in GM18 feature {feat_def['name']} at ({r_idx},{c_idx}): {e_feat}", extra={'request_id': request_id})
            
            # current_value_estimate is a raw sum. Expected range can be approx sum of |weights| if features are 0-1.
            # Here, sum of positive weights ~0.9, negative ~ -0.1. Range could be roughly -0.1 to 0.9 + bias.
            # Sigmoid to map to 0-1.
            scores[r_idx, c_idx] = MathUtils.sigmoid(current_value_estimate, k=2.5) # k adjusts steepness around bias
            
    return scores

# (End of EXT_GM module definitions for this part)
# ... (æ¥çºç¬¬ä¸é¨åçç¨å¼ç¢¼: MathUtils, BoardAnalyzerUtils, ext_a2 å° ext_gm18 çå½å¼å®ç¾©) ...

# -----------------------------------------------------------------------------
# 3. "Industry Extreme" Module Registration and Weights
# -----------------------------------------------------------------------------

EXTREME_MODULE_FUNCS_VEC: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
    "EXT_A2_Proximity": ext_a2_weighted_proximity_vec,
    "EXT_M3_Heterogeneity": ext_m3_local_heterogeneity_vec,
    "EXT_D3_PotentialField": ext_d3_potential_field_vec,
    "EXT_F10_Discontinuity": ext_f10_structural_discontinuity_vec,
    "EXT_GM1_RowControl": ext_gm1_adv_row_control_vec,
    "EXT_GM2_ColFlow": ext_gm2_adv_col_flow_vec,
    "EXT_GM3_ConnectedComp": ext_gm3_adv_connected_comp_vec, # Assuming GM3 was defined in a previous part
    "EXT_GM4_SpatialAutoCorr": ext_gm4_spatial_autocorrelation_vec, # Assuming GM4 was defined
    "EXT_GM5_LocalExtremum": ext_gm5_local_extremum_detector_vec, # Assuming GM5 was defined
    "EXT_GM6_PatternMatch": ext_gm6_local_pattern_match_vec, # Assuming GM6 was defined
    "EXT_GM7_Accessibility": ext_gm7_accessibility_pathfinding_vec, # Assuming GM7 was defined
    "EXT_GM8_MarginalDensity": ext_gm8_marginal_density_vec, # Assuming GM8 was defined
    "EXT_GM9_GradientFlow": ext_gm9_value_gradient_flow_vec, # Assuming GM9 was defined
    "EXT_GM10_InfluenceMap": ext_gm10_influence_mapping_vec, # Assuming GM10 was defined
    "EXT_GM11_MultiScaleSig": ext_gm11_multi_scale_significance_vec, # Assuming GM11 was defined
    "EXT_GM12_Texture": ext_gm12_local_texture_analyzer_vec, # Assuming GM12 was defined
    "EXT_GM13_SpatialFill": ext_gm13_spatial_filling_complexity_vec, # Assuming GM13 was defined
    "EXT_GM14_NicheCompetition": ext_gm14_ecological_niche_vec, # Assuming GM14 was defined
    "EXT_GM15_SecureTerritory": ext_gm15_secure_territory_vec,
    "EXT_GM16_Bottleneck": ext_gm16_bottleneck_vec,
    "EXT_GM17_NetworkHub": ext_gm17_network_flow_hub_vec,
    "EXT_GM18_RLValueEst": ext_gm18_rl_value_estimator_vec,
}

# Define weights for these new extreme modules
# These are illustrative; optimal weights depend on the specific game/application
# and should be tuned via testing or machine learning if possible.
EXTREME_MODULE_WEIGHTS: Dict[str, float] = {
    "EXT_A2_Proximity": 1.2,
    "EXT_M3_Heterogeneity": 1.0,
    "EXT_D3_PotentialField": 1.1,
    "EXT_F10_Discontinuity": 0.9,
    "EXT_GM1_RowControl": 1.0,
    "EXT_GM2_ColFlow": 1.0,
    "EXT_GM3_ConnectedComp": 1.1,
    "EXT_GM4_SpatialAutoCorr": 0.8,
    "EXT_GM5_LocalExtremum": 1.2, # Detecting peaks can be very important
    "EXT_GM6_PatternMatch": 0.9,
    "EXT_GM7_Accessibility": 1.3, # Pathfinding often crucial
    "EXT_GM8_MarginalDensity": 0.7,
    "EXT_GM9_GradientFlow": 0.8,
    "EXT_GM10_InfluenceMap": 1.2,
    "EXT_GM11_MultiScaleSig": 0.9,
    "EXT_GM12_Texture": 0.7,
    "EXT_GM13_SpatialFill": 0.8,
    "EXT_GM14_NicheCompetition": 1.0,
    "EXT_GM15_SecureTerritory": 1.3, # Eye formation is very strong
    "EXT_GM16_Bottleneck": 1.4,     # Critical paths are highly strategic
    "EXT_GM17_NetworkHub": 1.1,
    "EXT_GM18_RLValueEst": 1.5,     # AI-like value estimation can be powerful if well-configured
}
# Ensure all defined modules have weights
for mod_key in EXTREME_MODULE_FUNCS_VEC:
    if mod_key not in EXTREME_MODULE_WEIGHTS:
        logger.warning(f"Weight not defined for extreme module: {mod_key}. Defaulting to 1.0.")
        EXTREME_MODULE_WEIGHTS[mod_key] = 1.0

ANALYSIS_ENGINE_VERSION_EXTREME = "2.0_extreme" # Version for this new analyzer


# --- Modified Tensor Flow Scoring (Detailed) for Extreme Modules ---
def extreme_tensor_flow_score_detailed(grid: np.ndarray, request_id: str) -> Tuple[np.ndarray, List[List[List[TensorRuleContribution]]]]:
    """
    Calculates scores for each cell based on a set of "extreme" modules.
    Each module returns a float score grid. These are weighted and summed.
    """
    rows, cols = grid.shape
    total_score_grid_agg = np.zeros((rows, cols), dtype=float) # Renamed to avoid confusion
    # rule_contributions_grid[r][c] will be List[TensorRuleContribution]
    rule_contributions_grid_agg: List[List[List[TensorRuleContribution]]] = [[[] for _ in range(cols)] for _ in range(rows)] # Renamed

    active_modules = EXTREME_MODULE_FUNCS_VEC # Use the new extreme modules
    active_weights = EXTREME_MODULE_WEIGHTS

    for name, func in active_modules.items():
        try:
            # Each extreme module function now returns a grid of float scores (0-1 ideally)
            module_score_grid = func(grid, request_id) # (R, C) float scores
            
            weight = active_weights.get(name, 1.0)
            if math.isclose(weight, 0): continue # Skip if weight is effectively zero

            current_module_weighted_scores = module_score_grid * weight
            total_score_grid_agg += current_module_weighted_scores

            # Store contributions for transparency
            for r_idx in range(rows):
                for c_idx in range(cols):
                    raw_module_score_for_cell = module_score_grid[r_idx, c_idx]
                    # Only add contribution if the raw score is significant (e.g., > 0.01) 
                    # or if a boolean mask style was intended, where any non-zero score is an "application"
                    # For float scores, it's better to always add if weight is non-zero,
                    # or define a threshold for "significant contribution".
                    # Here, adding if raw_module_score_for_cell contributed something meaningful (e.g. > epsilon)
                    if abs(raw_module_score_for_cell) > 1e-4 or abs(weight * raw_module_score_for_cell) > 1e-4 : # or just always add if weight!=0
                        contribution = TensorRuleContribution(
                            rule_name=name,
                            score_if_applied=round(raw_module_score_for_cell, 4), # This is now the module's direct score
                            weight=weight,
                            weighted_score=round(current_module_weighted_scores[r_idx, c_idx], 4)
                        )
                        rule_contributions_grid_agg[r_idx][c_idx].append(contribution)
        except Exception as e:
            logger.error(f"Error processing EXTREME rule '{name}': {e}", exc_info=True, extra={'request_id': request_id})
            pass # Optionally skip failing module or handle error more gracefully
            
    return total_score_grid_agg, rule_contributions_grid_agg


# --- Original Helper Functions (get_card_max_value, get_legal_values_for_placement) ---
# (User's original helper functions - kept as is, assuming they are still relevant)
def get_card_max_value(grid: np.ndarray) -> int:
    if grid.size == 0: return 0
    valid_values = grid[grid != -1]
    return int(np.max(valid_values)) if valid_values.size > 0 else 0

def get_legal_values_for_placement(grid: np.ndarray) -> set:
    card_max_val = get_card_max_value(grid)
    upper_bound = card_max_val + 1 if card_max_val > 0 else 10 
    all_possible_values = set(range(1, upper_bound + 1))
    # Assuming for placement, we are interested in positive values not yet on board
    # If game allows re-placing existing values (e.g. to move them), this logic changes.
    # Current: only allows placing new, unique positive values up to max+1.
    used_values = set(grid.flatten())
    used_values.discard(-1) 
    # This means legal values are those NOT YET on board. This might conflict with mem_score
    # if mem_score is about frequency of existing values.
    # For "extreme" analyzer, "legal values" for placement might need more game-specific rules.
    # Let's assume this function correctly defines what can be newly placed.
    return all_possible_values - used_values


# --- CP Solver Logic ---
# (User's original CP Solver logic - kept mostly as is, uses CandidateDetail.final_objective_score)
def solve_cp_for_candidates(
    grid_shape: Tuple[int, int],
    current_grid_state: np.ndarray,
    candidates_to_evaluate: List[CandidateDetail],
    request_id: str
) -> List[CandidateDetail]:
    model = cp_model.CpModel()
    num_candidates = len(candidates_to_evaluate)
    if num_candidates == 0: return candidates_to_evaluate

    x = [model.NewBoolVar(f"x_{i}") for i in range(num_candidates)]
    model.Add(sum(x) == 1) # Pick exactly one

    objective_terms = [x[i] * int(candidates_to_evaluate[i].final_objective_score * 1000) for i in range(num_candidates)]
    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0.5 
    solver.parameters.num_search_workers = os.cpu_count() or 1
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        logger.info(f"CP Solver found a solution. Status: {solver.StatusName(status)}", extra={'request_id': request_id})
        for i in range(num_candidates):
            if solver.Value(x[i]) == 1:
                candidates_to_evaluate[i].is_selected_by_cp = True
                candidates_to_evaluate[i].cp_solver_notes = f"Selected by CP ({solver.StatusName(status)})"
            elif candidates_to_evaluate[i].cp_solver_notes is None:
                 candidates_to_evaluate[i].cp_solver_notes = "Not selected by CP"
    else:
        logger.warning(f"CP Solver no optimal/feasible solution. Status: {solver.StatusName(status)}", extra={'request_id': request_id})
        for cand_detail in candidates_to_evaluate:
            if cand_detail.cp_solver_notes is None:
                cand_detail.cp_solver_notes = f"No CP solution ({solver.StatusName(status)})"
    return candidates_to_evaluate


# --- API Endpoint: /analyze (Updated to use Extreme Analyzer) ---
@app.post("/analyze",
          response_model=AnalyzeSuccessResponse,
          responses={ # (User's original responses - kept as is)
              200: {"model": AnalyzeSuccessResponse},
              400: {"model": AnalyzeErrorResponse, "description": "Invalid input or no valid candidates"},
              422: {"model": AnalyzeErrorResponse, "description": "Validation error (Pydantic)"},
              500: {"model": AnalyzeErrorResponse, "description": "Internal server error"}
          },
          tags=["Analysis Engine vExtreme"]) # Updated tag
async def analyze(req: AnalyzeRequest, request: Request): # Renamed from analyze_v2 for consistency
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))

    try:
        logger.info(f"EXTREME Analyzer: Received request. Grid: {len(req.new_card)}x{len(req.new_card[0]) if req.new_card else 0}. Proposals: {len(req.proposed_values)}.", extra={'request_id': request_id})
        grid = np.array(req.new_card, dtype=int)

        # 1. Use the NEW EXTREME tensor flow detailed scoring
        raw_tf_scores_grid_extreme, rule_contributions_grid_extreme = await run_in_threadpool(
            extreme_tensor_flow_score_detailed, grid, request_id
        )
        
        # 2. Determine legal values (original logic)
        globally_legal_values_for_new_placement = get_legal_values_for_placement(grid)
        if not globally_legal_values_for_new_placement and any(pv.value != -1 for pv in req.proposed_values): # If proposing to place positive values
            logger.warning("EXTREME Analyzer: No globally legal positive values to place.", extra={'request_id': request_id})

        # 3. Prepare CandidateDetail objects
        all_evaluated_candidates: List[CandidateDetail] = []
        
        for pv_idx, pv in enumerate(req.proposed_values):
            r, c = pv.pos[0], pv.pos[1]
            val_proposed = pv.value
            is_valid_proposal_flag = True # Assume true initially
            validation_notes = ""

            # Perform proposal validation (can this value be placed at this spot?)
            if val_proposed != -1: # If proposing to place a positive value
                if grid[r,c] != -1:
                    is_valid_proposal_flag = False
                    validation_notes = f"Cell [{r},{c}] already filled with {grid[r,c]}."
                elif val_proposed not in globally_legal_values_for_new_placement:
                    is_valid_proposal_flag = False
                    validation_notes = f"Value {val_proposed} not in globally legal set {globally_legal_values_for_new_placement} for new placement."
            elif val_proposed == -1: # If proposing to clear a cell
                if grid[r,c] == -1:
                    is_valid_proposal_flag = False
                    validation_notes = f"Cell [{r},{c}] is already empty."
            
            if not is_valid_proposal_flag:
                 logger.warning(f"EXTREME Analyzer: Proposal ({val_proposed} at [{r},{c}]) invalid. {validation_notes}", extra={'request_id': request_id})
                 cand_detail = CandidateDetail(
                    pos=[r,c], value=val_proposed, is_valid_proposal=False,
                    raw_tensor_flow_score=0, mem_score_value=0, final_objective_score=0,
                    cp_solver_notes=validation_notes or "Invalid proposal."
                 )
                 all_evaluated_candidates.append(cand_detail)
                 continue

            # If proposal is valid up to this point:
            # Get score for the TARGET cell of the proposal from the pre-calculated extreme score grid
            raw_tf_score_cell = raw_tf_scores_grid_extreme[r, c]
            tf_contrib_cell = rule_contributions_grid_extreme[r][c]
            
            # Calculate mem_score (using original logic)
            # mem_score's legal_values_for_position might need to be more specific than globally_legal_values...
            # For now, using globally_legal_values or allowing any positive int if the game logic is different.
            # If val_proposed is -1 (clearing), mem_score might be calculated differently or be 0.
            current_mem_score = 0.0
            if val_proposed != -1: # Only calc mem_score for placing positive values
                # Ensure mem_score gets a relevant set of "legal values" for its context
                # If get_legal_values_for_placement filters out already used values, then a proposed value
                # that is already on board (but valid for *this specific proposal* if rules allow e.g. strengthening)
                # would not be in globally_legal_values_for_new_placement.
                # This interaction needs careful game-specific definition.
                # For now, let's use a broad set for mem_score calculation or assume val_proposed is valid for mem_score.
                mem_score_legal_set = set(range(1, (get_card_max_value(grid) or 9) + 2)) # Example: 1 to max+1 or 1 to 10
                if val_proposed in mem_score_legal_set: # Ensure proposed value is in the domain mem_score considers
                    current_mem_score = mem_score(r, c, val_proposed, mem_score_legal_set)
            
            # Define final objective score for CP
            # This could be adjusted, e.g. if clearing a cell has a different objective impact.
            # If val_proposed is -1 (clear), raw_tf_score_cell might represent "badness" of current filled cell.
            # The current extreme modules score "goodness" of a cell if it were filled positively.
            # This needs adaptation if clearing moves are scored differently.
            # Assuming for now, proposal is for positive values.
            mem_score_factor = 5.0 
            if val_proposed == -1: mem_score_factor = 0 # No mem score benefit for clearing in this example

            final_obj_for_cp = raw_tf_score_cell + (mem_score_factor * current_mem_score)

            cand_detail = CandidateDetail(
                pos=[r, c], value=val_proposed, is_valid_proposal=True,
                tensor_flow_contributions=tf_contrib_cell,
                raw_tensor_flow_score=round(raw_tf_score_cell, 4),
                mem_score_value=round(current_mem_score, 4),
                final_objective_score=round(final_obj_for_cp, 4)
            )
            all_evaluated_candidates.append(cand_detail)

        candidates_for_cp_solver = [cd for cd in all_evaluated_candidates if cd.is_valid_proposal]

        if not candidates_for_cp_solver:
            logger.warning("EXTREME Analyzer: No valid proposed values for CP solver.", extra={'request_id': request_id})
            return AnalyzeSuccessResponse(
                request_id=request_id, status="no_valid_candidates_for_cp",
                analysis_engine_version=ANALYSIS_ENGINE_VERSION_EXTREME,
                message="No valid proposals could be submitted to the solver.",
                result=None, all_candidates_evaluated=all_evaluated_candidates
            )

        updated_candidate_details = await run_in_threadpool(
            solve_cp_for_candidates, grid.shape, grid, candidates_for_cp_solver, request_id
        )
        
        # Merge CP results back
        processed_map_cp = {(cand.pos[0], cand.pos[1], cand.value): cand for cand in updated_candidate_details}
        for i, cand_orig in enumerate(all_evaluated_candidates):
            if cand_orig.is_valid_proposal:
                key_cp = (cand_orig.pos[0], cand_orig.pos[1], cand_orig.value)
                if key_cp in processed_map_cp:
                    all_evaluated_candidates[i] = processed_map_cp[key_cp]

        selected_by_cp_list = [cand for cand in all_evaluated_candidates if cand.is_selected_by_cp] # Renamed list
        
        final_result_detail: Optional[AnalyzeResultDetail] = None
        response_message = "EXTREME Analysis complete."
        status_val = "success" # Default success

        if not selected_by_cp_list:
            response_message = "EXTREME CP Solver did not select any candidate."
            logger.info(response_message, extra={'request_id': request_id})
            status_val = "fail_no_selection_cp_extreme"
        elif len(selected_by_cp_list) == 1:
            final_result_detail = AnalyzeResultDetail(**selected_by_cp_list[0].model_dump())
            response_message = f"EXTREME Analyzer successfully selected: Pos={final_result_detail.pos}, Val={final_result_detail.value}."
        else: # Should not happen with sum(x)==1
            final_result_detail = AnalyzeResultDetail(**max(selected_by_cp_list, key=lambda cd: cd.final_objective_score).model_dump())
            response_message = f"EXTREME CP Solver: Multiple options. Selected highest score: Pos={final_result_detail.pos}, Val={final_result_detail.value}."
            status_val = "success_multiple_options_extreme"
        
        logger.info(response_message, extra={'request_id': request_id})
        return AnalyzeSuccessResponse(
            request_id=request_id, status=status_val,
            analysis_engine_version=ANALYSIS_ENGINE_VERSION_EXTREME,
            message=response_message, result=final_result_detail,
            all_candidates_evaluated=all_evaluated_candidates
        )

    except HTTPException as he: raise he 
    except ValueError as ve:
        logger.warning(f"EXTREME Analyzer: ValueError: {str(ve)}", exc_info=True, extra={'request_id': request_id})
        # Pydantic validation errors usually result in 422 status code by FastAPI automatically.
        # If this is a custom ValueError, we might want to control the status code.
        # For now, let FastAPI handle it or raise HTTPException for specific codes.
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"EXTREME Analyzer: Unexpected error: {str(e)}", exc_info=True, extra={'request_id': request_id})
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred in the extreme analyzer.")


# --- Health Check Endpoint ---
# (User's original health check endpoint - can be kept as is, or updated to reflect new module set)
@app.get("/health/analyze", response_model=AnalyzeHealthStatus, tags=["Health & Monitoring"])
async def health_analyze(request: Request):
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    logger.info("Health check requested for /analyze components (EXTREME version).", extra={'request_id': request_id})
    checks = {}
    overall_status = "UP"

    # Check 1: Extreme Module functions and weights
    if not EXTREME_MODULE_FUNCS_VEC:
        checks["extreme_module_functions_load"] = "FAIL: EXTREME_MODULE_FUNCS_VEC is empty"
        overall_status = "DEGRADED"
    else:
        checks["extreme_module_functions_load"] = f"OK: {len(EXTREME_MODULE_FUNCS_VEC)} extreme functions loaded"

    if not EXTREME_MODULE_WEIGHTS:
        checks["extreme_module_weights_load"] = "FAIL: EXTREME_MODULE_WEIGHTS is empty"
        overall_status = "DEGRADED"
    else:
        checks["extreme_module_weights_load"] = f"OK: {len(EXTREME_MODULE_WEIGHTS)} extreme weights loaded"

    if EXTREME_MODULE_FUNCS_VEC and EXTREME_MODULE_WEIGHTS:
        missing_weights = [name for name in EXTREME_MODULE_FUNCS_VEC if name not in EXTREME_MODULE_WEIGHTS]
        if missing_weights:
            checks["extreme_functions_weights_match"] = f"WARN: Extreme functions missing weights: {missing_weights}"
            if overall_status == "UP": overall_status = "DEGRADED"
        else:
            checks["extreme_functions_weights_match"] = "OK"
    
    # Check 2: Memory data (same as before)
    if not os.path.exists(MEM_PATH):
        checks["memory_data_file_exists"] = f"FAIL: {MEM_PATH} not found"; overall_status = "DEGRADED"
    else:
        checks["memory_data_file_exists"] = "OK"
        if _total_samples_in_memory == 0 and os.path.getsize(MEM_PATH) > 0 :
             checks["memory_data_load_status"] = "WARN: Memory file exists but no samples loaded."
             if overall_status == "UP": overall_status = "DEGRADED"
        else: checks["memory_data_load_status"] = f"OK: {_total_samples_in_memory} samples currently loaded."

    # Check 3: Basic EXTREME tensor flow functionality
    try:
        dummy_grid = np.array([[-1, 1, 5], [2, -1, 8], [4, 6, -1]], dtype=int) # 3x3 for more complex modules
        _, _ = extreme_tensor_flow_score_detailed(dummy_grid, "health_check_extreme_tf")
        checks["extreme_tensor_flow_execution_test"] = "OK"
    except Exception as e:
        checks["extreme_tensor_flow_execution_test"] = f"FAIL: {str(e)}"
        logger.error("Health check: extreme_tensor_flow_execution_test failed.", exc_info=True, extra={'request_id': request_id})
        overall_status = "ERROR"

    # Check 4: CP Solver (same as before)
    try:
        _ = cp_model.CpModel()
        checks["cp_solver_availability_test"] = "OK"
    except Exception as e:
        checks["cp_solver_availability_test"] = f"FAIL: CP Model basic test failed - {str(e)}"
        logger.error("Health check: cp_solver_availability_test failed.", exc_info=True, extra={'request_id': request_id})
        overall_status = "ERROR"

    components_info = {
        "numpy_version": np.__version__,
        "ortools_version": getattr(cp_model, '__version__', "unknown"), # Try to get ortools version
        "analyzer_type": "Extreme Logic Modules"
    }
    return AnalyzeHealthStatus(
        status=overall_status,
        analysis_engine_version=ANALYSIS_ENGINE_VERSION_EXTREME, # Use new version
        checks=checks, components=components_info
    )

# --- Main execution for local testing (optional) ---
if __name__ == "__main__":
    import uvicorn
    if not os.path.exists(MEM_PATH):
        logger.info(f"Creating dummy {MEM_PATH} for testing.")
        dummy_mem_data = {"memory_cards": [[[1,2,-1,4],[-1,3,1,5],[2,-1,3,6],[7,8,9,-1]]]} # 4x4 example
        with open(MEM_PATH, "w") as f_mem: json.dump(dummy_mem_data, f_mem) # Renamed f to f_mem
        load_memory_data("main_startup")

    logger.info(f"Starting Uvicorn server for EXTREME Analyzer FastAPI app (main.py). Access OpenAPI docs at /docs.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
