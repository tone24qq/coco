# main.py

import json
import os
import uuid # For request IDs
import time # For process time
import logging # For enhanced logging
from datetime import datetime # For health check timestamp

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, validator, Field
from typing import List, Dict, Tuple, Callable, Optional, Any
import numpy as np
from ortools.sat.python import cp_model # Assuming Google OR-Tools is used
from celery.result import AsyncResult # Kept as per original, though not used in /analyze
from celery_worker import solve_task  # Kept as per original

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - RequestID: %(request_id)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# --- Application Setup ---
app = FastAPI(
    title="Plug-in權重 + 張量流 + 自動數字範圍 - AI Manager Enabled",
    version="3.2", # Incremented version
    description="Enhanced analysis API with AI Manager capabilities: transparent decision-making, health checks, and detailed traceability."
)

# --- Middleware for Request ID and Logging ---
class RequestContextLogMiddleware: # Simplified from BaseHTTPMiddleware for direct use or inspiration
    async def __call__(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        # Make request_id available to loggers
        # This is a common pattern; for a robust solution, contextvars might be used.
        # For simplicity here, we'll pass it explicitly where needed or rely on a custom log filter if set up globally.
        # A simple way for this example: attach to request.state
        request.state.request_id = request_id

        # For global logging access to request_id, a custom logging Filter can be set up.
        # Example:
        # class RequestIdFilter(logging.Filter):
        #     def filter(self, record):
        #         record.request_id = getattr(request.state, 'request_id', 'N/A') # This line needs access to current request, tricky for global filter
        #         return True
        # logging.getLogger().addFilter(RequestIdFilter()) -> This setup is more involved.

        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}" # ensure string
        
        # Standard log for every request
        logger.info(f"Request processed: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s", extra={'request_id': request_id})
        return response

app.middleware("http")(RequestContextLogMiddleware())


# --- Core Logic (TensorFlow Rules, Weights, etc.) ---
# (Original numpy vectorized rules: a6_fixed_position_vec, m3_interval_consistency_vec_full, etc. remain unchanged)
def a6_fixed_position_vec(grid: np.ndarray) -> np.ndarray:
    return grid == -1

def m3_interval_consistency_vec_full(grid: np.ndarray) -> np.ndarray:
    R, C = grid.shape
    result = np.full((R, C), False, dtype=bool)
    for r in range(R):
        row = grid[r]
        vals = np.unique(row[row != -1])
        for val in vals:
            positions = np.where(row == val)[0]
            if len(positions) < 2:
                continue
            intervals = np.diff(positions)
            if intervals.min() <= 3: # Original logic: if any interval is <= 3, mark all positions of this value
                for pos_idx in range(len(positions)):
                    result[r, positions[pos_idx]] = True
    return result

def a9_diagonal_symmetry_vec(grid: np.ndarray) -> np.ndarray:
    R, C = grid.shape
    result = np.zeros((R, C), dtype=bool)
    for i in range(min(R, C)):
        if grid[i, i] != -1:
            result[i, i] = True
    return result

def m5_sequence_direction_vec(grid: np.ndarray) -> np.ndarray:
    R, C = grid.shape
    result = np.zeros((R, C), dtype=bool)
    for r in range(R):
        row = grid[r]
        valid_indices = np.where(row != -1)[0]
        if len(valid_indices) > 1:
            vals = row[valid_indices]
            if np.all(np.diff(vals) > 0): # Checks if sequence is strictly increasing
                 result[r, valid_indices] = True
    return result

def m14_mirror_diff_vec(grid: np.ndarray) -> np.ndarray:
    R, C = grid.shape
    mirror_c = C - 1 - np.arange(C)
    result = np.zeros((R, C), dtype=bool)
    for r in range(R):
        row = grid[r]
        mirrored_row = row[mirror_c]
        valid = (row != -1) & (mirrored_row != -1)
        diff = np.abs(row - mirrored_row)
        result[r, valid] = diff[valid] <= 2
    return result

def m15_parity_block_vec(grid: np.ndarray) -> np.ndarray:
    R, C = grid.shape
    pos_sum = np.add.outer(np.arange(R), np.arange(C))
    parity_pos = (pos_sum % 2 == 0)
    parity_val = (grid % 2 == 0)
    valid = (grid != -1)
    return valid & (parity_val == parity_pos)


MODULE_FUNCS_VEC: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "A6": a6_fixed_position_vec,
    "M3": m3_interval_consistency_vec_full,
    "A9": a9_diagonal_symmetry_vec,
    "M5": m5_sequence_direction_vec,
    "M14": m14_mirror_diff_vec,
    "M15": m15_parity_block_vec,
}

MODULE_WEIGHTS = {
    "A6": 1.0, "M3": 1.2, "A9": 1.0, "M5": 1.1,
    "M14": 1.0, "M15": 1.1,
}
# --- Pydantic Models for API Request and Response ---
class ProposedValue(BaseModel):
    pos: List[int] = Field(..., min_items=2, max_items=2, description="Position [row, col]")
    value: int = Field(..., gt=0, description="Proposed value for the cell")

class AnalyzeRequest(BaseModel):
    new_card: List[List[int]] = Field(..., description="The game grid, -1 for empty cells")
    proposed_values: List[ProposedValue] = Field(..., description="List of proposed values to analyze")

    @validator("new_card")
    def check_rectangular_and_numeric(cls, g):
        if not g:
            raise ValueError("new_card cannot be empty")
        if any(not isinstance(row, list) for row in g):
            raise ValueError("new_card must be a list of lists")
        if any(len(row) != len(g[0]) for row in g):
            raise ValueError("new_card must be a rectangular grid")
        if any(not isinstance(val, int) for row in g for val in row):
            raise ValueError("All values in new_card must be integers")
        return g

    @validator("proposed_values", each_item=True)
    def check_pv_bounds_and_value(cls, pv, values): # Pydantic v1 style validator
        grid = values.get("new_card")
        if grid: # Grid has already been validated by this point if it's present
            rows, cols = len(grid), len(grid[0])
            r, c = pv.pos
            if not (0 <= r < rows and 0 <= c < cols):
                raise ValueError(f"Proposed position {pv.pos} is out of bounds for grid size {rows}x{cols}")
            
            # Check if cell is already filled
            if grid[r][c] != -1:
                 raise ValueError(f"Cell at {pv.pos} is already filled with {grid[r][c]}, cannot propose a new value.")

            # card_max = 0
            # for row_data in grid:
            #     for cell_val in row_data:
            #         if cell_val != -1 and cell_val > card_max:
            #             card_max = cell_val
            # if card_max == 0 and pv.value > 0 : # Empty grid, any positive value is fine relative to "max existing"
            #     pass
            # elif pv.value < 1 or pv.value > card_max : # Card max should be used if there are values.
            #     # This validation needs careful thought: should proposed value be capped by existing max?
            #     # Original: if pv.value < 1 or pv.value > card_max: -> This means you can't propose a number higher than what's on board.
            #     # Let's assume for now the user's original intent for card_max was related to game rules.
            #     # The get_card_max_value is used for legal_values generation range, not strict proposal cap.
            #     pass # Relaxing this specific validation for now as get_legal_values will handle ranges
        return pv

class TensorRuleContribution(BaseModel):
    rule_name: str
    score_if_applied: float # The raw score (typically 1.0 if boolean mask)
    weight: float
    weighted_score: float # score_if_applied * weight

class CandidateDetail(BaseModel):
    pos: List[int]
    value: int
    is_valid_proposal: bool = True # Was it a valid spot and value initially?
    tensor_flow_contributions: List[TensorRuleContribution] = Field(default_factory=list)
    raw_tensor_flow_score: float # Sum of weighted_scores from contributions for this cell
    mem_score_value: float
    final_objective_score: float # Score used in CP model (e.g., raw_tensor_flow_score + factor * mem_score_value)
    is_selected_by_cp: bool = False
    cp_solver_notes: Optional[str] = None # e.g., "Optimal", "Feasible", "Not selected"

class AnalyzeResultDetail(CandidateDetail): # The chosen one inherits from CandidateDetail
    pass

class AnalyzeSuccessResponse(BaseModel):
    request_id: str
    status: str = "success"
    main_module_version: str = Field(default=app.version)
    analysis_engine_version: str = Field(default="1.1") # Version of this analysis logic
    message: Optional[str] = None
    result: Optional[AnalyzeResultDetail] = None # Optional if no valid candidate could be chosen
    all_candidates_evaluated: List[CandidateDetail]

class AnalyzeErrorResponse(BaseModel):
    request_id: str
    status: str # e.g., "error", "fail"
    main_module_version: str = Field(default=app.version)
    analysis_engine_version: str = Field(default="1.1")
    message: str
    error_type: Optional[str] = None
    details: Optional[Any] = None

# --- Memory Handling ---
MEM_PATH = os.path.join(os.path.dirname(__file__), "memory_cards.json")
_memory_freq: Dict[Tuple[int, int, int], int] = {} # (r, c, v) -> count
_total_samples_in_memory = 0

def load_memory_data(req_id: str = "startup"):
    global _memory_freq, _total_samples_in_memory
    _memory_freq = {}
    _total_samples_in_memory = 0
    try:
        if os.path.exists(MEM_PATH):
            with open(MEM_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for card in data.get("memory_cards", []):
                for r, row_data in enumerate(card):
                    for c, v in enumerate(row_data):
                        if v != -1: # Assuming -1 is empty, other numbers are values
                            _memory_freq[(r, c, v)] = _memory_freq.get((r, c, v), 0) + 1
                            _total_samples_in_memory += 1
            logger.info(f"Memory data loaded: {_total_samples_in_memory} samples from {len(data.get('memory_cards', []))} cards.", extra={'request_id': req_id})
        else:
            logger.warning(f"Memory file not found: {MEM_PATH}. Mem score will be 0.", extra={'request_id': req_id})
    except Exception as e:
        logger.error(f"Error loading memory data from {MEM_PATH}: {e}", exc_info=True, extra={'request_id': req_id})

load_memory_data() # Load at startup

def mem_score(r: int, c: int, v: int, legal_values_for_position: set) -> float:
    if v not in legal_values_for_position: # Should not happen if legal_values check is done prior
        return 0.0
    if _total_samples_in_memory == 0:
        return 0.0
    
    count = _memory_freq.get((r, c, v), 0)
    # Consider if mem_score should be normalized differently, e.g. by total occurrences at (r,c) or total occurrences of v.
    # Original: cnt / _total_samples. This normalizes by total entries across all positions/values.
    return float(count) / float(_total_samples_in_memory)

# --- Tensor Flow Scoring (Detailed) ---
def tensor_flow_score_vec_detailed(grid: np.ndarray, request_id: str) -> Tuple[np.ndarray, List[List[List[TensorRuleContribution]]]]:
    R, C = grid.shape
    total_score_grid = np.zeros((R, C), dtype=float)
    # rule_contributions_grid[r][c] will be List[TensorRuleContribution]
    rule_contributions_grid: List[List[List[TensorRuleContribution]]] = [[[] for _ in range(C)] for _ in range(R)]

    for name, func in MODULE_FUNCS_VEC.items():
        try:
            mask = func(grid) # boolean mask (R, C)
            weight = MODULE_WEIGHTS.get(name, 1.0)
            if weight == 0: continue # Skip if weight is zero

            current_rule_weighted_scores = mask.astype(float) * weight
            total_score_grid += current_rule_weighted_scores

            for r_idx in range(R):
                for c_idx in range(C):
                    if mask[r_idx, c_idx]: # If rule applied at this cell
                        contribution = TensorRuleContribution(
                            rule_name=name,
                            score_if_applied=1.0, # Mask is boolean, so applied score is 1.0
                            weight=weight,
                            weighted_score=current_rule_weighted_scores[r_idx, c_idx] # This is essentially 'weight'
                        )
                        rule_contributions_grid[r_idx][c_idx].append(contribution)
        except Exception as e:
            logger.error(f"Error processing rule '{name}' in tensor_flow_score_vec_detailed: {e}", exc_info=True, extra={'request_id': request_id})
            # Depending on desired robustness, either skip rule or assign neutral score, or raise
            pass
    return total_score_grid, rule_contributions_grid


def get_card_max_value(grid: np.ndarray) -> int:
    # Returns the maximum value currently on the card, or 0 if empty or only -1.
    if grid.size == 0: return 0
    valid_values = grid[grid != -1]
    return int(np.max(valid_values)) if valid_values.size > 0 else 0


def get_legal_values_for_placement(grid: np.ndarray) -> set:
    """
    Determines the set of numbers that can be legally placed on the board.
    This version assumes values are 1 up to card_max + 1 (allowing a new highest number),
    and not already present on the board.
    Modify this function based on actual game rules for what "legal values" means.
    """
    card_max_val = get_card_max_value(grid)
    # Example: legal values are 1 to (current max on card + k), or 1 to N if fixed range.
    # For this example, let's say up to max_val_on_card + 1, or minimum 1 up to a default like 10 if card is empty.
    upper_bound = card_max_val + 1 if card_max_val > 0 else 10 # Default upper if card empty

    all_possible_values = set(range(1, upper_bound + 1))
    
    # Values already used on the grid cannot be placed again (assuming Sudoku-like unique value constraint)
    # This needs clarification: are values unique across the *entire* grid, or per row/col/box?
    # Original get_legal_values seemed to imply global uniqueness for placement.
    used_values = set(grid.flatten())
    used_values.discard(-1) # Remove placeholder for empty

    return all_possible_values - used_values


# --- CP Solver Logic ---
def solve_cp_for_candidates(
    grid_shape: Tuple[int, int], # R, C
    current_grid_state: np.ndarray, # For context if needed by advanced constraints
    candidates_to_evaluate: List[CandidateDetail],
    request_id: str
) -> List[CandidateDetail]:
    """
    Uses Constraint Programming to select the best candidate.
    Modifies 'is_selected_by_cp' and 'cp_solver_notes' in the input list.
    """
    model = cp_model.CpModel()
    num_candidates = len(candidates_to_evaluate)

    if num_candidates == 0:
        return candidates_to_evaluate

    # Create one Boolean variable for each candidate proposal
    x = [model.NewBoolVar(f"x_{i}") for i in range(num_candidates)]

    # Constraint: Exactly one candidate must be chosen
    # (Or modify if multiple placements are allowed, or zero if none are good enough)
    model.Add(sum(x) == 1)

    # Constraints based on game rules (e.g., value uniqueness if not already handled)
    # Example: If two candidates propose the same value for different cells, only one can be chosen
    # This is implicitly handled if candidates are for *one* cell choice.
    # If candidates could be for *multiple* cells in one go, this needs more complex constraints.
    # The current setup `sum(x)==1` means we pick one (pos,value) pair from the proposed list.

    # Objective: Maximize the sum of scores of chosen candidates
    # The 'final_objective_score' in CandidateDetail is already prepared.
    objective_terms = []
    for i in range(num_candidates):
        # Ensure score is integer for CP solver if it requires it (ortools often prefers ints for objectives)
        # Original scaled by 1000. Let's maintain that for consistency.
        objective_terms.append(x[i] * int(candidates_to_evaluate[i].final_objective_score * 1000))
    
    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0.5  # Timeout for the solver
    solver.parameters.num_search_workers = os.cpu_count() or 1 # Use available cores
    # solver.parameters.log_search_progress = True # For debugging CP solver

    status = solver.Solve(model)
    
    solution_found = False
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        solution_found = True
        logger.info(f"CP Solver found a solution. Status: {solver.StatusName(status)}", extra={'request_id': request_id})
        for i in range(num_candidates):
            if solver.Value(x[i]) == 1:
                candidates_to_evaluate[i].is_selected_by_cp = True
                candidates_to_evaluate[i].cp_solver_notes = f"Selected by CP Solver ({solver.StatusName(status)})"
                logger.info(
                    f"Candidate {i} selected: Pos={candidates_to_evaluate[i].pos}, Val={candidates_to_evaluate[i].value}, Score={candidates_to_evaluate[i].final_objective_score}",
                    extra={'request_id': request_id}
                )
            elif candidates_to_evaluate[i].cp_solver_notes is None: # Don't overwrite if already failed validation
                 candidates_to_evaluate[i].cp_solver_notes = "Not selected by CP Solver"

    else:
        logger.warning(f"CP Solver did not find an optimal/feasible solution. Status: {solver.StatusName(status)}", extra={'request_id': request_id})
        for cand_detail in candidates_to_evaluate:
            if cand_detail.cp_solver_notes is None:
                cand_detail.cp_solver_notes = f"No solution found by CP Solver ({solver.StatusName(status)})"
    
    return candidates_to_evaluate


# --- API Endpoint: /analyze ---
@app.post("/analyze",
          response_model=AnalyzeSuccessResponse,
          responses={
              200: {"model": AnalyzeSuccessResponse},
              400: {"model": AnalyzeErrorResponse, "description": "Invalid input or no valid candidates"},
              422: {"model": AnalyzeErrorResponse, "description": "Validation error (Pydantic)"},
              500: {"model": AnalyzeErrorResponse, "description": "Internal server error"}
          },
          tags=["Analysis Engine"])
async def analyze(req: AnalyzeRequest, request: Request):
    """
    Analyzes the given grid and proposed values to suggest the best move.
    Implements "AI Manager" principles for transparency and robustness.
    """
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())) # Get from middleware or generate

    try:
        logger.info(f"Received analysis request. Grid size: {len(req.new_card)}x{len(req.new_card[0]) if req.new_card else 0}. Proposed values: {len(req.proposed_values)}.", extra={'request_id': request_id})
        grid = np.array(req.new_card, dtype=int)

        # 1. Calculate detailed tensor flow scores for the entire grid
        raw_tensor_scores_grid, rule_contributions_grid = tensor_flow_score_vec_detailed(grid, request_id)
        
        # 2. Determine legal values that can be placed (context-dependent based on game rules)
        # This function needs to be robust and reflect actual game rules.
        # For this example, using the refined get_legal_values_for_placement.
        globally_legal_values_for_new_placement = get_legal_values_for_placement(grid)
        if not globally_legal_values_for_new_placement:
            logger.warning("No globally legal values can be placed on the current grid.", extra={'request_id': request_id})
            # This might be an early exit or just a warning depending on rules.

        # 3. Prepare CandidateDetail objects for each valid proposed value
        all_evaluated_candidates: List[CandidateDetail] = []
        
        for pv_idx, pv in enumerate(req.proposed_values):
            r, c = pv.pos[0], pv.pos[1]
            val_proposed = pv.value

            # Basic validation: is the spot empty? (Pydantic validator should catch filled spots already)
            # Pydantic validator already checks if grid[r][c] != -1.
            # is_valid_proposal_flag = grid[r,c] == -1 and val_proposed in globally_legal_values_for_new_placement
            
            # More precise check for a specific proposal:
            # Is the proposed value itself one of the currently legal values for placement?
            if val_proposed not in globally_legal_values_for_new_placement:
                 logger.warning(f"Proposed value {val_proposed} at [{r},{c}] is not in the set of globally legal values for new placement. Skipping.", extra={'request_id': request_id})
                 cand_detail = CandidateDetail(
                    pos=[r,c], value=val_proposed,
                    is_valid_proposal=False,
                    raw_tensor_flow_score=0, mem_score_value=0, final_objective_score=0,
                    cp_solver_notes=f"Value {val_proposed} not in globally legal set {globally_legal_values_for_new_placement}"
                 )
                 all_evaluated_candidates.append(cand_detail)
                 continue

            # If spot is valid and value is generally legal:
            raw_tf_score_cell = raw_tensor_scores_grid[r, c]
            tf_contrib_cell = rule_contributions_grid[r][c] # This is List[TensorRuleContribution]
            
            # mem_score needs the set of values that would be legal AT THAT POSITION if it were empty
            # For simplicity, we use globally_legal_values_for_new_placement here,
            # but a more refined mem_score might get legal values specific to (r,c) if rules are localized.
            current_mem_score = mem_score(r, c, val_proposed, globally_legal_values_for_new_placement)
            
            # Define how raw_tf_score and mem_score combine for CP objective
            # Original: tensor_score + 5.0 * mem_score
            final_obj_for_cp = raw_tf_score_cell + (5.0 * current_mem_score)

            cand_detail = CandidateDetail(
                pos=[r, c],
                value=val_proposed,
                is_valid_proposal=True,
                tensor_flow_contributions=tf_contrib_cell,
                raw_tensor_flow_score=round(raw_tf_score_cell, 4),
                mem_score_value=round(current_mem_score, 4),
                final_objective_score=round(final_obj_for_cp, 4),
                is_selected_by_cp=False # Default
            )
            all_evaluated_candidates.append(cand_detail)

        # Filter out candidates that were initially invalid for CP solver
        candidates_for_cp_solver = [cd for cd in all_evaluated_candidates if cd.is_valid_proposal]

        if not candidates_for_cp_solver:
            logger.warning("No valid proposed values to submit to CP solver after initial checks.", extra={'request_id': request_id})
            return AnalyzeSuccessResponse(
                request_id=request_id,
                status="no_valid_candidates",
                message="None of the proposed values were valid for consideration by the solver.",
                result=None,
                all_candidates_evaluated=all_evaluated_candidates # Show why they were invalid
            )

        # 4. Run CP Solver (in a thread pool to not block event loop for CPU-bound task)
        # The CP solver will update the 'is_selected_by_cp' and 'cp_solver_notes' fields.
        # It's important that solve_cp_for_candidates modifies the list items in place or returns a new list
        # with these items correctly updated. Assuming it modifies in place.
        
        # Pass grid_shape and current_grid_state if CP constraints need them
        updated_candidate_details = await run_in_threadpool(
            solve_cp_for_candidates,
            grid.shape,
            grid,
            candidates_for_cp_solver, # Send only valid ones to CP
            request_id
        )
        
        # Merge back results from CP solver into the main list if it returned a subset
        # This loop ensures all_evaluated_candidates reflects CP results for those it processed.
        processed_map = {(cand.pos[0], cand.pos[1], cand.value): cand for cand in updated_candidate_details}
        for i, cand in enumerate(all_evaluated_candidates):
            if cand.is_valid_proposal:
                key = (cand.pos[0], cand.pos[1], cand.value)
                if key in processed_map:
                    all_evaluated_candidates[i] = processed_map[key]


        # 5. Determine the final selected result
        selected_by_cp = [cand for cand in all_evaluated_candidates if cand.is_selected_by_cp]
        
        final_result_detail: Optional[AnalyzeResultDetail] = None
        response_message = "Analysis complete."

        if not selected_by_cp:
            response_message = "CP Solver did not select any candidate."
            logger.info(response_message, extra={'request_id': request_id})
            status_val = "fail_no_selection_cp"
        elif len(selected_by_cp) == 1:
            # Convert chosen CandidateDetail to AnalyzeResultDetail (they are compatible)
            final_result_detail = AnalyzeResultDetail(**selected_by_cp[0].model_dump())
            response_message = f"Successfully selected a candidate: Pos={final_result_detail.pos}, Val={final_result_detail.value}."
            logger.info(response_message, extra={'request_id': request_id})
            status_val = "success"
        else: # Multiple selected (should not happen with sum(x)==1, but good to handle)
            # Picking the one with highest objective score if multiple (tie-breaking)
            final_result_detail = AnalyzeResultDetail(**max(selected_by_cp, key=lambda cd: cd.final_objective_score).model_dump())
            response_message = f"CP Solver returned multiple options. Selected one with highest score: Pos={final_result_detail.pos}, Val={final_result_detail.value}."
            logger.warning(response_message, extra={'request_id': request_id})
            status_val = "success_multiple_options"


        return AnalyzeSuccessResponse(
            request_id=request_id,
            status=status_val,
            message=response_message,
            result=final_result_detail,
            all_candidates_evaluated=all_evaluated_candidates
        )

    except HTTPException as he: # Re-raise HTTPExceptions
        logger.warning(f"HTTPException in /analyze: {he.detail}", extra={'request_id': request_id})
        # This will be handled by FastAPI's default error handling for HTTPExceptions
        # Or, you can return a custom AnalyzeErrorResponse
        raise he 
    except ValueError as ve: # Typically from Pydantic or our own validation
        logger.warning(f"ValueError in /analyze: {str(ve)}", exc_info=True, extra={'request_id': request_id})
        return AnalyzeErrorResponse(
            request_id=request_id, status="fail", message=str(ve), error_type="ValueError"
        ) # This will be a 400 or 422 effectively if FastAPI sends it like that.
          # To control status code: raise HTTPException(status_code=400, detail=...)
    except Exception as e:
        logger.error(f"Unexpected error in /analyze: {str(e)}", exc_info=True, extra={'request_id': request_id})
        # In a production system, you might want to hide details of unexpected errors from the client
        # and log them for backend investigation.
        return AnalyzeErrorResponse(
            request_id=request_id,
            status="error",
            message="An unexpected internal server error occurred.",
            error_type=e.__class__.__name__,
            # details={"exception_message": str(e)} # Optionally include more detail, carefully
        ) # This should ideally be a 500. FastAPI does this if an unhandled exception bubbles up.
          # For direct return, you'd use JSONResponse with status_code=500.

# --- Health Check Endpoint ---
class AnalyzeHealthStatus(BaseModel):
    status: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    main_module_version: str = Field(default=app.version)
    analysis_engine_version: str = Field(default="1.1")
    checks: Dict[str, str]
    components: Dict[str, Any] = Field(default_factory=dict)


@app.get("/health/analyze", response_model=AnalyzeHealthStatus, tags=["Health & Monitoring"])
async def health_analyze(request: Request):
    """Provides a health check for the analysis submodule's components and configuration."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    logger.info("Health check requested for /analyze components.", extra={'request_id': request_id})

    checks = {}
    overall_status = "UP" # States: UP, DEGRADED, DOWN/ERROR

    # Check 1: Module functions and weights consistency
    if not MODULE_FUNCS_VEC:
        checks["module_functions_load"] = "FAIL: MODULE_FUNCS_VEC is empty"
        overall_status = "DEGRADED"
    else:
        checks["module_functions_load"] = f"OK: {len(MODULE_FUNCS_VEC)} functions loaded"

    if not MODULE_WEIGHTS:
        checks["module_weights_load"] = "FAIL: MODULE_WEIGHTS is empty"
        overall_status = "DEGRADED"
    else:
        checks["module_weights_load"] = f"OK: {len(MODULE_WEIGHTS)} weights loaded"

    if MODULE_FUNCS_VEC and MODULE_WEIGHTS:
        missing_weights = [name for name in MODULE_FUNCS_VEC if name not in MODULE_WEIGHTS]
        # extra_weights = [name for name in MODULE_WEIGHTS if name not in MODULE_FUNCS_VEC] # Also good to check
        if missing_weights:
            checks["functions_weights_match"] = f"WARN: Functions missing weights: {missing_weights}"
            if overall_status == "UP": overall_status = "DEGRADED"
        else:
            checks["functions_weights_match"] = "OK"

    # Check 2: Memory data
    # Re-load memory data for health check to test loading mechanism, or just check current state
    # For this example, check current state. Consider if a reload test is needed.
    if not os.path.exists(MEM_PATH):
        checks["memory_data_file_exists"] = f"FAIL: {MEM_PATH} not found"
        overall_status = "DEGRADED"
    else:
        checks["memory_data_file_exists"] = "OK"
        if _total_samples_in_memory == 0 and os.path.getsize(MEM_PATH) > 0 : # File exists and not empty, but no samples
             checks["memory_data_load_status"] = "WARN: Memory data file exists but no samples loaded (_total_samples_in_memory is 0). Potential load issue or empty valid file."
             if overall_status == "UP": overall_status = "DEGRADED"
        else:
            checks["memory_data_load_status"] = f"OK: {_total_samples_in_memory} samples currently loaded."


    # Check 3: Basic tensor flow functionality (quick test)
    try:
        dummy_grid = np.array([[-1, 1], [2, -1]], dtype=int)
        _, _ = tensor_flow_score_vec_detailed(dummy_grid, request_id="health_check_tf")
        checks["tensor_flow_execution_test"] = "OK"
    except Exception as e:
        checks["tensor_flow_execution_test"] = f"FAIL: {str(e)}"
        logger.error("Health check: tensor_flow_execution_test failed.", exc_info=True, extra={'request_id': request_id})
        overall_status = "ERROR"

    # Check 4: CP Solver availability (basic instantiation)
    try:
        _ = cp_model.CpModel()
        # Test solve a trivial model?
        # model = cp_model.CpModel()
        # x = model.NewBoolVar('x')
        # model.Maximize(x)
        # solver = cp_model.CpSolver()
        # status = solver.Solve(model)
        # if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        #    raise RuntimeError(f"CP Solver trivial model failed with status: {solver.StatusName(status)}")
        checks["cp_solver_availability_test"] = "OK"
    except Exception as e:
        checks["cp_solver_availability_test"] = f"FAIL: CP Model basic test failed - {str(e)}"
        logger.error("Health check: cp_solver_availability_test failed.", exc_info=True, extra={'request_id': request_id})
        overall_status = "ERROR"

    # Component versions (example)
    components_info = {
        "numpy_version": np.__version__,
        "ortools_version": cp_model.SetVersionNumber if hasattr(cp_model, 'SetVersionNumber') else "unknown" # Or-tools version is harder to get directly, this is a placeholder
    }


    return AnalyzeHealthStatus(
        status=overall_status,
        checks=checks,
        components=components_info
    )

# --- Main execution for local testing (optional) ---
if __name__ == "__main__":
    import uvicorn
    # Example: Generate a dummy memory_cards.json if it doesn't exist for testing
    if not os.path.exists(MEM_PATH):
        logger.info(f"Creating dummy {MEM_PATH} for testing.")
        dummy_mem_data = {"memory_cards": [[[1,2,-1],[-1,3,1],[2,-1,3]]]}
        with open(MEM_PATH, "w") as f:
            json.dump(dummy_mem_data, f)
        load_memory_data() # Reload after creating

    logger.info("Starting Uvicorn server for main_enhanced.py. Access OpenAPI docs at /docs.")
    uvicorn.run(app, host="0.0.0.0", port=8000)

