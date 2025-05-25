import json
import os
import time
import logging
import uuid # For unique log IDs
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, validator, Field
from typing import List, Dict, Tuple, Callable, Any, Optional
import numpy as np
from ortools.sat.python import cp_model
from tabulate import tabulate

# ── Logging Configuration ───────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s")
logger = logging.getLogger(__name__)

# ── File Paths ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEM_PATH = os.path.join(BASE_DIR, "memory_cards.json")
REASONING_LOG_PATH = os.path.join(BASE_DIR, "reasoning_log.jsonl")
MODULE_WEIGHTS_PATH = os.path.join(BASE_DIR, "module_weights.json")

# ── Table Formatting Utility ────────────────────────────────────────
def format_data_as_table(data_to_format: Any, headers_option: Any = None, tablefmt: str = "grid", floatfmt: str = ".2f", generate_default_headers_if_numpy_2d_and_no_headers: bool = False) -> str:
    current_headers = headers_option
    if isinstance(data_to_format, np.ndarray):
        final_data = data_to_format.tolist()
        if generate_default_headers_if_numpy_2d_and_no_headers and \
           (headers_option is None or headers_option == []) and \
           data_to_format.ndim == 2:
            num_cols = data_to_format.shape[1]
            current_headers = [f"Col {i+1}" for i in range(num_cols)]
    elif isinstance(data_to_format, list):
        final_data = data_to_format
    else:
        logger.warning(f"Unsupported data type for table formatting: {type(data_to_format)}")
        return "Unsupported data type for table formatting."
    if not final_data or (isinstance(final_data, list) and all(not row for row in final_data)):
        return "No data to format."
    if current_headers is None: actual_tabulate_headers = []
    else: actual_tabulate_headers = current_headers
    try:
        return tabulate(final_data, headers=actual_tabulate_headers, tablefmt=tablefmt, floatfmt=floatfmt if floatfmt else None)
    except Exception as e:
        logger.error(f"Error during table formatting: {e}", exc_info=True)
        return f"Error formatting table: {str(e)}"

app = FastAPI(title="MetaCognitive Scratch Card Solver (v5.0)", version="5.0")

# ── 1. Vectorized Module Functions & Weights Management ─────────────
def a6_fixed_position_vec(grid: np.ndarray) -> np.ndarray:
    return grid == -1

def b1_row_feature_vec(grid: np.ndarray) -> np.ndarray:
    feature_map = np.zeros_like(grid, dtype=float)
    for r_idx in range(grid.shape[0]):
        cnt = np.sum(grid[r_idx, :] != -1)
        feature_map[r_idx, :] = cnt
    return feature_map

def c2_col_feature_vec(grid: np.ndarray) -> np.ndarray:
    feature_map = np.zeros_like(grid, dtype=float)
    for c_idx in range(grid.shape[1]):
        cnt = np.sum(grid[:, c_idx] != -1)
        feature_map[:, c_idx] = cnt
    return feature_map

MODULE_FUNCS_VEC: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "A6": a6_fixed_position_vec,
    "B1": b1_row_feature_vec,
    "C2": c2_col_feature_vec,
}

# Global module weights, loaded from file
MODULE_WEIGHTS: Dict[str, float] = {}

def _load_module_weights() -> None:
    global MODULE_WEIGHTS
    default_weights = {
        "A6": 1.0, # Default if file not found or key missing
        "B1": 0.5,
        "C2": 0.8,
    }
    if os.path.exists(MODULE_WEIGHTS_PATH):
        try:
            with open(MODULE_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                loaded_weights = json.load(f)
                # Merge with defaults: loaded overrides default, but keeps defaults for new keys
                MODULE_WEIGHTS = {**default_weights, **loaded_weights}
            logger.info(f"Loaded module weights ({len(MODULE_WEIGHTS)} entries) from {MODULE_WEIGHTS_PATH}.")
        except Exception as e:
            logger.error(f"Failed to load module weights from {MODULE_WEIGHTS_PATH}: {e}. Using default weights.", exc_info=True)
            MODULE_WEIGHTS = default_weights
    else:
        logger.info(f"Module weights file {MODULE_WEIGHTS_PATH} not found. Initializing with default weights and creating file.")
        MODULE_WEIGHTS = default_weights
        _save_module_weights() # Create the file with defaults

def _save_module_weights() -> None:
    try:
        with open(MODULE_WEIGHTS_PATH, "w", encoding="utf-8") as f:
            json.dump(MODULE_WEIGHTS, f, indent=4, sort_keys=True)
        logger.info(f"Saved module weights to {MODULE_WEIGHTS_PATH}.")
    except Exception as e:
        logger.error(f"Failed to save module weights to {MODULE_WEIGHTS_PATH}: {e}", exc_info=True)

_load_module_weights() # Load weights at startup

# tensor_flow_score_vec_all is not directly used by CP-SAT now, but can be for other heuristics
def tensor_flow_score_vec_all(grid: np.ndarray) -> np.ndarray:
    total_score_map = np.zeros(grid.shape, dtype=float)
    active_weights = MODULE_WEIGHTS # Use the global, potentially adjusted weights
    for name, func in MODULE_FUNCS_VEC.items():
        if name in active_weights: # Check if weight exists
            total_score_map += func(grid).astype(float) * active_weights[name]
        else:
            logger.warning(f"Weight for module {name} not found in MODULE_WEIGHTS. Skipping.")
    return total_score_map

# ── 2. Enhanced Feature Tensor ────────────────────────────────────
def build_feature_tensor(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    valid_values = grid[grid != -1]
    max_val_in_grid = int(np.max(valid_values)) if valid_values.size > 0 else 1 # Ensure max_val is at least 1
    
    # C: 0:normalized_value, 1:is_empty, 2:norm_row_pos, 3:norm_col_pos, 4+N:one-hot_value
    num_one_hot_channels = max_val_in_grid 
    C = 4 + num_one_hot_channels 
    
    tensor = np.zeros((H, W, C), dtype=float)
    for r in range(H):
        for c in range(W):
            val = grid[r, c]
            # Channel 0: Normalized value (or 0 if empty)
            tensor[r, c, 0] = (float(val) / max_val_in_grid) if val != -1 else 0.0
            # Channel 1: Is empty cell
            tensor[r, c, 1] = 1.0 if val == -1 else 0.0
            # Channel 2: Normalized row position
            tensor[r, c, 2] = float(r) / (H - 1) if H > 1 else 0.0
            # Channel 3: Normalized column position
            tensor[r, c, 3] = float(c) / (W - 1) if W > 1 else 0.0
            # Channels 4 to 4+max_val_in_grid-1: One-hot encoding of the number itself
            if val != -1:
                if 1 <= val <= max_val_in_grid: # ensure val is within expected one-hot range
                    tensor[r, c, 4 + int(val) - 1] = 1.0
    return tensor

def calculate_scores_from_tensor(feature_tensor: np.ndarray, grid: np.ndarray) -> np.ndarray:
    # Simple sum of features for now. Could be weighted by MODULE_WEIGHTS related to tensor channels.
    # These weights could also be learned.
    # Example: Base weight + specific channel weights
    base_channel_weights = { # Example conceptual weights for tensor channels
        "value_norm": 0.2,
        "is_empty": 0.1, # Might be less useful for scoring filled cells, but good for context
        "pos_norm": 0.05, # Small position bias
        "one_hot_base": 0.5 # Base weight for specific number features
    }
    H, W, C = feature_tensor.shape
    
    # For simplicity, let's use a dynamic weighting based on current MODULE_WEIGHTS
    # This is a placeholder for a more sophisticated tensor channel weighting strategy
    # For now, using a simple sum as before, but acknowledging this is where per-channel weights would go.
    
    num_static_channels = 4 # value_norm, is_empty, norm_row_pos, norm_col_pos
    # Dynamic weights for one-hot encoded channels, could be linked to MODULE_WEIGHTS if they represent concepts
    # For now, a simple approach:
    weights = np.ones(C, dtype=float) 
    # Example: if MODULE_WEIGHTS["B1"] (row feature) is high, maybe upweight row-related tensor channels.
    # This part needs more sophisticated design for actual "rule-driven tensor weighting".
    # For now, we just sum, which is equivalent to weights = np.ones(C).
    
    # A more direct interpretation of existing MODULE_WEIGHTS for tensor calculation:
    # If we map MODULE_WEIGHTS concepts to tensor channels. E.g. A6 (empty), B1 (row), C2 (col)
    # This part is highly experimental and depends on the semantic meaning of your module weights
    # and tensor channels. The original implementation was a simple sum.
    # Let's keep it simple for now to avoid overcomplicating without clear semantics.
    # weights[1] = MODULE_WEIGHTS.get("A6", 1.0) # If A6 relates to "is_empty"

    return np.tensordot(feature_tensor, weights, axes=([2], [0]))


# ── 3. Lightweight Memory Module ───────────────────────────────────
_memory: Dict[str, Dict[str, Any]] = {} # Stores {board_id: {action_key: {count, total_score}}}

def _load_memory() -> None:
    global _memory
    if os.path.exists(MEM_PATH):
        try:
            with open(MEM_PATH, "r", encoding="utf-8") as f: _memory = json.load(f)
            logger.info(f"Loaded memory ({len(_memory)} entries) from {MEM_PATH}.")
        except Exception as e:
            logger.error(f"Failed to load memory from {MEM_PATH}: {e}. Initializing empty memory.", exc_info=True)
            _memory = {}
    else:
        logger.info(f"Memory file {MEM_PATH} not found. Initializing empty memory.")
        _memory = {}

_load_memory() # Load memory at startup

def _make_board_id(grid: np.ndarray) -> str:
    # Consider using a more robust hash if grid states can be very similar but not identical
    # For now, HxW_emptyCount is a reasonable simplification.
    # from hashlib import sha256
    # return sha256(grid.tobytes()).hexdigest() # More robust, but slower
    H, W = grid.shape
    empty_count = int(np.sum(grid == -1))
    non_empty_sum = int(np.sum(grid[grid != -1])) # Add more features to ID
    return f"{H}x{W}_empty{empty_count}_sum{non_empty_sum}"


def get_legal_values(grid: np.ndarray) -> List[int]:
    valid_numbers = grid[grid != -1]
    if valid_numbers.size == 0: return [1] # Default if grid is empty
    max_val = int(np.max(valid_numbers))
    return list(range(1, max_val + 1))

def update_memory(grid: np.ndarray, r: int, c: int, v: int, score_of_action: float) -> None:
    board_id = _make_board_id(grid)
    action_key = f"{r}_{c}_{v}"
    if board_id not in _memory: _memory[board_id] = {}
    entry = _memory[board_id].setdefault(action_key, {"count": 0, "total_score": 0.0})
    entry["count"] += 1
    entry["total_score"] += score_of_action
    # logger.debug(f"Updated memory for {board_id} - {action_key}: {entry}")
    # Note: To make memory adaptive to correctness, entry could include "correct_count".
    # This would require feedback on whether the action (r,c,v) was ultimately good.

def _save_memory() -> None:
    try:
        with open(MEM_PATH, "w", encoding="utf-8") as f:
            json.dump(_memory, f, indent=2, sort_keys=True) # indent=2 for smaller files
        logger.info(f"Saved memory ({len(_memory)} board states) to {MEM_PATH}.")
    except Exception as e:
        logger.error(f"Failed to save memory to {MEM_PATH}: {e}", exc_info=True)

def mem_score(grid: np.ndarray, r: int, c: int, v: int) -> float:
    board_id = _make_board_id(grid)
    action_key = f"{r}_{c}_{v}"
    if board_id in _memory and action_key in _memory[board_id]:
        entry = _memory[board_id][action_key]
        if entry["count"] > 0:
            return entry["total_score"] / entry["count"]
    return 0.0

# ── 3.5 MetaCognition Log Module ────────────────────────────────────
class MetaCognitionLog:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self._log_buffer: List[Dict[str, Any]] = []
        self._load_log_on_startup()

    def _load_log_on_startup(self):
        # Typically, .jsonl is append-only, but if needed for analysis, could load.
        # For now, we just ensure the directory exists.
        # If we wanted to load it all into memory (not recommended for large logs):
        # if os.path.exists(self.log_path):
        #    with open(self.log_path, "r", encoding="utf-8") as f:
        #        for line in f:
        #            try: self._log_buffer.append(json.loads(line))
        #            except json.JSONDecodeError: logger.warning(f"Skipping malformed line in {self.log_path}")
        # logger.info(f"Loaded {len(self._log_buffer)} entries from {self.log_path} into buffer (if implemented).")
        pass # Not loading into memory buffer by default to save resources

    def log_event(self, event_data: Dict[str, Any]):
        event_data["log_id"] = str(uuid.uuid4()) # Unique ID for each event
        event_data["timestamp"] = time.time()
        # Ensure all parts of event_data are JSON serializable (e.g. convert numpy types)
        def sanitize_for_json(item):
            if isinstance(item, np.integer): return int(item)
            if isinstance(item, np.floating): return float(item)
            if isinstance(item, np.ndarray): return item.tolist()
            if isinstance(item, tuple): return list(item) # Convert tuples to lists
            return item

        sanitized_event_data = {k: sanitize_for_json(v) for k, v in event_data.items()}
        
        self._log_buffer.append(sanitized_event_data)
        # logger.debug(f"Logged event {sanitized_event_data['log_id']}")
        # For immediate persistence (optional, can be batched)
        # self.save_log_entry(sanitized_event_data)


    def save_log_entry(self, event_data_json_serializable: Dict[str, Any]):
        """Appends a single, already serialized event to the log file."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_data_json_serializable) + "\n")
        except Exception as e:
            logger.error(f"Failed to append event to {self.log_path}: {e}", exc_info=True)
            
    def flush_buffer_to_log(self):
        """Saves all events currently in the buffer to the log file and clears buffer."""
        if not self._log_buffer:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                for event_data in self._log_buffer:
                    f.write(json.dumps(event_data) + "\n")
            logger.info(f"Flushed {len(self._log_buffer)} events to {self.log_path}.")
            self._log_buffer = [] # Clear buffer after saving
        except Exception as e:
            logger.error(f"Failed to flush buffer to {self.log_path}: {e}", exc_info=True)

# Instantiate the logger
meta_logger = MetaCognitionLog(REASONING_LOG_PATH)

# ── 4. CP-SAT Solver ──────────────────────────────────────────────
def build_and_solve_cp_vec(grid: np.ndarray, candidates: List[Tuple[int,int,int]], legal_vals: List[int]):
    t_start_total = time.time()
    CP_SOLVER_TIME_LIMIT_SECONDS = 5.0
    SCORE_NORMALIZATION_FACTOR = 10000

    t0_ft = time.time()
    feature_tensor = build_feature_tensor(grid)
    t1_ft = time.time()
    tf_scores_map = calculate_scores_from_tensor(feature_tensor, grid) # This is a map HxW
    t2_score_calc = time.time()

    # logger.info(f"TF scores map for grid {grid.shape}:\n{format_data_as_table(tf_scores_map, floatfmt='.3f', generate_default_headers_if_numpy_2d_and_no_headers=True)}")
    logger.debug(f"Time - Feature Tensor build: {t1_ft - t0_ft:.4f}s, TF Score calculation: {t2_score_calc - t1_ft:.4f}s")

    if not candidates:
        logger.warning("No candidates provided for CP-SAT solver.")
        return []

    model = cp_model.CpModel()
    num_candidates = len(candidates)
    chosen_idx_var = model.NewIntVar(0, num_candidates - 1, "chosen_idx")

    # Extract TF scores for each candidate (r,c)
    candidate_tf_scores_raw = [tf_scores_map[r_cand, c_cand] for r_cand, c_cand, _ in candidates]
    
    # Calculate memory scores for each candidate (r,c,v)
    candidate_mem_scores_raw = [mem_score(grid, r_cand, c_cand, v_cand) for r_cand, c_cand, v_cand in candidates]

    # Combine scores (TF score is for position, Mem score is for specific action r,c,v)
    # The philosophy here is that tf_score is a general heuristic for the cell,
    # and mem_score is a learned value for the specific action.
    candidate_total_scores_raw = [
        tf + mem for tf, mem in zip(candidate_tf_scores_raw, candidate_mem_scores_raw)
    ]
    
    candidate_total_scores_int = [int(s * SCORE_NORMALIZATION_FACTOR) for s in candidate_total_scores_raw]
    candidate_tf_scores_int = [int(s * SCORE_NORMALIZATION_FACTOR) for s in candidate_tf_scores_raw]
    # No need for candidate_mem_scores_int directly in model if total and tf are used to derive it later.

    min_total_score = min(candidate_total_scores_int) if candidate_total_scores_int else 0
    max_total_score = max(candidate_total_scores_int) if candidate_total_scores_int else 0
    objective_var = model.NewIntVar(min_total_score, max_total_score, "objective_score")
    model.AddElement(chosen_idx_var, candidate_total_scores_int, objective_var)

    min_tf_score = min(candidate_tf_scores_int) if candidate_tf_scores_int else 0
    max_tf_score = max(candidate_tf_scores_int) if candidate_tf_scores_int else 0
    chosen_tf_score_var = model.NewIntVar(min_tf_score, max_tf_score, "chosen_tf_score")
    model.AddElement(chosen_idx_var, candidate_tf_scores_int, chosen_tf_score_var)
    
    model.Maximize(objective_var)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = CP_SOLVER_TIME_LIMIT_SECONDS
    solver.parameters.num_workers = os.cpu_count() or 1
    # solver.parameters.log_search_progress = True # Enable for debugging solver
    
    t_before_solve = time.time()
    status = solver.Solve(model)
    t_after_solve = time.time()
    
    logger.info(f"CP-SAT Solve time: {t_after_solve - t_before_solve:.4f}s. Status: {solver.StatusName(status)}")

    results = []
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        selected_idx = solver.Value(chosen_idx_var)
        r_sol, c_sol, v_sol = candidates[selected_idx]
        
        final_total_score_norm = solver.Value(objective_var) / SCORE_NORMALIZATION_FACTOR
        tf_score_of_chosen_norm = solver.Value(chosen_tf_score_var) / SCORE_NORMALIZATION_FACTOR
        
        # Recalculate/fetch mem_score for the chosen one to be exact, or derive
        # mem_score_of_chosen_norm = candidate_mem_scores_raw[selected_idx] # More direct
        mem_score_of_chosen_norm = final_total_score_norm - tf_score_of_chosen_norm # Derived

        results.append(
            (r_sol, c_sol, v_sol, final_total_score_norm, tf_score_of_chosen_norm, mem_score_of_chosen_norm)
        )
        logger.info(
            f"CP-SAT solution: Best candidate index {selected_idx} -> ({r_sol},{c_sol},{v_sol}), "
            f"TotalScore: {final_total_score_norm:.4f} (TF: {tf_score_of_chosen_norm:.4f}, Mem: {mem_score_of_chosen_norm:.4f})"
        )
    else:
        # Log model details if invalid
        if status == cp_model.MODEL_INVALID: logger.error(f"CP-SAT Model Invalid. Validation: {model.Validate()}")
        elif status == cp_model.INFEASIBLE: logger.warning("CP-SAT Model Infeasible: No solution satisfies all constraints.")
        else: logger.warning(f"CP-SAT solver did not find an optimal/feasible solution. Status: {solver.StatusName(status)}")

    logger.debug(f"Total time for build_and_solve_cp_vec: {time.time() - t_start_total:.4f}s. Found {len(results)} solution(s).")
    return results, solver.StatusName(status), feature_tensor.shape[-1] # Return solver status and num_channels for logging


# ── 5. API Endpoints & Pydantic Models ───────────────────────────
class ProposedValue(BaseModel):
    pos: List[int]
    value: int
    @validator("pos")
    def _check_pos_length(cls, p_val: List[int]) -> List[int]:
        if len(p_val) != 2: raise ValueError("pos must contain [row, col]")
        if not all(isinstance(x, int) for x in p_val): raise ValueError("pos elements must be integers")
        return p_val

class AnalyzeRequest(BaseModel):
    new_card: List[List[int]]
    proposed_values: List[ProposedValue]

    @validator("new_card")
    def _check_grid_is_valid_rectangle(cls, grid_list: List[List[int]]) -> List[List[int]]:
        if not grid_list: raise ValueError("new_card cannot be empty")
        if not isinstance(grid_list, list) or not all(isinstance(row, list) for row in grid_list):
            raise ValueError("new_card must be a list of lists")
        if not grid_list[0]: raise ValueError("new_card rows cannot be empty; grid must have columns")
        first_row_len = len(grid_list[0])
        if not all(len(row) == first_row_len for row in grid_list):
            raise ValueError("new_card 必須是矩形 (all rows must have the same length)")
        return grid_list

    @validator("proposed_values", each_item=True)
    def _check_proposed_value_bounds(cls, pv: ProposedValue, values: Dict[str, Any]) -> ProposedValue:
        grid_list = values.get("new_card")
        if grid_list and isinstance(grid_list, list) and grid_list:
            try:
                grid_np = np.array(grid_list, dtype=int)
                rows, cols = grid_np.shape
                r, c = pv.pos
                if not (0 <= r < rows and 0 <= c < cols):
                    raise ValueError(f"Proposed position [{r},{c}] out of bounds for grid {rows}x{cols}")
                legal_game_vals = get_legal_values(grid_np) # Expensive to call per item, but ensures correctness
                if grid_np[r,c] != -1:
                     raise ValueError(f"Proposed position [{r},{c}] is not empty (current value: {grid_np[r,c]}).")
                if pv.value not in legal_game_vals:
                     raise ValueError(f"Proposed value {pv.value} is not a legal value {legal_game_vals} for the current grid state.")
            except Exception as e: # Catch broad exceptions if grid_np itself is malformed from earlier validation
                logger.error(f"Error during proposed_values validation with grid: {e}")
                raise ValueError(f"Invalid grid data encountered while validating proposed_values: {e}") from e
        return pv

@app.post("/analyze")
async def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    try:
        grid_np = np.array(req.new_card, dtype=int)
    except ValueError as ve:
        logger.error(f"Error converting new_card to NumPy array: {ve}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Invalid new_card format: {ve}")

    logger.info(f"Received grid ({grid_np.shape[0]}x{grid_np.shape[1]}):\n{format_data_as_table(grid_np, generate_default_headers_if_numpy_2d_and_no_headers=True)}")
    grid_id = _make_board_id(grid_np) # For logging

    current_legal_game_values = get_legal_values(grid_np)
    logger.info(f"Legal game values for the grid: {current_legal_game_values}")

    valid_candidates = []
    for pv_item in req.proposed_values:
        r, c = pv_item.pos[0], pv_item.pos[1]
        v = pv_item.value
        if 0 <= r < grid_np.shape[0] and 0 <= c < grid_np.shape[1]:
            if grid_np[r, c] == -1 and v in current_legal_game_values:
                valid_candidates.append((r, c, v))
            else:
                logger.warning(f"Skipping proposed value (pos [{r},{c}] (val: {grid_np[r,c]}), value {v}). Cell not empty or value not in legal game values {current_legal_game_values}.")
        else:
            logger.warning(f"Skipping proposed value due to out-of-bounds position: pos [{r},{c}] for grid {grid_np.shape}.")

    if not valid_candidates:
        logger.warning("No valid candidates derived from proposed_values for CP-SAT solver.")
        # Log this event too
        event_data_no_cand = {
            "grid_snapshot_id": grid_id, "grid_shape": grid_np.shape,
            "candidates_considered": [], "chosen_action": None,
            "solver_status": "NO_VALID_CANDIDATES", "predicted_total_score": None,
            "tf_score_component": None, "mem_score_component": None,
            "rules_and_weights_snapshot": MODULE_WEIGHTS.copy(),
            "feature_tensor_channels": None, "feedback_correct": None, "feedback_notes": "No valid candidates"
        }
        meta_logger.log_event(event_data_no_cand)
        background_tasks.add_task(meta_logger.flush_buffer_to_log) # Save log
        return {"status": "no_valid_candidates", "result": None, "message": "No valid candidates to process after filtering."}

    # Run solver in threadpool
    solver_results, solver_status_str, ft_channels = await run_in_threadpool(
        build_and_solve_cp_vec, grid_np, valid_candidates, current_legal_game_values
    )

    if not solver_results:
        logger.warning("CP-SAT Solver returned no solution.")
        event_data_fail = {
            "grid_snapshot_id": grid_id, "grid_shape": grid_np.shape,
            "candidates_considered": valid_candidates, "chosen_action": None,
            "solver_status": solver_status_str, "predicted_total_score": None,
            "tf_score_component": None, "mem_score_component": None,
            "rules_and_weights_snapshot": MODULE_WEIGHTS.copy(),
            "feature_tensor_channels": ft_channels, "feedback_correct": None, "feedback_notes": "Solver failed or no feasible solution"
        }
        meta_logger.log_event(event_data_fail)
        background_tasks.add_task(meta_logger.flush_buffer_to_log)
        return {"status": "solver_fail", "result": None, "message": f"Solver did not find a solution. Status: {solver_status_str}"}

    r_best, c_best, v_best, final_total_score, final_tf_score, final_mem_score = solver_results[0]

    # Update simple memory with the predicted total score
    update_memory(grid_np, r_best, c_best, v_best, final_total_score)
    
    # Log detailed reasoning event
    event_data_success = {
        "grid_snapshot_id": grid_id,
        "grid_shape": grid_np.shape,
        "raw_grid_input": req.new_card, # For exact reproducibility
        "candidates_considered": valid_candidates,
        "chosen_action": (r_best, c_best, v_best),
        "solver_status": solver_status_str,
        "predicted_total_score": final_total_score, # This is the CONFIDENCE SCORE
        "tf_score_component": final_tf_score,
        "mem_score_component": final_mem_score,
        "rules_and_weights_snapshot": MODULE_WEIGHTS.copy(), # Snapshot of weights used
        "feature_tensor_channels": ft_channels,
        "feedback_correct": None, # To be filled by feedback mechanism
        "feedback_notes": None
    }
    meta_logger.log_event(event_data_success)
    
    # Add tasks to run in background (saving memory, saving log)
    background_tasks.add_task(_save_memory)
    background_tasks.add_task(meta_logger.flush_buffer_to_log)
    # Periodically save module weights if they were changed by some online process
    # background_tasks.add_task(_save_module_weights) # Only if weights can change during runtime without feedback endpoint

    return {
        "status": "success",
        "result": {
            "pos": [r_best, c_best],
            "value": v_best,
            "confidence_score": round(final_total_score, 4), # Explicitly named
            "tensor_flow_score_component": round(final_tf_score, 4),
            "memory_score_component": round(final_mem_score, 4)
        },
        "log_id": event_data_success["log_id"] # Return log_id for potential feedback linkage
    }

# ── 6. Feedback and Meta-Learning (Conceptual Placeholders) ───────
class FeedbackRequest(BaseModel):
    log_id: str = Field(..., description="The unique ID of the reasoning event to provide feedback for.")
    is_correct: bool = Field(..., description="Was the prediction correct?")
    notes: Optional[str] = Field(None, description="Optional notes about this feedback.")

# @app.post("/feedback")
# async def submit_feedback(req: FeedbackRequest, background_tasks: BackgroundTasks):
#     logger.info(f"Received feedback for log_id {req.log_id}: correct={req.is_correct}, notes='{req.notes}'.")
#     
#     # 1. Update the specific log entry in reasoning_log.jsonl
#     # This is complex as jsonl is append-only. Would typically involve:
#     # - Reading the file, finding the line, updating it, writing to a new file, then replacing.
#     # - Or, using a proper database for the reasoning log if frequent updates are needed.
#     # - Simpler for now: add a NEW log entry indicating feedback for a previous one.
#     feedback_event = {
#         "feedback_for_log_id": req.log_id,
#         "is_correct": req.is_correct,
#         "notes": req.notes,
#         "feedback_timestamp": time.time()
#     }
#     # meta_logger.log_event(feedback_event_for_separate_logging) # If logging feedback separately
#     # meta_logger.flush_buffer_to_log() # if used
#
#     # For demonstration, assume we can find and update the original event (pseudo-code)
#     # updated_log_count = update_reasoning_log_entry(req.log_id, req.is_correct, req.notes)
#     # if updated_log_count == 0:
#     #     raise HTTPException(status_code=404, detail=f"Log event with ID {req.log_id} not found.")
#
#     # 2. Trigger weight adjustment based on this feedback
#     #    This needs the original event details (rules used, etc.)
#     # original_event = get_reasoning_log_entry(req.log_id) # Fetch from log
#     # if original_event:
#     # background_tasks.add_task(
#     # adjust_module_weights_based_on_feedback,
#     # original_event["rules_and_weights_snapshot"],
#     # original_event["chosen_action"], # Or more specific features
#     # req.is_correct
#     # )
#     # background_tasks.add_task(_save_module_weights) # Save new weights
#     # logger.info(f"Weight adjustment process initiated for log_id {req.log_id}.")
#     # return {"status": "feedback_received", "log_id": req.log_id, "message": "Feedback processed and weight adjustment initiated."}
#     raise HTTPException(status_code=501, detail="Feedback endpoint is conceptual and not fully implemented.")

# def adjust_module_weights_based_on_feedback(
#     rules_snapshot: Dict[str, float],
#     # chosen_action_features: Any, # Could be the action, parts of grid, etc.
#     is_correct: bool
# ):
#     global MODULE_WEIGHTS
#     ADJUSTMENT_RATE = 0.01 # Small learning rate
#     MIN_WEIGHT = 0.01
#     MAX_WEIGHT = 5.0
#
#     logger.info(f"Adjusting weights based on feedback (correct={is_correct}). Current weights: {MODULE_WEIGHTS}")
#     
#     # Example: Adjust weights of all rules active in the snapshot
#     # A more sophisticated approach would identify which rules were most influential or uncertain.
#     for rule_name, original_weight_in_snapshot in rules_snapshot.items():
#         if rule_name in MODULE_WEIGHTS:
#             current_weight = MODULE_WEIGHTS[rule_name]
#             if is_correct:
#                 # Reinforce: increase weight slightly
#                 new_weight = current_weight * (1 + ADJUSTMENT_RATE)
#             else:
#                 # Punish: decrease weight slightly
#                 new_weight = current_weight * (1 - ADJUSTMENT_RATE * 1.5) # Punish more heavily
#
#             MODULE_WEIGHTS[rule_name] = max(MIN_WEIGHT, min(MAX_WEIGHT, new_weight))
#             logger.info(f"Adjusted weight for '{rule_name}': {current_weight:.4f} -> {MODULE_WEIGHTS[rule_name]:.4f}")
#     logger.info(f"Finished adjusting weights. New weights: {MODULE_WEIGHTS}")
#     # Note: This is a very basic adjustment. Real systems might use more complex credit assignment.

# def analyze_reasoning_log_insights(log_file_path: str):
#    # This function would be run offline or periodically.
#    # 1. Load all log entries from log_file_path.
#    # 2. For entries with feedback (is_correct is not None):
#    #    - Calculate overall accuracy.
#    #    - For each rule in MODULE_WEIGHTS:
#    #      - What's the correlation between this rule's weight (from snapshot) and correctness?
#    #      - How often is this rule present in correct vs. incorrect predictions?
#    #    - Analyze tf_score_component vs mem_score_component:
#    #      - Which component is a better predictor of correctness?
#    #      - Are there cases where they strongly disagree and one is usually right?
#    #    - Grid characteristics:
#    #      - Are there specific grid_shapes or grid_snapshot_ids (patterns) where errors are common?
#    #      - This could hint at missing features or rules.
#    #    - Confidence calibration:
#    #      - Is high `predicted_total_score` (confidence) actually correlated with correctness?
#    # 3. Print reports, suggest weight changes, or identify grid types needing special attention.
#    # Example:
#    # if rule 'B1' is often in error logs with high weight, its default might be too high,
#    # or it's being misapplied in certain contexts.
#    logger.info(f"Placeholder for analyzing insights from {log_file_path}")
#    pass


# ── Application Shutdown Event ───────────────────────────────────
@app.on_event("shutdown")
async def on_shutdown_event():
    logger.info("Application shutting down. Performing final saves...")
    _save_memory()
    meta_logger.flush_buffer_to_log() # Ensure all logs are written
    _save_module_weights() # Save potentially adjusted weights (if online adjustment was implemented)
    logger.info("Final saves complete. Exiting.")

# ── Main Execution (for local testing) ───────────────────────────
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local testing: http://127.0.0.1:8000")
    # Ensure log directory exists if RELATIVE_LOG_DIR is used
    # Path(BASE_DIR / RELATIVE_LOG_DIR).mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=8000)

