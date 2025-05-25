import json
import os
import time
import logging
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, validator
from typing import List, Dict, Tuple, Callable
import numpy as np
from ortools.sat.python import cp_model
from tabulate import tabulate # <--- 新增 tabulate 載入

# ── Logging Configuration ───────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Table Formatting Utility ────────────────────────────────────────
def format_data_as_table(data_to_format, headers_option=None, tablefmt="grid", floatfmt=".2f", generate_default_headers_if_numpy_2d_and_no_headers=False):
    """
    Formats given data (list of lists or NumPy array) into a table string.

    Args:
        data_to_format: The data to format. Can be a list of lists or a NumPy array.
        headers_option: Headers for the table.
                        - A list of strings: explicitly sets column headers.
                        - "firstrow": uses the first row of data_to_format as headers.
                        - None or []: no headers will be printed (unless generate_default_headers... is True).
        tablefmt: The table format (e.g., "grid", "pipe", "simple").
        floatfmt: Format string for floating point numbers (e.g., ".2f" for 2 decimal places).
        generate_default_headers_if_numpy_2d_and_no_headers:
            If True, headers_option is None or [], and data is a 2D NumPy array,
            generates default headers like "Col 1", "Col 2".

    Returns:
        A string representing the formatted table.
    """
    current_headers = headers_option # Default to user-provided headers_option

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
        logger.warning("Unsupported data type for table formatting.")
        return "Unsupported data type for table formatting."

    if not final_data:
        return "No data to format."
    
    # tabulate expects headers to be a list/tuple of strings, or special strings like "firstrow" or "keys".
    # If current_headers is None from the logic above (and not "firstrow", etc.),
    # it's often better to pass an empty list for "no headers" explicitly.
    if current_headers is None:
        actual_tabulate_headers = []
    else:
        actual_tabulate_headers = current_headers
        
    try:
        # The floatfmt argument in tabulate applies to Python float objects.
        return tabulate(final_data, headers=actual_tabulate_headers, tablefmt=tablefmt, floatfmt=floatfmt if floatfmt else None)
    except Exception as e:
        logger.error(f"Error during table formatting: {e}", exc_info=True)
        return f"Error formatting table: {str(e)}"

app = FastAPI(title="Plug-in權重 + 張量流 + 多強化", version="4.0")

# ── 1. 原本的向量化模組函數們 ────────────────────────────────────
def a6_fixed_position_vec(grid: np.ndarray) -> np.ndarray:
    return grid == -1

def b1_row_feature_vec(grid: np.ndarray) -> np.ndarray:
    feature_map = np.zeros_like(grid, dtype=float)
    for r in range(grid.shape[0]):
        cnt = np.sum(grid[r, :] != -1)
        feature_map[r, :] = cnt
    return feature_map

def c2_col_feature_vec(grid: np.ndarray) -> np.ndarray:
    feature_map = np.zeros_like(grid, dtype=float)
    for c in range(grid.shape[1]):
        cnt = np.sum(grid[:, c] != -1)
        feature_map[:, c] = cnt
    return feature_map

MODULE_FUNCS_VEC: Dict[str, Callable] = {
    "A6": a6_fixed_position_vec,
    "B1": b1_row_feature_vec,
    "C2": c2_col_feature_vec,
}
MODULE_WEIGHTS = {
    "A6": 1.0,
    "B1": 0.5,
    "C2": 0.8,
}

def tensor_flow_score_vec_all(grid: np.ndarray) -> np.ndarray:
    total = np.zeros(grid.shape, dtype=float)
    for name, fn in MODULE_FUNCS_VEC.items():
        total += fn(grid).astype(float) * MODULE_WEIGHTS[name]
    return total

# ── 2. 增強版特徵張量 ────────────────────────────────────────────
def build_feature_tensor(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    # Handle case where grid might be all -1 or empty before finding max
    valid_values = grid[grid != -1]
    maxv = int(np.max(valid_values)) if valid_values.size > 0 else 1 # Ensure maxv is at least 1
    
    C = 4 + maxv # Number of channels
    tensor = np.zeros((H, W, C), dtype=float)

    for r in range(H):
        for c in range(W):
            v = grid[r, c]
            
            # Channel 0: Normalized value (0 if -1)
            tensor[r, c, 0] = (float(v) / maxv) if v != -1 else 0.0
            
            # Channel 1: Is fixed/empty cell (-1)
            tensor[r, c, 1] = 1.0 if v == -1 else 0.0
            
            # Channel 2: Normalized row index
            tensor[r, c, 2] = float(r) / (H - 1) if H > 1 else 0.0
            
            # Channel 3: Normalized column index
            tensor[r, c, 3] = float(c) / (W - 1) if W > 1 else 0.0
            
            # Channels 4 to 4+maxv-1: One-hot encoding of the value (if not -1)
            if v != -1:
                if 1 <= v <= maxv: # Ensure v is within the expected one-hot range
                    tensor[r, c, 4 + v - 1] = 1.0
                # else: logger.warning(f"Value {v} at ({r},{c}) out of expected range [1, {maxv}] for one-hot encoding.")
    return tensor

def calculate_scores_from_tensor(ft: np.ndarray, grid: np.ndarray) -> np.ndarray:
    # Assuming weights are dynamically sized based on the last dimension of the feature tensor
    weights = np.ones(ft.shape[-1], dtype=float) 
    # Example: customize weights, e.g., weights[0] = 1.5, weights[1] = 0.5 etc.
    # Ensure ft.shape[-1] matches len(weights)
    return np.tensordot(ft, weights, axes=([2], [0]))

# ── 3. 轻量记忆模块 ────────────────────────────────────────────
MEM_PATH = os.path.join(os.path.dirname(__file__), "memory_cards.json")
_memory: Dict[str, Dict[str, float]] = {}

def _load_memory():
    global _memory
    if os.path.exists(MEM_PATH):
        try:
            with open(MEM_PATH, "r", encoding="utf-8") as f:
                _memory = json.load(f)
            logger.info(f"Loaded memory ({len(_memory)} entries).")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {MEM_PATH}. Initializing empty memory.")
            _memory = {}
        except Exception as e:
            logger.error(f"Failed to load memory: {e}. Initializing empty memory.")
            _memory = {}
    else:
        logger.info(f"Memory file {MEM_PATH} not found. Initializing empty memory.")
        _memory = {}

_load_memory()

def _make_board_id(grid: np.ndarray) -> str:
    H, W = grid.shape
    empty_count = int(np.sum(grid == -1))
    return f"{H}x{W}_e{empty_count}"

def get_legal_values(grid: np.ndarray) -> List[int]:
    ev = grid[grid != -1]
    mv = int(np.max(ev)) if ev.size > 0 else 1 # Default to 1 if no numbers other than -1
    return list(range(1, mv + 1))

def mem_score(grid: np.ndarray, r: int, c: int, v: int) -> float:
    bid = _make_board_id(grid)
    key = f"{bid}_{r}_{c}_{v}"
    entry = _memory.get(key)
    return (entry["total_score"] / entry["count"]) if entry and entry.get("count", 0) > 0 else 0.0

def update_memory(grid: np.ndarray, r: int, c: int, v: int, score: float):
    bid = _make_board_id(grid)
    key = f"{bid}_{r}_{c}_{v}"
    entry = _memory.setdefault(key, {"count":0, "total_score":0.0})
    entry["count"] += 1
    entry["total_score"] += score

def _save_memory():
    try:
        with open(MEM_PATH, "w", encoding="utf-8") as f:
            json.dump(_memory, f, indent=4)
        logger.info(f"Saved memory ({len(_memory)} entries).")
    except Exception as e:
        logger.error(f"Failed to save memory: {e}")

@app.on_event("shutdown")
def _on_shutdown():
    _save_memory()

# ── 4. CP-SAT 解算 ────────────────────────────────────────────
def build_and_solve_cp_vec(grid: np.ndarray, candidates: List[Tuple[int,int,int]], legal_values: List[int]): # Renamed _ to legal_values for clarity
    t_start_total = time.time()
    
    t0_ft = time.time()
    ft = build_feature_tensor(grid)
    t1_ft = time.time()
    
    tf_scores = calculate_scores_from_tensor(ft, grid)
    t2_score_calc = time.time()
    
    # Log the tf_scores as a table <--- 新增表格日誌
    logger.info(f"TensorFlow scores (tf_scores):\n{format_data_as_table(tf_scores, floatfmt='.3f', generate_default_headers_if_numpy_2d_and_no_headers=True)}")

    logger.info(f"Time - Feature Tensor build: {t1_ft - t0_ft:.4f}s, Score calculation: {t2_score_calc - t1_ft:.4f}s")

    if not candidates:
        logger.warning("No candidates provided for CP-SAT solver.")
        return []

    model = cp_model.CpModel()
    n_candidates = len(candidates)
    
    # Decision variable: index of the chosen candidate
    chosen_idx_var = model.NewIntVar(0, n_candidates - 1, "chosen_idx")
    
    # Variables to store properties of the chosen candidate
    # These are not strictly necessary as decision variables if only used for objective/logging,
    # but AddElement makes the model explicit.
    # chosen_r_var = model.NewIntVar(0, grid.shape[0] - 1, "chosen_r")
    # chosen_c_var = model.NewIntVar(0, grid.shape[1] - 1, "chosen_c")
    # max_val_in_grid = int(np.max(grid[grid != -1])) if np.any(grid != -1) else 1
    # chosen_v_var = model.NewIntVar(1, max_val_in_grid, "chosen_v")

    # Score factor for converting floats to integers
    SCORE_FACTOR = 10000 
    
    # Precompute scores for all candidates
    candidate_tf_scores_int = [int(tf_scores[r, c] * SCORE_FACTOR) for r, c, v_cand in candidates]
    candidate_total_scores_int = [
        int((tf_scores[r, c] + mem_score(grid, r, c, v_cand)) * SCORE_FACTOR)
        for r, c, v_cand in candidates
    ]

    # Objective variable: score of the chosen candidate
    # Ensure min_score and max_score are valid even if candidate_total_scores_int is empty
    min_total_score_int = min(candidate_total_scores_int) if candidate_total_scores_int else 0
    max_total_score_int = max(candidate_total_scores_int) if candidate_total_scores_int else 0
    
    objective_var = model.NewIntVar(min_total_score_int, max_total_score_int, "objective_score")
    model.AddElement(chosen_idx_var, candidate_total_scores_int, objective_var)
    
    # Store tf_score of the chosen candidate for logging/returning (optional in model)
    min_tf_score_int = min(candidate_tf_scores_int) if candidate_tf_scores_int else 0
    max_tf_score_int = max(candidate_tf_scores_int) if candidate_tf_scores_int else 0
    chosen_tf_score_var = model.NewIntVar(min_tf_score_int, max_tf_score_int, "chosen_tf_score")
    model.AddElement(chosen_idx_var, candidate_tf_scores_int, chosen_tf_score_var)

    model.Maximize(objective_var)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    solver.parameters.num_workers = os.cpu_count() or 1
    
    t_before_solve = time.time()
    status = solver.Solve(model)
    t_after_solve = time.time()
    
    logger.info(f"CP-SAT Solve time: {t_after_solve - t_before_solve:.4f}s")

    results = []
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        selected_candidate_index = solver.Value(chosen_idx_var)
        r_sol, c_sol, v_sol = candidates[selected_candidate_index]
        
        # Get scores from solver (could also re-calculate from selected_candidate_index)
        final_score_solved = solver.Value(objective_var) / SCORE_FACTOR
        tf_score_solved = solver.Value(chosen_tf_score_var) / SCORE_FACTOR
        
        results.append((r_sol, c_sol, v_sol, final_score_solved, tf_score_solved))
        logger.info(f"CP-SAT solution: Best candidate index {selected_candidate_index} -> ({r_sol},{c_sol},{v_sol}), TotalScore: {final_score_solved:.4f}, TF_Score: {tf_score_solved:.4f}")
    elif status == cp_model.INFEASIBLE:
        logger.warning("CP-SAT solver: Model is infeasible.")
    elif status == cp_model.MODEL_INVALID:
        logger.error("CP-SAT solver: Model is invalid.")
        logger.error(f"Model validation: {model.Validate()}")
    else:
        logger.warning(f"CP-SAT solver: No optimal/feasible solution found. Status: {solver.StatusName(status)}")

    logger.info(f"Total time for build_and_solve_cp_vec: {time.time() - t_start_total:.4f}s => {results}")
    return results

# ── 5. /analyze API ────────────────────────────────────────────
class ProposedValue(BaseModel):
    pos: List[int] # Expected [row, col]
    value: int

    @validator("pos")
    def _check_pos_length(cls, p):
        if len(p) != 2:
            raise ValueError("pos must contain two integers: [row, col]")
        return p

class AnalyzeRequest(BaseModel):
    new_card: List[List[int]]
    proposed_values: List[ProposedValue]

    @validator("new_card")
    def _check_rect(cls, g_list):
        if not g_list: # Allow empty grid? Or raise error? For now, assume non-empty is typical.
            # raise ValueError("new_card cannot be empty") # Or handle as needed
            return g_list # Or np.array([]) if that's preferred for empty.
        if any(len(r) != len(g_list[0]) for r in g_list):
            raise ValueError("new_card 必須是矩形 (all rows must have same length)")
        return g_list # Keep as list for now, convert to np.array in endpoint

    @validator("proposed_values", each_item=True)
    def _check_pv(cls, pv: ProposedValue, values): # Pydantic v2: values is a dict of validated fields
        # `values` contains already validated fields, e.g., `new_card` if it's listed before `proposed_values`
        # For simplicity, let's assume new_card might not be processed yet or rely on endpoint logic
        # This validator primarily checks pv's internal consistency for now.
        # Cross-field validation is better handled in the endpoint or a root_validator if complex.
        g_list = values.get("new_card")
        if g_list and pv: # pv is not None
            grid_np = np.array(g_list, dtype=int) # Convert for checks
            rows, cols = grid_np.shape
            r, c = pv.pos
            if not (0 <= r < rows and 0 <= c < cols):
                raise ValueError(f"Proposed position [{r},{c}] is out of bounds for grid {rows}x{cols}")

            valid_nums_in_grid = grid_np[grid_np != -1]
            mv = int(np.max(valid_nums_in_grid)) if valid_nums_in_grid.size > 0 else 1
            if not (1 <= pv.value <= mv):
                raise ValueError(f"Proposed value {pv.value} is out of range [1, {mv}] for the current grid")
        return pv

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    grid = np.array(req.new_card, dtype=int)
    
    # Log the input grid as a table <--- 新增表格日誌
    logger.info(f"Received grid ({grid.shape[0]}x{grid.shape[1]}):\n{format_data_as_table(grid, generate_default_headers_if_numpy_2d_and_no_headers=True)}")

    legal_game_values = get_legal_values(grid)
    if not legal_game_values: # Should not happen if get_legal_values defaults to [1] for empty/all -1 grid
        logger.warning("No legal game values determined for the grid.")
        # Decide how to handle, maybe it implies no valid moves.

    valid_candidates = []
    for pv in req.proposed_values:
        r, c = pv.pos[0], pv.pos[1]
        v = pv.value
        
        # Check if position is empty and value is legal for the game
        if grid[r, c] == -1 and v in legal_game_values:
            valid_candidates.append((r, c, v))
        else:
            logger.warning(f"Skipping invalid proposed value: pos [{r},{c}] (current: {grid[r,c]}), value {v}. Legal game values: {legal_game_values}")

    if not valid_candidates:
        # Consider if this is a client error or just no valid moves from proposed.
        # If it's expected that proposed_values might all be invalid, then HTTPException might be too strong.
        # For now, keeping it as an error if *no* valid candidates can be processed.
        raise HTTPException(status_code=400, detail="沒有合法候選 (No valid candidates after filtering proposed values)")

    # Pass legal_game_values to build_and_solve_cp_vec, though it might not use it directly if candidates are pre-filtered
    best_move_info = await run_in_threadpool(build_and_solve_cp_vec, grid, valid_candidates, legal_game_values)
    
    if not best_move_info: # If solver returns empty list
        return {"status": "fail", "result": None, "message": "Solver did not find a solution or no valid candidates."}

    # Assuming best_move_info contains [(r, c, v, total_score, tf_score)]
    r_best, c_best, v_best, final_total_score, final_tf_score = best_move_info[0]
    
    # Update memory with the chosen move and its evaluated total score
    update_memory(grid, r_best, c_best, v_best, final_total_score)
    # _save_memory() # Consider if memory should be saved after every update or just on shutdown

    return {
        "status": "success",
        "result": {
            "pos": [r_best, c_best],
            "value": v_best,
            "score": round(final_total_score, 4),
            "tensor_flow_score": round(final_tf_score, 4)
        }
    }

# Example of running the server (for local testing)
if __name__ == "__main__":
    import uvicorn
    # Make sure to save this file as, e.g., main.py
    # Then run: uvicorn main:app --reload
    logger.info("Starting Uvicorn server for local testing: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)

