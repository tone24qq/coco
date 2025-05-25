import json
import os
import time
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks # <--- 載入 BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, validator
from typing import List, Dict, Tuple, Callable, Any
import numpy as np
from ortools.sat.python import cp_model
from tabulate import tabulate

# ── Logging Configuration ───────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Table Formatting Utility ────────────────────────────────────────
def format_data_as_table(data_to_format: Any, headers_option: Any = None, tablefmt: str = "grid", floatfmt: str = ".2f", generate_default_headers_if_numpy_2d_and_no_headers: bool = False) -> str:
    """
    Formats given data (list of lists or NumPy array) into a table string.
    (詳細註解同前一版本)
    """
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

    # --- 優化後的空資料判斷 ---
    if not final_data or (isinstance(final_data, list) and all(not row for row in final_data)):
        return "No data to format."
    # --- --- --- --- --- ---

    if current_headers is None:
        actual_tabulate_headers = []
    else:
        actual_tabulate_headers = current_headers

    try:
        return tabulate(final_data, headers=actual_tabulate_headers, tablefmt=tablefmt, floatfmt=floatfmt if floatfmt else None)
    except Exception as e:
        logger.error(f"Error during table formatting: {e}", exc_info=True)
        return f"Error formatting table: {str(e)}"

app = FastAPI(title="Plug-in權重 + 張量流 + 多強化 (背景存檔優化版)", version="4.2")

# ── 1. 原本的向量化模組函數們 ────────────────────────────────────
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
MODULE_WEIGHTS: Dict[str, float] = {
    "A6": 1.0,
    "B1": 0.5,
    "C2": 0.8,
}

def tensor_flow_score_vec_all(grid: np.ndarray) -> np.ndarray:
    total_score_map = np.zeros(grid.shape, dtype=float)
    for name, func in MODULE_FUNCS_VEC.items():
        total_score_map += func(grid).astype(float) * MODULE_WEIGHTS[name]
    return total_score_map

# ── 2. 增強版特徵張量 ────────────────────────────────────────────
def build_feature_tensor(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    valid_values = grid[grid != -1]
    maxv = int(np.max(valid_values)) if valid_values.size > 0 else 1
    C = 4 + maxv
    tensor = np.zeros((H, W, C), dtype=float)
    for r in range(H):
        for c in range(W):
            val = grid[r, c]
            tensor[r, c, 0] = (float(val) / maxv) if val != -1 else 0.0
            tensor[r, c, 1] = 1.0 if val == -1 else 0.0
            tensor[r, c, 2] = float(r) / (H - 1) if H > 1 else 0.0
            tensor[r, c, 3] = float(c) / (W - 1) if W > 1 else 0.0
            if val != -1:
                if 1 <= val <= maxv:
                    tensor[r, c, 4 + int(val) - 1] = 1.0
    return tensor

def calculate_scores_from_tensor(feature_tensor: np.ndarray, grid: np.ndarray) -> np.ndarray:
    num_channels = feature_tensor.shape[-1]
    weights = np.ones(num_channels, dtype=float)
    return np.tensordot(feature_tensor, weights, axes=([2], [0]))

# ── 3. 轻量记忆模块 ────────────────────────────────────────────
MEM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_cards.json")
_memory: Dict[str, Dict[str, Any]] = {}

def _load_memory() -> None:
    global _memory
    if os.path.exists(MEM_PATH):
        try:
            with open(MEM_PATH, "r", encoding="utf-8") as f:
                _memory = json.load(f)
            logger.info(f"Loaded memory ({len(_memory)} entries) from {MEM_PATH}.")
        except json.JSONDecodeError as jde:
            logger.error(f"Error decoding JSON from {MEM_PATH}: {jde}. Initializing empty memory.", exc_info=True)
            _memory = {}
        except Exception as e:
            logger.error(f"Failed to load memory from {MEM_PATH}: {e}. Initializing empty memory.", exc_info=True)
            _memory = {}
    else:
        logger.info(f"Memory file {MEM_PATH} not found. Initializing empty memory.")
        _memory = {}

_load_memory()

def _make_board_id(grid: np.ndarray) -> str:
    H, W = grid.shape
    empty_count = int(np.sum(grid == -1))
    return f"{H}x{W}_empty{empty_count}"

def get_legal_values(grid: np.ndarray) -> List[int]:
    valid_numbers = grid[grid != -1]
    if valid_numbers.size == 0:
        return [1]
    max_val = int(np.max(valid_numbers))
    return list(range(1, max_val + 1))

def update_memory(grid: np.ndarray, r: int, c: int, v: int, score: float) -> None:
    board_id = _make_board_id(grid)
    action_key = f"{r}_{c}_{v}" # More descriptive key part

    if board_id not in _memory:
        _memory[board_id] = {}

    entry = _memory[board_id].setdefault(action_key, {"count": 0, "total_score": 0.0})

    entry["count"] = entry.get("count",0) + 1
    entry["total_score"] = entry.get("total_score", 0.0) + score
    # logger.debug(f"Updated memory for {board_id} - {action_key}: {entry}")

def _save_memory() -> None: # _save_memory 本身不需改變
    try:
        with open(MEM_PATH, "w", encoding="utf-8") as f:
            json.dump(_memory, f, indent=4, sort_keys=True)
        logger.info(f"Saved memory ({len(_memory)} board states) to {MEM_PATH}.")
    except Exception as e:
        logger.error(f"Failed to save memory to {MEM_PATH}: {e}", exc_info=True)

@app.on_event("shutdown")
async def on_shutdown_event():
    logger.info("Application shutting down. Performing final save of memory...")
    _save_memory() # 保留 shutdown 時的最終存檔

# --- 新增 mem_score 函數 ---
def mem_score(grid: np.ndarray, r: int, c: int, v: int) -> float:
    """
    Retrieves the average score for a given action (r, c, v) on a specific board state from memory.
    Returns 0.0 if the board state or action is not found, or if the action has not been performed.
    """
    board_id = _make_board_id(grid)
    action_key = f"{r}_{c}_{v}"

    if board_id in _memory and action_key in _memory[board_id]:
        entry = _memory[board_id][action_key]
        if "count" in entry and entry["count"] > 0 and "total_score" in entry:
            # Return average score
            return entry["total_score"] / entry["count"]
    return 0.0 # Return 0 if no memory or action not found or count is zero

# ── 4. CP-SAT 解算 ────────────────────────────────────────────
def build_and_solve_cp_vec(grid: np.ndarray, candidates: List[Tuple[int,int,int]], _: List[int]):
    t_start_total = time.time()
    CP_SOLVER_TIME_LIMIT_SECONDS = 5.0
    SCORE_NORMALIZATION_FACTOR = 10000
    t0_ft = time.time()
    feature_tensor = build_feature_tensor(grid)
    t1_ft = time.time()
    tf_scores = calculate_scores_from_tensor(feature_tensor, grid)
    t2_score_calc = time.time()
    logger.info(f"TensorFlow scores (tf_scores) for grid {grid.shape}:\n{format_data_as_table(tf_scores, floatfmt='.3f', generate_default_headers_if_numpy_2d_and_no_headers=True)}")
    logger.info(f"Time - Feature Tensor build: {t1_ft - t0_ft:.4f}s, TF Score calculation: {t2_score_calc - t1_ft:.4f}s")
    if not candidates:
        logger.warning("No candidates provided for CP-SAT solver. Returning empty result.")
        return []
    model = cp_model.CpModel()
    num_candidates = len(candidates)
    chosen_idx_var = model.NewIntVar(0, num_candidates - 1, "chosen_idx")
    candidate_tf_scores_int = [int(tf_scores[r, c] * SCORE_NORMALIZATION_FACTOR) for r, c, v_cand in candidates]

    # --- 修改此處以包含 mem_score ---
    candidate_total_scores_int = [
        int((tf_scores[r_cand, c_cand] + mem_score(grid, r_cand, c_cand, v_cand)) * SCORE_NORMALIZATION_FACTOR)
        for r_cand, c_cand, v_cand in candidates
    ]
    # --- --- --- --- --- --- ---

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
    solver.parameters.log_search_progress = False
    t_before_solve = time.time()
    status = solver.Solve(model)
    t_after_solve = time.time()
    logger.info(f"CP-SAT Solve time: {t_after_solve - t_before_solve:.4f}s. Status: {solver.StatusName(status)}")
    results = []
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        selected_idx = solver.Value(chosen_idx_var)
        r_sol, c_sol, v_sol = candidates[selected_idx]
        final_score = solver.Value(objective_var) / SCORE_NORMALIZATION_FACTOR
        tf_score_of_chosen = solver.Value(chosen_tf_score_var) / SCORE_NORMALIZATION_FACTOR
        results.append((r_sol, c_sol, v_sol, final_score, tf_score_of_chosen))
        logger.info(f"CP-SAT solution: Best candidate index {selected_idx} -> ({r_sol},{c_sol},{v_sol}), TotalScore: {final_score:.4f}, TF_Score: {tf_score_of_chosen:.4f}")
    else:
        if status == cp_model.MODEL_INVALID: logger.error(f"CP-SAT Model Invalid. Validation: {model.Validate()}")
        elif status == cp_model.INFEASIBLE: logger.warning("CP-SAT Model Infeasible: No solution satisfies all constraints.")
        else: logger.warning(f"CP-SAT solver did not find an optimal/feasible solution. Status: {solver.StatusName(status)}")
    logger.info(f"Total time for build_and_solve_cp_vec: {time.time() - t_start_total:.4f}s. Found {len(results)} solution(s).")
    return results

# ── 5. /analyze API ────────────────────────────────────────────
class ProposedValue(BaseModel):
    pos: List[int]
    value: int

    @validator("pos")
    def _check_pos_length(cls, p_val: List[int]) -> List[int]:
        if len(p_val) != 2:
            raise ValueError("pos must contain exactly two integers: [row, col]")
        if not all(isinstance(x, int) for x in p_val):
            raise ValueError("pos elements must be integers")
        return p_val

class AnalyzeRequest(BaseModel):
    new_card: List[List[int]]
    proposed_values: List[ProposedValue]

    @validator("new_card")
    def _check_grid_is_valid_rectangle(cls, grid_list: List[List[int]]) -> List[List[int]]:
        if not grid_list:
            raise ValueError("new_card cannot be empty")
        if not isinstance(grid_list, list) or not all(isinstance(row, list) for row in grid_list):
            raise ValueError("new_card must be a list of lists")
        if not grid_list[0]:
             raise ValueError("new_card rows cannot be empty; grid must have columns")
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
                    raise ValueError(f"Proposed position [{r},{c}] is out of bounds for grid {rows}x{cols}")
                legal_game_vals = get_legal_values(grid_np)
                if grid_np[r,c] != -1:
                     raise ValueError(f"Proposed position [{r},{c}] is not empty (current value: {grid_np[r,c]}).")
                if pv.value not in legal_game_vals:
                     raise ValueError(f"Proposed value {pv.value} is not a legal value {legal_game_vals} for the current grid state.")
            except Exception as e:
                logger.error(f"Error during proposed_values validation with grid: {e}")
                raise ValueError(f"Invalid grid data encountered while validating proposed_values: {e}") from e
        return pv

@app.post("/analyze")
async def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    try:
        grid = np.array(req.new_card, dtype=int)
    except ValueError as ve:
        logger.error(f"Error converting new_card to NumPy array: {ve}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Invalid new_card format: {ve}")

    logger.info(f"Received grid ({grid.shape[0]}x{grid.shape[1]}):\n{format_data_as_table(grid, generate_default_headers_if_numpy_2d_and_no_headers=True)}")

    current_legal_game_values = get_legal_values(grid)
    logger.info(f"Legal game values for the grid: {current_legal_game_values}")

    valid_candidates = []
    for pv_item in req.proposed_values:
        r, c = pv_item.pos[0], pv_item.pos[1]
        v = pv_item.value
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            if grid[r, c] == -1 and v in current_legal_game_values:
                valid_candidates.append((r, c, v))
            else:
                logger.warning(f"Skipping proposed value (pos [{r},{c}] (val: {grid[r,c]}), value {v}). Cell not empty or value not in legal game values {current_legal_game_values}.")
        else:
            logger.warning(f"Skipping proposed value due to out-of-bounds position: pos [{r},{c}] for grid {grid.shape}.")


    if not valid_candidates:
        logger.warning("No valid candidates derived from proposed_values for CP-SAT solver.")
        return {"status": "no_valid_candidates", "result": None, "message": "No valid candidates to process after filtering."}

    best_move_info_list = await run_in_threadpool(build_and_solve_cp_vec, grid, valid_candidates, current_legal_game_values)

    if not best_move_info_list:
        logger.warning("CP-SAT Solver returned no solution.")
        return {"status": "solver_fail", "result": None, "message": "Solver did not find a solution."}

    r_best, c_best, v_best, final_total_score, final_tf_score = best_move_info_list[0]

    update_memory(grid, r_best, c_best, v_best, final_total_score)
    background_tasks.add_task(_save_memory)

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
    logger.info("Starting Uvicorn server for local testing: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
