# ------------------- dependencies -------------------
# pip install fastapi uvicorn ortools tabulate numpy

import json
import os
import time
import logging
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, validator, Field
from typing import List, Dict, Tuple, Callable, Any, Optional
import numpy as np
from ortools.sat.python import cp_model
from tabulate import tabulate

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)

# --- File paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEM_PATH = os.path.join(BASE_DIR, "memory_cards.json")
REASONING_LOG_PATH = os.path.join(BASE_DIR, "reasoning_log.jsonl")
MODULE_WEIGHTS_PATH = os.path.join(BASE_DIR, "module_weights.json")

# --- Table formatting utility ---
def format_data_as_table(
    data_to_format: Any,
    headers_option: Any = None,
    tablefmt: str = "grid",
    floatfmt: str = ".2f",
    generate_default_headers_if_numpy_2d_and_no_headers: bool = False
) -> str:
    headers = headers_option
    if isinstance(data_to_format, np.ndarray):
        data = data_to_format.tolist()
        if generate_default_headers_if_numpy_2d_and_no_headers and headers in (None, []) and data_to_format.ndim == 2:
            cols = data_to_format.shape[1]
            headers = [f"Col {i+1}" for i in range(cols)]
    elif isinstance(data_to_format, list):
        data = data_to_format
    else:
        logger.warning(f"Unsupported data type for table formatting: {type(data_to_format)}")
        return "Unsupported data type for table formatting."
    if not data or (isinstance(data, list) and all(not row for row in data)):
        return "No data to format."
    actual_headers = headers if headers is not None else []
    try:
        return tabulate(data, headers=actual_headers, tablefmt=tablefmt, floatfmt=floatfmt)
    except Exception as e:
        logger.error(f"Error during table formatting: {e}", exc_info=True)
        return f"Error formatting table: {e}"

app = FastAPI(
    title="MetaCognitive Scratch Card Solver (Combined v1+v2)",
    version="1.0"
)

# -----------------------------------------------------------------------------
# 1. Memory module (from version1)
# -----------------------------------------------------------------------------
_memory: Dict[str, Dict[str, Any]] = {}

def _make_board_id(grid: np.ndarray) -> str:
    H, W = grid.shape
    empty = int(np.sum(grid == -1))
    total = int(np.sum(grid[grid != -1]))
    return f"{H}x{W}_empty{empty}_sum{total}"

def _load_memory() -> None:
    global _memory
    if os.path.exists(MEM_PATH):
        try:
            with open(MEM_PATH, "r", encoding="utf-8") as f:
                _memory = json.load(f)
            logger.info(f"Loaded memory ({len(_memory)}) from {MEM_PATH}")
        except Exception as e:
            logger.error(f"Failed to load memory: {e}", exc_info=True)
            _memory = {}
    else:
        _memory = {}
        logger.info("No memory file found; starting fresh.")

def _save_memory() -> None:
    try:
        with open(MEM_PATH, "w", encoding="utf-8") as f:
            json.dump(_memory, f, indent=2, sort_keys=True)
        logger.info(f"Saved memory ({len(_memory)}) to {MEM_PATH}")
    except Exception as e:
        logger.error(f"Failed to save memory: {e}", exc_info=True)

def update_memory(grid: np.ndarray, r: int, c: int, v: int, score: float) -> None:
    bid = _make_board_id(grid)
    key = f"{r}_{c}_{v}"
    if bid not in _memory:
        _memory[bid] = {}
    entry = _memory[bid].setdefault(key, {"count": 0, "total_score": 0.0})
    entry["count"] += 1
    entry["total_score"] += score

def mem_score(grid: np.ndarray, r: int, c: int, v: int) -> float:
    bid = _make_board_id(grid)
    key = f"{r}_{c}_{v}"
    if bid in _memory and key in _memory[bid]:
        ent = _memory[bid][key]
        if ent["count"] > 0:
            return ent["total_score"] / ent["count"]
    return 0.0

_load_memory()

# -----------------------------------------------------------------------------
# 2. Meta-cognition log (from version1)
# -----------------------------------------------------------------------------
class MetaCognitionLog:
    def __init__(self, path: str):
        self.path = path
        self.buffer: List[Dict[str, Any]] = []

    def log_event(self, event: Dict[str, Any]):
        event["log_id"] = str(uuid.uuid4())
        event["timestamp"] = time.time()
        # sanitize
        for k, v in list(event.items()):
            if isinstance(v, np.integer):
                event[k] = int(v)
            elif isinstance(v, np.floating):
                event[k] = float(v)
            elif isinstance(v, np.ndarray):
                event[k] = v.tolist()
            elif isinstance(v, tuple):
                event[k] = list(v)
        self.buffer.append(event)

    def flush(self):
        if not self.buffer:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                for ev in self.buffer:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            logger.info(f"Flushed {len(self.buffer)} events to {self.path}")
            self.buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush log: {e}", exc_info=True)

meta_logger = MetaCognitionLog(REASONING_LOG_PATH)

# -----------------------------------------------------------------------------
# 3. Module weights management (from version1)
# -----------------------------------------------------------------------------
MODULE_WEIGHTS: Dict[str, float] = {}

def _load_module_weights() -> None:
    global MODULE_WEIGHTS
    defaults = {
        "A2": 0.7, "A5": 0.8, "A6": 0.6, "A8": 0.5,
        "M1": 0.6, "M2": 0.8, "M3": 0.9, "M4": 0.5, "M5": 0.5,
        "M6": 0.5, "M7": 0.5, "M8": 0.5, "M9": 0.6, "M10": 0.5,
        "M11": 0.5, "F2": 0.5, "F3": 0.5, "R2": 0.5, "R7": 0.5,
        "D3": 0.7,
    }
    if os.path.exists(MODULE_WEIGHTS_PATH):
        try:
            with open(MODULE_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            MODULE_WEIGHTS = {**defaults, **loaded}
            logger.info(f"Loaded module weights from {MODULE_WEIGHTS_PATH}")
        except Exception as e:
            logger.error(f"Error loading weights: {e}", exc_info=True)
            MODULE_WEIGHTS = defaults
    else:
        MODULE_WEIGHTS = defaults
        _save_module_weights()

def _save_module_weights() -> None:
    try:
        with open(MODULE_WEIGHTS_PATH, "w", encoding="utf-8") as f:
            json.dump(MODULE_WEIGHTS, f, indent=2, sort_keys=True)
        logger.info(f"Saved module weights to {MODULE_WEIGHTS_PATH}")
    except Exception as e:
        logger.error(f"Failed to save module weights: {e}", exc_info=True)

_load_module_weights()

# -----------------------------------------------------------------------------
# 4. Heuristic functions (A/M/F/R/D series, vectorized where possible)
# -----------------------------------------------------------------------------
def a2_center_radial_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    center = np.array([(H-1)/2, (W-1)/2])
    dist = np.sqrt((np.arange(H)[:,None]-center[0])**2 + (np.arange(W)-center[1])**2)
    norm = np.max(dist) or 1
    score = 1 - dist/norm
    return score * (grid==-1)

def a5_adj_density_vec(grid: np.ndarray) -> np.ndarray:
    padded = np.pad(grid!=-1, ((1,1),(1,1)), 'constant')
    dens = (
        padded[:-2,1:-1] + padded[2:,1:-1] +
        padded[1:-1,:-2] + padded[1:-1,2:]
    ) / 4.0
    return dens * (grid==-1)

def a6_fixed_position_vec(grid: np.ndarray) -> np.ndarray:
    return (grid==-1).astype(float)

def a8_symmetry_vec(grid: np.ndarray) -> np.ndarray:
    flip = np.fliplr(grid)
    return ((grid==flip).astype(float)) * (grid==-1)

def m1_uni_gap_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        filled = np.where(grid[i]!=-1)[0]
        if len(filled)>1:
            gaps = np.diff(filled)
            s = 1 - np.std(gaps)/(W or 1)
            score[i,:] = max(0,s)
    return score * (grid==-1)

def m2_seq_pattern_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        vals = np.sort(grid[i][grid[i]!=-1])
        if len(vals)>2:
            d = np.diff(vals)
            score[i,:] += 1.0/(1+np.std(d))
    for j in range(W):
        vals = np.sort(grid[:,j][grid[:,j]!=-1])
        if len(vals)>2:
            d = np.diff(vals)
            score[:,j] += 1.0/(1+np.std(d))
    return score * (grid==-1)

def m3_diff_band_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        for j in range(W):
            if grid[i,j]==-1:
                vals=[]
                for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni,nj=i+di,j+dj
                    if 0<=ni<H and 0<=nj<W and grid[ni,nj]!=-1:
                        vals.append(abs(grid[ni,nj]))
                if vals:
                    m = np.mean(vals)
                    score[i,j] = 1.0 if 5<=m<=20 else 0.3
    return score * (grid==-1)

def m4_biaxial_stat_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    row_d = np.sum(grid!=-1,axis=1)/(W or 1)
    col_d = np.sum(grid!=-1,axis=0)/(H or 1)
    mask = (grid==-1)
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        for j in range(W):
            if mask[i,j] and 0.5<row_d[i]<0.8 and 0.5<col_d[j]<0.8:
                score[i,j]=1.0
    return score

def m5_bar_focus_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    mask = (grid==-1)
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        if np.sum(grid[i]!=-1)>W//2:
            score[i,:]+=1
    for j in range(W):
        if np.sum(grid[:,j]!=-1)>H//2:
            score[:,j]+=1
    return score*mask

def m6_neighbor_cycle_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        for j in range(W):
            if grid[i,j]==-1:
                cnt=0; tot=0
                for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni,nj=i+di,j+dj
                    if 0<=ni<H and 0<=nj<W:
                        tot+=1
                        if grid[ni,nj]!=-1: cnt+=1
                score[i,j]=cnt/(tot or 1)
    return score*(grid==-1)

def m7_bisec_zone_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    rows = np.array([np.sum(grid[i]!=-1) for i in range(H)])
    cols = np.array([np.sum(grid[:,j]!=-1) for j in range(W)])
    metric = (np.std(rows)+np.std(cols))/(H+W or 1)
    s = 1.0/(1+metric)
    return np.full_like(grid, s, dtype=float)*(grid==-1)

def m8_repeat_gap_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        filled = np.where(grid[i]!=-1)[0]
        if len(filled)>2:
            gaps=np.diff(filled)
            s = 1 - np.std(gaps)/(np.mean(gaps) or 1)
            score[i,:]=max(0,s)
    return score*(grid==-1)

def m9_double_rule_overlap_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        for j in range(W):
            if grid[i,j]==-1:
                if j>0 and grid[i,j-1]!=-1: score[i,j]+=0.5
                if j<W-1 and grid[i,j+1]!=-1: score[i,j]+=0.5
    return score*(grid==-1)

def m10_seq_order_match_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        vals = grid[i][grid[i]!=-1]
        if len(vals)>1 and (np.all(np.diff(vals)>0) or np.all(np.diff(vals)<0)):
            score[i,:]=1.0
    return score*(grid==-1)

def m11_block_match_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    bs = max(min(H,W)//4, 2)
    score = np.zeros_like(grid, dtype=float)
    for i in range(0,H-bs+1,bs):
        for j in range(0,W-bs+1,bs):
            block = grid[i:i+bs,j:j+bs]
            den = np.sum(block!=-1)/(block.size or 1)
            score[i:i+bs,j:j+bs] = den
    return score*(grid==-1)

def f2_row_rotate_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(1,H):
        if grid[i-1,W-1]!=-1 and grid[i,0]==-1:
            score[i,0]=1.0
    return score*(grid==-1)

def f3_col_rotate_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for j in range(1,W):
        if grid[H-1,j-1]!=-1 and grid[0,j]==-1:
            score[0,j]=1.0
    return score*(grid==-1)

def r2_rev_diff_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        vals=grid[i][grid[i]!=-1]
        if len(vals)>1 and np.all(np.diff(vals)<0):
            score[i,:]=1.0
    return score*(grid==-1)

def r7_odd_even_dist_vec(grid: np.ndarray) -> np.ndarray:
    filled = grid[grid!=-1]
    if filled.size>0:
        odd = np.sum(filled%2==1)
        even = np.sum(filled%2==0)
        ratio = abs(odd-even)/(odd+even)
        s = 1.0-ratio if ratio<0.3 else 0.1
    else:
        s = 0.5
    return np.full_like(grid, s, dtype=float)*(grid==-1)

def d3_pair_freq_vec(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(H):
        for j in range(W-1):
            if grid[i,j]==-1:
                if grid[i,j+1]!=-1 and abs(grid[i,j+1]) in [1,9,10]:
                    score[i,j]=1.0
    return score*(grid==-1)

MODULE_FUNCS_VEC: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    **{f"A{idx}": fn for idx, fn in zip([2,5,6,8],
        [a2_center_radial_vec, a5_adj_density_vec, a6_fixed_position_vec, a8_symmetry_vec])},
    **{f"M{idx}": fn for idx, fn in zip(range(1,12),
        [m1_uni_gap_vec, m2_seq_pattern_vec, m3_diff_band_vec, m4_biaxial_stat_vec,
         m5_bar_focus_vec, m6_neighbor_cycle_vec, m7_bisec_zone_vec, m8_repeat_gap_vec,
         m9_double_rule_overlap_vec, m10_seq_order_match_vec, m11_block_match_vec])},
    **{f"F{idx}": fn for idx, fn in zip([2,3],[f2_row_rotate_vec,f3_col_rotate_vec])},
    **{f"R{idx}": fn for idx in [2,7] for fn in ([r2_rev_diff_vec] if idx==2 else [r7_odd_even_dist_vec])},
    "D3": d3_pair_freq_vec
}

# -----------------------------------------------------------------------------
# 5. Combined score function
# -----------------------------------------------------------------------------
def tensor_flow_score_vec_all(grid: np.ndarray) -> np.ndarray:
    total = np.zeros_like(grid, dtype=float)
    for name, fn in MODULE_FUNCS_VEC.items():
        w = MODULE_WEIGHTS.get(name, 0.0)
        if w and grid.ndim==2 and fn in MODULE_FUNCS_VEC.values():
            try:
                score_map = fn(grid.copy())
                if score_map.shape == grid.shape:
                    total += score_map * w
                else:
                    logger.error(f"Heuristic {name} returned shape {score_map.shape}, expected {grid.shape}")
            except Exception as e:
                logger.error(f"Error in heuristic {name}: {e}", exc_info=True)
    return total

# -----------------------------------------------------------------------------
# 6. Pydantic models & CP-SAT solve step (from version2, with memory & log)
# -----------------------------------------------------------------------------
class GridInput(BaseModel):
    grid: List[List[int]] = Field(..., description="Current grid, -1 for empty")
    num_to_place: int = Field(1, gt=0, description="How many cells to fill")
    value_domain_min: int = Field(1, description="Min value")
    value_domain_max: int = Field(20, description="Max value")

    @validator("grid")
    def check_grid(cls, v):
        if not v or not all(isinstance(row, list) for row in v):
            raise ValueError("Grid must be non-empty list of lists")
        length = len(v[0])
        if any(len(r)!=length for r in v):
            raise ValueError("Grid must be rectangular")
        return v

    @validator("value_domain_max")
    def check_domain(cls, vmax, values):
        vmin = values.get("value_domain_min", None)
        if vmin is not None and vmax < vmin:
            raise ValueError("value_domain_max must >= value_domain_min")
        return vmax

class SolveStepResponse(BaseModel):
    new_grid: List[List[int]]
    chosen_cells: List[Tuple[int,int,int]]
    solver_log: str
    status: str
    computed_scores_table: Optional[str] = None
    log_id: Optional[str] = None

@app.post("/solve_step", response_model=SolveStepResponse)
async def solve_step_endpoint(grid_input: GridInput, background_tasks: BackgroundTasks):
    grid_np = np.array(grid_input.grid)
    H, W = grid_np.shape
    empties = list(zip(*np.where(grid_np==-1)))
    n = grid_input.num_to_place
    if not empties:
        return SolveStepResponse(
            new_grid=grid_np.tolist(),
            chosen_cells=[],
            solver_log="No empty cells",
            status="NO_EMPTY_CELLS"
        )
    if n>len(empties):
        n = len(empties)

    # compute heuristic scores
    score_map = await run_in_threadpool(tensor_flow_score_vec_all, grid_np.copy())
    raw_scores = [score_map[r,c] for r,c in empties]
    scale = 1000.0
    scaled = [int(s*scale) for s in raw_scores]
    # prepare table
    table_data = [[r,c,raw_scores[i],scaled[i]] for i,(r,c) in enumerate(empties)]
    scores_table = format_data_as_table(
        table_data,
        headers_option=["Row","Col","Raw","Scaled"],
        tablefmt="pipe"
    )

    # build CP-SAT model
    model = cp_model.CpModel()
    idx_vars = [model.NewIntVar(0, len(empties)-1, f"idx{i}") for i in range(n)]
    val_vars = [model.NewIntVar(grid_input.value_domain_min, grid_input.value_domain_max, f"val{i}") for i in range(n)]
    if n>1:
        model.AddAllDifferent(idx_vars)
        model.AddAllDifferent(val_vars)

    terms = []
    for i in range(n):
        term = model.NewIntVar(min(scaled), max(scaled), f"term{i}")
        model.AddElement(idx_vars[i], scaled, term)
        terms.append(term)
    if terms:
        model.Maximize(sum(terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    new_grid = grid_np.copy()
    chosen = []
    log = f"Status: {solver.StatusName(status)}\n"
    if terms:
        obj = solver.ObjectiveValue()
        log += f"Objective (scaled): {obj}, raw sum: {obj/scale}\n"
    log += f"Scores table:\n{scores_table}\n"

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(n):
            ei = solver.Value(idx_vars[i])
            r,c = empties[ei]
            v = solver.Value(val_vars[i])
            new_grid[r,c] = v
            chosen.append((int(r),int(c),int(v)))
            # update memory
            update_memory(grid_np, r, c, v, raw_scores[ei])
        # log event
        event = {
            "grid_id": _make_board_id(grid_np),
            "chosen_cells": chosen,
            "status": solver.StatusName(status),
            "scores": raw_scores,
            "weights_snapshot": MODULE_WEIGHTS.copy()
        }
        meta_logger.log_event(event)
        background_tasks.add_task(_save_memory)
        background_tasks.add_task(meta_logger.flush)
    else:
        log += "No solution or infeasible.\n"

    return SolveStepResponse(
        new_grid=new_grid.tolist(),
        chosen_cells=chosen,
        solver_log=log,
        status=solver.StatusName(status),
        computed_scores_table=scores_table,
        log_id=event.get("log_id")
    )

@app.post("/analyze_scores")
async def analyze_scores_endpoint(grid_input: GridInput):
    grid_np = np.array(grid_input.grid)
    score_map = await run_in_threadpool(tensor_flow_score_vec_all, grid_np.copy())
    empties = list(zip(*np.where(grid_np==-1)))
    data = [[r,c, score_map[r,c]] for r,c in empties]
    table = format_data_as_table(data, headers_option=["Row","Col","Score"], tablefmt="pipe")
    return {
        "message": "Scores computed",
        "scores_table": table,
        "raw_score_map": score_map.tolist()
    }

class FeedbackRequest(BaseModel):
    log_id: str
    is_correct: bool
    notes: Optional[str] = None

@app.post("/feedback")
async def feedback_endpoint(req: FeedbackRequest, background_tasks: BackgroundTasks):
    # Conceptual placeholder: we simply log the feedback
    fb = {
        "feedback_for": req.log_id,
        "correct": req.is_correct,
        "notes": req.notes
    }
    meta_logger.log_event(fb)
    background_tasks.add_task(meta_logger.flush)
    return {"status": "feedback recorded", "log_id": req.log_id}

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down: saving memory, logs, weights")
    _save_memory()
    meta_logger.flush()
    _save_module_weights()

if __name__ == "__main__":
    import uvicorn
    logger.info("Running local server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)