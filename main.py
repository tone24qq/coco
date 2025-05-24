import json
import os
import time
import logging
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, validator
from typing import List, Dict, Tuple, Callable # Removed Set as it wasn't explicitly used
import numpy as np
from ortools.sat.python import cp_model
# Removed: from celery.result import AsyncResult
# Removed: from celery_worker import solve_task

# ── Logging Configuration ───────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Plug-in權重 + 張量流 + 多強化", version="4.0")

# ── 1. 原本的向量化模組函數們（保留舊有規則） ────────────────────
def a6_fixed_position_vec(grid: np.ndarray) -> np.ndarray:
    """
    示例：返回 grid 中 -1 的位置的布林遮罩。
    """
    return grid == -1

def b1_row_feature_vec(grid: np.ndarray) -> np.ndarray:
    """
    示例：基於行中非 -1 數值的數量來生成特徵。
    """
    feature_map = np.zeros_like(grid, dtype=float)
    for r_idx in range(grid.shape[0]):
        non_neg_count = np.sum(grid[r_idx, :] != -1)
        feature_map[r_idx, :] = non_neg_count
    return feature_map

def c2_col_feature_vec(grid: np.ndarray) -> np.ndarray:
    """
    示例：基於列中非 -1 數值的數量來生成特徵。
    """
    feature_map = np.zeros_like(grid, dtype=float)
    for c_idx in range(grid.shape[1]):
        non_neg_count = np.sum(grid[:, c_idx] != -1)
        feature_map[:, c_idx] = non_neg_count
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
    """
    計算所有模組函數加權後的總分數網格。
    """
    total_score_val = np.zeros(grid.shape, dtype=float)
    for name, func in MODULE_FUNCS_VEC.items():
        mask = func(grid)
        total_score_val += mask.astype(float) * MODULE_WEIGHTS.get(name, 1.0)
    return total_score_val

# ── 2. 增強版特徵張量（集中於 main.py） ────────────────────────────
def build_feature_tensor(grid: np.ndarray) -> np.ndarray:
    """
    增強版特徵張量示例，通道說明：
    - 0: 標準化數值 (val/max_val)
    - 1: 是否空格 (1/0)
    - 2: row position normalized
    - 3: col position normalized
    - 4~(4+max_val-1): one-hot 值通道 (針對 1 到 max_val 的數值)
    """
    H, W = grid.shape
    max_val_tensor = int(np.max(grid[grid != -1])) if np.any(grid != -1) else 1
    C_tensor = 4 + max_val_tensor
    tensor = np.zeros((H, W, C_tensor), dtype=float)

    for r_idx in range(H):
        for c_idx in range(W):
            val = grid[r_idx, c_idx]
            tensor[r_idx, c_idx, 0] = (val / max_val_tensor) if val != -1 else 0.0
            tensor[r_idx, c_idx, 1] = 1.0 if val == -1 else 0.0
            tensor[r_idx, c_idx, 2] = r_idx / (H - 1) if H > 1 else 0.0
            tensor[r_idx, c_idx, 3] = c_idx / (W - 1) if W > 1 else 0.0
            if val != -1 and 1 <= val <= max_val_tensor:
                tensor[r_idx, c_idx, 4 + val - 1] = 1.0
    return tensor

def calculate_scores_from_tensor(feature_tensor: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    計算分數網格，示例：各通道等權重加總。
    """
    weights = np.ones(feature_tensor.shape[-1], dtype=float)
    return np.tensordot(feature_tensor, weights, axes=([2], [0]))

# ── 3. 記憶模組 & 其他輔助函數 ────────────────────────────────
MEM_PATH = os.path.join(os.path.dirname(__file__), "memory_cards.json")
_memory_data: Dict[str, Dict[str, float]] = {}

def _load_memory():
    global _memory_data
    if os.path.exists(MEM_PATH):
        try:
            with open(MEM_PATH, "r", encoding="utf-8") as f:
                _memory_data = json.load(f)
            logger.info(f"Memory loaded from {MEM_PATH} with {len(_memory_data)} entries.")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {MEM_PATH}. Initializing empty memory.")
            _memory_data = {}
        except Exception as e_load:
            logger.error(f"Failed to load memory from {MEM_PATH}: {e_load}. Initializing empty memory.")
            _memory_data = {}
    else:
        logger.info(f"Memory file {MEM_PATH} not found. Starting with empty memory.")
        _memory_data = {}

_load_memory()

def get_legal_values(grid: np.ndarray) -> List[int]:
    existing_values = grid[grid != -1]
    max_val_legal = int(np.max(existing_values)) if existing_values.size > 0 else 1
    return list(range(1, max_val_legal + 1))

def mem_score(grid: np.ndarray, r_mem: int, c_mem: int, v_mem: int) -> float:
    move_key = f"{r_mem}_{c_mem}_{v_mem}"
    if move_key in _memory_data:
        data = _memory_data[move_key]
        if data.get("count", 0) > 0:
            return data.get("total_score", 0.0) / data["count"]
    return 0.0

def update_memory(r_update: int, c_update: int, v_update: int, outcome_score: float):
    global _memory_data
    move_key = f"{r_update}_{c_update}_{v_update}"
    if move_key not in _memory_data:
        _memory_data[move_key] = {"count": 0, "total_score": 0.0}
    _memory_data[move_key]["count"] += 1
    _memory_data[move_key]["total_score"] += outcome_score
    logger.info(f"Memory updated for {move_key}. New count: {_memory_data[move_key]['count']}, Total score: {_memory_data[move_key]['total_score']:.2f}")

def _save_memory():
    try:
        with open(MEM_PATH, "w", encoding="utf-8") as f:
            json.dump(_memory_data, f, indent=4)
        logger.info(f"Memory saved to {MEM_PATH}.")
    except Exception as e_save:
        logger.error(f"Failed to save memory to {MEM_PATH}: {e_save}")

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Application shutdown event: Saving memory...")
    _save_memory()

# ── 4. CP-SAT 解算與性能記錄 ─────────────────────────────────
def build_and_solve_cp_vec(
    grid: np.ndarray,
    candidates: List[Tuple[int,int,int]],
    legal_values: List[int]
) -> List[Tuple[int,int,int,float,float]]:
    start_all = time.time()
    if not candidates:
        logger.info("沒有提供有效的候選給 CP-SAT 解算器。")
        return []

    max_val_cp = int(np.max(grid[grid != -1])) if np.any(grid != -1) else 1
    if max_val_cp == 0: max_val_cp = 1

    t0 = time.time()
    feat = build_feature_tensor(grid)
    t1 = time.time()
    base_tf_scores = calculate_scores_from_tensor(feat, grid)
    t2 = time.time()
    logger.info(f"張量建構: {t1-t0:.4f}s, 分數計算: {t2-t1:.4f}s")

    model = cp_model.CpModel()
    num_candidates = len(candidates)
    chosen_idx = model.NewIntVar(0, num_candidates - 1, 'chosen_candidate_idx')
    r_chosen = model.NewIntVar(0, grid.shape[0] - 1, 'r_chosen')
    c_chosen = model.NewIntVar(0, grid.shape[1] - 1, 'c_chosen')
    v_chosen = model.NewIntVar(1, max_val_cp, 'v_chosen')

    candidate_tf_scores_scaled = []
    candidate_total_scores_scaled = []
    SCORE_SCALE_FACTOR = 10000

    for r_cand, c_cand, v_cand in candidates:
        tf_score_val = base_tf_scores[r_cand, c_cand]
        candidate_tf_scores_scaled.append(int(tf_score_val * SCORE_SCALE_FACTOR))
        mem_score_val = mem_score(grid, r_cand, c_cand, v_cand)
        total_score_val = tf_score_val + mem_score_val
        candidate_total_scores_scaled.append(int(total_score_val * SCORE_SCALE_FACTOR))

    min_total_score_scaled = min(candidate_total_scores_scaled) if candidate_total_scores_scaled else 0
    max_total_score_scaled = max(candidate_total_scores_scaled) if candidate_total_scores_scaled else 0
    # Ensure min <= max if list was empty or all same
    if min_total_score_scaled > max_total_score_scaled: max_total_score_scaled = min_total_score_scaled

    chosen_total_score = model.NewIntVar(min_total_score_scaled, max_total_score_scaled, 'chosen_total_score')

    # Corrected AddElement calls
    model.AddElement(chosen_idx, [cand[0] for cand in candidates], r_chosen)
    model.AddElement(chosen_idx, [cand[1] for cand in candidates], c_chosen)
    model.AddElement(chosen_idx, [cand[2] for cand in candidates], v_chosen)
    model.AddElement(chosen_idx, candidate_total_scores_scaled, chosen_total_score)

    model.Maximize(chosen_total_score)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    solver.parameters.log_search_progress = True
    solver.parameters.num_workers = os.cpu_count() or 1
    res = solver.Solve(model)

    best_list = []
    if res == cp_model.OPTIMAL or res == cp_model.FEASIBLE:
        idx = solver.Value(chosen_idx)
        r_sol, c_sol, v_sol = candidates[idx]
        final_total_score = solver.Value(chosen_total_score) / SCORE_SCALE_FACTOR
        final_tf_score = candidate_tf_scores_scaled[idx] / SCORE_SCALE_FACTOR
        best_list.append((r_sol, c_sol, v_sol, final_total_score, final_tf_score))
        logger.info(f"CP-SAT 找到一個解: ({r_sol}, {c_sol}, {v_sol}) 總分數 {final_total_score:.4f}")
    else:
        logger.warning(f"CP-SAT 未能找到最佳或可行解。狀態: {solver.StatusName(res)}")

    logger.info(f"總解算時間: {time.time() - start_all:.4f}s")
    return best_list

# ── 5. /analyze API ───────────────────────────────────────────────
class ProposedValue(BaseModel):
    pos: List[int]
    value: int

class AnalyzeRequest(BaseModel):
    new_card: List[List[int]]
    proposed_values: List[ProposedValue]

    @validator("new_card")
    def check_rectangular(cls, g_val):
        if not g_val or any(len(row) != len(g_val[0]) for row in g_val):
            raise ValueError("new_card 必須是矩形")
        return g_val

    @validator("proposed_values", each_item=True)
    def check_pv(cls, pv_val, values_val):
        grid_val = values_val.get("new_card")
        if grid_val:
            rows, cols = len(grid_val), len(grid_val[0])
            r_pv, c_pv = pv_val.pos
            if not (0 <= r_pv < rows and 0 <= c_pv < cols):
                raise ValueError(f"pos 越界：{pv_val.pos}")
            maxv_pv = int(np.max(np.array(grid_val)[np.array(grid_val) != -1])) if np.any(np.array(grid_val) != -1) else 1
            if pv_val.value < 1 or pv_val.value > maxv_pv: # Ensure maxv_pv is at least 1
                raise ValueError(f"value 超出範圍：1~{max(1, maxv_pv)}")
        return pv_val

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        grid_req = np.array(req.new_card, dtype=int)
        legal_req = get_legal_values(grid_req)
        
        candidates_req = []
        for pv_req in req.proposed_values:
            r_req, c_req = pv_req.pos
            v_req = pv_req.value
            if grid_req[r_req, c_req] == -1 and v_req in legal_req:
                candidates_req.append((r_req, c_req, v_req))
            else:
                logger.warning(f"跳過無效候選: 位置={pv_req.pos}, 數值={v_req}。單元格非空或數值不合法。")

        if not candidates_req:
            raise HTTPException(status_code=400, detail="沒有合法候選可以分析。請確保 proposed_values 指向空位且數值合法。")
        
        best_req = await run_in_threadpool(build_and_solve_cp_vec, grid_req, candidates_req, legal_req)
        
        if not best_req:
            return {"status": "fail", "result": None, "message": "CP-SAT 解算器未能找到最佳解。"}
        
        r_res, c_res, v_res, score_res, tf_score_res = best_req[0] 
        
        # Update and save memory
        update_memory(r_res, c_res, v_res, score_res)
        _save_memory()

        return {
            "status": "success",
            "result": {
                "pos": [r_res, c_res],
                "value": v_res,
                "score": round(score_res, 4),
                "tensor_flow_score": round(tf_score_res, 4)
            }
        }
    except HTTPException:
        raise
    except Exception as e_analyze:
        logger.exception("在 /analyze 端點發生意外錯誤。")
        raise HTTPException(status_code=500, detail=f"內部伺服器錯誤: {str(e_analyze)}")

if __name__ == "__main__":
    # This part is for local testing if you run the script directly
    # It won't run when deployed with Uvicorn/Gunicorn
    # Example: uvicorn main:app --reload
    import uvicorn
    logger.info("Starting Uvicorn server for local development...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
