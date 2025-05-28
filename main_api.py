# main_api.py
# coding: utf-8

from fastapi import FastAPI, HTTPException, Body, Request
from typing import List, Dict, Optional, Any, Tuple, Callable # Added Tuple, Callable
import logging
import uuid
import os
import numpy as np # Required by /health/analyze
from pydantic import BaseModel

import logging

# --- Logging Setup (一定要在 try/except 前) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger("extreme_api_service")

# --- Module Imports ---
# Assuming analyzer.py and main.py (as main_logic_module) are in the Python path
from analyzer import Analyzer, InitializationError, InvalidInputError, ModuleError, ModuleNotFoundError, ModuleExecutionError, VisualizationError
import main as main_logic_module  # TODO: User to verify this is the correct GM logic module

try:
    EXTREME_MODULE_FUNCS_VEC = list(main_logic_module.registered_modules.keys())
    # 預設所有權重都是 1.0（你也可以寫成 main.py 給的自訂權重）
    EXTREME_MODULE_WEIGHTS = {k: 1.0 for k in EXTREME_MODULE_FUNCS_VEC}
except Exception as e:
    EXTREME_MODULE_FUNCS_VEC = []
    EXTREME_MODULE_WEIGHTS = {}
    logger.error("自動同步 EXTREME_MODULE_FUNCS_VEC/WEIGHTS 失敗: %s", e)
# --- Placeholder for Global Constants & Functions ---
# These MUST be defined by the user with their actual values/implementations
ANALYSIS_ENGINE_VERSION_EXTREME: str = "1.1.0-extreme" # Placeholder
EXTREME_MODULE_FUNCS_VEC: List[str] = ["gm_func_1", "gm_func_2"] # Placeholder
EXTREME_MODULE_WEIGHTS: Dict[str, Any] = {"gm_func_1": "weights1.bin"} # Placeholder
MEM_PATH: str = "data/persistent_memory.json" # Placeholder

class MockCPModel: # Placeholder for ortools.sat.python.cp_model
    __version__: str = "9.9.mock"
    def CpModel(self):
        logger.info("[Placeholder] MockCPModel.CpModel() invoked.")
        pass
cp_model = MockCPModel()

def extreme_tensor_flow_score_detailed(grid: np.ndarray, request_id_context: str) -> Tuple[np.ndarray, List[List[Dict[str, Any]]]]:
    """Placeholder for user's core TensorFlow scoring logic."""
    logger.info(f"[Placeholder] extreme_tensor_flow_score_detailed for {request_id_context}, grid: {grid.shape}")
    # Return dummy scores and contributions matching expected structure
    scores = np.random.rand(*grid.shape).astype(np.float32) * 10
    contributions = [[{"rule": f"dummy_r{r}_c{c}", "value": random.random()} for c in range(grid.shape[1])] for r in range(grid.shape[0])]
    return scores, contributions

async def run_in_threadpool(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Placeholder for running blocking IO or CPU-bound tasks in a thread pool."""
    # In production, use: from fastapi.concurrency import run_in_threadpool
    # Or for Python 3.9+: import asyncio; await asyncio.to_thread(func, *args, **kwargs)
    logger.info(f"[Placeholder] Synchronously executing {func.__name__} via run_in_threadpool placeholder.")
    return func(*args, **kwargs)

def get_legal_values_for_placement(grid: np.ndarray) -> set:
    """Placeholder for user's logic to get legal placement values."""
    logger.info(f"[Placeholder] get_legal_values_for_placement for grid: {grid.shape}")
    return set(range(1, int(np.max(grid) if grid.size > 0 else 9) + 1)) # Example

def get_card_max_value(grid: np.ndarray) -> Optional[int]:
    """Placeholder for user's logic to get max value on card."""
    logger.info(f"[Placeholder] get_card_max_value for grid: {grid.shape}")
    return int(np.max(grid)) if grid.size > 0 else 9 # Example

def mem_score(r: int, c: int, val: int, context_set: set) -> float:
    """Placeholder for user's memory scoring logic."""
    logger.info(f"[Placeholder] mem_score for ({r},{c}) val {val} context_size {len(context_set)}")
    return random.random() * 5.0
# --- End of Placeholders ---


# --- Pydantic Models ---
class AnalyzeHealthStatus(BaseModel):
    status: str
    analysis_engine_version: str
    checks: Dict[str, str]
    components: Dict[str, str]

class CandidateDetail(BaseModel):
    pos: List[int]
    value: int
    is_valid_proposal: bool
    raw_tensor_flow_score: float
    mem_score_value: float
    final_objective_score: float
    cp_solver_notes: Optional[str] = None

class AnalyzeSuccessResponse(BaseModel):
    request_id: str
    message: str
    grid_shape: Tuple[int, ...]
    evaluated_candidates: List[CandidateDetail]

class AnalyzeErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None

class ProposedValue(BaseModel):
    pos: Tuple[int, int]
    value: int

class AnalysisRequest(BaseModel): # Renamed from user's example for clarity
    new_card: List[List[int]]
    proposed_values: List[ProposedValue] # Changed from List[int] to List[ProposedValue]
    active_modules: Optional[List[str]] = None
    module_weights: Optional[Dict[str, float]] = None
    top_n: Optional[int] = None
# --- Pydantic Models End ---


# --- FastAPI App Setup ---
app = FastAPI(
    title="智慧評分系統 API (Extreme Edition)",
    description="提供基於進階 N 維張量運算與 AI 模組的盤面分析與評分建議 API 服務。",
    version=ANALYSIS_ENGINE_VERSION_EXTREME
)
# --- FastAPI App Setup End ---


# --- Analyzer Instance ---
analyzer_instance: Optional[Analyzer] = None
try:
    analyzer_instance = Analyzer(main_module=main_logic_module, default_top_n=3)
    logger.info("Analyzer instance created successfully for API, using logic module: %s", getattr(main_logic_module, '__name__', 'N/A'))
except InitializationError as e_init:
    logger.critical("CRITICAL_API_STARTUP_ERROR: Failed to initialize Analyzer: %s", e_init, exc_info=True)
except Exception as e:
    logger.critical("CRITICAL_API_STARTUP_ERROR: Unexpected error during Analyzer initialization: %s", e, exc_info=True)
# --- Analyzer Instance End ---


# --- API Endpoints ---
@app.get("/", status_code=200, tags=["Utilities"], summary="Root Path / Basic Health Ping")
async def read_root():
    """服務根路徑，用於平台健康檢查或提供簡單的歡迎訊息。"""
    return {"message": "Smart Scoring System API (Extreme Edition) is running and healthy!"}

@app.get("/health", response_model=HealthResponse, tags=["Utilities"], summary="Simple Analyzer Health Check")
async def health_check_simple(request: Request): # Added request for consistency if needed
    """提供簡單的健康狀態檢查，確認 Analyzer 核心實例是否已成功初始化。"""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())) # Example: Use request_id
    if analyzer_instance is None:
        logger.warning(f"HEALTH_CHECK_SIMPLE /health: Failed - Analyzer not initialized. RequestID: {request_id}")
        return HealthResponse(status="unhealthy", reason="Analyzer core component not initialized.", analyzer_status="Not Initialized")
    logger.info(f"HEALTH_CHECK_SIMPLE /health: Successful - Analyzer is initialized. RequestID: {request_id}")
    return HealthResponse(status="ok", message="Analyzer API is running and Analyzer core is initialized.", analyzer_status="Initialized")

@app.get("/health/analyze", response_model=AnalyzeHealthStatus, tags=["Health & Monitoring"], summary="Detailed System Health Analysis")
async def health_analyze_detailed(request: Request):
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    logger.info(f"HEALTH_CHECK_DETAILED /health/analyze: Request received. RequestID: {request_id}")
    checks: Dict[str, str] = {}
    overall_status: str = "UP"

    if not EXTREME_MODULE_FUNCS_VEC:
        checks["extreme_module_funcs_load"] = "FAIL"; overall_status = "DEGRADED"
    else:
        checks["extreme_module_funcs_load"] = f"OK: {len(EXTREME_MODULE_FUNCS_VEC)} funcs"

    if not EXTREME_MODULE_WEIGHTS:
        checks["extreme_module_weights_load"] = "FAIL"; overall_status = "DEGRADED"
    else:
        checks["extreme_module_weights_load"] = f"OK: {len(EXTREME_MODULE_WEIGHTS)} weights"

    if EXTREME_MODULE_FUNCS_VEC and EXTREME_MODULE_WEIGHTS:
        missing = [n for n in EXTREME_MODULE_FUNCS_VEC if n not in EXTREME_MODULE_WEIGHTS]
        if missing:
            checks["extreme_funcs_weights_match"] = f"WARN: Missing weights for: {', '.join(missing[:3])}{'...' if len(missing)>3 else ''}"
            overall_status = "DEGRADED"
        else:
            checks["extreme_funcs_weights_match"] = "OK"

    if not os.path.exists(MEM_PATH):
        checks["memory_file_exists"] = f"FAIL: {MEM_PATH} not found"; overall_status="DEGRADED"
    else:
        checks["memory_file_exists"] = "OK (Path exists)"

    try:
        dummy_grid_data = [[-1,1,5,0],[2,-1,8,3],[4,6,-1,7],[0,0,0,0]]
        dummy_grid_np = np.array(dummy_grid_data, dtype=np.int32)
        await run_in_threadpool(extreme_tensor_flow_score_detailed, dummy_grid_np, f"health_tf_{request_id}")
        checks["extreme_tf_execution_test"] = "OK"
    except Exception as e:
        checks["extreme_tf_execution_test"] = f"FAIL: {str(e)}"; logger.error(f"HEALTH_ERROR /health/analyze: extreme_tf test FAIL. RequestID: {request_id}", exc_info=True); overall_status="ERROR"

    try:
        _ = cp_model.CpModel()
        checks["cp_solver_avail_test"] = "OK"
    except Exception as e:
        checks["cp_solver_avail_test"] = f"FAIL: {str(e)}"; logger.error(f"HEALTH_ERROR /health/analyze: CP Solver test FAIL. RequestID: {request_id}", exc_info=True); overall_status="ERROR"

    return AnalyzeHealthStatus(
        status=overall_status,
        analysis_engine_version=ANALYSIS_ENGINE_VERSION_EXTREME,
        checks=checks,
        components={
            "numpy_version": np.__version__,
            "ortools_version": getattr(cp_model, '__version__', "unknown"),
            "analyzer_type": "Extreme Logic Modules v22"
        }
    )

@app.post("/analyze",
            response_model=AnalyzeSuccessResponse,
            responses={
                400: {"model": AnalyzeErrorResponse, "description": "Invalid input data (client-side error)"},
                422: {"model": AnalyzeErrorResponse, "description": "Validation error in request data (unprocessable entity)"},
                500: {"model": AnalyzeErrorResponse, "description": "Internal server processing error"},
                503: {"model": AnalyzeErrorResponse, "description": "Service temporarily unavailable (e.g., Analyzer not initialized)"}
            },
            tags=["Analysis Engine vExtreme"],
            summary="Perform Extreme N-Dimensional Tensor Analysis")
async def analyze_board_main(req: AnalysisRequest, request: Request): # Renamed to avoid conflict with module 'analyze' if any
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    logger.info(f"API_CALL /analyze: RequestID: {request_id}. Grid: {len(req.new_card)}x{len(req.new_card[0]) if req.new_card and req.new_card[0] else 'empty'}. Proposals: {len(req.proposed_values)}.")

    if analyzer_instance is None:
        logger.error(f"API_ERROR /analyze: Analyzer instance not available. RequestID: {request_id}")
        raise HTTPException(status_code=503, detail="Analysis service is temporarily unavailable due to initialization failure.")

    if not req.new_card or not req.new_card[0]:
        logger.warning(f"API_VALIDATION_ERROR /analyze: Empty new_card received. RequestID: {request_id}")
        raise HTTPException(status_code=400, detail="Input 'new_card' cannot be empty or contain empty rows.")
    
    try:
        grid_np = np.array(req.new_card, dtype=np.int32)
    except Exception as e:
        logger.error(f"API_VALIDATION_ERROR /analyze: Failed to convert new_card to NumPy array. RequestID: {request_id}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid data format in 'new_card': {str(e)}")

    # Call the separate Analyzer class method
    try:
        # Assuming analyzer_instance.analyze_board is an async or can be run in threadpool
        analysis_result = await run_in_threadpool( # Or directly await if analyze_board is async
            analyzer_instance.analyze_board,
            new_card=req.new_card, # Analyzer might expect list of lists
            proposed_values=req.proposed_values, # Pass the Pydantic list directly
            active_modules=req.active_modules,
            module_weights=req.module_weights,
            top_n=req.top_n
        )
        # Assuming analysis_result is a list of CandidateDetail-like dicts or objects
        # If it's not exactly List[CandidateDetail], you might need to transform it here.
        # For this example, assuming it can be directly used.
        # If analyzer_instance.analyze_board returns a complex dict that matches AnalyzeSuccessResponse:
        # return analysis_result

        # If analyze_board returns just the list of candidates, construct the full response:
        processed_candidates = []
        if isinstance(analysis_result, list): # Basic check, adjust based on actual return
            for cand_data in analysis_result:
                if isinstance(cand_data, dict): # If analyze_board returns dicts
                    processed_candidates.append(CandidateDetail(**cand_data))
                elif isinstance(cand_data, CandidateDetail): # If it already returns Pydantic models
                    processed_candidates.append(cand_data)
                else:
                    logger.warning(f"API_RESULT_WARN /analyze: Unexpected candidate data type: {type(cand_data)}. RequestID: {request_id}")
        else: # Handle if result is not a list (e.g. a full dict response)
             logger.warning(f"API_RESULT_WARN /analyze: Unexpected result type from analyzer: {type(analysis_result)}. RequestID: {request_id}")
             # If analysis_result is expected to be the full AnalyzeSuccessResponse dict:
             # return analysis_result
             # For now, assume it should be a list of candidates:
             raise HTTPException(status_code=500, detail="Internal error: Unexpected analysis result format.")


        logger.info(f"API_SUCCESS /analyze: Analysis complete. RequestID: {request_id}. Evaluated {len(processed_candidates)} candidates.")
        return AnalyzeSuccessResponse(
            request_id=request_id,
            message="Analysis successfully completed.",
            grid_shape=grid_np.shape,
            evaluated_candidates=processed_candidates
        )

    except InvalidInputError as e:
        logger.warning(f"API_VALIDATION_ERROR /analyze: Invalid input from Analyzer: {e}. RequestID: {request_id}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Invalid Input Parameters: {str(e)}")
    except (ModuleNotFoundError, ModuleExecutionError, ModuleError, VisualizationError) as e:
        logger.error(f"API_MODULE_ERROR /analyze: Analyzer module error: {e}. RequestID: {request_id}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Module Error during analysis ({type(e).__name__}): {str(e)}")
    except Exception as e:
        logger.critical(f"API_UNEXPECTED_ERROR /analyze: Unexpected critical error. RequestID: {request_id}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected internal server error: {type(e).__name__} - {str(e)}")
# --- API 端點結束 ---


# --- 主程式執行入口 (用於本地開發) ---
if __name__ == "__main__":
    logger.info("Executing main_api.py directly (intended for local Uvicorn launch instruction).")
    print("\nTo run this FastAPI application locally for development:")
    print("1. Ensure all dependencies are installed: pip install fastapi uvicorn numpy pydantic")
    print("2. If using ortools, ensure it's installed: pip install ortools")
    print("3. Ensure 'new_module.py' (with PuzzleTensorOps) and 'analyzer.py' (with Analyzer) are present.")
    print("4. In your terminal, run: uvicorn main_api:app --reload --host 0.0.0.0 --port 8000")
    print("   (Assuming this file is named main_api.py and the FastAPI instance is 'app')")
    print("5. Open your browser to http://127.0.0.1:8000/docs to interact with the API.")
    print("\nFor deployment on platforms like Render, use the platform's start command, e.g.:")
    print("  uvicorn main_api:app --host 0.0.0.0 --port ${PORT} --workers 1")

    # Programmatic start for Uvicorn (optional, for specific local test scenarios)
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000) # Or "main_api:app" for reload
