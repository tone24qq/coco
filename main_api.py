# main_api_alt_corrected.py
# coding: utf-8

from fastapi import FastAPI, HTTPException, Body, Request, Depends, status
from typing import List, Dict, Optional, Any, Tuple, Callable, Union # Added Union
import logging
import uuid
import os # os.path was used
import numpy as np

# Project-specific imports
import brain # Assuming brain.py is in the same directory or PYTHONPATH
from analyzer import Analyzer, InitializationError, InvalidInputError, ModuleError, VisualizationError # Import relevant exceptions

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - RequestID: %(request_id)s - %(message)s'
)
logger = logging.getLogger("extreme_api_service_alt") # Renamed logger for clarity if both run

# --- Global Variables & Initializations ---
ANALYSIS_ENGINE_VERSION_EXTREME: str = "1.1.0-extreme-corrected"
MEM_PATH: str = "data/persistent_memory.json" # Placeholder, ensure this path is meaningful or handled

# --- Analyzer Instance ---
analyzer_instance_alt: Optional[Analyzer] = None
try:
    # Use the imported 'brain' module directly
    analyzer_instance_alt = Analyzer(main_module=brain, default_top_n=3)
    logger.info("Alternate Analyzer instance (analyzer_instance_alt) created successfully, using 'brain' module.")
except InitializationError as e_init:
    logger.critical("CRITICAL_ALT_API_STARTUP_ERROR: Failed to initialize Alternate Analyzer: %s", e_init, exc_info=True)
    analyzer_instance_alt = None
except Exception as e_unexpected:
    logger.critical("CRITICAL_ALT_API_STARTUP_ERROR: Unexpected error during Alternate Analyzer initialization: %s", e_unexpected, exc_info=True)
    analyzer_instance_alt = None


# --- Mock/Placeholder Components (from original file, kept for structural integrity if needed by user) ---
class MockCPModel: # This was in the original PDF.
    _version: str = "9.9.mock"
    def CpModel(self):
        logger.info("[Placeholder] MockCPModel.CpModel() invoked.")
        pass # It's a mock, so no actual model returned

cp_model = MockCPModel() # Instantiated mock

def extreme_tensor_flow_score_detailed(grid: np.ndarray, request_id_context: str) -> Tuple[np.ndarray, List[List[Dict[str, Any]]]]:
    logger.info(f"[Placeholder] extreme_tensor_flow_score_detailed for {request_id_context}, grid: {grid.shape}")
    scores = np.random.rand(*grid.shape).astype(np.float32) * 10
    contributions = [[{"rule": f"dummy_r{r}_c{c}", "value": np.random.random()} for c in range(grid.shape[1])] for r in range(grid.shape[0])]
    return scores, contributions


# --- Pydantic Models ---
# Models for health checks
class HealthResponse(BaseModel):
    status: str
    message: Optional[str] = None
    reason: Optional[str] = None
    analyzer_status: Optional[str] = None

class AnalyzeHealthStatus(BaseModel):
    status: str
    analysis_engine_version: str
    checks: Dict[str, str]
    components: Dict[str, str]

# Models for /analyze endpoint (corrected to match Analyzer's actual input/output)
class AltAnalysisRequest(BaseModel):
    new_card: List[List[int]] = Field(..., example=[[1,-1,0],[-1,2,-1]])
    # Corrected: Analyzer expects List[int] for proposed_values
    proposed_values: List[int] = Field(..., example=[3,5])
    active_modules: Optional[List[str]] = None
    module_weights: Optional[Dict[str, float]] = None
    top_n: Optional[int] = Field(None, gt=0)
    client_request_id: Optional[str] = None # Added for consistency


# Reusing models from main.py for Analyzer response structure
class SuggestionItem(BaseModel):
    position: List[int]
    score: float

class ProcessedParams(BaseModel):
    requested_top_n: Union[int, str]
    actual_top_n: int
    requested_active_modules: Optional[List[str]]
    effective_active_modules: List[str]
    requested_module_weights: Optional[Dict[str, float]]
    final_module_weights: Dict[str, float]
    request_id: str
    error: Optional[bool] = None

class BoardDimensions(BaseModel):
    rows: int
    cols: int

class AltAnalyzeResponse(BaseModel):
    request_id: str
    # Mirroring Analyzer's output structure
    suggestions: Optional[Dict[int, List[SuggestionItem]]] = None # Key from PV (int)
    visualization: Optional[str] = None
    board_dimensions: Optional[BoardDimensions] = None
    processed_params: Optional[ProcessedParams] = None
    error: Optional[str] = None # For high-level errors or errors from analyzer response
    message: Optional[str] = None # General message


class AltAnalyzeErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None

# --- FastAPI App Instance ---
app_alt = FastAPI(
    title="智慧評分系統 API (Extreme Edition - Corrected)",
    description="提供基於進階分析與AI模組的盤面分析與評分建議API服務 (修正版)。",
    version=ANALYSIS_ENGINE_VERSION_EXTREME
)

# --- Middleware for Request ID (Simplified) ---
@app_alt.middleware("http")
async def add_request_id_middleware(request: Request, call_next: Callable):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id # Make it accessible in request state
    
    # For logging, pass 'extra' dict.
    # This simple middleware doesn't auto-inject into all loggers.
    # Log calls in endpoints should include extra={'request_id': request_id}

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# --- API Endpoints ---
@app_alt.get("/", status_code=200, tags=["Utilities (Alt)"], summary="Root Path / Basic Health Ping (Alt)")
async def read_root_alt(request: Request):
    req_id = getattr(request.state, 'request_id', "N/A_alt_root")
    logger.info("Alternate API root accessed.", extra={'request_id': req_id})
    return {"message": "Smart Scoring System API (Extreme Edition - Corrected) is running!"}

@app_alt.get("/health", response_model=HealthResponse, tags=["Utilities (Alt)"], summary="Simple Analyzer Health Check (Alt)")
async def health_check_simple_alt(request: Request):
    req_id = getattr(request.state, 'request_id', "N/A_alt_health")
    if analyzer_instance_alt is None:
        logger.warning(f"HEALTH_CHECK_SIMPLE (Alt) /health: Failed - Analyzer not initialized.", extra={'request_id': req_id})
        return HealthResponse(status="unhealthy", reason="Alternate Analyzer core component not initialized.", analyzer_status="Not Initialized")
    logger.info(f"HEALTH_CHECK_SIMPLE (Alt) /health: Successful - Alternate Analyzer is initialized.", extra={'request_id': req_id})
    return HealthResponse(status="ok", message="Alternate Analyzer API is running and core is initialized.", analyzer_status="Initialized")


@app_alt.post("/analyze", 
             response_model=AltAnalyzeResponse,
             responses={
                 400: {"model": AltAnalyzeErrorResponse, "description": "Invalid input data (client-side error)"},
                 422: {"model": AltAnalyzeErrorResponse, "description": "Validation error in request data"},
                 500: {"model": AltAnalyzeErrorResponse, "description": "Internal server processing error"},
                 503: {"model": AltAnalyzeErrorResponse, "description": "Service temporarily unavailable"}
             },
             tags=["Analysis Engine vExtreme (Alt)"], 
             summary="Perform Board Analysis using Analyzer (Corrected Alternate)")
async def analyze_board_main_alt(
    payload: AltAnalysisRequest, 
    request: Request # For request_id
):
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    log_extra = {'request_id': request_id, 'client_request_id': payload.client_request_id}
    
    logger.info(
        f"ALT_API_CALL /analyze: RequestID: {request_id}. Grid: "
        f"{len(payload.new_card)}x{len(payload.new_card[0]) if payload.new_card and payload.new_card[0] else 'empty'}. "
        f"Proposals: {len(payload.proposed_values)}.",
        extra=log_extra
    )

    if analyzer_instance_alt is None:
        logger.error(f"ALT_API_ERROR /analyze: Alternate Analyzer instance not available. RequestID: {request_id}", extra=log_extra)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                            detail="Analysis service (alt) is temporarily unavailable due to initialization failure.")

    try:
        # Correctly await the async method from Analyzer
        analysis_result_dict = await analyzer_instance_alt.analyze_board(
            new_card=payload.new_card,
            proposed_values=payload.proposed_values, # Now List[int]
            active_modules=payload.active_modules,
            module_weights=payload.module_weights,
            top_n=payload.top_n,
            request_id_for_logging=request_id # Pass API request_id
        )
        
        # Check if analyzer returned an internal error structure
        if 'error' in analysis_result_dict and isinstance(analysis_result_dict['error'], str):
            logger.warning(f"ALT_API_ANALYZER_HANDLED_ERROR /analyze: {analysis_result_dict['error']}. RequestID: {request_id}", extra=log_extra)
            return AltAnalyzeResponse(
                request_id=request_id,
                error=analysis_result_dict['error'],
                suggestions=None, # Or extract from dict if available
                visualization=analysis_result_dict.get('visualization'),
                board_dimensions=BoardDimensions(**analysis_result_dict.get('board_dimensions', {'rows':0, 'cols':0})),
                processed_params=ProcessedParams(**analysis_result_dict.get('processed_params', {})),
                message="Analysis completed with input validation issues."
            )

        # Adapt analyzer_result_dict to AltAnalyzeResponse
        suggestions_raw = analysis_result_dict.get('suggestions', {})
        suggestions_typed: Dict[int, List[SuggestionItem]] = {}
        if isinstance(suggestions_raw, dict):
            for pv_key, sugg_list_raw in suggestions_raw.items():
                try:
                    pv_int_key = int(pv_key)
                    suggestions_typed[pv_int_key] = [SuggestionItem(**sugg) for sugg in sugg_list_raw]
                except (ValueError, TypeError) as e:
                     logger.warning(f"Could not parse suggestion for PV key '{pv_key}' in Alt API: {e}", extra=log_extra)
        
        logger.info(f"ALT_API_SUCCESS /analyze: Analysis complete. RequestID: {request_id}", extra=log_extra)
        return AltAnalyzeResponse(
            request_id=request_id,
            message="Analysis successfully completed.",
            suggestions=suggestions_typed,
            visualization=analysis_result_dict.get('visualization'),
            board_dimensions=BoardDimensions(**analysis_result_dict.get('board_dimensions', {'rows':0, 'cols':0})),
            processed_params=ProcessedParams(**analysis_result_dict.get('processed_params', {}))
        )

    except InvalidInputError as e_analyzer_invalid: # Errors from Analyzer's validation
        logger.warning(f"ALT_API_VALIDATION_ERROR /analyze: Invalid input for Analyzer: {e_analyzer_invalid}. RequestID: {request_id}", 
                       exc_info=True, extra=log_extra)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid Input Parameters for Analyzer: {str(e_analyzer_invalid)}")
    
    except (ModuleError, VisualizationError) as e_analyzer_runtime: # Other specific errors from Analyzer
        logger.error(f"ALT_API_MODULE_ERROR /analyze: Analyzer runtime error: {e_analyzer_runtime}. RequestID: {request_id}", 
                     exc_info=True, extra=log_extra)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analyzer Error during analysis ({type(e_analyzer_runtime).__name__}): {str(e_analyzer_runtime)}")
    
    except Exception as e_unexpected: # Catch-all for other errors
        logger.critical(f"ALT_API_UNEXPECTED_ERROR /analyze: Unexpected critical error. RequestID: {request_id}", 
                        exc_info=True, extra=log_extra)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected internal server error: {type(e_unexpected).__name__} - {str(e_unexpected)}")


if __name__ == "__main__":
    import uvicorn
    # Add a simple handler for direct run, similar to the main.py
    if not logger.hasHandlers():
        _stdout_handler_alt = logging.StreamHandler()
        _stdout_handler_alt.setFormatter(logging.Formatter(
             '%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - RequestID: %(request_id)s - %(message)s',
            '%Y-%m-%d %H:%M:%S'
        ))
        class RequestIDLogFilterAlt(logging.Filter):
            def filter(self, record):
                record.request_id = getattr(record, 'request_id', 'system_alt')
                return True
        logger.addFilter(RequestIDLogFilterAlt())
        logger.addHandler(_stdout_handler_alt)

    logger.info("Executing main_api_alt_corrected.py directly.", extra={'request_id': 'startup_alt'})
    logger.info("This is an ALTERNATE API implementation. The 'main.py' is the primary recommended API.", extra={'request_id': 'startup_alt'})
    if analyzer_instance_alt is None:
        logger.error("ALTERNATE ANALYZER NOT INITIALIZED. /analyze endpoint will fail.", extra={'request_id': 'startup_alt'})
    else:
        logger.info("Alternate Analyzer instance is initialized.", extra={'request_id': 'startup_alt'})
    
    print("\nTo run this ALTERNATE FastAPI application (main_api_alt_corrected.py):")
    print("1. Ensure dependencies are installed: fastapi uvicorn numpy matplotlib pydantic pydantic-settings")
    print("2. Ensure 'analyzer.py' and 'brain.py' are in the same directory or Python path.")
    print("3. In your terminal, run: uvicorn main_api_alt_corrected:app_alt --reload --host 0.0.0.0 --port 8001")
    print("   (Using port 8001 to avoid conflict if main.py runs on 8000)")
    print("4. Open browser to http://127.0.0.1:8001/docs for this alternate API.")
    
    # To run programmatically (example, usually Uvicorn CLI is preferred for dev)
    # uvicorn.run(app_alt, host="0.0.0.0", port=8001)