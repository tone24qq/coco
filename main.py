# main.py
import asyncio
import datetime # Not directly used in this snippet, but often useful
import logging
import time
import uuid
from typing import Any, Coroutine, Dict, List, Optional, Union, Tuple # Added Tuple

import numpy as np
from fastapi import (BackgroundTasks, Body, Depends, FastAPI, HTTPException,
                     Path, Query, Request, Security, status)
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKey, APIKeyHeader, APIKeyQuery
from pydantic import BaseModel, Field, field_validator, HttpUrl # HttpUrl used in Settings
from pydantic_settings import BaseSettings

# For Prometheus metrics
from prometheus_client import Counter, Gauge, Histogram, Summary # type: ignore
from starlette_prometheus import PrometheusMiddleware, handle_metrics # type: ignore

# Project-specific imports
import brain # type: ignore
from analyzer import Analyzer, InitializationError, InvalidInputError, ModuleError # Import necessary items

# --- Application Settings ---
class Settings(BaseSettings):
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    APP_TITLE: str = "進階自動補格評分 API (Analyzer Integrated)"
    APP_DESCRIPTION: str = "提供強化盤面評分模組的API接口,包含完整分析流程、批次處理、背景任務與安全性增強。"
    APP_VERSION: str = "2.1.0" # Updated version

    API_KEY: str = "YOUR_SECRET_API_KEY_HERE" # Default, should be set via environment
    API_KEY_NAME: str = "X-API-KEY"

    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    TASK_CALLBACK_URL_ENABLED: bool = False
    TASK_CALLBACK_URL: Optional[HttpUrl] = None

    class Config:
        # env_file = ".env" # Uncomment to load from .env file
        # env_file_encoding = 'utf-8'
        pass

settings = Settings()

# --- Logging Configuration ---
# Basic config for stdout, can be customized further (e.g., with handlers, formatters for files)
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - RequestID: %(request_id)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
# Create a logger for this module
logger = logging.getLogger(__name__)

# --- Analyzer Instance ---
analyzer_instance: Optional[Analyzer] = None
try:
    analyzer_instance = Analyzer(main_module=brain, default_top_n=3)
    logger.info(f"Analyzer instance created successfully with 'brain' module. Default top_n: {analyzer_instance.default_top_n}")
except InitializationError as e_init:
    logger.critical("CRITICAL_API_STARTUP_ERROR: Failed to initialize Analyzer: %s", e_init, exc_info=True)
    analyzer_instance = None
except Exception as e_unexpected: # Catch any other error during Analyzer init
    logger.critical("CRITICAL_API_STARTUP_ERROR: Unexpected error during Analyzer initialization: %s", e_unexpected, exc_info=True)
    analyzer_instance = None


# --- Prometheus Metrics Definition ---
REQUEST_COUNT = Counter(
    "api_request_count",
    "Total number of API requests processed",
    ["method", "endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "endpoint"]
)
ACTIVE_BACKGROUND_TASKS = Gauge(
    "api_active_background_tasks",
    "Number of currently active background scoring tasks"
)
MODULE_USAGE_COUNT = Counter(
    "api_module_usage_count",
    "Count of how many times each scoring module is used (direct calls)",
    ["module_name"]
)
ANALYZER_REQUEST_COUNT = Counter(
    "api_analyzer_request_count",
    "Total number of full analysis requests processed via /analyze"
)
ANALYZER_REQUEST_LATENCY = Histogram(
    "api_analyzer_request_latency_seconds",
    "Full analysis request latency via /analyze in seconds"
)


# --- API Key Authentication ---
api_key_query = APIKeyQuery(name=settings.API_KEY_NAME, auto_error=False)
api_key_header = APIKeyHeader(name=settings.API_KEY_NAME, auto_error=False)

async def get_api_key(
    key_query: Optional[str] = Security(api_key_query),
    key_header: Optional[str] = Security(api_key_header),
) -> str:
    if key_query == settings.API_KEY:
        return key_query
    if key_header == settings.API_KEY:
        return key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key"
    )

# --- Pydantic Models for API ---

# Models for existing /score endpoints
class GridDataBase(BaseModel):
    grid_data: List[List[Union[int, float]]] = Field(..., example=[[-1, 1, -1], [2, -1, 3], [-1, 4, -1]])

    @field_validator('grid_data')
    def validate_grid_data(cls, v: List[List[Union[int, float]]]) -> List[List[Union[int, float]]]:
        if not v or not all(isinstance(row, list) for row in v): # Allow v[0] to be empty for Nx0 grids
            raise ValueError("Grid data must be a non-empty list of lists.")
        
        # Allow empty rows if all rows are empty (e.g. [[], []] for a 2x0 grid)
        if not any(v): # All rows are empty lists
            if not all(not row for row in v): # check if some are non-empty
                 raise ValueError("Grid data has inconsistent empty/non-empty rows.")
            return v


        # If first row is not empty, expect rectangular
        if v[0]:
            num_cols = len(v[0])
            if not all(len(row) == num_cols for row in v):
                raise ValueError("All non-empty rows must have the same number of columns.")
        else: # First row is empty, means all rows must be empty (Nx0 grid)
            if not all(not row for row in v):
                 raise ValueError("If first row is empty, all rows must be empty for an Nx0 grid.")


        for r_idx, row in enumerate(v):
            for c_idx, cell_val in enumerate(row):
                if not isinstance(cell_val, (int, float)):
                    raise ValueError(f"Cell ({r_idx}, {c_idx}) type invalid: {type(cell_val)}. Must be number.")
        return v

class GridInput(GridDataBase):
    client_request_id: Optional[str] = Field(None, description="Optional client-provided request ID for tracing.")

class BatchGridItem(GridDataBase):
    item_id: str = Field(description="Unique identifier for this item in the batch.")
    module_name: str = Field(description="Scoring module to use for this item.")

class BatchGridInput(BaseModel):
    grids: List[BatchGridItem] = Field(..., max_items=50) # Use max_items for Pydantic v2
    client_request_id: Optional[str] = Field(None, description="Optional client-provided request ID for the batch.")

class ScoreOutput(BaseModel): # For direct brain module calls (conceptual, not used by /score's TaskAcceptedResponse)
    module_name: str
    score_grid: List[List[float]]
    message: Optional[str] = None
    error: Optional[str] = None

class TaskAcceptedResponse(BaseModel):
    task_id: str
    status: str = "accepted"
    message: str
    client_request_id: Optional[str] = None

class ModuleInfo(BaseModel):
    name: str
    description: Optional[str] = "No description available."
    version: Optional[str] = "N/A"


# Models for new /analyze endpoint using Analyzer
class AnalyzeBoardApiRequest(BaseModel):
    new_card: List[List[int]] = Field(..., example=[[1, -1, 0], [-1, 2, -1]])
    proposed_values: List[int] = Field(..., example=[3, 5])
    active_modules: Optional[List[str]] = Field(None, example=["GM1_Random", "GM2_TargetTopLeft"])
    module_weights: Optional[Dict[str, float]] = Field(None, example={"GM1_Random": 0.5, "GM2_TargetTopLeft": 1.5})
    top_n: Optional[int] = Field(None, example=5, gt=0)
    client_request_id: Optional[str] = Field(None, description="Optional client-provided request ID for tracing.")

class SuggestionItem(BaseModel):
    position: List[int] = Field(..., example=[0,1])
    score: float = Field(..., example=0.75)

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

class AnalyzeBoardApiResponse(BaseModel):
    # For success case
    suggestions: Optional[Dict[int, List[SuggestionItem]]] = None # Key is proposed_value (int)
    visualization: Optional[str] = None # base64 encoded image or error string
    board_dimensions: Optional[BoardDimensions] = None
    processed_params: Optional[ProcessedParams] = None
    # For error case (e.g. input validation before analyzer call)
    error: Optional[str] = None
    request_id: str # Always include request_id

# --- Rate Limiter (Simple In-Memory Implementation) ---
request_counts: Dict[str, List[float]] = {}

# --- Helper Functions for Scoring Task (Background for /score direct brain calls) ---
async def run_scoring_task(
    task_id: str,
    module_name: str,
    grid_data: List[List[Union[int, float]]], # From GridDataBase
    original_request_id: str, # Main request ID
    client_request_id: Optional[str] = None
):
    """Performs actual single-module scoring (brain.get_module_score) in a background task."""
    ACTIVE_BACKGROUND_TASKS.inc()
    log_extra = {'request_id': original_request_id, 'task_id': task_id, 'module_name': module_name}
    logger.info("Background task started for direct brain module scoring.", extra=log_extra)
    
    try:
        # Ensure grid_data is suitable for np.array(dtype=float) or brain.py's expectations
        # brain.py's get_module_score expects np.ndarray. Individual modules often process ints.
        # Let's assume modules handle float input if that's what grid_data provides, or convert to int if appropriate.
        # For now, assume brain.py handles conversion or modules are robust.
        np_grid = np.array(grid_data) # Let brain.py handle dtype specifics if needed.
                                      # Or np.array(grid_data, dtype=np.int32) if all brain modules expect int.

        if np_grid.size == 0:
            raise ValueError("Input grid is empty after numpy conversion.")

        start_time = time.monotonic()
        # kwargs for brain.get_module_score
        brain_kwargs = {'request_id': task_id} # No proposed_value for direct single module scoring here
        
        score_np_array = await asyncio.to_thread(brain.get_module_score, module_name, np_grid, **brain_kwargs)
        duration = time.monotonic() - start_time
        
        # score_list_of_lists = score_np_array.tolist() # This would be part of actual result delivery
        result_message = f"Direct brain scoring successful for module {module_name}."
        logger.info(result_message + f" Duration: {duration:.4f}s", extra=log_extra)

        if settings.TASK_CALLBACK_URL_ENABLED and settings.TASK_CALLBACK_URL:
            logger.info(f"Simulating callback to {settings.TASK_CALLBACK_URL} with result for task {task_id}.", extra=log_extra)
            # callback_payload = {"task_id": task_id, "status": "completed", 
            #                     "result": score_list_of_lists, # Example
            #                     "client_request_id": client_request_id,
            #                     "original_request_id": original_request_id}
            # async with httpx.AsyncClient() as client:
            #     await client.post(str(settings.TASK_CALLBACK_URL), json=callback_payload)

    except Exception as e:
        error_message = f"Error in background direct brain scoring task for module {module_name}: {str(e)}"
        logger.error(error_message, exc_info=True, extra=log_extra)
    finally:
        ACTIVE_BACKGROUND_TASKS.dec()
        logger.info("Background task for direct brain module scoring finished.", extra=log_extra)

# --- FastAPI Application Instance & Middlewares ---
app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
)
app.add_middleware(PrometheusMiddleware) # Exposes /metrics

@app.middleware("http")
async def base_middleware(request: Request, call_next: Callable[[Request], Coroutine[Any, Any, Any]]) -> Any:
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    
    # Attach request_id to request.state for access in routes/dependencies
    request.state.request_id = request_id

    # For logging, pass 'extra' dict. The logger format includes '%(request_id)s'.
    # We need a way for the logger to access this. Using a filter is more robust.
    # For now, rely on passing 'extra' in each log call within endpoints.

    # Rate Limiting
    client_ip = request.client.host if request.client else "unknown_client"
    current_time = time.time()
    
    # Clean up old timestamps
    if client_ip in request_counts: # Ensure key exists before list comp
        request_counts[client_ip] = [t for t in request_counts.get(client_ip, []) if t > current_time - settings.RATE_LIMIT_WINDOW_SECONDS]
    
    if len(request_counts.get(client_ip, [])) >= settings.RATE_LIMIT_REQUESTS:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}", extra={'request_id': request_id})
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status_code=429).inc()
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Too many requests, please try again later.", "request_id": request_id}
        )
    request_counts.setdefault(client_ip, []).append(current_time)

    start_time_metric = time.monotonic()
    
    try:
        response = await call_next(request)
    except Exception as exc: # Catch unhandled exceptions from routes to ensure metrics/headers
        duration_metric = time.monotonic() - start_time_metric
        logger.error(
            f"Unhandled exception during request processing: {request.method} {request.url.path} - Error: {exc}",
            exc_info=True, extra={'request_id': request_id}
        )
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status_code=500).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(duration_metric)
        # Let the global_exception_handler format the JSON response
        raise # Re-raise to be caught by global_exception_handler or FastAPI's default.

    duration_metric = time.monotonic() - start_time_metric

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none';"
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    REQUEST_COUNT.labels(method=request.method, endpoint=str(request.url.path), status_code=response.status_code).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=str(request.url.path)).observe(duration_metric)
    
    logger.info(
        f"Request: {request.method} {request.url.path} - Response: {response.status_code} - Duration: {duration_metric:.4f}s",
        extra={'request_id': request_id}
    )
    return response

# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())) # Ensure request_id is available
    logger.error(f"Global unhandled exception: {exc}", exc_info=True, extra={'request_id': request_id})
    
    # Ensure metric is counted if not already by middleware for some specific exception paths
    # Typically, middleware would catch it before this, but as a fallback:
    # REQUEST_COUNT.labels(method=request.method, endpoint=str(request.url.path), status_code=500).inc()

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "request_id": request_id,
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please contact support.",
            "detail": str(exc) if settings.LOG_LEVEL.upper() == "DEBUG" else None
        }
    )

# --- API Endpoints ---
@app.get("/", tags=["Utility"], summary="Root endpoint providing a welcome message.")
async def root(request: Request):
    request_id = getattr(request.state, 'request_id', "N/A")
    logger.info("Root endpoint accessed.", extra={'request_id': request_id})
    return {
        "message": f"Welcome to {settings.APP_TITLE} v{settings.APP_VERSION}",
        "docs_url": str(request.url.replace(path="/docs")),
        "openapi_url": str(request.url.replace(path="/openapi.json")),
        "analyzer_status": "Initialized" if analyzer_instance else "Not Initialized"
    }

app.add_route("/metrics", handle_metrics, methods=["GET"], tags=["Utility"], summary="Prometheus metrics endpoint.")

@app.get("/modules", response_model=List[ModuleInfo], tags=["Modules"], summary="List all available scoring modules in brain.py.")
async def list_available_modules(request: Request, api_key: APIKey = Depends(get_api_key)):
    request_id = getattr(request.state, 'request_id', "N/A")
    logger.info("Listing available brain modules.", extra={'request_id': request_id})
    modules_info: List[ModuleInfo] = []
    if hasattr(brain, 'REGISTERED_MODULES_BRAIN'):
        for module_name in brain.REGISTERED_MODULES_BRAIN.keys():
            # In a real scenario, brain module might have a way to provide descriptions
            modules_info.append(ModuleInfo(name=module_name, description=f"Details for {module_name} (if available in brain.py)"))
    return modules_info

@app.post("/analyze", response_model=AnalyzeBoardApiResponse, tags=["Analysis Engine"], 
          summary="Perform comprehensive board analysis using Analyzer.")
async def analyze_board_endpoint(
    payload: AnalyzeBoardApiRequest,
    request: Request, # To get request.state.request_id
    api_key: APIKey = Depends(get_api_key) # Ensures API key authentication
):
    request_id = getattr(request.state, 'request_id', "N/A") # Get request_id from middleware
    log_extra = {'request_id': request_id, 'client_request_id': payload.client_request_id}
    logger.info(f"Received request for /analyze.", extra=log_extra)

    if analyzer_instance is None:
        logger.error("Analyzer instance not available for /analyze.", extra=log_extra)
        ANALYZER_REQUEST_COUNT.inc() # Count attempt
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                            detail="Analysis service is temporarily unavailable due to initialization failure.")
    
    start_time = time.monotonic()
    try:
        analysis_result_dict = await analyzer_instance.analyze_board(
            new_card=payload.new_card,
            proposed_values=payload.proposed_values,
            active_modules=payload.active_modules,
            module_weights=payload.module_weights,
            top_n=payload.top_n,
            request_id_for_logging=request_id # Pass the API's request_id to analyzer
        )
        duration = time.monotonic() - start_time
        ANALYZER_REQUEST_LATENCY.observe(duration)
        ANALYZER_REQUEST_COUNT.inc()
        logger.info(f"/analyze call successful. Duration: {duration:.4f}s", extra=log_extra)
        
        # Adapt analyzer_result_dict to AnalyzeBoardApiResponse
        # The 'error' key in analysis_result_dict indicates an error *within* analyzer (e.g. invalid input handled by analyzer)
        if analysis_result_dict.get('error') and isinstance(analysis_result_dict.get('error'), str) : # Analyzer handled an input error.
            return AnalyzeBoardApiResponse(
                error=analysis_result_dict['error'], # Error message from analyzer
                visualization=analysis_result_dict.get('visualization'), # Error visualization
                board_dimensions=BoardDimensions(**analysis_result_dict.get('board_dimensions', {'rows':0, 'cols':0})),
                processed_params=ProcessedParams(**analysis_result_dict.get('processed_params', {})), # Empty if error early
                request_id=request_id
            )

        # Ensure keys exist before unpacking for Pydantic models
        suggestions_raw = analysis_result_dict.get('suggestions', {})
        suggestions_typed: Dict[int, List[SuggestionItem]] = {}
        if isinstance(suggestions_raw, dict):
            for pv_key, sugg_list_raw in suggestions_raw.items():
                try:
                    pv_int_key = int(pv_key) # Ensure key is int
                    suggestions_typed[pv_int_key] = [SuggestionItem(**sugg) for sugg in sugg_list_raw]
                except (ValueError, TypeError) as e:
                     logger.warning(f"Could not parse suggestion for PV key '{pv_key}': {e}", extra=log_extra)


        return AnalyzeBoardApiResponse(
            suggestions=suggestions_typed,
            visualization=analysis_result_dict.get('visualization'),
            board_dimensions=BoardDimensions(**analysis_result_dict.get('board_dimensions', {'rows':0, 'cols':0})),
            processed_params=ProcessedParams(**analysis_result_dict.get('processed_params', {})),
            request_id=request_id
        )

    except InvalidInputError as e_analyzer_invalid_input: # Errors from analyzer's _validate_inputs
        duration = time.monotonic() - start_time
        ANALYZER_REQUEST_LATENCY.observe(duration) # Observe latency even for handled errors
        ANALYZER_REQUEST_COUNT.inc()
        logger.warning(f"Invalid input for /analyze, caught from Analyzer: {e_analyzer_invalid_input}", extra=log_extra, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=AnalyzeBoardApiResponse(
                error=f"Invalid Input Parameters: {str(e_analyzer_invalid_input)}",
                request_id=request_id
            ).model_dump(exclude_none=True) # Use model_dump for Pydantic v2
        )
    except ModuleError as e_analyzer_module_error: # Errors related to brain modules from Analyzer
        duration = time.monotonic() - start_time
        ANALYZER_REQUEST_LATENCY.observe(duration)
        ANALYZER_REQUEST_COUNT.inc()
        logger.error(f"Module error during /analyze from Analyzer: {e_analyzer_module_error}", extra=log_extra, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=AnalyzeBoardApiResponse(
                error=f"Module Error during analysis ({type(e_analyzer_module_error).__name__}): {str(e_analyzer_module_error)}",
                request_id=request_id
            ).model_dump(exclude_none=True)
        )
    except Exception as e_general: # Catch any other unexpected error from analyzer or this endpoint
        duration = time.monotonic() - start_time
        ANALYZER_REQUEST_LATENCY.observe(duration)
        ANALYZER_REQUEST_COUNT.inc() # Count as an attempt
        logger.critical(f"Unexpected critical error during /analyze: {e_general}", extra=log_extra, exc_info=True)
        # Let the global exception handler manage this response for consistency
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected internal server error during analysis: {str(e_general)}")


# Endpoints for direct brain module scoring (from original main_fastapi.py)
@app.post("/score/{module_name}", response_model=TaskAcceptedResponse,
          status_code=status.HTTP_202_ACCEPTED, tags=["Scoring (Direct Brain Module)"], 
          summary="Submit a single grid for direct brain module scoring (background task).")
async def score_grid_background(
    request: Request, # To get request.state.request_id
    payload: GridInput, # Reusing GridInput which has grid_data and client_request_id
    module_name: str = Path(..., description="The name of the scoring module in brain.py to use."),
    background_tasks: BackgroundTasks = Depends(), # FastAPI injects this
    api_key: APIKey = Depends(get_api_key)
):
    req_id = getattr(request.state, 'request_id', "N/A")
    client_req_id = payload.client_request_id
    task_id = str(uuid.uuid4())
    log_extra = {'request_id': req_id, 'task_id': task_id, 
                 'module_name': module_name, 'client_request_id': client_req_id}

    if module_name not in brain.REGISTERED_MODULES_BRAIN:
        logger.warning(f"Module '{module_name}' not found for direct scoring task.", extra=log_extra)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Module '{module_name}' not found in brain.py.")

    background_tasks.add_task(
        run_scoring_task,
        task_id=task_id,
        module_name=module_name,
        grid_data=payload.grid_data,
        original_request_id=req_id,
        client_request_id=client_req_id
    )
    MODULE_USAGE_COUNT.labels(module_name=module_name).inc()
    logger.info("Direct brain module scoring task enqueued.", extra=log_extra)
    return TaskAcceptedResponse(
        task_id=task_id,
        message=f"Scoring task for module '{module_name}' accepted (direct brain call) and is being processed in the background.",
        client_request_id=client_req_id
    )

@app.post("/score/batch", response_model=List[TaskAcceptedResponse],
          status_code=status.HTTP_202_ACCEPTED, tags=["Scoring (Direct Brain Module)"], 
          summary="Submit multiple grids for batch direct brain module scoring (background tasks).")
async def score_batch_grid_background(
    request: Request, # To get request.state.request_id
    payload: BatchGridInput,
    background_tasks: BackgroundTasks = Depends(),
    api_key: APIKey = Depends(get_api_key)
):
    req_id = getattr(request.state, 'request_id', "N/A")
    client_req_id = payload.client_request_id
    responses: List[TaskAcceptedResponse] = []
    log_extra_batch = {'request_id': req_id, 'batch_size': len(payload.grids), 'client_request_id': client_req_id}
    logger.info("Batch direct brain module scoring task received.", extra=log_extra_batch)

    for item in payload.grids:
        task_id = str(uuid.uuid4())
        log_extra_item = {**log_extra_batch, 'task_id': task_id, 'item_id': item.item_id, 'module_name': item.module_name}
        
        if item.module_name not in brain.REGISTERED_MODULES_BRAIN:
            logger.warning(f"Module '{item.module_name}' not found for batch item ID '{item.item_id}'.", extra=log_extra_item)
            responses.append(TaskAcceptedResponse(
                task_id=f"error_invalid_module_{item.item_id}",
                status="rejected",
                message=f"Module '{item.module_name}' for item_id '{item.item_id}' not found.",
                client_request_id=client_req_id
            ))
            continue

        background_tasks.add_task(
            run_scoring_task,
            task_id=task_id,
            module_name=item.module_name,
            grid_data=item.grid_data,
            original_request_id=req_id,
            client_request_id=client_req_id
        )
        MODULE_USAGE_COUNT.labels(module_name=item.module_name).inc()
        responses.append(TaskAcceptedResponse(
            task_id=task_id,
            message=f"Scoring task for item_id '{item.item_id}' (module '{item.module_name}') accepted (direct brain call).",
            client_request_id=client_req_id
        ))
        logger.info("Batch item for direct brain module scoring enqueued.", extra=log_extra_item)
    return responses


# --- Main Execution Block ---
if __name__ == "__main__":
    import uvicorn
    # Setup a simple StreamHandler for the logger if no handlers are configured by basicConfig
    # This is mostly for when running uvicorn programmatically without its own log config.
    if not logger.hasHandlers():
        _stdout_handler = logging.StreamHandler()
        _stdout_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - RequestID: %(request_id)s - %(message)s',
            '%Y-%m-%d %H:%M:%S'
        ))
        # Add a filter to inject request_id if not present (though middleware should handle for API calls)
        class RequestIDLogFilter(logging.Filter):
            def filter(self, record):
                record.request_id = getattr(record, 'request_id', 'system')
                return True
        logger.addFilter(RequestIDLogFilter())
        logger.addHandler(_stdout_handler)


    logger.info(f"Starting {settings.APP_TITLE} v{settings.APP_VERSION} on {settings.APP_HOST}:{settings.APP_PORT}", extra={'request_id': 'startup'})
    logger.info(f"Default API Key: {settings.API_KEY[:4]}... (Ensure this is changed for production!)", extra={'request_id': 'startup'})
    logger.info(f"Rate Limiting: {settings.RATE_LIMIT_REQUESTS} requests per {settings.RATE_LIMIT_WINDOW_SECONDS} seconds (in-memory).", extra={'request_id': 'startup'})
    if settings.TASK_CALLBACK_URL_ENABLED:
        logger.info(f"Task callback enabled, will attempt to POST to: {settings.TASK_CALLBACK_URL}", extra={'request_id': 'startup'})
    else:
        logger.info("Task callback is disabled. Background task results will only be logged.", extra={'request_id': 'startup'})
    
    if analyzer_instance is None:
        logger.error("ANALYZER NOT INITIALIZED. API WILL HAVE LIMITED FUNCTIONALITY (/analyze endpoint will fail).", extra={'request_id': 'startup'})
    else:
        logger.info("Analyzer instance is initialized and available.", extra={'request_id': 'startup'})

    # Use Uvicorn's standard logging config by default if not specified otherwise
    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT)