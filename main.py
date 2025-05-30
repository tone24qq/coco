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
# Corrected import for starlette_prometheus:
# PrometheusMiddleware will expose /metrics by default.
# No need to import handle_metrics if it's causing an error,
# unless a very specific version or custom routing is intended.
from starlette_prometheus import PrometheusMiddleware # type: ignore

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
    APP_VERSION: str = "2.1.1" # Updated version to reflect fix

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
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - RequestID: %(request_id)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# --- Analyzer Instance ---
analyzer_instance: Optional[Analyzer] = None
try:
    analyzer_instance = Analyzer(main_module=brain, default_top_n=3) # [cite: 1]
    logger.info(f"Analyzer instance created successfully with 'brain' module. Default top_n: {analyzer_instance.default_top_n}") # [cite: 3]
except InitializationError as e_init:
    logger.critical("CRITICAL_API_STARTUP_ERROR: Failed to initialize Analyzer: %s", e_init, exc_info=True) # [cite: 3]
    analyzer_instance = None
except Exception as e_unexpected:
    logger.critical("CRITICAL_API_STARTUP_ERROR: Unexpected error during Analyzer initialization: %s", e_unexpected, exc_info=True) # [cite: 3]
    analyzer_instance = None


# --- Prometheus Metrics Definition ---
REQUEST_COUNT = Counter(
    "api_request_count",
    "Total number of API requests processed", # [cite: 3]
    ["method", "endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds", # [cite: 4]
    ["method", "endpoint"]
)
ACTIVE_BACKGROUND_TASKS = Gauge(
    "api_active_background_tasks",
    "Number of currently active background scoring tasks" # [cite: 4]
)
MODULE_USAGE_COUNT = Counter(
    "api_module_usage_count",
    "Count of how many times each scoring module is used (direct calls)", # [cite: 4]
    ["module_name"]
)
ANALYZER_REQUEST_COUNT = Counter(
    "api_analyzer_request_count",
    "Total number of full analysis requests processed via /analyze" # [cite: 5]
)
ANALYZER_REQUEST_LATENCY = Histogram(
    "api_analyzer_request_latency_seconds",
    "Full analysis request latency via /analyze in seconds" # [cite: 5]
)


# --- API Key Authentication ---
api_key_query = APIKeyQuery(name=settings.API_KEY_NAME, auto_error=False) # [cite: 5]
api_key_header = APIKeyHeader(name=settings.API_KEY_NAME, auto_error=False) # [cite: 5]

async def get_api_key(
    key_query: Optional[str] = Security(api_key_query),
    key_header: Optional[str] = Security(api_key_header),
) -> str:
    if key_query == settings.API_KEY: # [cite: 5]
        return key_query
    if key_header == settings.API_KEY: # [cite: 5]
        return key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, # [cite: 5]
        detail="Invalid or missing API Key" # [cite: 5]
    )

# --- Pydantic Models for API ---

class GridDataBase(BaseModel):
    grid_data: List[List[Union[int, float]]] = Field(..., example=[[-1, 1, -1], [2, -1, 3], [-1, 4, -1]]) # [cite: 5]

    @field_validator('grid_data')
    def validate_grid_data(cls, v: List[List[Union[int, float]]]) -> List[List[Union[int, float]]]:
        if not v or not all(isinstance(row, list) for row in v): # [cite: 5, 6]
            raise ValueError("Grid data must be a non-empty list of lists.") # [cite: 6]
        
        if not any(v): # [cite: 7]
            if not all(not row for row in v): # [cite: 7]
                 raise ValueError("Grid data has inconsistent empty/non-empty rows.") # [cite: 7]
            return v # [cite: 7]

        if v[0]: # [cite: 8]
            num_cols = len(v[0]) # [cite: 8]
            if not all(len(row) == num_cols for row in v): # [cite: 8]
                raise ValueError("All non-empty rows must have the same number of columns.") # [cite: 8]
        else: 
            if not all(not row for row in v): # [cite: 8]
                 raise ValueError("If first row is empty, all rows must be empty for an Nx0 grid.") # [cite: 8]

        for r_idx, row in enumerate(v): # [cite: 8]
            for c_idx, cell_val in enumerate(row): # [cite: 8]
                if not isinstance(cell_val, (int, float)): # [cite: 8]
                    raise ValueError(f"Cell ({r_idx}, {c_idx}) type invalid: {type(cell_val)}. Must be number.") # [cite: 8]
        return v # [cite: 8]

class GridInput(GridDataBase):
    client_request_id: Optional[str] = Field(None, description="Optional client-provided request ID for tracing.") # [cite: 8]

class BatchGridItem(GridDataBase):
    item_id: str = Field(description="Unique identifier for this item in the batch.") # [cite: 8, 9]
    module_name: str = Field(description="Scoring module to use for this item.") # [cite: 9]

class BatchGridInput(BaseModel):
    grids: List[BatchGridItem] = Field(..., max_items=50) # [cite: 9]
    client_request_id: Optional[str] = Field(None, description="Optional client-provided request ID for the batch.") # [cite: 9]

class ScoreOutput(BaseModel): 
    module_name: str # [cite: 9]
    score_grid: List[List[float]] # [cite: 9]
    message: Optional[str] = None # [cite: 9]
    error: Optional[str] = None # [cite: 9]

class TaskAcceptedResponse(BaseModel):
    task_id: str # [cite: 9]
    status: str = "accepted" # [cite: 9]
    message: str # [cite: 9]
    client_request_id: Optional[str] = None # [cite: 9]

class ModuleInfo(BaseModel):
    name: str # [cite: 9]
    description: Optional[str] = "No description available." # [cite: 9]
    version: Optional[str] = "N/A" # [cite: 9]

class AnalyzeBoardApiRequest(BaseModel):
    new_card: List[List[int]] = Field(..., example=[[1, -1, 0], [-1, 2, -1]]) # [cite: 10]
    proposed_values: List[int] = Field(..., example=[3, 5]) # [cite: 10]
    active_modules: Optional[List[str]] = Field(None, example=["GM1_Random", "GM2_TargetTopLeft"]) # [cite: 10]
    module_weights: Optional[Dict[str, float]] = Field(None, example={"GM1_Random": 0.5, "GM2_TargetTopLeft": 1.5}) # [cite: 10]
    top_n: Optional[int] = Field(None, example=5, gt=0) # [cite: 10]
    client_request_id: Optional[str] = Field(None, description="Optional client-provided request ID for tracing.") # [cite: 10]

class SuggestionItem(BaseModel):
    position: List[int] = Field(..., example=[0,1]) # [cite: 10]
    score: float = Field(..., example=0.75) # [cite: 10]

class ProcessedParams(BaseModel):
    requested_top_n: Union[int, str] # [cite: 10]
    actual_top_n: int # [cite: 10]
    requested_active_modules: Optional[List[str]] # [cite: 10]
    effective_active_modules: List[str] # [cite: 10]
    requested_module_weights: Optional[Dict[str, float]] # [cite: 10]
    final_module_weights: Dict[str, float] # [cite: 10]
    request_id: str # [cite: 10]
    error: Optional[bool] = None # [cite: 10]

class BoardDimensions(BaseModel):
    rows: int # [cite: 10]
    cols: int # [cite: 10]

class AnalyzeBoardApiResponse(BaseModel):
    suggestions: Optional[Dict[int, List[SuggestionItem]]] = None # [cite: 10]
    visualization: Optional[str] = None # [cite: 10, 11]
    board_dimensions: Optional[BoardDimensions] = None # [cite: 11]
    processed_params: Optional[ProcessedParams] = None # [cite: 11]
    error: Optional[str] = None # [cite: 12]
    request_id: str # [cite: 12]

request_counts: Dict[str, List[float]] = {} # [cite: 12]

async def run_scoring_task(
    task_id: str, # [cite: 12]
    module_name: str, # [cite: 12]
    grid_data: List[List[Union[int, float]]], # [cite: 13]
    original_request_id: str, # [cite: 13]
    client_request_id: Optional[str] = None # [cite: 13]
):
    ACTIVE_BACKGROUND_TASKS.inc() # [cite: 13]
    log_extra = {'request_id': original_request_id, 'task_id': task_id, 'module_name': module_name} # [cite: 13]
    logger.info("Background task started for direct brain module scoring.", extra=log_extra) # [cite: 13]
    
    try:
        np_grid = np.array(grid_data) # [cite: 17]
        if np_grid.size == 0: # [cite: 18]
            raise ValueError("Input grid is empty after numpy conversion.") # [cite: 18]

        start_time = time.monotonic() # [cite: 18]
        brain_kwargs = {'request_id': task_id} # [cite: 18]
        
        score_np_array = await asyncio.to_thread(brain.get_module_score, module_name, np_grid, **brain_kwargs) # [cite: 18]
        duration = time.monotonic() - start_time # [cite: 18]
        
        result_message = f"Direct brain scoring successful for module {module_name}." # [cite: 18]
        logger.info(result_message + f" Duration: {duration:.4f}s", extra=log_extra) # [cite: 18]

        if settings.TASK_CALLBACK_URL_ENABLED and settings.TASK_CALLBACK_URL: # [cite: 18]
            logger.info(f"Simulating callback to {settings.TASK_CALLBACK_URL} with result for task {task_id}.", extra=log_extra) # [cite: 18]
            # Example callback payload structure
            # callback_payload = {
            #     "task_id": task_id, 
            #     "status": "completed", 
            #     "result": score_np_array.tolist(), # Or some other representation
            #     "client_request_id": client_request_id,
            #     "original_request_id": original_request_id
            # }
            # Example usage of httpx (would need to be installed and imported)
            # import httpx 
            # async with httpx.AsyncClient() as client: # [cite: 19]
            #     try:
            #         await client.post(str(settings.TASK_CALLBACK_URL), json=callback_payload) # [cite: 19]
            #     except httpx.RequestError as exc_httpx:
            #         logger.error(f"Callback failed for task {task_id}: {exc_httpx}", extra=log_extra)

    except Exception as e: # [cite: 20]
        error_message = f"Error in background direct brain scoring task for module {module_name}: {str(e)}" # [cite: 20]
        logger.error(error_message, exc_info=True, extra=log_extra) # [cite: 20]
    finally:
        ACTIVE_BACKGROUND_TASKS.dec() # [cite: 20]
        logger.info("Background task for direct brain module scoring finished.", extra=log_extra) # [cite: 20]

app = FastAPI(
    title=settings.APP_TITLE, # [cite: 20]
    description=settings.APP_DESCRIPTION, # [cite: 20]
    version=settings.APP_VERSION, # [cite: 20]
)
app.add_middleware(PrometheusMiddleware) # [cite: 20]

@app.middleware("http")
async def base_middleware(request: Request, call_next: Callable[[Request], Coroutine[Any, Any, Any]]) -> Any: # [cite: 20]
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4()) # [cite: 20]
    request.state.request_id = request_id # [cite: 20]

    client_ip = request.client.host if request.client else "unknown_client" # [cite: 23]
    current_time = time.time() # [cite: 23]
    
    if client_ip in request_counts: # [cite: 23]
        request_counts[client_ip] = [t for t in request_counts.get(client_ip, []) if t > current_time - settings.RATE_LIMIT_WINDOW_SECONDS] # [cite: 23]
    
    if len(request_counts.get(client_ip, [])) >= settings.RATE_LIMIT_REQUESTS: # [cite: 23]
        logger.warning(f"Rate limit exceeded for IP: {client_ip}", extra={'request_id': request_id}) # [cite: 23]
        REQUEST_COUNT.labels(method=request.method, endpoint=str(request.url.path), status_code=429).inc() # Corrected: ensure endpoint is str
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, # [cite: 23]
            content={"detail": "Too many requests, please try again later.", "request_id": request_id} # [cite: 23]
        )
    request_counts.setdefault(client_ip, []).append(current_time) # [cite: 23]

    start_time_metric = time.monotonic() # [cite: 24]
    
    try:
        response = await call_next(request) # [cite: 24]
    except Exception as exc: # [cite: 24]
        duration_metric = time.monotonic() - start_time_metric # [cite: 24]
        logger.error(
            f"Unhandled exception during request processing: {request.method} {request.url.path} - Error: {exc}", # [cite: 24]
            exc_info=True, extra={'request_id': request_id}
        )
        REQUEST_COUNT.labels(method=request.method, endpoint=str(request.url.path), status_code=500).inc() # [cite: 24] Corrected: ensure endpoint is str
        REQUEST_LATENCY.labels(method=request.method, endpoint=str(request.url.path)).observe(duration_metric) # [cite: 24] Corrected: ensure endpoint is str
        raise # [cite: 24]

    duration_metric = time.monotonic() - start_time_metric # [cite: 24]

    response.headers["X-Request-ID"] = request_id # [cite: 24]
    response.headers["X-Content-Type-Options"] = "nosniff" # [cite: 24]
    response.headers["X-Frame-Options"] = "DENY" # [cite: 24]
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none';" # [cite: 24, 25]
    if request.url.scheme == "https": # [cite: 25]
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains" # [cite: 25]

    REQUEST_COUNT.labels(method=request.method, endpoint=str(request.url.path), status_code=response.status_code).inc() # [cite: 25]
    REQUEST_LATENCY.labels(method=request.method, endpoint=str(request.url.path)).observe(duration_metric) # [cite: 25]
    
    logger.info(
        f"Request: {request.method} {request.url.path} - Response: {response.status_code} - Duration: {duration_metric:.4f}s", # [cite: 25]
        extra={'request_id': request_id}
    )
    return response # [cite: 25]

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception): # [cite: 25]
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())) # [cite: 25]
    logger.error(f"Global unhandled exception: {exc}", exc_info=True, extra={'request_id': request_id}) # [cite: 26]
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # [cite: 26]
        content={ # [cite: 26]
            "request_id": request_id, # [cite: 26]
            "error": "Internal Server Error", # [cite: 26]
            "message": "An unexpected error occurred. Please contact support.", # [cite: 26]
            "detail": str(exc) if settings.LOG_LEVEL.upper() == "DEBUG" else None # [cite: 26]
        }
    )

@app.get("/", tags=["Utility"], summary="Root endpoint providing a welcome message.")
async def root(request: Request): # [cite: 26]
    request_id = getattr(request.state, 'request_id', "N/A") # [cite: 26]
    logger.info("Root endpoint accessed.", extra={'request_id': request_id}) # [cite: 26]
    return { # [cite: 26]
        "message": f"Welcome to {settings.APP_TITLE} v{settings.APP_VERSION}", # [cite: 26]
        "docs_url": str(request.url.replace(path="/docs")), # [cite: 26]
        "openapi_url": str(request.url.replace(path="/openapi.json")), # [cite: 26]
        "analyzer_status": "Initialized" if analyzer_instance else "Not Initialized" # [cite: 26]
    }

# The PrometheusMiddleware will add the /metrics endpoint by default.
# Explicitly adding it via app.add_route can be removed if not customizing.
# For simplicity and to avoid potential conflicts with `handle_metrics` not being found:
# app.add_route("/metrics", handle_metrics, methods=["GET"], tags=["Utility"], summary="Prometheus metrics endpoint.") [cite: 26]
# This line is removed. The middleware itself should expose it.

@app.get("/modules", response_model=List[ModuleInfo], tags=["Modules"], summary="List all available scoring modules in brain.py.")
async def list_available_modules(request: Request, api_key: APIKey = Depends(get_api_key)): # [cite: 27]
    request_id = getattr(request.state, 'request_id', "N/A") # [cite: 27]
    logger.info("Listing available brain modules.", extra={'request_id': request_id}) # [cite: 27]
    modules_info: List[ModuleInfo] = [] # [cite: 27]
    if hasattr(brain, 'REGISTERED_MODULES_BRAIN'): # [cite: 27]
        for module_name in brain.REGISTERED_MODULES_BRAIN.keys(): # [cite: 27, 28]
            modules_info.append(ModuleInfo(name=module_name, description=f"Details for {module_name} (if available in brain.py)")) # [cite: 28]
    return modules_info # [cite: 28]

@app.post("/analyze", response_model=AnalyzeBoardApiResponse, tags=["Analysis Engine"], 
          summary="Perform comprehensive board analysis using Analyzer.")
async def analyze_board_endpoint(
    payload: AnalyzeBoardApiRequest, # [cite: 29]
    request: Request, # [cite: 29]
    api_key: APIKey = Depends(get_api_key) # [cite: 29]
):
    request_id = getattr(request.state, 'request_id', "N/A") # [cite: 29]
    log_extra = {'request_id': request_id, 'client_request_id': payload.client_request_id} # [cite: 29]
    logger.info(f"Received request for /analyze.", extra=log_extra) # [cite: 29]

    if analyzer_instance is None: # [cite: 29]
        logger.error("Analyzer instance not available for /analyze.", extra=log_extra) # [cite: 29]
        ANALYZER_REQUEST_COUNT.inc() # [cite: 29]
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, # [cite: 29]
                            detail="Analysis service is temporarily unavailable due to initialization failure.") # [cite: 29]
    
    start_time = time.monotonic() # [cite: 29]
    try:
        analysis_result_dict = await analyzer_instance.analyze_board( # [cite: 29]
            new_card=payload.new_card, # [cite: 29]
            proposed_values=payload.proposed_values, # [cite: 29]
            active_modules=payload.active_modules, # [cite: 29]
            module_weights=payload.module_weights, # [cite: 29]
            top_n=payload.top_n, # [cite: 29]
            request_id_for_logging=request_id # [cite: 29]
        )
        duration = time.monotonic() - start_time # [cite: 29]
        ANALYZER_REQUEST_LATENCY.observe(duration) # [cite: 29]
        ANALYZER_REQUEST_COUNT.inc() # [cite: 29]
        logger.info(f"/analyze call successful. Duration: {duration:.4f}s", extra=log_extra) # [cite: 29, 30]
        
        if analysis_result_dict.get('error') and isinstance(analysis_result_dict.get('error'), str): # [cite: 30]
            return AnalyzeBoardApiResponse( # [cite: 30]
                error=analysis_result_dict['error'], # [cite: 30]
                visualization=analysis_result_dict.get('visualization'), # [cite: 30]
                board_dimensions=BoardDimensions(**analysis_result_dict.get('board_dimensions', {'rows':0, 'cols':0})), # [cite: 30]
                processed_params=ProcessedParams(**analysis_result_dict.get('processed_params', {})), # [cite: 30]
                request_id=request_id # [cite: 31]
            )

        suggestions_raw = analysis_result_dict.get('suggestions', {}) # [cite: 31]
        suggestions_typed: Dict[int, List[SuggestionItem]] = {} # [cite: 31]
        if isinstance(suggestions_raw, dict): # [cite: 31]
            for pv_key, sugg_list_raw in suggestions_raw.items(): # [cite: 31]
                try:
                    pv_int_key = int(pv_key) # [cite: 31]
                    suggestions_typed[pv_int_key] = [SuggestionItem(**sugg) for sugg in sugg_list_raw] # [cite: 31]
                except (ValueError, TypeError) as e: # [cite: 31]
                     logger.warning(f"Could not parse suggestion for PV key '{pv_key}': {e}", extra=log_extra) # [cite: 31]

        return AnalyzeBoardApiResponse( # [cite: 31]
            suggestions=suggestions_typed, # [cite: 31]
            visualization=analysis_result_dict.get('visualization'), # [cite: 31]
            board_dimensions=BoardDimensions(**analysis_result_dict.get('board_dimensions', {'rows':0, 'cols':0})), # [cite: 31]
            processed_params=ProcessedParams(**analysis_result_dict.get('processed_params', {})), # [cite: 31]
            request_id=request_id # [cite: 31]
        )

    except InvalidInputError as e_analyzer_invalid_input: # [cite: 31]
        duration = time.monotonic() - start_time # [cite: 31]
        ANALYZER_REQUEST_LATENCY.observe(duration) # [cite: 31]
        ANALYZER_REQUEST_COUNT.inc() # [cite: 31]
        logger.warning(f"Invalid input for /analyze, caught from Analyzer: {e_analyzer_invalid_input}", extra=log_extra, exc_info=True) # [cite: 31]
        return JSONResponse( # [cite: 31]
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, # [cite: 31]
            content=AnalyzeBoardApiResponse( # [cite: 31]
                error=f"Invalid Input Parameters: {str(e_analyzer_invalid_input)}", # [cite: 31]
                request_id=request_id # [cite: 31]
            ).model_dump(exclude_none=True) # [cite: 31]
        )
    except ModuleError as e_analyzer_module_error: # [cite: 31, 32]
        duration = time.monotonic() - start_time # [cite: 32]
        ANALYZER_REQUEST_LATENCY.observe(duration) # [cite: 32]
        ANALYZER_REQUEST_COUNT.inc() # [cite: 32]
        logger.error(f"Module error during /analyze from Analyzer: {e_analyzer_module_error}", extra=log_extra, exc_info=True) # [cite: 33]
        return JSONResponse( # [cite: 33]
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # [cite: 33]
            content=AnalyzeBoardApiResponse( # [cite: 33]
                error=f"Module Error during analysis ({type(e_analyzer_module_error).__name__}): {str(e_analyzer_module_error)}", # [cite: 33]
                request_id=request_id # [cite: 33]
            ).model_dump(exclude_none=True) # [cite: 33]
        )
    except Exception as e_general: # [cite: 33]
        duration = time.monotonic() - start_time # [cite: 33]
        ANALYZER_REQUEST_LATENCY.observe(duration) # [cite: 33]
        ANALYZER_REQUEST_COUNT.inc() # [cite: 33]
        logger.critical(f"Unexpected critical error during /analyze: {e_general}", extra=log_extra, exc_info=True) # [cite: 33]
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected internal server error during analysis: {str(e_general)}") # [cite: 33]

@app.post("/score/{module_name}", response_model=TaskAcceptedResponse,
          status_code=status.HTTP_202_ACCEPTED, tags=["Scoring (Direct Brain Module)"], 
          summary="Submit a single grid for direct brain module scoring (background task).")
async def score_grid_background(
    request: Request, # [cite: 33]
    payload: GridInput, # [cite: 33, 34]
    module_name: str = Path(..., description="The name of the scoring module in brain.py to use."), # [cite: 34]
    background_tasks: BackgroundTasks = Depends(), # [cite: 34]
    api_key: APIKey = Depends(get_api_key) # [cite: 34]
):
    req_id = getattr(request.state, 'request_id', "N/A") # [cite: 34]
    client_req_id = payload.client_request_id # [cite: 34]
    task_id = str(uuid.uuid4()) # [cite: 34]
    log_extra = {'request_id': req_id, 'task_id': task_id, 'module_name': module_name, 'client_request_id': client_req_id} # [cite: 34]

    if module_name not in brain.REGISTERED_MODULES_BRAIN: # [cite: 34, 35]
        logger.warning(f"Module '{module_name}' not found for direct scoring task.", extra=log_extra) # [cite: 35]
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Module '{module_name}' not found in brain.py.") # [cite: 35]

    background_tasks.add_task( # [cite: 35]
        run_scoring_task, # [cite: 35]
        task_id=task_id, # [cite: 35, 36]
        module_name=module_name, # [cite: 36]
        grid_data=payload.grid_data, # [cite: 36]
        original_request_id=req_id, # [cite: 36]
        client_request_id=client_req_id # [cite: 36]
    )
    MODULE_USAGE_COUNT.labels(module_name=module_name).inc() # [cite: 36]
    logger.info("Direct brain module scoring task enqueued.", extra=log_extra) # [cite: 36]
    return TaskAcceptedResponse( # [cite: 36]
        task_id=task_id, # [cite: 36]
        message=f"Scoring task for module '{module_name}' accepted (direct brain call) and is being processed in the background.", # [cite: 36]
        client_request_id=client_req_id # [cite: 36]
    )

@app.post("/score/batch", response_model=List[TaskAcceptedResponse],
          status_code=status.HTTP_202_ACCEPTED, tags=["Scoring (Direct Brain Module)"], 
          summary="Submit multiple grids for batch direct brain module scoring (background tasks).")
async def score_batch_grid_background(
    request: Request, # [cite: 36]
    payload: BatchGridInput, # [cite: 36]
    background_tasks: BackgroundTasks = Depends(), # [cite: 36]
    api_key: APIKey = Depends(get_api_key) # [cite: 36]
):
    req_id = getattr(request.state, 'request_id', "N/A") # [cite: 36]
    client_req_id = payload.client_request_id # [cite: 36]
    responses: List[TaskAcceptedResponse] = [] # [cite: 36]
    log_extra_batch = {'request_id': req_id, 'batch_size': len(payload.grids), 'client_request_id': client_req_id} # [cite: 36]
    logger.info("Batch direct brain module scoring task received.", extra=log_extra_batch) # [cite: 36]

    for item in payload.grids: # [cite: 36]
        task_id = str(uuid.uuid4()) # [cite: 36]
        log_extra_item = {**log_extra_batch, 'task_id': task_id, 'item_id': item.item_id, 'module_name': item.module_name} # [cite: 36]
        
        if item.module_name not in brain.REGISTERED_MODULES_BRAIN: # [cite: 36]
            logger.warning(f"Module '{item.module_name}' not found for batch item ID '{item.item_id}'.", extra=log_extra_item) # [cite: 36, 37]
            responses.append(TaskAcceptedResponse( # [cite: 37]
                task_id=f"error_invalid_module_{item.item_id}", # [cite: 37]
                status="rejected", # [cite: 37]
                message=f"Module '{item.module_name}' for item_id '{item.item_id}' not found.", # [cite: 37]
                client_request_id=client_req_id # [cite: 37]
            ))
            continue # [cite: 37]

        background_tasks.add_task( # [cite: 38]
            run_scoring_task, # [cite: 38]
            task_id=task_id, # [cite: 38]
            module_name=item.module_name, # [cite: 38]
            grid_data=item.grid_data, # [cite: 38]
            original_request_id=req_id, # [cite: 38]
            client_request_id=client_req_id # [cite: 38]
        )
        MODULE_USAGE_COUNT.labels(module_name=item.module_name).inc() # [cite: 38]
        responses.append(TaskAcceptedResponse( # [cite: 38]
            task_id=task_id, # [cite: 38]
            message=f"Scoring task for item_id '{item.item_id}' (module '{item.module_name}') accepted (direct brain call).", # [cite: 38]
            client_request_id=client_req_id # [cite: 38]
        ))
        logger.info("Batch item for direct brain module scoring enqueued.", extra=log_extra_item) # [cite: 38]
    return responses # [cite: 38]

if __name__ == "__main__": # [cite: 38]
    import uvicorn # [cite: 38]
    if not logger.hasHandlers(): # [cite: 39]
        _stdout_handler = logging.StreamHandler() # [cite: 39]
        _stdout_handler.setFormatter(logging.Formatter( # [cite: 39]
            '%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - RequestID: %(request_id)s - %(message)s', # [cite: 39]
            '%Y-%m-%d %H:%M:%S' # [cite: 39]
        ))
        class RequestIDLogFilter(logging.Filter): # [cite: 39]
            def filter(self, record): # [cite: 39]
                record.request_id = getattr(record, 'request_id', 'system') # [cite: 39]
                return True # [cite: 39]
        logger.addFilter(RequestIDLogFilter()) # [cite: 39]
        logger.addHandler(_stdout_handler) # [cite: 39]

    logger.info(f"Starting {settings.APP_TITLE} v{settings.APP_VERSION} on {settings.APP_HOST}:{settings.APP_PORT}", extra={'request_id': 'startup'}) # [cite: 39]
    logger.info(f"Default API Key: {settings.API_KEY[:4]}... (Ensure this is changed for production!)", extra={'request_id': 'startup'}) # [cite: 39]
    logger.info(f"Rate Limiting: {settings.RATE_LIMIT_REQUESTS} requests per {settings.RATE_LIMIT_WINDOW_SECONDS} seconds (in-memory).", extra={'request_id': 'startup'}) # [cite: 40]
    if settings.TASK_CALLBACK_URL_ENABLED: # [cite: 40]
        logger.info(f"Task callback enabled, will attempt to POST to: {settings.TASK_CALLBACK_URL}", extra={'request_id': 'startup'}) # [cite: 40]
    else:
        logger.info("Task callback is disabled. Background task results will only be logged.", extra={'request_id': 'startup'}) # [cite: 40]
    
    if analyzer_instance is None: # [cite: 40]
        logger.error("ANALYZER NOT INITIALIZED. API WILL HAVE LIMITED FUNCTIONALITY (/analyze endpoint will fail).", extra={'request_id': 'startup'}) # [cite: 40]
    else:
        logger.info("Analyzer instance is initialized and available.", extra={'request_id': 'startup'}) # [cite: 40]

    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT) # [cite: 40]