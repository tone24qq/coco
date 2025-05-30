import asyncio
import datetime
import logging
import time
import uuid
import brain
from typing import Any, Coroutine, Dict, List, Optional, Union

import numpy as np
from fastapi import (BackgroundTasks, Body, Depends, FastAPI, HTTPException,
                     Path, Query, Request, Security, status)
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKey, APIKeyHeader, APIKeyQuery
from pydantic import BaseModel, Field, field_validator, HttpUrl
from pydantic_settings import BaseSettings
# For Prometheus metrics
from prometheus_client import Counter, Gauge, Histogram, Summary # type: ignore
from starlette_prometheus import PrometheusMiddleware, handle_metrics # type: ignore

# Assume brain.py is in the same directory or PYTHONPATH
import brain # type: ignore

# --- Application Settings ---
class Settings(BaseSettings):
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    APP_TITLE: str = "橘子專案-進階自動補格評分API (Júzi Zhuānxàn - Advanced Scoring API)"
    APP_DESCRIPTION: str = "提供強化盤面評分模組的API接口，包含批次處理、背景任務與安全性增強。"
    APP_VERSION: str = "2.0.0"

    # --- Security Settings ---
    API_KEY: str = "YOUR_SECRET_API_KEY_HERE"  # Default, should be set via environment
    API_KEY_NAME: str = "X-API-KEY"

    # --- Rate Limiting Settings (Simple In-Memory) ---
    RATE_LIMIT_REQUESTS: int = 100  # Max requests
    RATE_LIMIT_WINDOW_SECONDS: int = 60  # Per window

    # --- Task Management (Simulated) ---
    # In a real scenario, task results would be stored in a DB or Redis
    # For now, we'll just log background task completion
    TASK_CALLBACK_URL_ENABLED: bool = False # Enable to simulate callback
    TASK_CALLBACK_URL: Optional[HttpUrl] = None # e.g., http://localhost:8001/task_result

    # To load from .env file (create a .env file in the same directory)
    # class Config:
    #     env_file = ".env"
    #     env_file_encoding = 'utf-8'

settings = Settings()

# --- Logging Configuration ---
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - RequestID: %(request_id)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

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
# Gauge for active background tasks (example, requires manual management)
ACTIVE_BACKGROUND_TASKS = Gauge(
    "api_active_background_tasks",
    "Number of currently active background scoring tasks"
)
MODULE_USAGE_COUNT = Counter(
    "api_module_usage_count",
    "Count of how many times each scoring module is used",
    ["module_name"]
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

# --- Pydantic Models ---
class GridDataBase(BaseModel):
    grid_data: List[List[Union[int, float]]] = Field(..., example=[[-1, 1, -1], [2, -1, 3], [-1, 4, -1]])

    @field_validator('grid_data')
    def validate_grid_data(cls, v: List[List[Union[int, float]]]) -> List[List[Union[int, float]]]:
        if not v or not all(isinstance(row, list) for row in v) or not v[0]:
            raise ValueError("Grid data must be a non-empty list of non-empty lists.")
        num_cols = len(v[0])
        if num_cols == 0:
             raise ValueError("Grid columns cannot be empty (first row is empty).")
        if not all(len(row) == num_cols for row in v):
            raise ValueError("All rows must have the same number of columns.")
        for r_idx, row in enumerate(v):
            for c_idx, cell_val in enumerate(row):
                if not isinstance(cell_val, (int, float)):
                    raise ValueError(f"Cell ({r_idx}, {c_idx}) type invalid: {type(cell_val)}. Must be number.")
        return v

class GridInput(GridDataBase):
    # Optional client-provided request_id, otherwise one will be generated
    client_request_id: Optional[str] = Field(None, description="Optional client-provided request ID for tracing.")

class BatchGridItem(GridDataBase):
    item_id: str = Field(description="Unique identifier for this item in the batch.")
    module_name: str = Field(description="Scoring module to use for this item.")

class BatchGridInput(BaseModel):
    grids: List[BatchGridItem] = Field(..., max_length=50) # Limit batch size
    client_request_id: Optional[str] = Field(None, description="Optional client-provided request ID for the batch.")

class ScoreOutput(BaseModel):
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
    # Add other relevant details brain module might provide
    # e.g. input_constraints: Optional[Dict[str, Any]] = None

# --- Rate Limiter (Simple In-Memory Implementation) ---
# WARNING: This in-memory limiter is not suitable for multi-process/multi-worker deployments.
# For robust rate limiting, use solutions like slowapi with Redis.
request_counts: Dict[str, List[float]] = {}

# --- Helper Functions for Scoring Task (Background) ---
async def run_scoring_task(
    task_id: str,
    module_name: str,
    grid_data: List[List[Union[int, float]]],
    original_request_id: str,
    client_request_id: Optional[str] = None
):
    """
    Performs the actual scoring in a background task.
    In a real application, results might be stored in a DB or sent to a callback.
    """
    ACTIVE_BACKGROUND_TASKS.inc()
    log_extra = {'request_id': original_request_id, 'task_id': task_id, 'module_name': module_name}
    logger.info(f"Background task started for scoring.", extra=log_extra)

    try:
        np_grid = np.array(grid_data, dtype=np.float32)
        if np_grid.size == 0:
            raise ValueError("Input grid is empty after numpy conversion.")

        # Simulate call to brain module (replace with actual call)
        # score_np_array = brain.get_module_score(module_name, np_grid, request_id=task_id)
        # For demonstration, let's assume it returns a processed grid or raises an error.
        # This part should ideally be non-blocking if brain.get_module_score is I/O bound and not CPU bound.
        # If CPU bound and long, asyncio.to_thread would be used here INSIDE the background task for true parallelism.
        # However, BackgroundTasks themselves run in a thread pool if they are sync functions.
        # If brain.get_module_score is async, it can be awaited directly.
        
        # Assuming brain.get_module_score is synchronous and potentially CPU-bound:
        start_time = time.monotonic()
        score_np_array = await asyncio.to_thread(brain.get_module_score, module_name, np_grid, request_id=task_id)
        duration = time.monotonic() - start_time
        
        score_list_of_lists = score_np_array.tolist()
        result_message = f"Scoring successful for module {module_name}."
        logger.info(result_message + f" Duration: {duration:.4f}s", extra=log_extra)
        
        # Placeholder for result handling (e.g., save to DB, notify, callback)
        if settings.TASK_CALLBACK_URL_ENABLED and settings.TASK_CALLBACK_URL:
            # In a real app, use an HTTP client like httpx to send this
            logger.info(f"Simulating callback to {settings.TASK_CALLBACK_URL} with result.", extra=log_extra)
            # callback_payload = {"task_id": task_id, "status": "completed", "result": score_list_of_lists, "client_request_id": client_request_id}
            # async with httpx.AsyncClient() as client:
            #    await client.post(str(settings.TASK_CALLBACK_URL), json=callback_payload)

    except Exception as e:
        error_message = f"Error in background scoring task for module {module_name}: {str(e)}"
        logger.error(error_message, exc_info=True, extra=log_extra)
        # Placeholder for error handling (e.g., save error status to DB)
    finally:
        ACTIVE_BACKGROUND_TASKS.dec()
        logger.info(f"Background task finished.", extra=log_extra)


# --- FastAPI Application Instance & Middlewares ---
app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    #openapi_tags= ... # Can define tags metadata here
)

app.add_middleware(PrometheusMiddleware) # Exposes /metrics

@app.middleware("http")
async def base_middleware(request: Request, call_next: Coroutine[Any, Any, Any]) -> Any:
    # 1. Manage Request ID
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id # For access in routes/dependencies

    # Attach request_id to log records (for current request context)
    # This is a simplified approach. For robust contextvars-based logging:
    # https://www.python.org/dev/peps/pep-0567/
    # https://github.com/encode/starlette/issues/893
    # Currently, the logging format directly includes %(request_id)s,
    # which needs to be injected. The previous log_requests_middleware did this.
    # For simplicity here, we rely on passing `extra` to logger calls.

    # 2. Simple In-Memory Rate Limiting (Basic, for demonstration)
    # Not suitable for distributed systems; use Redis-backed for that.
    client_ip = request.client.host if request.client else "unknown_client"
    current_time = time.time()
    
    # Clean up old timestamps for the IP
    request_counts[client_ip] = [t for t in request_counts.get(client_ip, []) if t > current_time - settings.RATE_LIMIT_WINDOW_SECONDS]

    if len(request_counts.get(client_ip, [])) >= settings.RATE_LIMIT_REQUESTS:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}", extra={'request_id': request_id})
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status_code=429).inc()
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Too many requests, please try again later."}
        )
    request_counts.setdefault(client_ip, []).append(current_time)

    # 3. Add Security Headers
    start_time_metric = time.monotonic()
    response = await call_next(request) # Process request
    duration_metric = time.monotonic() - start_time_metric

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none';"
    if request.url.scheme == "https": # Only add HSTS if served over HTTPS
         response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    # 4. Metrics Recording
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status_code=response.status_code).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(duration_metric)
    
    logger.info(
        f"Request: {request.method} {request.url.path} - Response: {response.status_code} - Duration: {duration_metric:.4f}s",
        extra={'request_id': request_id}
    )
    return response

# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    logger.error(f"Unhandled exception: {exc}", exc_info=True, extra={'request_id': request_id})
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status_code=500).inc()
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "request_id": request_id,
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please contact support.",
            "detail": str(exc) if settings.LOG_LEVEL == "DEBUG" else None # Show details only in DEBUG
        }
    )

# --- API Endpoints ---
@app.get("/", tags=["Utility"], summary="Root endpoint providing a welcome message.")
async def root(request: Request):
    return {"message": f"Welcome to {settings.APP_TITLE} v{settings.APP_VERSION}", "docs": str(request.url.replace(path="/docs"))}

app.add_route("/metrics", handle_metrics, methods=["GET"], tags=["Utility"], summary="Prometheus metrics endpoint.")


@app.get("/modules", response_model=List[ModuleInfo], tags=["Modules"], summary="List all available scoring modules.")
async def list_available_modules(api_key: APIKey = Depends(get_api_key)):
    # Assume brain.REGISTERED_MODULES_BRAIN.keys() gives names
    # and brain.get_module_details(name) provides more info.
    # This is a hypothetical extension of brain module's capabilities.
    modules_info = []
    for module_name in brain.REGISTERED_MODULES_BRAIN.keys():
        try:
            # Hypothetical function in brain module
            details = getattr(brain, 'get_module_details', lambda name: {})(module_name)
            modules_info.append(ModuleInfo(
                name=module_name,
                description=details.get('description'),
                version=details.get('version')
            ))
        except Exception: # Fallback if get_module_details is not available or fails
             modules_info.append(ModuleInfo(name=module_name))
    return modules_info

@app.get("/modules/{module_name}", response_model=ModuleInfo, tags=["Modules"], summary="Get details for a specific scoring module.")
async def get_module_info(
    module_name: str = Path(..., description="The name of the scoring module."),
    api_key: APIKey = Depends(get_api_key)
):
    if module_name not in brain.REGISTERED_MODULES_BRAIN:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Module '{module_name}' not found.")
    try:
        details = getattr(brain, 'get_module_details', lambda name: {})(module_name)
        return ModuleInfo(
            name=module_name,
            description=details.get('description'),
            version=details.get('version')
        )
    except Exception:
        return ModuleInfo(name=module_name)


@app.post("/score/{module_name}", response_model=TaskAcceptedResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Scoring"], summary="Submit a single grid for scoring (background task).")
async def score_grid_background(
    request: Request,
    payload: GridInput,
    module_name: str = Path(..., description="The name of the scoring module to use."),
    background_tasks: BackgroundTasks = Depends(), # FastAPI injects this
    api_key: APIKey = Depends(get_api_key)
):
    """
    Accepts a grid for scoring. The scoring is performed as a background task.
    The API immediately returns a task ID.
    """
    req_id = request.state.request_id
    client_req_id = payload.client_request_id
    task_id = str(uuid.uuid4())
    log_extra = {'request_id': req_id, 'task_id': task_id, 'module_name': module_name, 'client_request_id': client_req_id}

    if module_name not in brain.REGISTERED_MODULES_BRAIN:
        logger.warning(f"Module '{module_name}' not found for task.", extra=log_extra)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Module '{module_name}' not found.")

    # Add scoring job to background tasks
    background_tasks.add_task(
        run_scoring_task,
        task_id=task_id,
        module_name=module_name,
        grid_data=payload.grid_data,
        original_request_id=req_id, # Pass main request_id for linking logs
        client_request_id=client_req_id
    )
    MODULE_USAGE_COUNT.labels(module_name=module_name).inc()
    logger.info(f"Grid scoring task enqueued.", extra=log_extra)
    
    return TaskAcceptedResponse(
        task_id=task_id,
        message=f"Scoring task for module '{module_name}' accepted and is being processed in the background.",
        client_request_id=client_req_id
    )


@app.post("/score/batch", response_model=List[TaskAcceptedResponse], status_code=status.HTTP_202_ACCEPTED, tags=["Scoring"], summary="Submit multiple grids for batch scoring (background tasks).")
async def score_batch_grid_background(
    request: Request,
    payload: BatchGridInput,
    background_tasks: BackgroundTasks = Depends(),
    api_key: APIKey = Depends(get_api_key)
):
    req_id = request.state.request_id
    client_req_id = payload.client_request_id
    responses = []
    
    log_extra_batch = {'request_id': req_id, 'batch_size': len(payload.grids), 'client_request_id': client_req_id}
    logger.info(f"Batch grid scoring task received.", extra=log_extra_batch)

    for item in payload.grids:
        task_id = str(uuid.uuid4())
        log_extra_item = {**log_extra_batch, 'task_id': task_id, 'item_id': item.item_id, 'module_name': item.module_name}

        if item.module_name not in brain.REGISTERED_MODULES_BRAIN:
            logger.warning(f"Module '{item.module_name}' not found for batch item ID '{item.item_id}'.", extra=log_extra_item)
            # For batch, we might skip invalid items or reject the whole batch.
            # Here, we'll create a task that will internally fail or just return an error marker.
            # Or, simply don't enqueue and return an error for this item.
            # For now, let's make a response indicating failure for this item.
            responses.append(TaskAcceptedResponse(
                task_id=f"error_invalid_module_{item.item_id}", # Special task_id to indicate pre-check failure
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
            original_request_id=req_id, # Link to main batch request
            client_request_id=client_req_id # Link to overall client request ID
        )
        MODULE_USAGE_COUNT.labels(module_name=item.module_name).inc()
        responses.append(TaskAcceptedResponse(
            task_id=task_id,
            message=f"Scoring task for item_id '{item.item_id}' (module '{item.module_name}') accepted.",
            client_request_id=client_req_id
        ))
        logger.info(f"Batch item enqueued for scoring.", extra=log_extra_item)

    return responses

# --- Caching (Conceptual) ---
# To implement caching (e.g., for brain.get_module_score results):
# 1. Setup a Redis client (e.g., using 'redis-py' library).
# 2. Before calling brain.get_module_score, generate a cache key (e.g., from module_name and a hash of grid_data).
# 3. Try to get the result from Redis using this key.
# 4. If cache hit, return the cached result.
# 5. If cache miss, call brain.get_module_score, then store its result in Redis with an expiry time.
# Example conceptual placement within `run_scoring_task`:
#
# import hashlib
# import json
# # Assuming `redis_client` is an initialized Redis client instance
#
# async def get_cached_score(cache_key: str):
#    # cached_result = await redis_client.get(cache_key)
#    # if cached_result: return json.loads(cached_result)
#    return None
#
# async def set_cached_score(cache_key: str, result: Any, ttl_seconds: int = 3600):
#    # await redis_client.setex(cache_key, ttl_seconds, json.dumps(result))
#    pass
#
# In `run_scoring_task`, before `brain.get_module_score`:
#   cache_key_payload = {"module": module_name, "grid": grid_data}
#   cache_key_string = json.dumps(cache_key_payload, sort_keys=True)
#   cache_key = f"score_cache:{hashlib.md5(cache_key_string.encode()).hexdigest()}"
#   cached = await get_cached_score(cache_key)
#   if cached:
#       logger.info(f"Cache hit for task {task_id}", extra=log_extra)
#       # Process `cached` as if it came from `brain.get_module_score`
#       # ... then return or handle callback ...
#       return
#   ...
#   # After successful scoring:
#   await set_cached_score(cache_key, score_list_of_lists)


# --- Main Execution Block ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting {settings.APP_TITLE} v{settings.APP_VERSION} on {settings.APP_HOST}:{settings.APP_PORT}")
    logger.info(f"Default API Key: {settings.API_KEY[:4]}... (Ensure this is changed for production!)") # Log partial key for awareness
    logger.info(f"Rate Limiting: {settings.RATE_LIMIT_REQUESTS} requests per {settings.RATE_LIMIT_WINDOW_SECONDS} seconds (in-memory).")
    if settings.TASK_CALLBACK_URL_ENABLED:
        logger.info(f"Task callback enabled, will attempt to POST to: {settings.TASK_CALLBACK_URL}")
    else:
        logger.info("Task callback is disabled. Background task results will only be logged.")

    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT)