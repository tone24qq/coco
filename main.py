"""
main.py: FastAPI Application Entry Point.

Handles API requests, validation, and calls the analyzer.
"""
import logging
import time
import uuid
from contextvars import ContextVar

import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from pythonjsonlogger import jsonlogger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette_prometheus import PrometheusMiddleware, metrics

from analyzer import analyze_grid, OverallAnalysisResult

# --- Request ID ContextVar ---
# Used to store request_id for logging and passing through layers.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


# --- Application Configuration ---
class AppSettings(BaseSettings):
    app_name: str = "Numeric Panel Analyzer AI"
    log_level: str = "INFO"
    request_id_header: str = "X-Request-ID"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = AppSettings()


# --- Logging Configuration ---
# Based on python-json-logger to include request_id
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["level"] = record.levelname
        log_record["logger_name"] = record.name
        rid = request_id_var.get()
        if rid:
            log_record["request_id"] = rid
        # Avoid KeyError if extra is not set or request_id is not in extra
        if "request_id" not in log_record and hasattr(record, "request_id"):
             log_record["request_id"] = record.request_id # type: ignore


log = logging.getLogger(__name__)
log_handler = logging.StreamHandler()
formatter = CustomJsonFormatter(
    "%(asctime)s %(level)s %(logger_name)s %(request_id)s %(message)s"
)
log_handler.setFormatter(formatter)
logging.basicConfig(handlers=[log_handler], level=settings.log_level.upper())
# Configure uvicorn loggers if needed to use this format
logging.getLogger("uvicorn.access").handlers = [log_handler]
logging.getLogger("uvicorn.error").handlers = [log_handler]


# --- FastAPI Application Setup ---
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="API for analyzing numeric panel grids using AI modules.",
)

# --- Middleware ---
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        # Get request_id from header or generate one
        request_id = request.headers.get(settings.request_id_header)
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Set request_id in context var
        token = request_id_var.set(request_id)

        start_time = time.perf_counter()
        
        response = await call_next(request)
        
        process_time = (time.perf_counter() - start_time) * 1000  # ms
        response.headers[settings.request_id_header] = request_id # Also add to response
        log.info( # Use the root logger configured with CustomJsonFormatter
            "Request processed",
            extra={
                "request_id": request_id, # Ensure it's in extra
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time_ms": f"{process_time:.2f}",
                "client_host": request.client.host if request.client else "unknown",
            },
        )
        request_id_var.reset(token) # Reset context var
        return response

app.add_middleware(RequestIDMiddleware)
app.add_middleware(PrometheusMiddleware) # Exposes /metrics endpoint

# Custom exception handler for validation errors to include request_id
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    rid = request_id_var.get() or "not-set"
    log.error("Request validation error", extra={"request_id": rid, "errors": exc.errors()})
    return JSONResponse(
        status_code=422,
        content={
            "request_id": rid,
            "detail": exc.errors(),
            "body": exc.body
        },
    )

# General error handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse: # pragma: no cover
    rid = request_id_var.get() or "not-set"
    log.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={"request_id": rid}
    )
    return JSONResponse(
        status_code=500,
        content={
            "request_id": rid,
            "detail": "Internal Server Error",
            "error_type": type(exc).__name__,
        },
    )


# --- API Models (Pydantic V2) ---
class GridInput(BaseModel):
    grid: list[list[int | float]] = Field(
        ...,
        description="A 2D list representing the numeric panel. Use -1 for empty cells.",
        example=[[1, -1, 3], [-1, 5, -1], [7, 8, -1]],
    )

    @validator("grid")
    def validate_grid_structure(cls, value: list[list[int | float]]) -> list[list[int | float]]:
        if not value: # pragma: no cover
            raise ValueError("Grid cannot be empty.")
        if not all(isinstance(row, list) for row in value): # pragma: no cover
            raise ValueError("Grid must be a list of lists.")
        
        first_row_len = len(value[0])
        if not all(len(row) == first_row_len for row in value): # pragma: no cover
            raise ValueError("All rows in the grid must have the same length.")
        if first_row_len == 0 and len(value) >0 : #e.g. [[]]
             pass # Allow list of empty lists if that's a valid state (e.g. 0xN grid)
        
        # Check for valid numbers (e.g. within a certain range, or -1)
        # This depends on game rules, for now, just type check is handled by Pydantic
        return value


class AnalysisResponse(BaseModel):
    message: str = "Analysis complete"
    analysis_results: OverallAnalysisResult


# --- API Endpoints ---
@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Performs a health check of the service."""
    rid = request_id_var.get() # Get request_id if set by middleware
    log.info("Health check performed", extra={"request_id": rid})
    return {"status": "healthy", "app_name": settings.app_name}

# Add Prometheus metrics endpoint manually if not using add_middleware for some reason
# (starlette-prometheus add_middleware already does this)
app.add_route("/metrics", metrics)


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_panel_endpoint(grid_input: GridInput) -> AnalysisResponse:
    """
    Analyzes a numeric panel grid using a suite of AI modules.
    """
    current_request_id = request_id_var.get() # Should be set by middleware
    if not current_request_id: # pragma: no cover
        # Fallback if middleware didn't run or context was lost
        current_request_id = str(uuid.uuid4()) 
        request_id_var.set(current_request_id)


    log.info(
        "Received analysis request",
        extra={"request_id": current_request_id, "input_grid_shape_0": len(grid_input.grid[0]) if grid_input.grid else 0},
    )

    try:
        # Convert input to numpy array
        # Ensure consistent dtype, e.g., float64 for calculations, or int if appropriate
        grid_array = np.array(grid_input.grid, dtype=np.float64)
    except ValueError as e: # pragma: no cover
        log.error(f"Error converting grid to numpy array: {e}", extra={"request_id": current_request_id})
        # This should ideally be caught by Pydantic validation or return 422
        raise RequestValidationError(errors=[{"loc": ["body", "grid"], "msg": f"Invalid grid format: {e}", "type": "value_error"}])


    analysis_output: OverallAnalysisResult = await analyze_grid(
        grid_array, current_request_id
    )

    return AnalysisResponse(analysis_results=analysis_output)


if __name__ == "__main__": # pragma: no cover
    import uvicorn
    # For development, run with: uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=settings.log_level.lower())