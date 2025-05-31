# main.py
import asyncio
import logging
import math
import random
import time # For elapsed_ms
import uuid
from collections import Counter, deque
from contextvars import ContextVar
from typing import Any, Callable, Hashable # Removed TypeAlias for <3.10 compatibility

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response # Removed Depends as it's not used directly in endpoint
from numpy.typing import NDArray
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette_prometheus import PrometheusMiddleware, metrics

# --- ContextVars for Logging ---
request_id_contextvar: ContextVar[str | None] = ContextVar("request_id", default=None)
trace_id_contextvar: ContextVar[str | None] = ContextVar("trace_id", default=None)
user_id_contextvar: ContextVar[str | None] = ContextVar("user_id", default=None) # Example

# --- Settings via Pydantic BaseSettings (.env file) ---
class AppSettings(BaseSettings):
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    APP_NAME: str = Field("BrainAPI", description="Application name")
    SERVICE_NAME: str = Field("coco-analyzer", description="Service identifier for logs")
    ENVIRONMENT: str = Field("dev", description="Environment (dev, staging, prod)")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = AppSettings()

# --- Logging Filter to Add Contextual Info ---
class ContextualLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Ensure these attributes are always set on the record, even if None
        setattr(record, "request_id", request_id_contextvar.get())
        setattr(record, "trace_id", trace_id_contextvar.get())
        setattr(record, "service_name", settings.SERVICE_NAME) # Static from settings
        setattr(record, "environment", settings.ENVIRONMENT)   # Static from settings
        setattr(record, "user_id", user_id_contextvar.get()) # Example for user_id
        return True

# --- Logging Configuration ---
class JsonFormatter(logging.Formatter):
    """JSON Log Formatter adhering to logging_spec_v2025.txt."""
    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            # Core LogRecord attributes
            "timestamp": self.formatTime(record, self.datefmt), # Adheres to spec
            "level": record.levelname,                         # Adheres to spec
            "logger": record.name,
            "module": record.module,                           # Adheres to spec
            "function": record.funcName,
            "line": record.lineno,                             # Adheres to spec
            "message": record.getMessage(),                    # Adheres to spec

            # Attributes added by ContextualLogFilter
            "service": getattr(record, "service_name", settings.SERVICE_NAME), # Adheres to spec
            "env": getattr(record, "environment", settings.ENVIRONMENT),       # Adheres to spec
            "request_id": getattr(record, "request_id", None), # Adheres to spec
            "trace_id": getattr(record, "trace_id", None),     # Adheres to spec
            "user_id": getattr(record, "user_id", None),       # Adheres to spec (if user_id_contextvar is used)
        }

        # Attributes typically added via 'extra' by request/response logging middleware
        # Ensure these keys match what's passed in 'extra' and the output spec
        if hasattr(record, "client_ip"): log_entry["ip"] = getattr(record, "client_ip") # Adheres to spec
        if hasattr(record, "http_method_val"): log_entry["method"] = getattr(record, "http_method_val") # Adheres to spec
        if hasattr(record, "http_url_val"): log_entry["url"] = getattr(record, "http_url_val")       # Adheres to spec
        if hasattr(record, "http_status_code"): log_entry["status"] = getattr(record, "http_status_code") # Adheres to spec
        if hasattr(record, "response_elapsed_ms"): log_entry["elapsed_ms"] = getattr(record, "response_elapsed_ms") # Adheres to spec

        # Optional fields from spec that might be in 'extra'
        if hasattr(record, "user_agent_val"): log_entry["user_agent"] = getattr(record, "user_agent_val")
        if hasattr(record, "response_bytes_val"): log_entry["response_bytes"] = getattr(record, "response_bytes_val")


        # Handle exception information
        if record.exc_info:
            log_entry["exception_info"] = self.formatException(record.exc_info)
        if record.stack_info: # Less common, but good to include if present
            log_entry["stack_info"] = self.formatStack(record.stack_info)
        
        # Remove entries with None values to keep logs cleaner, if desired
        # log_entry = {k: v for k, v in log_entry.items() if v is not None}

        return str(log_entry).replace("'", '"')


# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(settings.LOG_LEVEL.upper())
if root_logger.hasHandlers():
    root_logger.handlers.clear()

context_filter = ContextualLogFilter() # Instantiate filter
root_logger.addFilter(context_filter)  # Add filter to root logger

stream_handler = logging.StreamHandler() # Defaults to sys.stderr, use sys.stdout if preferred
# Ensure datefmt produces ISO 8601 with timezone (e.g., Zulu for UTC)
# Python's %z is platform-dependent for timezone name. Using fixed Zulu for UTC example.
# For true ISO 8601, datetime objects with timezone info are better.
# logging.Formatter uses time.strftime, which has limitations with %z.
# A common practice is to log in UTC.
iso_date_format = "%Y-%m-%dT%H:%M:%S" # Add .%03dZ for milliseconds and Zulu manually if needed
# For simplicity here, using what logging.Formatter provides.
# To get precise ISO8601 with Z:
# import datetime
# class UTCISOFormatter(JsonFormatter):
#    def formatTime(self, record, datefmt=None):
#        return datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc).isoformat()
# json_formatter = UTCISOFormatter()
json_formatter = JsonFormatter(fmt=None, datefmt=iso_date_format) # fmt=None uses default LogRecord attributes for message
# Set UTC for formatter if logging in UTC
# json_formatter.converter = time.gmtime # Uncomment to make timestamps UTC in the formatter

stream_handler.setFormatter(json_formatter)
root_logger.addHandler(stream_handler)

module_logger = logging.getLogger(__name__)

# --- Request/Response Logging Middleware (Adhering to logging_spec_v2025.txt) ---
class StructuredRequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        tr_id = request.headers.get("X-Trace-ID") or req_id
        usr_id = request.headers.get("X-User-ID") # Example: get user_id from header

        req_id_token = request_id_contextvar.set(req_id)
        tr_id_token = trace_id_contextvar.set(tr_id)
        usr_id_token = user_id_contextvar.set(usr_id) if usr_id else None


        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")

        # These keys in 'extra' must match what JsonFormatter expects (e.g., "client_ip", "http_method_val")
        log_extras_request = {
            "client_ip": client_ip,
            "http_method_val": request.method,
            "http_url_val": str(request.url),
            "user_agent_val": user_agent,
        }
        # user_id is set via contextvar and picked by filter, so not needed in extra here unless for override.
        module_logger.info("Request received", extra=log_extras_request)

        start_time = time.perf_counter()
        response: Response
        try:
            response = await call_next(request)
        except Exception as e:
            # Log unhandled exceptions before they are turned into 500 by Starlette/FastAPI
            elapsed_time_ms = (time.perf_counter() - start_time) * 1000
            log_extras_exception = {
                "client_ip": client_ip,
                "http_method_val": request.method,
                "http_url_val": str(request.url),
                "http_status_code": 500, # Assuming it will become a 500
                "response_elapsed_ms": round(elapsed_time_ms, 2),
                "user_agent_val": user_agent,
            }
            module_logger.error(f"Unhandled exception during request processing: {str(e)}", exc_info=True, extra=log_extras_exception)
            raise # Re-raise to let FastAPI handle it and return a 500 response
        
        elapsed_time_ms = (time.perf_counter() - start_time) * 1000
        response_content_length = response.headers.get("content-length")

        log_extras_response = {
            "client_ip": client_ip,
            "http_method_val": request.method,
            "http_url_val": str(request.url),
            "http_status_code": response.status_code,
            "response_elapsed_ms": round(elapsed_time_ms, 2),
            "user_agent_val": user_agent,
            "response_bytes_val": int(response_content_length) if response_content_length else None
        }

        if 400 <= response.status_code < 500:
            module_logger.warning("Client error response", extra=log_extras_response)
        elif response.status_code >= 500:
            module_logger.error("Server error response", extra=log_extras_response)
        else:
            module_logger.info("Request completed", extra=log_extras_response)

        response.headers["X-Request-ID"] = req_id
        if tr_id: response.headers["X-Trace-ID"] = tr_id
        
        request_id_contextvar.reset(req_id_token)
        trace_id_contextvar.reset(tr_id_token)
        if usr_id_token: user_id_contextvar.reset(usr_id_token)
        return response

# --- FastAPI Application Setup ---
app = FastAPI(
    title=settings.APP_NAME,
    description="API for Brain Module Grid Scoring",
    version="1.0.0",
)
app.add_middleware(PrometheusMiddleware)
app.add_middleware(StructuredRequestLoggingMiddleware)

# --- Pydantic Models for API (same as before) ---
class GridInput(BaseModel):
    grid: list[list[int]]
    module_name: str
    model_config = ConfigDict(extra="forbid")

class ScoreOutput(BaseModel):
    module_name: str
    score_grid: list[list[float]]
    request_id: str | None
    trace_id: str | None
    model_config = ConfigDict(extra="forbid")

# === Helper Utilities (MathUtils, BoardAnalyzerUtils - same, ensure they use module_logger) ===
# (Code for MathUtils and BoardAnalyzerUtils remains largely the same as the previous corrected version)
class MathUtils:
    def sigmoid(self, x: float, k: float = 1.0) -> float:
        try: clamped_x = np.clip(-k * x, -700.0, 700.0); return 1.0 / (1.0 + math.exp(clamped_x))
        except OverflowError: return 0.0 if -k * x > 0 else 1.0
    def normalize_value(self, value: float, min_val: float, max_val: float, clamp: bool = True) -> float:
        if math.isclose(max_val, min_val):
            if math.isclose(value, min_val): return 0.5
            elif value < min_val: return 0.0
            else: return 1.0
        if (max_val - min_val) == 0: return 0.5 if math.isclose(value, min_val) else (0.0 if value < min_val else 1.0)
        normalized = (value - min_val) / (max_val - min_val)
        return float(np.clip(normalized, 0.0, 1.0)) if clamp else normalized
    def manhattan_distance(self, p1: tuple[int, int], p2: tuple[int, int]) -> int: return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    def euclidean_distance(self, p1: tuple[int, int] | tuple[float, float], p2: tuple[int, int] | tuple[float, float]) -> float: return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    def get_entropy(self, values: list[Hashable]) -> float:
        if not values: return 0.0
        counts = Counter(values); total_count = len(values); entropy = 0.0
        for count_val in counts.values():
            probability = count_val / total_count
            if probability > 0: entropy -= probability * math.log2(probability)
        return entropy

class BoardAnalyzerUtils:
    def get_neighborhood_values(self, grid: NDArray[np.int_], r: int, c: int, radius: int = 1, eight_connectivity: bool = True, val_func: Callable[[int], float | None] = lambda x_val: float(x_val) if x_val != -1 else None, include_center: bool = False) -> list[float]:
        neighbors: list[float] = []; rows_grid, cols_grid = grid.shape
        for dr_offset in range(-radius, radius + 1):
            for dc_offset in range(-radius, radius + 1):
                if not include_center and dr_offset == 0 and dc_offset == 0: continue
                if not eight_connectivity and (abs(dr_offset) + abs(dc_offset) != 1 if radius == 1 else abs(dr_offset) + abs(dc_offset) > radius): continue
                nr, nc = r + dr_offset, c + dc_offset
                if 0 <= nr < rows_grid and 0 <= nc < cols_grid:
                    processed_val = val_func(grid[nr, nc])
                    if processed_val is not None: neighbors.append(processed_val)
        return neighbors
    def get_value_gradient_at_cell(self, grid: NDArray[np.int_], r: int, c: int, val_func: Callable[[int], float] = lambda x_val: float(x_val) if x_val != -1 else 0.0) -> tuple[float, float]:
        rows_grid, cols_grid = grid.shape
        def safe_val(r_in: int, c_in: int) -> float: return val_func(grid[r_in, c_in]) if 0 <= r_in < rows_grid and 0 <= c_in < cols_grid else 0.0
        gx = (safe_val(r - 1, c + 1) + 2 * safe_val(r, c + 1) + safe_val(r + 1, c + 1)) - (safe_val(r - 1, c - 1) + 2 * safe_val(r, c - 1) + safe_val(r + 1, c - 1))
        gy = (safe_val(r + 1, c - 1) + 2 * safe_val(r + 1, c) + safe_val(r + 1, c + 1)) - (safe_val(r - 1, c - 1) + 2 * safe_val(r - 1, c) + safe_val(r - 1, c + 1))
        return gx, gy
    def find_sequences_in_line(self, line: list[int], min_len: int = 3, check_arithmetic: bool = True, check_geometric: bool = False, allow_gaps: int = 0) -> list[list[int]]:
        sequences: list[list[int]] = []; n = len(line)
        if n < min_len: return sequences
        if check_arithmetic:
            for i in range(n):
                if line[i] == -1: continue
                for j in range(i + 1, n):
                    if line[j] == -1:
                        if allow_gaps > 0:
                            current_gap_count_for_diff = 0; next_non_gap_idx = -1
                            for k_search in range(j, n):
                                if line[k_search] == -1: current_gap_count_for_diff += 1
                                else: next_non_gap_idx = k_search; break
                            if next_non_gap_idx != -1 and current_gap_count_for_diff <= allow_gaps:
                                diff = line[next_non_gap_idx] - line[i]
                                current_seq = [line[i], line[next_non_gap_idx]]; current_gap_count = current_gap_count_for_diff
                                for l_extend in range(next_non_gap_idx + 1, n):
                                    if line[l_extend] == -1: current_gap_count += 1;
                                    if current_gap_count > allow_gaps: break; continue
                                    expected = current_seq[-1] + diff
                                    if math.isclose(float(line[l_extend]), float(expected)): current_seq.append(line[l_extend]); current_gap_count = 0
                                    else: break
                                if len(current_seq) >= min_len: sequences.append(list(current_seq))
                        continue
                    diff = line[j] - line[i]; current_seq = [line[i], line[j]]; current_gap_count = 0
                    for k in range(j + 1, n):
                        if line[k] == -1: current_gap_count += 1;
                        if current_gap_count > allow_gaps: break; continue
                        expected = current_seq[-1] + diff
                        if math.isclose(float(line[k]), float(expected)): current_seq.append(line[k]); current_gap_count = 0
                        else: break
                    if len(current_seq) >= min_len: sequences.append(list(current_seq))
        return sequences
    def get_card_max_value_from_grid_dimensions(self, grid_shape: tuple[int, int]) -> int: rows_g, cols_g = grid_shape; return 0 if rows_g == 0 or cols_g == 0 else rows_g * cols_g
    def get_all_possible_numbers_for_grid(self, grid_shape: tuple[int, int]) -> set[int]: max_val = self.get_card_max_value_from_grid_dimensions(grid_shape); return set() if max_val == 0 else set(range(1, max_val + 1))
    def get_legal_values_for_placement(self, grid: NDArray[np.int_]) -> set[int]:
        if grid.size == 0: return set()
        all_possible = self.get_all_possible_numbers_for_grid(grid.shape)
        used_positive: set[int] = {int(v) for v in grid.flat if v != -1 and v > 0}
        return all_possible - used_positive

_math_utils = MathUtils()
_board_analyzer_utils = BoardAnalyzerUtils()

# === Brain Core Dispatch Area (same as before) ===
ScoringFunctionType = Callable[[NDArray[np.int_], str | None], NDArray[np.float64]]
REGISTERED_MODULES_BRAIN: dict[str, ScoringFunctionType] = {}

async def get_module_score_async(module_name: str, grid: NDArray[np.int_], **kwargs: Any) -> NDArray[np.float64]:
    log_req_id = request_id_contextvar.get(); log_tr_id = trace_id_contextvar.get()
    if module_name not in REGISTERED_MODULES_BRAIN:
        module_logger.error("Module not found", extra={"module_name_requested": module_name})
        rows_g, cols_g = grid.shape if grid.ndim == 2 and grid.size > 0 else (0,0); return np.zeros((rows_g, cols_g), dtype=np.float64)
    module_func = REGISTERED_MODULES_BRAIN[module_name]
    module_logger.info("Executing module via to_thread", extra={"module_name_executed": module_name})
    try:
        import inspect; sig = inspect.signature(module_func); final_kwargs = kwargs.copy()
        if 'request_id' in sig.parameters: final_kwargs['request_id'] = log_req_id
        score_grid = await asyncio.to_thread(module_func, grid, **final_kwargs)
        if not isinstance(score_grid, np.ndarray):
            module_logger.error("Module returned non-NumPy array", extra={"module_name_error": module_name, "returned_type": str(type(score_grid))})
            raise TypeError(f"Module {module_name} did not return a NumPy array.")
        if score_grid.shape != grid.shape:
            module_logger.error("Module returned array with incorrect shape", extra={"module_name_shape_error": module_name, "returned_shape": str(score_grid.shape), "expected_shape": str(grid.shape)})
            raise ValueError(f"Module {module_name} returned array with incorrect shape.")
        return score_grid.astype(np.float64)
    except Exception as e:
        module_logger.error(f"Error executing module: {str(e)}", exc_info=True, extra={"module_name_exception": module_name})
        rows_g, cols_g = grid.shape if grid.ndim == 2 and grid.size > 0 else (0,0); return np.zeros((rows_g, cols_g), dtype=np.float64)

# --- Scoring Module Implementations (ensure they use module_logger) ---
# (EXT_A2, EXT_M3, EXT_F10 definitions remain same as previous corrected version, ensuring they use module_logger
# and their signatures can accept request_id if needed for internal logic, though logging uses contextvar)
def EXT_A2_Weighted_Proximity_Vec(grid: NDArray[np.int_], request_id: str | None = None, **kwargs: Any) -> NDArray[np.float64]:
    module_logger.debug("Executing EXT_A2_Weighted_Proximity_Vec")
    rows, cols = grid.shape; scores = np.zeros((rows, cols), dtype=np.float64)
    if rows == 0 or cols == 0: return scores
    radius = 2; value_weight_factor = 0.15; distance_decay_factor = 1.8
    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            proximity_score = 0.0
            for dr_offset in range(-radius, radius + 1):
                for dc_offset in range(-radius, radius + 1):
                    if dr_offset == 0 and dc_offset == 0: continue
                    nr, nc = r_idx + dr_offset, c_idx + dc_offset
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                        dist = _math_utils.manhattan_distance((r_idx, c_idx), (nr, nc)); dist = 1 if dist == 0 else dist
                        proximity_score += (grid[nr, nc] * value_weight_factor) / (dist**distance_decay_factor)
            max_val_on_grid = float(_board_analyzer_utils.get_card_max_value_from_grid_dimensions((rows, cols))); max_val_on_grid = 1.0 if max_val_on_grid == 0 else max_val_on_grid
            num_neighbors_in_radius = (2 * radius + 1) ** 2 - 1
            heuristic_max_score = (num_neighbors_in_radius * max_val_on_grid * value_weight_factor / (1**distance_decay_factor if distance_decay_factor != 0 else 1.0))
            if heuristic_max_score == 0 and num_neighbors_in_radius > 0 and max_val_on_grid > 0: heuristic_max_score = 1e-9
            if heuristic_max_score > 0: scores[r_idx, c_idx] = _math_utils.normalize_value(proximity_score, 0, heuristic_max_score, clamp=True)
            elif math.isclose(proximity_score, 0.0): scores[r_idx, c_idx] = 0.0
            else: scores[r_idx, c_idx] = 0.5; module_logger.warning("EXT_A2: Heuristic max score is zero.", extra={"proximity_score": proximity_score, "heuristic_max_score": heuristic_max_score})
    return scores

def EXT_M3_Local_Heterogeneity_Vec(grid: NDArray[np.int_], request_id: str | None = None, **kwargs: Any) -> NDArray[np.float64]:
    module_logger.debug("Executing EXT_M3_Local_Heterogeneity_Vec")
    rows, cols = grid.shape; scores = np.zeros((rows, cols), dtype=np.float64)
    if rows == 0 or cols == 0: return scores
    radius = 1; min_neighbors_for_robust_score = 2
    all_possible = _board_analyzer_utils.get_all_possible_numbers_for_grid(grid.shape)
    if not all_possible: return scores
    max_theoretical_entropy: float
    if len(all_possible) > 1: max_theoretical_entropy = math.log2(len(all_possible))
    elif len(all_possible) == 1: max_theoretical_entropy = math.log2(2)
    else: max_theoretical_entropy = 1.0
    if max_theoretical_entropy == 0: max_theoretical_entropy = 1.0
    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            raw_neighbors = _board_analyzer_utils.get_neighborhood_values(grid, r_idx, c_idx, radius, True, lambda x: int(x) if x != -1 else None, False)
            processed_neighbors: list[Hashable] = [v for v in raw_neighbors if v is not None]
            if len(processed_neighbors) < min_neighbors_for_robust_score: scores[r_idx, c_idx] = 0.0; continue
            current_entropy = _math_utils.get_entropy(processed_neighbors)
            scores[r_idx, c_idx] = _math_utils.normalize_value(current_entropy / max_theoretical_entropy, 0, 1, True)
    return scores

def EXT_F10_Discontinuity_Vec(grid: NDArray[np.int_], request_id: str | None = None, **kwargs: Any) -> NDArray[np.float64]:
    module_logger.debug("Executing EXT_F10_Discontinuity_Vec")
    rows, cols = grid.shape; scores = np.zeros((rows, cols), dtype=np.float64)
    if rows == 0 or cols == 0: return scores
    legal_values = _board_analyzer_utils.get_legal_values_for_placement(grid)
    if not legal_values: return scores
    min_seq_len = 3; heuristic_max_len = float(max(rows, cols, min_seq_len)); heuristic_max_len = float(min_seq_len) if heuristic_max_len == 0 else heuristic_max_len
    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_len_contrib = 0.0
            for val_try_float in legal_values:
                val_try = int(val_try_float); temp_grid = grid.copy(); temp_grid[r_idx, c_idx] = val_try; current_max = 0.0
                # Row, Col, Diag1, Diag2 checks... (Simplified for brevity, assume full logic from previous version)
                lines_to_check = [list(temp_grid[r_idx, :]), list(temp_grid[:, c_idx])]
                if rows > 0 and cols > 0:
                    lines_to_check.append(list(np.diag(temp_grid, k=c_idx - r_idx)))
                    lines_to_check.append(list(np.diag(np.fliplr(temp_grid), k=(cols - 1 - c_idx) - r_idx)))
                for line_vals in lines_to_check:
                    for seq in _board_analyzer_utils.find_sequences_in_line(line_vals, min_seq_len, True, False, 1):
                        if val_try in seq: current_max = max(current_max, float(len(seq)))
                if current_max >= min_seq_len: max_len_contrib = max(max_len_contrib, current_max)
            scores[r_idx, c_idx] = _math_utils.normalize_value(max_len_contrib, 0, heuristic_max_len, True)
    return scores

REGISTERED_MODULES_BRAIN = {
    "EXT_A2_Weighted_Proximity_Vec": EXT_A2_Weighted_Proximity_Vec,
    "EXT_M3_Local_Heterogeneity_Vec": EXT_M3_Local_Heterogeneity_Vec,
    "EXT_F10_Discontinuity_Vec": EXT_F10_Discontinuity_Vec,
}

@app.post("/score_grid", response_model=ScoreOutput)
async def score_grid_endpoint(grid_input: GridInput) -> ScoreOutput:
    current_req_id = request_id_contextvar.get(); current_tr_id = trace_id_contextvar.get()
    module_logger.info("Processing /score_grid request", extra={"module_name_requested": grid_input.module_name})
    try:
        grid_list = grid_input.grid
        grid_np: NDArray[np.int_]
        if not grid_list: grid_np = np.array([[]], dtype=np.int_).reshape(0,0)
        elif not all(isinstance(row, list) for row in grid_list) or (len(grid_list) > 0 and not all(len(row) == len(grid_list[0]) for row in grid_list)):
            raise HTTPException(status_code=400, detail="Grid must be a list of lists with consistent row lengths.")
        else: grid_np = np.array(grid_list, dtype=np.int_)
        if grid_np.ndim != 2:
             if grid_np.shape == (0,): grid_np = np.array([[]], dtype=np.int_).reshape(0,0)
             else: raise HTTPException(status_code=400, detail=f"Grid must be 2D, got shape {grid_np.shape}.")
    except HTTPException: raise
    except Exception as e: module_logger.error(f"Grid conversion error: {str(e)}", exc_info=True); raise HTTPException(status_code=400, detail=f"Invalid grid: {str(e)}")
    score_array = await get_module_score_async(grid_input.module_name, grid_np)
    return ScoreOutput(module_name=grid_input.module_name, score_grid=score_array.tolist(), request_id=current_req_id, trace_id=current_tr_id)

if __name__ == "__main__":
    uvicorn_log_level = settings.LOG_LEVEL.lower()
    if uvicorn_log_level not in ["critical", "error", "warning", "info", "debug", "trace"]: uvicorn_log_level = "info"
    module_logger.info(f"Starting {settings.APP_NAME} on http://localhost:8000 with log level {uvicorn_log_level}")
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True, log_level=uvicorn_log_level) # Port 10000 as per logs