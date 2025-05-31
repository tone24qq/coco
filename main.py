# main.py
import asyncio
import logging
import math
import random
import time # For elapsed_ms
import uuid
from collections import Counter, deque
from contextvars import ContextVar
from typing import Any, Callable, Hashable, cast

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from numpy.typing import NDArray
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette_prometheus import PrometheusMiddleware, metrics

# --- ContextVars for Logging ---
request_id_contextvar: ContextVar[str | None] = ContextVar("request_id", default=None)
trace_id_contextvar: ContextVar[str | None] = ContextVar("trace_id", default=None)
# Add other contextvars if needed, e.g., user_id_contextvar

# --- Settings via Pydantic BaseSettings (.env file) ---
class AppSettings(BaseSettings):
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    APP_NAME: str = Field("BrainAPI", description="Application name")
    SERVICE_NAME: str = Field("coco-analyzer", description="Service identifier for logs") # From logging_spec_v2025.txt
    ENVIRONMENT: str = Field("dev", description="Environment (dev, staging, prod)") # From logging_spec_v2025.txt

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = AppSettings()

# --- Logging Filter to Add Contextual Info ---
class ContextualLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_contextvar.get() # type: ignore[attr-defined]
        record.trace_id = trace_id_contextvar.get() # type: ignore[attr-defined]
        record.service_name = settings.SERVICE_NAME # type: ignore[attr-defined]
        record.environment = settings.ENVIRONMENT # type: ignore[attr-defined]
        # record.user_id = user_id_contextvar.get() # Example if user_id is in contextvar
        return True

# --- Logging Configuration ---
class JsonFormatter(logging.Formatter):
    """JSON Log Formatter adhering to logging_spec_v2025.txt."""
    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "service": getattr(record, "service_name", settings.SERVICE_NAME),
            "env": getattr(record, "environment", settings.ENVIRONMENT),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "request_id": getattr(record, "request_id", None), # Populated by ContextualLogFilter
            "trace_id": getattr(record, "trace_id", None),     # Populated by ContextualLogFilter
            "message": record.getMessage(),
        }

        # Add fields from 'extra' if they exist and are part of the spec
        # These are typically added by specific logging calls, like the request/response logger
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id # type: ignore[attr-defined]
        if hasattr(record, "ip"):
            log_entry["ip"] = record.ip # type: ignore[attr-defined]
        if hasattr(record, "http_method"): # Renamed from "method" to avoid clash with LogRecord.method
            log_entry["method"] = record.http_method # type: ignore[attr-defined]
        if hasattr(record, "http_url"): # Renamed from "url"
            log_entry["url"] = record.http_url # type: ignore[attr-defined]
        if hasattr(record, "http_status"): # Renamed from "status"
            log_entry["status"] = record.http_status # type: ignore[attr-defined]
        if hasattr(record, "elapsed_ms"):
            log_entry["elapsed_ms"] = record.elapsed_ms # type: ignore[attr-defined]
        if hasattr(record, "user_agent"):
             log_entry["user_agent"] = record.user_agent # type: ignore[attr-defined]
        if hasattr(record, "response_bytes"):
            log_entry["response_bytes"] = record.response_bytes # type: ignore[attr-defined]


        # Include any other custom fields from 'extra' that aren't explicitly handled above
        # Be cautious with this part to not include overly verbose or sensitive data by default
        standard_attrs = set(logging.LogRecord('', '', '', '', '', '', '', '', '').__dict__.keys()) | set(log_entry.keys())
        for key, value in record.__dict__.items():
            if key not in standard_attrs and key not in ['args', 'exc_text', 'stack_info', 'msg', 'asctime', # common internal attrs
                                                          'service_name', 'environment']: # already handled or internal
                log_entry[f"extra_{key}"] = value


        if record.exc_info:
            log_entry["exception_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_entry["stack_info"] = self.formatStack(record.stack_info)

        return str(log_entry).replace("'", '"')


# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(settings.LOG_LEVEL.upper())
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# Add filter and formatter to a new stream handler
context_filter = ContextualLogFilter()
root_logger.addFilter(context_filter)

stream_handler = logging.StreamHandler()
json_formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S%z") # ISO 8601 format
stream_handler.setFormatter(json_formatter)
root_logger.addHandler(stream_handler)

# Specific logger for this module
module_logger = logging.getLogger(__name__)

# --- Request/Response Logging Middleware (Adhering to logging_spec_v2025.txt) ---
class StructuredRequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Set up contextvars for request_id and trace_id
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        tr_id = request.headers.get("X-Trace-ID") or req_id # Use req_id if trace_id is missing
        
        req_id_token = request_id_contextvar.set(req_id)
        tr_id_token = trace_id_contextvar.set(tr_id)

        # user_id_token = user_id_contextvar.set(request.headers.get("X-User-ID")) # Example

        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")

        log_extras_request = {
            "ip": client_ip,
            "http_method": request.method,
            "http_url": str(request.url),
            "user_agent": user_agent,
            # "user_id": user_id_contextvar.get() # if set
        }
        module_logger.info("Request received", extra=log_extras_request)

        start_time = time.perf_counter()
        response = await call_next(request)
        elapsed_time_ms = (time.perf_counter() - start_time) * 1000
        
        response_content_length = response.headers.get("content-length")

        log_extras_response = {
            "ip": client_ip,
            "http_method": request.method,
            "http_url": str(request.url),
            "http_status": response.status_code,
            "elapsed_ms": round(elapsed_time_ms, 2),
            "user_agent": user_agent,
            # "user_id": user_id_contextvar.get(), # if set
            "response_bytes": int(response_content_length) if response_content_length else None
        }
        # Customize message based on status code for better context from logging_spec
        if 400 <= response.status_code < 500:
            module_logger.warning("Client error response", extra=log_extras_response)
        elif response.status_code >= 500:
            module_logger.error("Server error response", extra=log_extras_response)
        else:
            module_logger.info("Request completed", extra=log_extras_response)


        response.headers["X-Request-ID"] = req_id
        if tr_id:
             response.headers["X-Trace-ID"] = tr_id
        
        request_id_contextvar.reset(req_id_token)
        trace_id_contextvar.reset(tr_id_token)
        # user_id_contextvar.reset(user_id_token) # if set
        return response


# --- FastAPI Application Setup ---
app = FastAPI(
    title=settings.APP_NAME,
    description="API for Brain Module Grid Scoring",
    version="1.0.0",
)

app.add_middleware(PrometheusMiddleware) # Should be one of the first ideally
app.add_middleware(StructuredRequestLoggingMiddleware) # Add our new logging middleware


# --- Pydantic Models for API (same as before) ---
class GridInput(BaseModel):
    grid: list[list[int]] = Field(..., description="The game grid, -1 for empty cells.")
    module_name: str = Field(..., description="Name of the scoring module to use.")
    # request_id is now handled by middleware and contextvars primarily for logging,
    # but can be optionally passed for external tracing correlation if needed.
    # It's not used by the endpoint logic directly if contextvars are primary.
    # passed_request_id: str | None = Field(None, alias="X-Request-ID", description="Optional external request ID.")


    model_config = ConfigDict(extra="forbid")

class ScoreOutput(BaseModel):
    module_name: str = Field(..., description="Name of the executed scoring module.")
    score_grid: list[list[float]] = Field(..., description="The resulting scores for each cell.")
    request_id: str | None = Field(None, description="Request ID associated with this scoring.")
    trace_id: str | None = Field(None, description="Trace ID associated with this scoring.")


    model_config = ConfigDict(extra="forbid")


# === Helper Utilities (MathUtils, BoardAnalyzerUtils - same as before, ensure they use module_logger) ===
class MathUtils:
    # ... (previous MathUtils code, ensure logging uses module_logger if any needed) ...
    def sigmoid(self, x: float, k: float = 1.0) -> float:
        try:
            clamped_x = np.clip(-k * x, -700.0, 700.0)
            return 1.0 / (1.0 + math.exp(clamped_x))
        except OverflowError:
            return 0.0 if -k * x > 0 else 1.0

    def normalize_value(
        self, value: float, min_val: float, max_val: float, clamp: bool = True
    ) -> float:
        if math.isclose(max_val, min_val):
            if math.isclose(value, min_val): return 0.5
            elif value < min_val: return 0.0
            else: return 1.0
        if (max_val - min_val) == 0:
             return 0.5 if math.isclose(value, min_val) else (0.0 if value < min_val else 1.0)
        normalized = (value - min_val) / (max_val - min_val)
        return float(np.clip(normalized, 0.0, 1.0)) if clamp else normalized

    def manhattan_distance(self, p1: tuple[int, int], p2: tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def euclidean_distance(self, p1: tuple[int, int] | tuple[float, float], p2: tuple[int, int] | tuple[float, float]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_entropy(self, values: list[Hashable]) -> float:
        if not values: return 0.0
        counts = Counter(values)
        total_count = len(values)
        entropy = 0.0
        for count_val in counts.values():
            probability = count_val / total_count
            if probability > 0: entropy -= probability * math.log2(probability)
        return entropy

class BoardAnalyzerUtils:
    # ... (previous BoardAnalyzerUtils code, ensure logging uses module_logger if any needed) ...
    def get_neighborhood_values(
        self, grid: NDArray[np.int_], r: int, c: int, radius: int = 1,
        eight_connectivity: bool = True,
        val_func: Callable[[int], float | None] = lambda x_val: float(x_val) if x_val != -1 else None,
        include_center: bool = False,
    ) -> list[float]:
        neighbors: list[float] = []
        rows_grid, cols_grid = grid.shape
        for dr_offset in range(-radius, radius + 1):
            for dc_offset in range(-radius, radius + 1):
                if not include_center and dr_offset == 0 and dc_offset == 0: continue
                if not eight_connectivity:
                    if radius == 1 and abs(dr_offset) + abs(dc_offset) != 1: continue
                    elif radius > 1 and abs(dr_offset) + abs(dc_offset) > radius: continue
                nr, nc = r + dr_offset, c + dc_offset
                if 0 <= nr < rows_grid and 0 <= nc < cols_grid:
                    processed_val = val_func(grid[nr, nc])
                    if processed_val is not None: neighbors.append(processed_val)
        return neighbors

    def get_value_gradient_at_cell(
        self, grid: NDArray[np.int_], r: int, c: int,
        val_func: Callable[[int], float] = lambda x_val: float(x_val) if x_val != -1 else 0.0,
    ) -> tuple[float, float]:
        rows_grid, cols_grid = grid.shape
        def safe_val(r_in: int, c_in: int) -> float:
            if 0 <= r_in < rows_grid and 0 <= c_in < cols_grid: return val_func(grid[r_in, c_in])
            return 0.0
        gx = (safe_val(r - 1, c + 1) + 2 * safe_val(r, c + 1) + safe_val(r + 1, c + 1)) - \
             (safe_val(r - 1, c - 1) + 2 * safe_val(r, c - 1) + safe_val(r + 1, c - 1))
        gy = (safe_val(r + 1, c - 1) + 2 * safe_val(r + 1, c) + safe_val(r + 1, c + 1)) - \
             (safe_val(r - 1, c - 1) + 2 * safe_val(r - 1, c) + safe_val(r - 1, c + 1))
        return gx, gy

    def find_sequences_in_line(
        self, line: list[int], min_len: int = 3, check_arithmetic: bool = True,
        check_geometric: bool = False, allow_gaps: int = 0,
    ) -> list[list[int]]:
        sequences: list[list[int]] = []
        n = len(line)
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
                                    if line[l_extend] == -1:
                                        current_gap_count += 1
                                        if current_gap_count > allow_gaps: break
                                        continue
                                    expected = current_seq[-1] + diff
                                    if math.isclose(float(line[l_extend]), float(expected)): current_seq.append(line[l_extend]); current_gap_count = 0
                                    else: break
                                if len(current_seq) >= min_len: sequences.append(list(current_seq))
                        continue
                    diff = line[j] - line[i]
                    current_seq = [line[i], line[j]]; current_gap_count = 0
                    for k in range(j + 1, n):
                        if line[k] == -1:
                            current_gap_count += 1
                            if current_gap_count > allow_gaps: break
                            continue
                        expected = current_seq[-1] + diff
                        if math.isclose(float(line[k]), float(expected)): current_seq.append(line[k]); current_gap_count = 0
                        else: break
                    if len(current_seq) >= min_len: sequences.append(list(current_seq))
        return sequences

    def get_card_max_value_from_grid_dimensions(self, grid_shape: tuple[int, int]) -> int:
        rows_grid, cols_grid = grid_shape
        return 0 if rows_grid == 0 or cols_grid == 0 else rows_grid * cols_grid

    def get_all_possible_numbers_for_grid(self, grid_shape: tuple[int, int]) -> set[int]:
        max_val = self.get_card_max_value_from_grid_dimensions(grid_shape)
        return set() if max_val == 0 else set(range(1, max_val + 1))

    def get_legal_values_for_placement(self, grid: NDArray[np.int_]) -> set[int]:
        if grid.size == 0: return set()
        rows_grid, cols_grid = grid.shape
        all_possible = self.get_all_possible_numbers_for_grid((rows_grid, cols_grid))
        used_positive: set[int] = {int(v) for v in grid.flat if v != -1 and v > 0}
        return all_possible - used_positive

_math_utils = MathUtils()
_board_analyzer_utils = BoardAnalyzerUtils()

# === Brain Core Dispatch Area (same as before) ===
ScoringFunctionType = Callable[[NDArray[np.int_], str | None], NDArray[np.float64]]
REGISTERED_MODULES_BRAIN: dict[str, ScoringFunctionType] = {}

async def get_module_score_async(
    module_name: str, grid: NDArray[np.int_], **kwargs: Any # request_id_val removed, filter handles it
) -> NDArray[np.float64]:
    # The effective_request_id for logging within this function will be picked up by the filter.
    # Specific module might still need request_id if it does specific logic with it,
    # but for logging, filter is primary.
    
    # For logging within this specific function call, if not relying purely on the filter:
    log_req_id = request_id_contextvar.get()
    log_tr_id = trace_id_contextvar.get()

    if module_name not in REGISTERED_MODULES_BRAIN:
        module_logger.error(
            f"Module not found", # Simpler message, context in structured log
            extra={"module_name_requested": module_name} # Filter adds req_id, tr_id
        )
        rows_grid, cols_grid = grid.shape if grid.ndim == 2 and grid.size > 0 else (0,0)
        return np.zeros((rows_grid, cols_grid), dtype=np.float64)

    module_func = REGISTERED_MODULES_BRAIN[module_name]
    module_logger.info(
        f"Executing module via to_thread",
        extra={"module_name_executed": module_name} # Filter adds req_id, tr_id
    )

    try:
        # Pass request_id explicitly ONLY if the module signature expects it
        # and it's used for more than just logging (which filter covers)
        import inspect
        sig = inspect.signature(module_func)
        final_kwargs = kwargs.copy()
        if 'request_id' in sig.parameters:
            final_kwargs['request_id'] = log_req_id # Pass the current request_id
        
        # Pass grid as the first positional argument
        score_grid = await asyncio.to_thread(module_func, grid, **final_kwargs)
        
        if not isinstance(score_grid, np.ndarray):
            module_logger.error(
                "Module returned non-NumPy array",
                extra={"module_name_error": module_name, "returned_type": str(type(score_grid))}
            )
            raise TypeError(f"Module {module_name} did not return a NumPy array.")
        if score_grid.shape != grid.shape:
            module_logger.error(
                "Module returned array with incorrect shape",
                extra={
                    "module_name_shape_error": module_name,
                    "returned_shape": str(score_grid.shape),
                    "expected_shape": str(grid.shape)
                }
            )
            raise ValueError(f"Module {module_name} returned array with incorrect shape.")
            
        return score_grid.astype(np.float64)
    except Exception as e:
        module_logger.error(
            f"Error executing module: {str(e)}",
            exc_info=True,
            extra={"module_name_exception": module_name}
        )
        rows_grid, cols_grid = grid.shape if grid.ndim == 2 and grid.size > 0 else (0,0)
        return np.zeros((rows_grid, cols_grid), dtype=np.float64)


# --- Scoring Module Implementations (ensure they use module_logger, no `extra` for req_id) ---
def EXT_A2_Weighted_Proximity_Vec(
    grid: NDArray[np.int_], request_id: str | None = None, **kwargs: Any # request_id can be kept for direct use if needed
) -> NDArray[np.float64]:
    # The filter will add request_id to logs. If this function uses request_id for non-logging purposes,
    # it can accept it. Otherwise, it can be removed from signature if only for logging.
    # For now, keeping it as per original structure, but logging will use contextvar one.
    module_logger.debug("Executing EXT_A2_Weighted_Proximity_Vec") # req_id added by filter
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=np.float64)
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
                        dist = _math_utils.manhattan_distance((r_idx, c_idx), (nr, nc))
                        if dist == 0: dist = 1
                        score_contribution = (grid[nr, nc] * value_weight_factor) / (dist**distance_decay_factor)
                        proximity_score += score_contribution
            max_val_on_grid = float(_board_analyzer_utils.get_card_max_value_from_grid_dimensions((rows, cols)))
            if max_val_on_grid == 0: max_val_on_grid = 1.0
            num_neighbors_in_radius = (2 * radius + 1) ** 2 - 1
            heuristic_max_score = (num_neighbors_in_radius * max_val_on_grid * value_weight_factor /
                                   (1**distance_decay_factor if distance_decay_factor != 0 else 1.0))
            if heuristic_max_score == 0 and num_neighbors_in_radius > 0 and max_val_on_grid > 0: heuristic_max_score = 1e-9
            if heuristic_max_score > 0:
                scores[r_idx, c_idx] = _math_utils.normalize_value(proximity_score, 0, heuristic_max_score, clamp=True)
            elif math.isclose(proximity_score, 0.0): scores[r_idx, c_idx] = 0.0
            else:
                scores[r_idx, c_idx] = 0.5
                module_logger.warning("EXT_A2: Heuristic max score is zero but proximity score is not.",
                                 extra={"proximity_score": proximity_score, "heuristic_max_score": heuristic_max_score})
    return scores

def EXT_M3_Local_Heterogeneity_Vec(
    grid: NDArray[np.int_], request_id: str | None = None, **kwargs: Any
) -> NDArray[np.float64]:
    module_logger.debug("Executing EXT_M3_Local_Heterogeneity_Vec")
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=np.float64)
    if rows == 0 or cols == 0: return scores
    radius = 1; min_neighbors_for_robust_score = 2
    all_possible_values_in_game = _board_analyzer_utils.get_all_possible_numbers_for_grid(grid.shape)
    if not all_possible_values_in_game: return scores
    max_theoretical_entropy: float
    if len(all_possible_values_in_game) > 1: max_theoretical_entropy = math.log2(len(all_possible_values_in_game))
    elif len(all_possible_values_in_game) == 1: max_theoretical_entropy = math.log2(2)
    else: max_theoretical_entropy = 1.0
    if max_theoretical_entropy == 0: max_theoretical_entropy = 1.0
    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            raw_neighbor_values = _board_analyzer_utils.get_neighborhood_values(
                grid, r_idx, c_idx, radius=radius, eight_connectivity=True,
                val_func=lambda x_val: int(x_val) if x_val != -1 else None, include_center=False)
            processed_neighbor_values: list[Hashable] = [val for val in raw_neighbor_values if val is not None]
            if len(processed_neighbor_values) < min_neighbors_for_robust_score:
                scores[r_idx, c_idx] = 0.0; continue
            current_entropy = _math_utils.get_entropy(processed_neighbor_values)
            normalized_score = current_entropy / max_theoretical_entropy
            scores[r_idx, c_idx] = _math_utils.normalize_value(normalized_score, 0, 1, clamp=True)
    return scores

def EXT_F10_Discontinuity_Vec(
    grid: NDArray[np.int_], request_id: str | None = None, **kwargs: Any
) -> NDArray[np.float64]:
    module_logger.debug("Executing EXT_F10_Discontinuity_Vec")
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=np.float64)
    if rows == 0 or cols == 0: return scores
    legal_values_for_placement = _board_analyzer_utils.get_legal_values_for_placement(grid)
    if not legal_values_for_placement: return scores
    min_sequence_len_to_score = 3
    heuristic_max_len = float(max(rows, cols, min_sequence_len_to_score))
    if heuristic_max_len == 0: heuristic_max_len = float(min_sequence_len_to_score)
    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_len_contribution_for_this_cell = 0.0
            for val_to_try_float in legal_values_for_placement:
                val_to_try = int(val_to_try_float)
                temp_grid = grid.copy(); temp_grid[r_idx, c_idx] = val_to_try
                current_val_max_len = 0.0
                # Row
                row_line = list(temp_grid[r_idx, :])
                for seq in _board_analyzer_utils.find_sequences_in_line(row_line, min_sequence_len_to_score, True, False, 1):
                    if val_to_try in seq: current_val_max_len = max(current_val_max_len, float(len(seq)))
                # Col
                col_line = list(temp_grid[:, c_idx])
                for seq in _board_analyzer_utils.find_sequences_in_line(col_line, min_sequence_len_to_score, True, False, 1):
                    if val_to_try in seq: current_val_max_len = max(current_val_max_len, float(len(seq)))
                # Diagonals
                if cols > 0 and rows > 0:
                    diag1_line = list(np.diag(temp_grid, k=c_idx - r_idx))
                    for seq in _board_analyzer_utils.find_sequences_in_line(diag1_line, min_sequence_len_to_score, True, False, 1):
                        if val_to_try in seq: current_val_max_len = max(current_val_max_len, float(len(seq)))
                    flipped_temp_grid = np.fliplr(temp_grid); flipped_c_idx = cols - 1 - c_idx
                    diag2_line = list(np.diag(flipped_temp_grid, k=flipped_c_idx - r_idx))
                    for seq in _board_analyzer_utils.find_sequences_in_line(diag2_line, min_sequence_len_to_score, True, False, 1):
                        if val_to_try in seq: current_val_max_len = max(current_val_max_len, float(len(seq)))
                if current_val_max_len >= min_sequence_len_to_score:
                    max_len_contribution_for_this_cell = max(max_len_contribution_for_this_cell, current_val_max_len)
            scores[r_idx, c_idx] = _math_utils.normalize_value(max_len_contribution_for_this_cell, 0, heuristic_max_len, clamp=True)
    return scores

REGISTERED_MODULES_BRAIN = {
    "EXT_A2_Weighted_Proximity_Vec": EXT_A2_Weighted_Proximity_Vec,
    "EXT_M3_Local_Heterogeneity_Vec": EXT_M3_Local_Heterogeneity_Vec,
    "EXT_F10_Discontinuity_Vec": EXT_F10_Discontinuity_Vec,
}

@app.post("/score_grid", response_model=ScoreOutput)
async def score_grid_endpoint(grid_input: GridInput) -> ScoreOutput:
    current_req_id = request_id_contextvar.get() # Should be set by middleware
    current_tr_id = trace_id_contextvar.get()   # Should be set by middleware

    module_logger.info(
        "Processing /score_grid request", # Filter adds req_id, tr_id
        extra={"module_name_requested": grid_input.module_name}
    )
    try:
        grid_list = grid_input.grid
        if not grid_list and not (isinstance(grid_list, list) and len(grid_list) == 0 and (len(grid_list[0])==0 if len(grid_list)>0 else True)): # Check for truly empty or 0x0
            grid_np: NDArray[np.int_] = np.array([[]], dtype=np.int_).reshape(0,0)
        elif not all(isinstance(row, list) for row in grid_list) or \
             (len(grid_list) > 0 and not all(len(row) == len(grid_list[0]) for row in grid_list if grid_list)): # Added check for grid_list not empty
            detail = "Grid must be a list of lists with consistent row lengths."
            module_logger.error(detail) # Filter adds req_id, tr_id
            raise HTTPException(status_code=400, detail=detail)
        else:
            grid_np = np.array(grid_list, dtype=np.int_)
        if grid_np.ndim != 2 :
             if grid_np.shape == (0,): # np.array([]) results in shape (0,)
                 grid_np = np.array([[]], dtype=np.int_).reshape(0,0) # Convert to 0x0
             else:
                detail = f"Grid must be 2-dimensional after conversion, got shape {grid_np.shape}."
                module_logger.error(detail, extra={"grid_shape_error": str(grid_np.shape)})
                raise HTTPException(status_code=400, detail=detail)
    except ValueError as ve:
        module_logger.error(f"Invalid grid format during NumPy conversion: {str(ve)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid grid format: {str(ve)}") from ve
    except HTTPException: raise
    except Exception as e_conv:
        module_logger.error(f"Unexpected error during grid conversion: {str(e_conv)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Unexpected error processing grid: {str(e_conv)}")

    score_array = await get_module_score_async(grid_input.module_name, grid_np)
    
    return ScoreOutput(
        module_name=grid_input.module_name,
        score_grid=score_array.tolist(),
        request_id=current_req_id,
        trace_id=current_tr_id
    )

if __name__ == "__main__":
    uvicorn_log_level = settings.LOG_LEVEL.lower()
    if uvicorn_log_level not in ["critical", "error", "warning", "info", "debug", "trace"]:
        uvicorn_log_level = "info"
    module_logger.info(f"Starting {settings.APP_NAME} on http://localhost:8000 with log level {uvicorn_log_level}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level=uvicorn_log_level)