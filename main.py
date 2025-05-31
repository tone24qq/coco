# main.py
import asyncio
import logging
import math
import random
import uuid
from collections import Counter, deque
from contextvars import ContextVar
from typing import Any, Callable, Hashable # Removed TypeAlias for <3.10 compatibility if needed elsewhere, but Callable is fine

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from numpy.typing import NDArray
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette_prometheus import PrometheusMiddleware, metrics

# --- Request ID ContextVar ---
request_id_contextvar: ContextVar[str | None] = ContextVar("request_id", default=None)

# --- Settings via Pydantic BaseSettings (.env file) ---
class AppSettings(BaseSettings):
    """Application settings."""

    LOG_LEVEL: str = Field("INFO", description="Logging level")
    APP_NAME: str = Field("BrainAPI", description="Application name")
    # Add other settings as needed, e.g., API keys, external URLs

    class Config:
        """Pydantic BaseSettings config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = AppSettings()

# --- Logging Configuration ---
class JsonFormatter(logging.Formatter):
    """JSON Log Formatter with request_id."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "N/A_system"),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        
        # Add other 'extra' attributes from the LogRecord if they exist
        # Be careful not to overwrite standard fields or include sensitive data without purpose
        for key, value in record.__dict__.items():
            if key not in log_record and key not in ("args", "asctime", "created", "exc_info", "exc_text", "filename",
                                                     "levelname", "levelno", "lineno", "message", "module", "msecs",
                                                     "msg", "name", "pathname", "process", "processName", "relativeCreated",
                                                     "stack_info", "thread", "threadName", "taskName", "request_id"): # exclude standard and already handled
                log_record[key] = value

        return str(log_record).replace("'", '"') # Basic JSON-like string

# Set up root logger
logger = logging.getLogger()
logger.setLevel(settings.LOG_LEVEL.upper())

# Remove existing handlers to avoid duplicate logs
if logger.hasHandlers():
    logger.handlers.clear()

# Add new handler with JSON formatter
handler = logging.StreamHandler() # Output to stdout
formatter = JsonFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)

# Specific logger for this module, will inherit root logger's config
module_logger = logging.getLogger(__name__)


# --- Request ID Middleware ---
class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware to handle request IDs."""

    async def dispatch(self, request: Request, call_next: Callable): # type: ignore[type-arg]
        """Attach a request ID to each request."""
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_contextvar.set(request_id)

        # Add request_id to the logging record for all subsequent logs in this request context
        # This can be done by adapting the logger or using a filter.
        # For simplicity, we rely on the formatter pulling from LogRecord.request_id
        # which is populated by passing `extra={'request_id': request_id}`
        # Or, as done here, the formatter can try to access the contextvar directly,
        # but it's cleaner if `extra` is used or a filter sets it on the record.
        # The JsonFormatter tries `getattr(record, "request_id", ...)`

        module_logger.debug( # Example of logging with explicit request_id in extra
            f"Request started: {request.method} {request.url.path}",
            extra={"request_id": request_id, "http_method": request.method, "http_path": request.url.path}
        )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        request_id_contextvar.reset(token)
        return response

# --- FastAPI Application Setup ---
app = FastAPI(
    title=settings.APP_NAME,
    description="API for Brain Module Grid Scoring",
    version="1.0.0",
)

app.add_middleware(PrometheusMiddleware)
app.add_middleware(RequestIdMiddleware)
app.add_route("/metrics", metrics)


# --- Pydantic Models for API ---
class GridInput(BaseModel):
    """Input for scoring a grid."""

    grid: list[list[int]] = Field(..., description="The game grid, -1 for empty cells.")
    module_name: str = Field(..., description="Name of the scoring module to use.")
    request_id: str | None = Field(None, description="Optional request ID to trace calls.")

    model_config = ConfigDict(extra="forbid")


class ScoreOutput(BaseModel):
    """Output containing the scores for the grid."""

    module_name: str = Field(..., description="Name of the executed scoring module.")
    score_grid: list[list[float]] = Field(..., description="The resulting scores for each cell.")
    request_id: str | None = Field(None, description="Request ID associated with this scoring.")

    model_config = ConfigDict(extra="forbid")


# === Helper Utilities ===
class MathUtils:
    """Provides common math tools, ensuring consistent calculation styles across modules."""

    def sigmoid(self, x: float, k: float = 1.0) -> float:
        """
        Safe sigmoid function, avoids overflow.

        Args:
            x: The input value.
            k: Scaling factor for x.

        Returns:
            The sigmoid value.
        """
        try:
            clamped_x = np.clip(-k * x, -700.0, 700.0) # 新寫法 ✅
            return 1.0 / (1.0 + math.exp(clamped_x))
        except OverflowError:
            return 0.0 if -k * x > 0 else 1.0 # 新寫法 ✅ (logic remains)


    def normalize_value(
        self, value: float, min_val: float, max_val: float, clamp: bool = True
    ) -> float:
        """
        Normalizes a value to the [0, 1] range.
        Handles cases where min_val equals max_val to prevent division by zero.
        Addresses Requirement 2.c (reasonable score distribution).
        強化:處理 min_val 和 max_val相等時,根據 value 與其的關係返回0.0,0.5,或1.0,更
        精確地處理邊界情況。

        Args:
            value: The value to normalize.
            min_val: The minimum possible value in the original range.
            max_val: The maximum possible value in the original range.
            clamp: If True, clamps the output to [0, 1].

        Returns:
            The normalized value.
        """
        if math.isclose(max_val, min_val):
            if math.isclose(value, min_val):
                return 0.5
            elif value < min_val:
                return 0.0
            else:  # value > max_val (which is min_val)
                return 1.0

        if (max_val - min_val) == 0: # Should be caught by isclose above, but as a safeguard
             return 0.5 if math.isclose(value, min_val) else (0.0 if value < min_val else 1.0)


        normalized = (value - min_val) / (max_val - min_val)
        if clamp:
            return float(np.clip(normalized, 0.0, 1.0)) # 新寫法 ✅ Ensure float return
        return normalized

    def manhattan_distance(self, p1: tuple[int, int], p2: tuple[int, int]) -> int:
        """Calculates Manhattan distance between two points (r, c)."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def euclidean_distance(self, p1: tuple[int, int] | tuple[float, float], p2: tuple[int, int] | tuple[float, float]) -> float:
        """Calculates Euclidean distance between two points (r, c)."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_entropy(self, values: list[Hashable]) -> float:
        """
        Calculates Shannon entropy for a list of values.
        Values can be numbers or any hashable type for frequency counting.
        """
        if not values:
            return 0.0

        counts = Counter(values)
        total_count = len(values)
        entropy = 0.0
        for count_val in counts.values():
            probability = count_val / total_count
            if probability > 0: # Avoid math.log2(0)
                entropy -= probability * math.log2(probability)
        return entropy


class BoardAnalyzerUtils:
    """
    Provides common board analysis utility functions.
    Used by modules to inspect grid neighborhoods, gradients, etc.
    """

    def get_neighborhood_values(
        self,
        grid: NDArray[np.int_],
        r: int,
        c: int,
        radius: int = 1,
        eight_connectivity: bool = True,
        val_func: Callable[[int], float | None] = lambda x_val: float(x_val)
        if x_val != -1
        else None,
        include_center: bool = False,
    ) -> list[float]:
        """
        Retrieves values from the neighborhood of a cell.
        Supports configurable radius, connectivity, and value processing.
        """
        neighbors: list[float] = []
        rows, cols = grid.shape

        for dr_offset in range(-radius, radius + 1):
            for dc_offset in range(-radius, radius + 1):
                if not include_center and dr_offset == 0 and dc_offset == 0:
                    continue

                if not eight_connectivity:
                    if radius == 1 and abs(dr_offset) + abs(dc_offset) != 1:
                        continue
                    elif radius > 1 and abs(dr_offset) + abs(dc_offset) > radius:
                        continue

                nr, nc = r + dr_offset, c + dc_offset

                if 0 <= nr < rows and 0 <= nc < cols:
                    processed_val = val_func(grid[nr, nc])
                    if processed_val is not None:
                        neighbors.append(processed_val)
        return neighbors

    def get_value_gradient_at_cell(
        self,
        grid: NDArray[np.int_],
        r: int,
        c: int,
        val_func: Callable[[int], float] = lambda x_val: float(x_val)
        if x_val != -1
        else 0.0,
    ) -> tuple[float, float]:
        """
        Calculates an approximate gradient (Sobel-like) at a cell.
        Useful for modules analyzing value changes.
        """
        rows, cols = grid.shape

        def safe_val(r_in: int, c_in: int) -> float:
            if 0 <= r_in < rows and 0 <= c_in < cols:
                return val_func(grid[r_in, c_in])
            return 0.0

        # Sobel operator like calculation
        gx = (
            safe_val(r - 1, c + 1)
            + 2 * safe_val(r, c + 1)
            + safe_val(r + 1, c + 1)
        ) - (
            safe_val(r - 1, c - 1)
            + 2 * safe_val(r, c - 1)
            + safe_val(r + 1, c - 1)
        )
        gy = (
            safe_val(r + 1, c - 1)
            + 2 * safe_val(r + 1, c)
            + safe_val(r + 1, c + 1)
        ) - (
            safe_val(r - 1, c - 1)
            + 2 * safe_val(r - 1, c)
            + safe_val(r - 1, c + 1)
        )
        return gx, gy

    def find_sequences_in_line(
        self,
        line: list[int],
        min_len: int = 3,
        check_arithmetic: bool = True,
        check_geometric: bool = False,
        allow_gaps: int = 0,
    ) -> list[list[int]]:
        """
        Finds arithmetic or geometric sequences in a 1D list of numbers.
        Simplified version focusing on arithmetic as geometric logic from PDF was complex.
        """
        sequences: list[list[int]] = []
        n = len(line)
        if n < min_len:
            return sequences

        if check_arithmetic:
            for i in range(n):
                if line[i] == -1:
                    continue
                for j in range(i + 1, n):
                    # Try to establish initial difference
                    if line[j] == -1:
                        if allow_gaps > 0: # Need to find the next non-gap number to establish diff
                            current_gap_count_for_diff = 0
                            next_non_gap_idx = -1
                            for k_search in range(j, n):
                                if line[k_search] == -1:
                                    current_gap_count_for_diff += 1
                                else:
                                    next_non_gap_idx = k_search
                                    break
                            if next_non_gap_idx != -1 and current_gap_count_for_diff <= allow_gaps:
                                diff = line[next_non_gap_idx] - line[i]
                                current_seq = [line[i], line[next_non_gap_idx]]
                                current_gap_count = current_gap_count_for_diff
                                # Extend sequence
                                for l_extend in range(next_non_gap_idx + 1, n):
                                    if line[l_extend] == -1:
                                        current_gap_count += 1
                                        if current_gap_count > allow_gaps:
                                            break
                                        continue
                                    expected = current_seq[-1] + diff
                                    if math.isclose(float(line[l_extend]), float(expected)):
                                        current_seq.append(line[l_extend])
                                        current_gap_count = 0
                                    else: # Sequence broken
                                        break
                                if len(current_seq) >= min_len:
                                    sequences.append(list(current_seq))
                            # else: no valid element to form diff or too many gaps
                        continue # Move to next j if initial j is a gap and no diff established

                    # line[j] is not -1
                    diff = line[j] - line[i]
                    current_seq = [line[i], line[j]]
                    current_gap_count = 0
                    for k in range(j + 1, n):
                        if line[k] == -1:
                            current_gap_count += 1
                            if current_gap_count > allow_gaps:
                                break
                            continue
                        expected = current_seq[-1] + diff
                        if math.isclose(float(line[k]), float(expected)):
                            current_seq.append(line[k])
                            current_gap_count = 0
                        else: # Sequence broken
                            break
                    if len(current_seq) >= min_len:
                        sequences.append(list(current_seq))
        # Geometric sequence check is omitted for brevity as per PDF's complexity and focus
        return sequences


    def get_card_max_value_from_grid_dimensions(
        self, grid_shape: tuple[int, int]
    ) -> int:
        """Calculates the maximum possible number on the card based on its dimensions."""
        rows, cols = grid_shape
        if rows == 0 or cols == 0:
            return 0
        return rows * cols

    def get_all_possible_numbers_for_grid(
        self, grid_shape: tuple[int, int]
    ) -> set[int]:
        """
        Returns a set of all numbers that could theoretically appear on a grid of given dimensions.
        """
        max_val = self.get_card_max_value_from_grid_dimensions(grid_shape)
        if max_val == 0:
            return set()
        return set(range(1, max_val + 1))

    def get_legal_values_for_placement(self, grid: NDArray[np.int_]) -> set[int]:
        """
        Determines the set of numbers that can be legally placed onto an empty cell in the grid.
        This adheres to the rule: numbers are 1 to R*C and no positive number can be repeated.
        """
        if grid.size == 0:
            return set()

        rows, cols = grid.shape
        all_possible_on_this_grid = self.get_all_possible_numbers_for_grid(
            (rows, cols)
        )
        
        used_positive_values_on_board: set[int] = set() # 新寫法 ✅
        for v_flat in grid.flat:
            v = int(v_flat) # Ensure it's int for comparison and set addition
            if v != -1 and v > 0:
                 used_positive_values_on_board.add(v)

        legal_placements = all_possible_on_this_grid - used_positive_values_on_board
        return legal_placements


# Initialize utility instances
_math_utils = MathUtils()
_board_analyzer_utils = BoardAnalyzerUtils()

# === Brain Core Dispatch Area ===
# Type alias for scoring functions
ScoringFunctionType = Callable[[NDArray[np.int_], str | None], NDArray[np.float64]] # Corrected np.float_ to np.float64
REGISTERED_MODULES_BRAIN: dict[str, ScoringFunctionType] = {}


async def get_module_score_async(
    module_name: str, grid: NDArray[np.int_], request_id_val: str | None, **kwargs: Any
) -> NDArray[np.float64]: # Corrected return type
    """
    Retrieves and executes a specific scoring module from the registry asynchronously.
    Uses asyncio.to_thread for potentially CPU-bound module functions.

    Args:
        module_name: The registered name of the module to execute.
        grid: The input numpy array representing the game board.
        request_id_val: The request ID for logging and tracing.
        kwargs: Additional keyword arguments for the module.

    Returns:
        A numpy array containing the scores for each cell, as computed by the module.
        Returns a zero array of the same shape if the module is not found or an error occurs.
    """
    effective_request_id = request_id_val or request_id_contextvar.get() or "N/A_brain_dispatch_async"

    if module_name not in REGISTERED_MODULES_BRAIN:
        module_logger.error(
            f"Module {module_name} not found in REGISTERED_MODULES_BRAIN.",
            extra={"request_id": effective_request_id, "module_name": module_name},
        )
        rows, cols = grid.shape if grid.ndim == 2 and grid.size > 0 else (0,0)
        return np.zeros((rows, cols), dtype=np.float64) # Corrected dtype

    module_func = REGISTERED_MODULES_BRAIN[module_name]
    module_logger.info(
        f"Executing module: {module_name} via to_thread",
        extra={"request_id": effective_request_id, "module_name": module_name},
    )

    try:
        kwargs_for_module = kwargs.copy()
        # Check if 'request_id' is an expected parameter by the module_func
        import inspect
        sig = inspect.signature(module_func)
        if 'request_id' in sig.parameters:
            kwargs_for_module['request_id'] = effective_request_id
        elif 'request_id' in kwargs_for_module: # remove if not in signature
            del kwargs_for_module['request_id']


        # Ensure grid is passed as the first positional argument
        score_grid = await asyncio.to_thread(module_func, grid, **kwargs_for_module)
        
        if not isinstance(score_grid, np.ndarray):
            module_logger.error(
                f"Module {module_name} returned type {type(score_grid)}, expected np.ndarray.",
                extra={"request_id": effective_request_id, "module_name": module_name, "returned_type": str(type(score_grid))},
            )
            raise TypeError(f"Module {module_name} did not return a NumPy array.")
        if score_grid.shape != grid.shape:
            module_logger.error(
                f"Module {module_name} returned shape {score_grid.shape}, expected {grid.shape}.",
                extra={"request_id": effective_request_id, "module_name": module_name, "returned_shape": str(score_grid.shape), "expected_shape": str(grid.shape)},
            )
            raise ValueError(f"Module {module_name} returned array with incorrect shape.")
            
        return score_grid.astype(np.float64) # Ensure correct dtype
    except Exception as e:
        module_logger.error(
            f"Error executing module {module_name}: {str(e)}",
            exc_info=True, # Provides stack trace
            extra={"request_id": effective_request_id, "module_name": module_name},
        )
        rows, cols = grid.shape if grid.ndim == 2 and grid.size > 0 else (0,0)
        return np.zeros((rows, cols), dtype=np.float64) # Corrected dtype


# --- Scoring Module Implementations (Modernized) ---
def EXT_A2_Weighted_Proximity_Vec(
    grid: NDArray[np.int_], request_id: str | None = None, **kwargs: Any # Added **kwargs
) -> NDArray[np.float64]: # Corrected return type
    """
    (A2-加權鄰近性)
    """
    effective_request_id = request_id or request_id_contextvar.get() or "N/A_brain_A2"
    module_logger.debug(
        "Executing EXT_A2_Weighted_Proximity_Vec",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=np.float64) # Corrected dtype
    if rows == 0 or cols == 0:
        return scores

    radius = 2
    value_weight_factor = 0.15
    distance_decay_factor = 1.8

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:
                continue

            proximity_score = 0.0
            for dr_offset in range(-radius, radius + 1):
                for dc_offset in range(-radius, radius + 1):
                    if dr_offset == 0 and dc_offset == 0:
                        continue
                    
                    nr, nc = r_idx + dr_offset, c_idx + dc_offset
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                        dist = _math_utils.manhattan_distance(
                            (r_idx, c_idx), (nr, nc)
                        )
                        if dist == 0:
                            dist = 1

                        score_contribution = (
                            grid[nr, nc] * value_weight_factor
                        ) / (dist**distance_decay_factor)
                        proximity_score += score_contribution
            
            max_val_on_grid = float(
                _board_analyzer_utils.get_card_max_value_from_grid_dimensions(
                    (rows, cols)
                )
            )
            if max_val_on_grid == 0:
                max_val_on_grid = 1.0

            num_neighbors_in_radius = (2 * radius + 1) ** 2 - 1
            heuristic_max_score = (
                num_neighbors_in_radius
                * max_val_on_grid
                * value_weight_factor
                / (1**distance_decay_factor if distance_decay_factor != 0 else 1.0) # Avoid 1**0 issues if decay is 0
            )
            if heuristic_max_score == 0 and num_neighbors_in_radius > 0 and max_val_on_grid > 0 : # If factors are non-zero but result is zero
                heuristic_max_score = 1e-9 # Small number to avoid div by zero if proximity_score can be non-zero

            if heuristic_max_score > 0: # or math.isclose(heuristic_max_score, 0.0) and proximity_score > 0 :
                scores[r_idx, c_idx] = _math_utils.normalize_value(
                    proximity_score, 0, heuristic_max_score, clamp=True
                )
            elif math.isclose(proximity_score, 0.0):
                 scores[r_idx, c_idx] = 0.0
            else: # proximity_score is non-zero but heuristic_max_score is zero, implies an issue or edge case.
                 # Default to 0 or 1 based on interpretation, or log a warning.
                 scores[r_idx, c_idx] = 0.5 # Neutral score if normalization is problematic
                 module_logger.warning(
                     "Heuristic max score is zero in EXT_A2, but proximity score is not. Check factors.",
                     extra={"request_id": effective_request_id, "proximity_score": proximity_score, "heuristic_max_score": heuristic_max_score}
                 )

    return scores


def EXT_M3_Local_Heterogeneity_Vec(
    grid: NDArray[np.int_], request_id: str | None = None, **kwargs: Any # Added **kwargs
) -> NDArray[np.float64]: # Corrected return type
    """
    (M3 - 局部異質性)
    """
    effective_request_id = request_id or request_id_contextvar.get() or "N/A_brain_M3"
    module_logger.debug("Executing EXT_M3_Local_Heterogeneity_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=np.float64) # Corrected dtype
    if rows == 0 or cols == 0:
        return scores

    radius = 1
    min_neighbors_for_robust_score = 2
    
    all_possible_values_in_game = _board_analyzer_utils.get_all_possible_numbers_for_grid(grid.shape)
    if not all_possible_values_in_game: # No numbers possible, so no diversity
        return scores # scores remain zeros

    max_theoretical_entropy: float
    if len(all_possible_values_in_game) > 1:
        max_theoretical_entropy = math.log2(len(all_possible_values_in_game))
    elif len(all_possible_values_in_game) == 1: # Only one possible number
        max_theoretical_entropy = math.log2(2) # Avoid log2(1)=0, or treat as 0 if N=1 means no diversity by definition
                                              # PDF implies giving some scale log2(2)
    else: # No possible values (empty set)
        max_theoretical_entropy = 1.0 # Fallback, though handled by early exit if not all_possible_values_in_game

    if max_theoretical_entropy == 0: # If it somehow becomes 0 (e.g. log2(1))
        max_theoretical_entropy = 1.0 # Prevent division by zero

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:
                continue

            # Ensure val_func returns hashable items for Counter in get_entropy
            raw_neighbor_values = _board_analyzer_utils.get_neighborhood_values(
                grid, r_idx, c_idx, radius=radius, eight_connectivity=True,
                val_func=lambda x_val: int(x_val) if x_val != -1 else None, # Produces int | None
                include_center=False
            )
            
            # Filter out Nones and ensure they are hashable (ints are)
            processed_neighbor_values: list[Hashable] = [val for val in raw_neighbor_values if val is not None]


            if len(processed_neighbor_values) < min_neighbors_for_robust_score:
                scores[r_idx, c_idx] = 0.0
                continue
            
            current_entropy = _math_utils.get_entropy(processed_neighbor_values)

            normalized_score = current_entropy / max_theoretical_entropy
            scores[r_idx, c_idx] = _math_utils.normalize_value(normalized_score, 0, 1, clamp=True)

    return scores


def EXT_F10_Discontinuity_Vec(
    grid: NDArray[np.int_], request_id: str | None = None, **kwargs: Any # Added **kwargs
) -> NDArray[np.float64]: # Corrected return type
    """
    (F10-不連續性修復/序列完成度)
    """
    effective_request_id = request_id or request_id_contextvar.get() or "N/A_brain_F10"
    module_logger.debug("Executing EXT_F10_Discontinuity_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=np.float64) # Corrected dtype
    if rows == 0 or cols == 0:
        return scores

    legal_values_for_placement = _board_analyzer_utils.get_legal_values_for_placement(grid)
    if not legal_values_for_placement:
        return scores

    min_sequence_len_to_score = 3
    heuristic_max_len = float(max(rows, cols, min_sequence_len_to_score))
    if heuristic_max_len == 0 : heuristic_max_len = float(min_sequence_len_to_score) # avoid div by zero

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:
                continue

            max_len_contribution_for_this_cell = 0.0
            for val_to_try_float in legal_values_for_placement: # legal_values are int
                val_to_try = int(val_to_try_float) # Ensure int for grid placement
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = val_to_try
                current_val_max_len = 0.0

                # Check Row
                row_line = list(temp_grid[r_idx, :])
                sequences_in_row = _board_analyzer_utils.find_sequences_in_line(
                    row_line, min_len=min_sequence_len_to_score, allow_gaps=1, check_arithmetic=True
                )
                for seq in sequences_in_row:
                    if val_to_try in seq:
                        current_val_max_len = max(current_val_max_len, float(len(seq)))
                
                col_line = list(temp_grid[:, c_idx])
                sequences_in_col = _board_analyzer_utils.find_sequences_in_line(
                    col_line, min_len=min_sequence_len_to_score, allow_gaps=1, check_arithmetic=True
                )
                for seq in sequences_in_col:
                    if val_to_try in seq:
                        current_val_max_len = max(current_val_max_len, float(len(seq)))

                if cols > 0 and rows > 0 : # Diagonals only make sense if 2D
                    diag1_line = list(np.diag(temp_grid, k=c_idx - r_idx))
                    sequences_in_diag1 = _board_analyzer_utils.find_sequences_in_line(
                        diag1_line, min_len=min_sequence_len_to_score, allow_gaps=1, check_arithmetic=True
                    )
                    for seq in sequences_in_diag1:
                        if val_to_try in seq:
                            current_val_max_len = max(current_val_max_len, float(len(seq)))

                    flipped_temp_grid = np.fliplr(temp_grid)
                    flipped_c_idx = cols - 1 - c_idx
                    diag2_line = list(np.diag(flipped_temp_grid, k=flipped_c_idx - r_idx))
                    sequences_in_diag2 = _board_analyzer_utils.find_sequences_in_line(
                        diag2_line, min_len=min_sequence_len_to_score, allow_gaps=1, check_arithmetic=True
                    )
                    for seq in sequences_in_diag2:
                        if val_to_try in seq:
                            current_val_max_len = max(current_val_max_len, float(len(seq)))

                if current_val_max_len >= min_sequence_len_to_score:
                    max_len_contribution_for_this_cell = max(max_len_contribution_for_this_cell, current_val_max_len)
            
            scores[r_idx, c_idx] = _math_utils.normalize_value(
                max_len_contribution_for_this_cell, 0, heuristic_max_len, clamp=True
            )
    return scores

REGISTERED_MODULES_BRAIN = {
    "EXT_A2_Weighted_Proximity_Vec": EXT_A2_Weighted_Proximity_Vec,
    "EXT_M3_Local_Heterogeneity_Vec": EXT_M3_Local_Heterogeneity_Vec,
    "EXT_F10_Discontinuity_Vec": EXT_F10_Discontinuity_Vec,
}

@app.post("/score_grid", response_model=ScoreOutput)
async def score_grid_endpoint(grid_input: GridInput):
    """
    Scores a given grid using the specified module.
    """
    current_request_id = request_id_contextvar.get() or grid_input.request_id or str(uuid.uuid4())
    # If middleware set it, contextvar.get() is enough.
    # If X-Request-ID header is used, middleware should set it.
    # If passed in body, use grid_input.request_id.
    # Fallback to new UUID.
    
    # Ensure contextvar is set for subsequent operations if not already by middleware
    token = request_id_contextvar.set(current_request_id)


    module_logger.info(
        f"Received scoring request for module: {grid_input.module_name}",
        extra={"request_id": current_request_id, "module_name": grid_input.module_name}
    )

    try:
        grid_list = grid_input.grid
        if not grid_list: # Empty list representing the grid
            if grid_input.module_name in REGISTERED_MODULES_BRAIN : # if module expects 0x0 grid
                 grid_np: NDArray[np.int_] = np.array([[]], dtype=np.int_).reshape(0,0) # Create 0x0 array
            else: # Should not happen if module expects a grid
                 module_logger.warning("Empty grid provided and module might not handle it.", extra={"request_id": current_request_id})
                 grid_np = np.array([[]], dtype=np.int_).reshape(0,0) # default to 0x0
        
        elif not all(isinstance(row, list) for row in grid_list) or \
             (len(grid_list) > 0 and not all(len(row) == len(grid_list[0]) for row in grid_list)):
            detail = "Grid must be a list of lists with consistent row lengths."
            module_logger.error(detail, extra={"request_id": current_request_id})
            raise HTTPException(status_code=400, detail=detail)
        else:
            grid_np = np.array(grid_list, dtype=np.int_)

        if grid_np.ndim != 2: #This check might be redundant if above list checks are robust
            detail = "Grid must be 2-dimensional after conversion."
            module_logger.error(detail, extra={"request_id": current_request_id, "grid_shape": str(grid_np.shape)})
            raise HTTPException(status_code=400, detail=detail)


    except ValueError as ve:
        module_logger.error(
            f"Invalid grid format during NumPy conversion: {str(ve)}", extra={"request_id": current_request_id}
        )
        raise HTTPException(status_code=400, detail=f"Invalid grid format: {str(ve)}") from ve
    except HTTPException: # Re-raise if it's already an HTTPException
        raise
    except Exception as e_conv: # Catch any other conversion errors
        module_logger.error(
            f"Unexpected error during grid conversion: {str(e_conv)}", exc_info=True, extra={"request_id": current_request_id}
        )
        raise HTTPException(status_code=400, detail=f"Unexpected error processing grid: {str(e_conv)}")


    score_array = await get_module_score_async(
        grid_input.module_name, grid_np, current_request_id
    )
    
    response = ScoreOutput(
        module_name=grid_input.module_name,
        score_grid=score_array.tolist(),
        request_id=current_request_id
    )
    request_id_contextvar.reset(token) # Reset contextvar
    return response


# --- Main execution for Uvicorn ---
if __name__ == "__main__":
    # Determine log level for uvicorn from settings, default to lowercase 'info' if not directly mappable
    uvicorn_log_level = settings.LOG_LEVEL.lower()
    valid_uvicorn_levels = ["critical", "error", "warning", "info", "debug", "trace"]
    if uvicorn_log_level not in valid_uvicorn_levels:
        uvicorn_log_level = "info" # Default for uvicorn

    module_logger.info(f"Starting {settings.APP_NAME} on http://localhost:8000 with log level {uvicorn_log_level}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level=uvicorn_log_level)