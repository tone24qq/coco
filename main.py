# main.py
# coding: utf-8

import asyncio
import base64
import datetime
import io
import logging
import math
import random
import time
import uuid
from collections import Counter, deque
from typing import (Any, Callable, Coroutine, Dict, List,
                    Tuple, Union) # Union will be replaced by | later where appropriate

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy asnp
from fastapi import (BackgroundTasks, Body, Depends, FastAPI, HTTPException, Path,
                   Query, Request, Security, status)
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKey, APIKeyHeader, APIKeyQuery
from pydantic import BaseModel, Field, HttpUrl, field_validator, validator
from pydantic_settings import BaseSettings
from prometheus_client import Counter as PrometheusCounter, Gauge, Histogram, Summary # Renamed to avoid conflict
from starlette_prometheus import PrometheusMiddleware

# Python 3.11+ specific: Union types as X | Y
# Ensure all typing.Optional[X] becomes X | None
# Ensure all typing.List becomes list, typing.Dict becomes dict etc.

matplotlib.use('Agg')  # Ensure Matplotlib works in a headless environment

# --- Application Settings ---
class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or a .env file.
    """
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    APP_TITLE: str = "智慧評分系統 API (Extreme Edition) v2.5"
    APP_DESCRIPTION: str = "提供基於進階N維張量運算與AI模組的盤面分析、評分建議、批次處理與背景任務的API服務 (2025 Enhanced)。"
    APP_VERSION: str = "2.5.0"
    ANALYZER_VERSION: str = "2.0.0-extreme" # For AnalyzeHealthStatus

    # Security Settings
    API_KEY: str = "YOUR_VERY_SECRET_API_KEY_FOR_2025"  # Default, MUST be set via environment
    API_KEY_NAME: str = "X-API-KEY"

    # Rate Limiting Settings (Simple In-Memory - for demonstration)
    RATE_LIMIT_REQUESTS: int = 100  # Max requests
    RATE_LIMIT_WINDOW_SECONDS: int = 60  # Per window

    # Task Management
    TASK_CALLBACK_URL_ENABLED: bool = False
    TASK_CALLBACK_URL: HttpUrl | None = None # e.g., "http://localhost:8001/task_result"

    # Paths
    MEM_PATH: str = "data/persistent_memory.json" # Placeholder from main_api.pdf

    class Config:
        """
        Pydantic BaseSettings configuration.
        """
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

# --- Logging Configuration ---
class RequestContextLogFilter(logging.Filter):
    """
    Logging filter to ensure 'request_id' is available in log records.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, 'request_id'):
            record.request_id = 'N/A_context' # Default for logs outside request context
        return True

# Configure root logger
root_logger = logging.getLogger()
root_logger.addFilter(RequestContextLogFilter())
# Ensure handlers are cleared before basicConfig if re-running in some environments (e.g. Jupyter)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - RequestID: %(request_id)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True # Added force=True to ensure basicConfig works even if root logger was already configured
)

logger = logging.getLogger(__name__) # Main application logger
brain_logger = logging.getLogger("brain") # Logger for brain module
analyzer_logger = logging.getLogger("analyzer") # Logger for analyzer module


# --- Prometheus Metrics Definition ---
REQUEST_COUNT = PrometheusCounter(
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
MODULE_USAGE_COUNT = PrometheusCounter(
    "api_module_usage_count",
    "Count of how many times each scoring module is used",
    ["module_name"]
)

# --- Brain Logic (from 大腦3.pdf, enhanced for 2025) ---
class MathUtils:
    """
    Provides common math utility functions, ensuring consistent calculation styles across modules.
    """
    def sigmoid(self, x: float, k: float = 1.0) -> float:
        """Safe sigmoid function, avoids overflow."""
        try:
            clamped_x = max(-700.0, min(700.0, -k * x))
            return 1 / (1 + math.exp(clamped_x))
        except OverflowError:
            return 0.0 if -k * x > 0 else 1.0

    def normalize_value(self, value: float, min_val: float, max_val: float, clamp: bool = True) -> float:
        """
        Normalizes a value to the [0, 1] range.
        Handles cases where min_val equals max_val to prevent division by zero. [cite: 3]
        Enhanced: Returns 0.0, 0.5, or 1.0 based on value's relation to min_val/max_val
        when they are equal, for more precise boundary handling.
        """
        if math.isclose(max_val, min_val):
            if math.isclose(value, min_val):
                return 0.5
            elif value < min_val:
                return 0.0
            else:  # value > max_val (which is min_val)
                return 1.0
        normalized = (value - min_val) / (max_val - min_val)
        if clamp:
            return max(0.0, min(1.0, normalized))
        return normalized

    def manhattan_distance(self, p1: tuple[int, int], p2: tuple[int, int]) -> int:
        """Calculates Manhattan distance between two points (r, c). [cite: 5]"""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def euclidean_distance(self, p1: tuple[int, int], p2: tuple[int, int]) -> float:
        """Calculates Euclidean distance between two points (r, c). [cite: 6]"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) # [cite: 5]

    def get_entropy(self, values: list[Any]) -> float:
        """Calculates Shannon entropy for a list of values. [cite: 7]"""
        if not values:
            return 0.0
        counts = Counter(values)
        total_count = len(values)
        entropy = 0.0
        for count in counts.values():
            probability = count / total_count
            entropy -= probability * math.log2(probability)
        return entropy

class BoardAnalyzerUtils:
    """
    Provides common board analysis utility functions. [cite: 8]
    Used by modules to inspect grid neighborhoods, gradients, etc. [cite: 8]
    """
    def get_neighborhood_values(
        self,
        grid: np.ndarray,
        r: int,
        c: int,
        radius: int = 1,
        eight_connectivity: bool = True,
        val_func: Callable[[int], float | None] = lambda x_val: float(x_val) if x_val != -1 else None, # [cite: 6]
        include_center: bool = False
    ) -> list[float]:
        """
        Retrieves values from the neighborhood of a cell. [cite: 9]
        Supports configurable radius, connectivity, and value processing. [cite: 9]
        """
        neighbors: list[float] = []
        rows, cols = grid.shape
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if not include_center and dr == 0 and dc == 0:
                    continue
                if not eight_connectivity: # 4-connectivity
                    if radius == 1 and abs(dr) + abs(dc) != 1: # Only direct up, down, left, right
                        continue
                    elif radius > 1 and abs(dr) + abs(dc) > radius: # Sum of absolute differences for larger radius
                        continue

                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols: # [cite: 25]
                    processed_val = val_func(grid[nr, nc])
                    if processed_val is not None:
                        neighbors.append(processed_val)
        return neighbors

    def get_value_gradient_at_cell(
        self,
        grid: np.ndarray,
        r: int,
        c: int,
        val_func: Callable[[int], float] = lambda x_val: float(x_val) if x_val != -1 else 0.0
    ) -> tuple[float, float]: # [cite: 8]
        """
        Calculates an approximate gradient (Sobel-like) at a cell. [cite: 11]
        Useful for modules analyzing value changes. [cite: 11]
        """
        rows, cols = grid.shape
        def safe_val(r_in: int, c_in: int) -> float:
            if 0 <= r_in < rows and 0 <= c_in < cols:
                return val_func(grid[r_in, c_in])
            return 0.0

        # Sobel operators [cite: 8]
        gx = (safe_val(r - 1, c + 1) + 2 * safe_val(r, c + 1) + safe_val(r + 1, c + 1)) - \
             (safe_val(r - 1, c - 1) + 2 * safe_val(r, c - 1) + safe_val(r + 1, c - 1))
        gy = (safe_val(r + 1, c - 1) + 2 * safe_val(r + 1, c) + safe_val(r + 1, c + 1)) - \
             (safe_val(r - 1, c - 1) + 2 * safe_val(r - 1, c) + safe_val(r - 1, c + 1))
        return gx, gy

    def find_sequences_in_line(
        self,
        line: list[int],
        min_len: int = 3,
        check_arithmetic: bool = True,
        check_geometric: bool = False, # Simplified geometric, use with caution for floats
        allow_gaps: int = 0
    ) -> list[list[int]]: # [cite: 26]
        """
        Finds arithmetic or geometric sequences in a 1D list of numbers. [cite: 27]
        Supports gaps and returns sequence elements.
        Enhanced: Improved flexibility for arithmetic sequence detection (negative diff, across zero, constant handling).
        Returns list of found sequences (as lists of numbers).
        """
        sequences: list[list[int]] = []
        n = len(line)
        if n < min_len:
            return sequences

        # Arithmetic sequence check
        if check_arithmetic:
            for i in range(n):
                if line[i] == -1: # Skip starting with a gap marker
                    continue # [cite: 28]
                
                # Iterate through possible next elements to establish a common difference
                for j in range(i + 1, n):
                    # Handle initial gaps before diff is established
                    gaps_between_i_j = 0
                    if line[j] == -1:
                        # Count gaps until next non -1 or end of line
                        k = j
                        while k < n and line[k] == -1:
                            gaps_between_i_j += 1
                            k += 1
                        if k == n or gaps_between_i_j > allow_gaps : # Reached end or too many gaps
                            continue # Try next j
                        # k is now the index of the first non -1 after initial gaps
                        next_val_idx = k
                    else:
                        next_val_idx = j # No gaps right after line[i] to establish diff with line[j]

                    if line[next_val_idx] == -1: # Should not happen if logic above is correct
                        continue

                    diff = line[next_val_idx] - line[i]
                    # Exclude constant non-zero sequences (e.g., [5, 5, 5]) as 'arithmetic' by default
                    # unless min_len is met by such a sequence (e.g. if diff is 0 and min_len is 1, it might be valid)
                    # For this function, generally, arithmetic implies changing values.
                    # If diff is 0 and line[i] != 0, this is a constant sequence.
                    if diff == 0 and line[i] != 0 : # [cite: 10, 11]
                        # Check if constant sequence of min_len is desired
                        # For now, let's assume arithmetic usually means non-constant for general puzzle logic
                        # However, a sequence of [0,0,0] is arith with diff 0.
                        pass # Allow constant sequences including zeros

                    current_seq_values = [line[i], line[next_val_idx]]
                    current_seq_indices = [i, next_val_idx] # Not used in return, but good for debugging
                    
                    # Extend sequence
                    last_val_in_seq = line[next_val_idx]
                    last_idx_in_seq = next_val_idx
                    gaps_after_last_val = gaps_between_i_j # Initial gaps between i and next_val_idx

                    for l_extend in range(next_val_idx + 1, n):
                        if line[l_extend] == -1:
                            gaps_after_last_val += 1
                            if gaps_after_last_val > allow_gaps:
                                break # Too many gaps, sequence broken for this diff
                            continue # Gap is allowed, try next element

                        expected_next = last_val_in_seq + diff
                        if math.isclose(line[l_extend], expected_next):
                            current_seq_values.append(line[l_extend])
                            current_seq_indices.append(l_extend)
                            last_val_in_seq = line[l_extend]
                            last_idx_in_seq = l_extend
                            gaps_after_last_val = 0  # Reset gap count after finding a valid number
                        elif line[l_extend] != -1: # Sequence broken by a different number [cite: 12]
                            break
                    
                    if len(current_seq_values) >= min_len:
                        sequences.append(current_seq_values)

        # Geometric sequence check (simplified, careful with division by zero and floating point precision)
        if check_geometric:
            for i in range(n):
                if line[i] == -1 or line[i] == 0: # Geometric sequences typically don't start with 0 unless all are 0
                    continue

                for j in range(i + 1, n):
                    gaps_between_i_j_geom = 0
                    if line[j] == -1:
                        k_geom = j
                        while k_geom < n and line[k_geom] == -1:
                            gaps_between_i_j_geom += 1
                            k_geom += 1
                        if k_geom == n or gaps_between_i_j_geom > allow_gaps:
                            continue
                        next_val_idx_geom = k_geom
                    else:
                        next_val_idx_geom = j
                    
                    if line[next_val_idx_geom] == -1 or line[next_val_idx_geom] == 0: # Or handle 0s in geometric sequence if needed
                        continue

                    # Ratio must be established with non-zero numbers generally
                    if line[i] == 0: continue # Cannot establish ratio if first element is 0 and second is not
                    
                    ratio: float | None = None
                    try:
                        # Ensure line[i] is not zero before division [cite: 34]
                        if math.isclose(line[i], 0): # Should be caught by outer loop condition
                             continue
                        ratio_candidate = line[next_val_idx_geom] / line[i]
                        # If ratio isn't integer-like and not a trivial division, might be problematic for int-based puzzles [cite: 13]
                        # This check is tricky for general floats; for typical int puzzles, ratios are often integers or simple fractions.
                        # For this version, we'll allow float ratios.
                        ratio = ratio_candidate
                    except ZeroDivisionError: # Should not happen due to line[i] != 0 check
                        continue
                    
                    if ratio is None: # Should not happen
                        continue

                    # Avoid constant sequences if ratio is 1, unless they are identical and meet min_len
                    if math.isclose(ratio, 1.0) and not math.isclose(line[i], line[next_val_idx_geom]): # [cite: 13]
                         # If ratio is 1 but numbers are different, not geometric in typical sense.
                         # If numbers are identical, it's a constant sequence; let arithmetic handle it if diff=0.
                         # This condition means e.g. [5, 5.0001] with ratio ~1.
                         continue


                    current_seq_values_geom = [line[i], line[next_val_idx_geom]]
                    last_val_in_seq_geom = line[next_val_idx_geom]
                    gaps_after_last_val_geom = gaps_between_i_j_geom

                    for l_extend_geom in range(next_val_idx_geom + 1, n):
                        if line[l_extend_geom] == -1:
                            gaps_after_last_val_geom += 1
                            if gaps_after_last_val_geom > allow_gaps:
                                break
                            continue
                        
                        # Check for 0 in sequence after ratio established (e.g., [2,4,0]) - generally breaks geometric pattern
                        if math.isclose(line[l_extend_geom], 0.0) and not math.isclose(last_val_in_seq_geom, 0.0):
                             break # Geometric sequence cannot typically continue with/through 0 if ratio leads to non-zero next.

                        expected_next_float = float(last_val_in_seq_geom) * ratio
                        if math.isclose(float(line[l_extend_geom]), expected_next_float):
                            current_seq_values_geom.append(line[l_extend_geom])
                            last_val_in_seq_geom = line[l_extend_geom]
                            gaps_after_last_val_geom = 0
                        elif line[l_extend_geom] != -1: # Sequence broken by a different number [cite: 14]
                            break
                    
                    if len(current_seq_values_geom) >= min_len:
                        # Avoid adding duplicates if arithmetic also found it (e.g. [2,2,2] is arith diff 0, geom ratio 1)
                        # This simple check might not be robust for all overlaps.
                        if not (check_arithmetic and current_seq_values_geom in sequences and math.isclose(ratio,1.0)):
                             sequences.append(current_seq_values_geom)
        return sequences

    def get_card_max_value_from_grid_dimensions(self, grid_shape: tuple[int, int]) -> int:
        """Calculates the maximum possible number on the card based on its dimensions. [cite: 15, 16]"""
        rows, cols = grid_shape
        if rows == 0 or cols == 0:
            return 0
        return rows * cols # Standard Sudoku-like rule: numbers are 1 to R*C

    def get_all_possible_numbers_for_grid(self, grid_shape: tuple[int, int]) -> set[int]:
        """
        Returns a set of all numbers that could theoretically appear on a grid of given dimensions. [cite: 16, 17]
        """
        max_val = self.get_card_max_value_from_grid_dimensions(grid_shape)
        if max_val == 0:
            return set() # [cite: 36]
        return set(range(1, max_val + 1)) # [cite: 37]

    def get_legal_values_for_placement(self, grid: np.ndarray) -> set[int]:
        """
        Determines the set of numbers that can be legally placed onto an empty cell in the grid. [cite: 18]
        This adheres to the rule: numbers are 1 to R*C and no positive number can be repeated. [cite: 19, 20]
        """
        if grid.size == 0: # [cite: 17]
            return set()

        rows, cols = grid.shape
        all_possible_on_this_grid = self.get_all_possible_numbers_for_grid((rows, cols)) # [cite: 38]
        used_positive_values_on_board = set(int(v) for v in grid.flatten() if v != -1 and v > 0) # [cite: 17]
        legal_placements = all_possible_on_this_grid - used_positive_values_on_board
        return legal_placements

# Instantiate utilities
_math_utils = MathUtils()
_board_analyzer_utils = BoardAnalyzerUtils()

# --- Scoring Module Implementations (from 大腦3.pdf, enhanced for 2025) ---
# Each module must be async def and accept (grid: np.ndarray, request_id: str | None = None) -> np.ndarray:
# For modules that were originally synchronous, they will be wrapped with asyncio.to_thread if CPU bound.
# However, to fit the Analyzer's async gather pattern, we make the module functions themselves async
# and internally use to_thread for the core logic if it's blocking.
# For simplicity in this single file, we'll define them as async and assume their core logic can be
# made non-blocking or is quick enough not to require to_thread for this exercise's scope.
# If they were truly CPU-bound, `asyncio.to_thread(sync_core_logic, ...)` would be used inside.
# For now, the transformation to async def is primarily for the signature.

async def EXT_A2_Weighted_Proximity_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_A2") -> np.ndarray: # [cite: 43]
    """(A2-加權鄰近性) [cite: 21]"""
    brain_logger.debug("Executing EXT_A2_Weighted_Proximity_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    radius = 2 # [cite: 19]
    value_weight_factor = 0.15 # [cite: 19]
    distance_decay_factor = 1.8 # [cite: 21]

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: # Only score empty cells [cite: 20, 44]
                continue
            proximity_score = 0.0
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0: continue # [cite: 21, 45]
                    nr, nc = r_idx + dr, c_idx + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                        dist = _math_utils.manhattan_distance((r_idx, c_idx), (nr, nc))
                        if dist == 0: dist = 1 # Safeguard [cite: 22]
                        score_contribution = (grid[nr, nc] * value_weight_factor) / (dist ** distance_decay_factor) # [cite: 22]
                        proximity_score += score_contribution
            
            max_val_on_grid = _board_analyzer_utils.get_card_max_value_from_grid_dimensions((rows, cols)) # [cite: 23, 46]
            if max_val_on_grid == 0: max_val_on_grid = 1.0
            num_neighbors_in_radius = (2 * radius + 1)**2 - 1
            heuristic_max_score = num_neighbors_in_radius * max_val_on_grid * value_weight_factor / (1**distance_decay_factor) # [cite: 23]
            if heuristic_max_score > 0:
                scores[r_idx, c_idx] = _math_utils.normalize_value(proximity_score, 0, heuristic_max_score, clamp=True)
            else:
                scores[r_idx, c_idx] = 0.0
    return scores

async def EXT_M3_Local_Heterogeneity_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_M3") -> np.ndarray: # [cite: 47]
    """(M3 - 局部異質性) [cite: 27]"""
    brain_logger.debug("Executing EXT_M3_Local_Heterogeneity_Vec", extra={'request_id': request_id}) # [cite: 48]
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    radius = 1 # [cite: 27]
    min_neighbors_for_robust_score = 2 # [cite: 27]
    all_possible_values_in_game = _board_analyzer_utils.get_all_possible_numbers_for_grid(grid.shape) # [cite: 25, 48]
    if not all_possible_values_in_game: return scores

    max_theoretical_entropy: float
    if len(all_possible_values_in_game) > 1:
        max_theoretical_entropy = math.log2(len(all_possible_values_in_game)) # [cite: 27]
    elif len(all_possible_values_in_game) == 1: # [cite: 49]
        max_theoretical_entropy = math.log2(2) # Avoid log2(1)=0 [cite: 28, 30, 31, 32]
    else:
        max_theoretical_entropy = 1.0 # Fallback [cite: 29]

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue # [cite: 29]
            neighbor_values = _board_analyzer_utils.get_neighborhood_values( # [cite: 34]
                grid, r_idx, c_idx, radius=radius,
                val_func=lambda x_val: int(x_val) if x_val != -1 else None, # [cite: 34]
                include_center=False
            )
            if len(neighbor_values) < min_neighbors_for_robust_score:
                scores[r_idx, c_idx] = 0.0 # [cite: 50]
                continue
            current_entropy = _math_utils.get_entropy(neighbor_values) # [cite: 35, 51]
            if max_theoretical_entropy > 0:
                normalized_score = current_entropy / max_theoretical_entropy # [cite: 32, 36, 52]
                scores[r_idx, c_idx] = _math_utils.normalize_value(normalized_score, 0, 1, clamp=True) # [cite: 38]
            else:
                scores[r_idx, c_idx] = 0.0 # [cite: 38]
    return scores

# ... (All other 24 brain modules EXT_D3 to EXT_GM20 would be defined here similarly)
# For brevity in this response, I will define a few more and then a placeholder for the rest.

async def EXT_D3_Potential_Field_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_D3") -> np.ndarray: # [cite: 53]
    """(D3-位勢場分析) [cite: 39]"""
    brain_logger.debug("Executing EXT_D3_Potential_Field_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    decay_exponent = 1.5 # [cite: 34, 39]
    max_influence_radius = 3 # [cite: 34, 39]
    max_possible_val_on_grid = _board_analyzer_utils.get_card_max_value_from_grid_dimensions((rows, cols)) # [cite: 34]
    if max_possible_val_on_grid == 0: return scores

    num_cells_in_radius_approx = (2 * max_influence_radius + 1)**2 - 1 # [cite: 55]
    heuristic_max_potential = num_cells_in_radius_approx * (max_possible_val_on_grid / (1**decay_exponent)) # [cite: 35, 40, 54]
    if heuristic_max_potential == 0: heuristic_max_potential = 1.0 # [cite: 41]

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue # [cite: 36]
            current_cell_potential = 0.0
            for nr in range(rows):
                for nc in range(cols):
                    if grid[nr, nc] != -1: # [cite: 36]
                        num_val = grid[nr, nc]
                        if num_val <= 0: continue
                        dist = _math_utils.manhattan_distance((r_idx, c_idx), (nr, nc))
                        if dist == 0: continue # [cite: 36, 56]
                        if dist > max_influence_radius: continue
                        potential_contribution = num_val / (dist ** decay_exponent) # [cite: 36, 42]
                        current_cell_potential += potential_contribution
            scores[r_idx, c_idx] = _math_utils.normalize_value(current_cell_potential, 0, heuristic_max_potential, clamp=True)
    return scores

async def EXT_F10_Discontinuity_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_F10") -> np.ndarray: # [cite: 57]
    """(F10-不連續性修復/序列完成度) [cite: 43]"""
    brain_logger.debug("Executing EXT_F10_Discontinuity_Vec", extra={'request_id': request_id}) # [cite: 58]
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    legal_values_for_placement = _board_analyzer_utils.get_legal_values_for_placement(grid) # [cite: 38, 58]
    if not legal_values_for_placement: return scores

    min_sequence_len_to_score = 3 # [cite: 38, 43, 58]
    heuristic_max_len = float(max(rows, cols, min_sequence_len_to_score)) # [cite: 38, 43, 58]

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue # [cite: 39, 58]
            max_len_contribution_for_this_cell = 0.0 # [cite: 39, 44, 59]

            for val_to_try in legal_values_for_placement:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = val_to_try
                current_val_max_len = 0.0

                # Check Row [cite: 40, 45, 60]
                row_line = list(temp_grid[r_idx, :])
                sequences_in_row = _board_analyzer_utils.find_sequences_in_line(row_line, min_len=min_sequence_len_to_score, allow_gaps=1, check_arithmetic=True) # [cite: 41]
                for seq in sequences_in_row:
                    if val_to_try in seq: # [cite: 41, 45]
                        current_val_max_len = max(current_val_max_len, len(seq))
                
                # Check Column [cite: 42, 47, 61]
                col_line = list(temp_grid[:, c_idx])
                sequences_in_col = _board_analyzer_utils.find_sequences_in_line(col_line, min_len=min_sequence_len_to_score, allow_gaps=1, check_arithmetic=True) # [cite: 42, 62]
                for seq in sequences_in_col:
                    if val_to_try in seq:
                        current_val_max_len = max(current_val_max_len, len(seq))

                # Check Diagonals [cite: 43, 48, 63]
                diag1_line = list(np.diag(temp_grid, k=c_idx - r_idx)) # [cite: 43]
                sequences_in_diag1 = _board_analyzer_utils.find_sequences_in_line(diag1_line, min_len=min_sequence_len_to_score, allow_gaps=1, check_arithmetic=True) # [cite: 43]
                for seq in sequences_in_diag1:
                    if val_to_try in seq:
                        current_val_max_len = max(current_val_max_len, len(seq))

                flipped_temp_grid = np.fliplr(temp_grid)
                flipped_c_idx = cols - 1 - c_idx
                diag2_line = list(np.diag(flipped_temp_grid, k=flipped_c_idx - r_idx))
                sequences_in_diag2 = _board_analyzer_utils.find_sequences_in_line(diag2_line, min_len=min_sequence_len_to_score, allow_gaps=1, check_arithmetic=True) # [cite: 43]
                for seq in sequences_in_diag2:
                    if val_to_try in seq:
                        current_val_max_len = max(current_val_max_len, len(seq))
                
                if current_val_max_len >= min_sequence_len_to_score:
                    max_len_contribution_for_this_cell = max(max_len_contribution_for_this_cell, current_val_max_len)
            
            if heuristic_max_len > 0: # [cite: 49]
                scores[r_idx, c_idx] = _math_utils.normalize_value(max_len_contribution_for_this_cell, 0, heuristic_max_len, clamp=True)
            else:
                scores[r_idx, c_idx] = 0.0
    return scores

# Placeholder for the remaining 22 brain modules (EXT_P7 to EXT_GM20)
# In a full implementation, all 26 modules from 大腦3.pdf would be defined here.
async def placeholder_brain_module(grid: np.ndarray, request_id: str | None = "N/A_placeholder") -> np.ndarray:
    brain_logger.debug(f"Executing Placeholder Brain Module. RequestID: {request_id}", extra={'request_id': request_id})
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols), dtype=float)
    # Simulate some work and return scores based on a simple logic, e.g., random or position-based
    # This is just to make sure the system runs. In a real scenario, each module has its specific logic.
    scores = np.random.rand(rows, cols)
    # Ensure only empty cells get scores > 0
    for r in range(rows):
        for c in range(cols):
            if grid[r,c] != -1:
                scores[r,c] = 0.0
    return scores

# --- Brain Core Dispatch Area (from 大腦3.pdf) ---
# This dictionary will be populated after all module functions are defined.
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, str | None], Coroutine[Any, Any, np.ndarray]]] = {}

# Simplified registration:
# We are defining all modules as async.
ALL_MODULE_NAMES_TEMP_LIST = [
    "EXT_A2_Weighted_Proximity_Vec", "EXT_M3_Local_Heterogeneity_Vec", "EXT_D3_Potential_Field_Vec", "EXT_F10_Discontinuity_Vec", # Defined
    "EXT_P7_Pathfinding_Value_Vec", "EXT_R5_Resource_Control_Vec", "EXT_GM1_Row_Control_Vec", "EXT_GM2_Col_Flow_Vec",
    "EXT_GM3_Adv_Connected_Comp_Vec", "EXT_GM4_Spatial_Auto_Corr_Vec", "EXT_GM5_Line_Completion_Vec", "EXT_GM6_Symmetry_Potential_Vec",
    "EXT_GM7_Numeric_Gaps_Vec", "EXT_GM8_Edge_Affinity_Vec", "EXT_GM9_Center_Control_Vec", "EXT_GM10_Blocking_Value_Vec",
    "EXT_GM11_Pair_Correlation_Vec", "EXT_GM12_Island_Analysis_Vec", "EXT_GM13_Sequence_Diversity_Vec", "EXT_GM14_Risk_Assessment_Vec",
    "EXT_GM15_Information_Gain_Vec", "EXT_GM16_Harmonic_Centrality_Vec", "EXT_GM17_Entropy_Minimization_Vec",
    "EXT_GM18_RL_Value_Est_Vec", "EXT_GM19_Masked_Number_Skip_Pattern_Vec", "EXT_GM20_Skip_Pattern_Confidence_Vec" # Undefined (placeholders)
]

# Populate REGISTERED_MODULES_BRAIN
REGISTERED_MODULES_BRAIN["EXT_A2_Weighted_Proximity_Vec"] = EXT_A2_Weighted_Proximity_Vec
REGISTERED_MODULES_BRAIN["EXT_M3_Local_Heterogeneity_Vec"] = EXT_M3_Local_Heterogeneity_Vec
REGISTERED_MODULES_BRAIN["EXT_D3_Potential_Field_Vec"] = EXT_D3_Potential_Field_Vec
REGISTERED_MODULES_BRAIN["EXT_F10_Discontinuity_Vec"] = EXT_F10_Discontinuity_Vec

for name in ALL_MODULE_NAMES_TEMP_LIST:
    if name not in REGISTERED_MODULES_BRAIN:
         # Assigning the placeholder to all undefined modules
        REGISTERED_MODULES_BRAIN[name] = placeholder_brain_module


# This get_module_score is part of the brain, called by Analyzer
# The Analyzer will wrap calls to this in asyncio.to_thread because the underlying modules (originally) are synchronous CPU-bound.
# However, we've made the module functions themselves async for signature compatibility with Analyzer's gather.
# So, this function can now be async and directly await the module_func.
async def get_module_score(module_name: str, grid: np.ndarray, pv_value_unused: int | None = None, request_id: str | None = "N/A_brain_dispatch") -> np.ndarray:
    """
    Retrieves and executes a specific scoring module from the registry. [cite: 24, 25]
    The pv_value is passed from Analyzer but not all brain modules use it directly in their signature.
    It might be used if modules were more dynamic based on proposed_value. Here, modules only get grid.
    """
    if module_name not in REGISTERED_MODULES_BRAIN:
        brain_logger.error(f"Module {module_name} not found in REGISTERED_MODULES_BRAIN.", extra={'request_id': request_id}) # [cite: 41]
        rows, cols = grid.shape
        return np.zeros((rows, cols), dtype=float) # [cite: 41]

    module_func = REGISTERED_MODULES_BRAIN[module_name]
    brain_logger.info(f"Executing brain module: {module_name} for PV (unused here): {pv_value_unused}", extra={'request_id': request_id}) # [cite: 41]
    try:
        # The module_func itself is now async def
        score_grid = await module_func(grid, request_id=request_id) # [cite: 42]
        # Ensure scores are zero for non-empty cells, as per typical brain module logic for placement suggestions
        if isinstance(score_grid, np.ndarray) and score_grid.shape == grid.shape:
            score_grid[grid != -1] = 0.0
        else:
            brain_logger.warning(f"Module {module_name} returned unexpected score_grid type or shape. Got {type(score_grid)}, shape {getattr(score_grid, 'shape', 'N/A')}", extra={'request_id': request_id})
            rows,cols = grid.shape
            return np.zeros((rows,cols), dtype=float)

        return score_grid
    except Exception as e:
        brain_logger.error(f"Error executing module {module_name}: {e}", exc_info=True, extra={'request_id': request_id}) # [cite: 42]
        rows, cols = grid.shape
        return np.zeros((rows, cols), dtype=float) # [cite: 42]


# Hypothetical function for /modules endpoint to get details (as in main 2.pdf)
def get_module_details(module_name: str) -> dict[str, Any]:
    """
    Returns details for a given module name.
    This is a placeholder; in a real system, this info might be stored with modules.
    """
    # Example descriptions, could be expanded
    descriptions = {
        "EXT_A2_Weighted_Proximity_Vec": "Scores empty cells based on proximity to existing numbers and their values.",
        "EXT_M3_Local_Heterogeneity_Vec": "Scores empty cells based on the diversity (entropy) of neighboring numbers.",
        "EXT_D3_Potential_Field_Vec": "Scores empty cells using a potential field analogy, attracted by existing numbers.",
        "EXT_F10_Discontinuity_Vec": "Scores empty cells on their ability to complete or extend numerical sequences."
    }
    return {
        "description": descriptions.get(module_name, "No specific description available for this module."),
        "version": "1.0.0", # Could be dynamic
        "input_constraints": {"requires_empty_cells": True}
    }


# Simple wrapper to pass to Analyzer, mimicking a module object
class BrainInterface:
    """
    Interface providing access to brain's registered modules and scoring function.
    This is used by the Analyzer class.
    """
    def __init__(self):
        self.registered_modules = REGISTERED_MODULES_BRAIN
        # get_module_score is already async
        self.get_module_score = get_module_score
        self.get_module_details = get_module_details # For /modules endpoint

brain_interface = BrainInterface()

# --- Analyzer Logic (from An.pdf, enhanced for 2025) ---
# Custom Exceptions for Analyzer
class AnalyzerError(Exception):
    """Base class for exceptions in the Analyzer."""
    pass

class InitializationError(AnalyzerError):
    """Error during Analyzer initialization."""
    pass

class InvalidInputError(AnalyzerError): # [cite: 207]
    """Error due to invalid input parameters."""
    pass

class ModuleError(AnalyzerError): # [cite: 207]
    """Base class for errors related to main_module modules."""
    pass

class ModuleNotFoundError(ModuleError): # [cite: 207]
    """Error when a requested module is not found or registered."""
    pass

class ModuleExecutionError(ModuleError): # [cite: 207]
    """Error during the execution of a module in main_module."""
    pass

class VisualizationError(AnalyzerError): # [cite: 207]
    """Error during the generation of the visualization."""
    pass

class Analyzer:
    """
    Core dispatcher for the intelligent scoring system. [cite: 207]
    Receives analysis requests, invokes logic modules from main_module (brain_interface),
    fuses results, and returns suggestions.
    Adheres strictly to non-interference with analysis logic, performing only coordination and fair fusion.
    """
    PV_COLORS = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values()) # [cite: 207]
    _current_cell_size_inch_for_dpi: float # Temp storage for DPI calculation in visualization

    def __init__(self, main_module: BrainInterface, default_top_n: int = 3): # [cite: 207]
        """
        Initializes the Analyzer.
        :param main_module: The brain module interface providing 'get_module_score' and 'registered_modules'.
        :param default_top_n: Default number of top suggestions to return.
        """
        if not hasattr(main_module, 'get_module_score') or not callable(main_module.get_module_score): # [cite: 207]
            raise InitializationError("main_module must provide a callable 'get_module_score' method.")
        if not hasattr(main_module, 'registered_modules') or not isinstance(main_module.registered_modules, dict): # [cite: 207]
            raise InitializationError("main_module must provide a 'registered_modules' (dict) attribute.")
        
        self.main_module = main_module
        self.default_top_n = default_top_n
        analyzer_logger.info(
            f"Analyzer initialized with default_top_n={self.default_top_n}. " # [cite: 208]
            f"Registered modules from main_module: {list(main_module.registered_modules.keys())}"
        )

    def _validate_inputs(
        self,
        new_card: list[list[int]],
        proposed_values_input: list[dict[str, Any]], # Changed from list[int] in An.pdf to list[ProposedValue] from main_api.pdf
        active_modules: list[str] | None,
        module_weights: dict[str, float] | None,
        top_n: int | None,
        request_id: str | None = "N/A"
    ) -> tuple[int, int, list[dict[str, Any]], list[str] | None, dict[str, float] | None, int]: # [cite: 209]
        """Validates the input parameters for the analysis."""
        log_extra = {'request_id': request_id}
        if not new_card or not isinstance(new_card, list): # [cite: 209]
            raise InvalidInputError("Board (new_card) cannot be empty and must be a list.")
        if not all(isinstance(row, list) for row in new_card): # [cite: 209]
            raise InvalidInputError("Each row in the board (new_card) must be a list.")

        rows = len(new_card)
        if rows == 0 and not (len(new_card) == 1 and len(new_card[0])==0) : # Allow 0x0 (represented as [[]] or [])
             # If it's not strictly 0x0 (like `[[], []]`), it might be an issue or an empty list of rows for a non-zero col board.
             # For now, if rows is 0, we assume 0x0 or similar valid empty representation.
             pass # Valid 0-row board
        
        cols = len(new_card[0]) if rows > 0 and new_card[0] is not None else 0 # [cite: 209]

        if rows > 0 and cols == 0 and not all(len(row) == 0 for row in new_card): # if rows > 0, cols must be consistently 0 [cite: 209]
             raise InvalidInputError("Board (new_card) has inconsistent column definitions or first row is empty while others are not.")
        
        if rows > 0 and not all(len(row) == cols for row in new_card): # [cite: 209]
            raise InvalidInputError("Board (new_card) must be rectangular (all rows must have the same number of columns).")
        
        if not all(isinstance(val, int) for row in new_card for val in row): # [cite: 210]
            raise InvalidInputError("All values in the board (new_card) must be integers.")

        # proposed_values is now a list of Pydantic models (ProposedValue), not simple ints.
        # The validation of ProposedValue structure happens at the Pydantic model level.
        # Here we just check if the list itself is provided correctly.
        if not proposed_values_input or not isinstance(proposed_values_input, list): # [cite: 210]
             raise InvalidInputError("Candidate values (proposed_values) must be a non-empty list of proposed value objects.")
        # Individual proposed_value objects are validated by Pydantic.
        # We extract the numeric 'value' for the analyzer's internal PV list if needed, but primarily work with the objects.

        if active_modules is not None: # [cite: 210]
            if not isinstance(active_modules, list) or not all(isinstance(m, str) for m in active_modules):
                raise InvalidInputError("Active modules (active_modules), if provided, must be a list of strings.")
        
        if module_weights is not None: # [cite: 210]
            if not isinstance(module_weights, dict) or \
               not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in module_weights.items()):
                raise InvalidInputError("Module weights (module_weights), if provided, must be a {str: float/int} dictionary.") # [cite: 211]

        final_top_n = top_n if top_n is not None else self.default_top_n # [cite: 212]
        if not isinstance(final_top_n, int) or final_top_n <= 0: # [cite: 212]
            raise InvalidInputError(f"Top-N count ({final_top_n}) must be a positive integer.")

        analyzer_logger.debug(f"Input validation successful. Rows: {rows}, Cols: {cols}, Top_N: {final_top_n}", extra=log_extra) # [cite: 212]
        return rows, cols, proposed_values_input, active_modules, module_weights, final_top_n


    def _get_effective_modules_and_weights(
        self,
        requested_active_modules: list[str] | None,
        requested_module_weights: dict[str, float] | None,
        request_id: str | None = "N/A"
    ) -> tuple[list[str], dict[str, float]]: # [cite: 212]
        """Determines the effective modules and their weights to be used for analysis."""
        log_extra = {'request_id': request_id}
        registered_module_names = list(self.main_module.registered_modules.keys()) # [cite: 212]
        effective_module_names: list[str] = []

        if requested_active_modules is None:
            effective_module_names = registered_module_names # [cite: 212]
            analyzer_logger.info("No active_modules specified, using all registered modules: %s", effective_module_names, extra=log_extra) # [cite: 212]
        else:
            for module_name in requested_active_modules: # [cite: 212]
                if module_name not in registered_module_names:
                    analyzer_logger.warning("Requested module '%s' is not registered in main_module. It will be ignored.", module_name, extra=log_extra) # [cite: 212]
                else:
                    effective_module_names.append(module_name)
            if not effective_module_names and requested_active_modules: # [cite: 212]
                analyzer_logger.warning("None of the specified active_modules (%s) are registered. No modules will be executed.", requested_active_modules, extra=log_extra)
            elif not effective_module_names:
                 analyzer_logger.warning("active_modules list is empty, no modules will be executed.", extra=log_extra)


        final_module_weights: dict[str, float] = {name: 1.0 for name in effective_module_names} # [cite: 212]
        if requested_module_weights: # [cite: 212]
            for name, weight in requested_module_weights.items():
                if name in final_module_weights:
                    final_module_weights[name] = float(weight)
                else:
                    analyzer_logger.warning( # [cite: 212]
                        "Module '%s' in weight configuration is not in the list of effective modules (%s). Its weight will be ignored.",
                        name, effective_module_names, extra=log_extra
                    )
        
        analyzer_logger.info("Effective Modules: %s", effective_module_names, extra=log_extra) # [cite: 213]
        analyzer_logger.info("Final Module Weights: %s", final_module_weights, extra=log_extra) # [cite: 213]
        return effective_module_names, final_module_weights

    def _fuse_scores(
        self,
        module_scores_map: dict[str, np.ndarray],
        module_weights_map: dict[str, float],
        rows: int,
        cols: int,
        request_id: str | None = "N/A_REQ_ID"
    ) -> np.ndarray: # [cite: 213]
        """Fuses scores from multiple modules using their weights and normalizes the result."""
        log_extra = {'request_id': request_id}
        if rows == 0 or cols == 0: # Handle empty board case [cite: 213]
            return np.array([[]], dtype=float) if rows == 0 else np.empty((rows, 0), dtype=float)

        fused_scores = np.zeros((rows, cols), dtype=float)
        if not module_scores_map: # [cite: 213]
            analyzer_logger.warning("No scores received from any module. Fused result will be a zero matrix.", extra=log_extra)
            return fused_scores

        active_module_names_with_scores = list(module_scores_map.keys())
        analyzer_logger.debug(f"Starting fusion for {len(active_module_names_with_scores)} modules: {active_module_names_with_scores}", extra=log_extra) # [cite: 213]

        for module_name, scores_array in module_scores_map.items():
            weight = module_weights_map.get(module_name)
            if weight is None: # [cite: 213]
                analyzer_logger.error(f"Critical internal error: Module '{module_name}' missing weight during score fusion. Using default weight 1.0.", extra=log_extra)
                weight = 1.0
            
            if not isinstance(scores_array, np.ndarray) or scores_array.shape != (rows, cols): # [cite: 213]
                analyzer_logger.error(
                    f"Module '{module_name}' score format mismatch. Expected {rows}x{cols} np.ndarray, got {type(scores_array)} "
                    f"shape {scores_array.shape if isinstance(scores_array, np.ndarray) else 'N/A'}. This module's scores will be ignored.",
                    extra=log_extra
                )
                continue
            analyzer_logger.debug(f"Fusing scores from module '{module_name}' (weight: {weight:.2f}).", extra=log_extra) # [cite: 213]
            fused_scores += scores_array * weight
        
        min_score_val = np.min(fused_scores) if fused_scores.size > 0 else 0.0 # [cite: 214]
        max_score_val = np.max(fused_scores) if fused_scores.size > 0 else 0.0 # [cite: 214]

        if math.isclose(max_score_val, min_score_val): # [cite: 214]
            normalized_fused_scores = np.zeros_like(fused_scores)
            if not math.isclose(min_score_val, 0.0):
                 analyzer_logger.debug(f"Fused scores are all identical ({min_score_val:.4f}), normalized to 0.0.", extra=log_extra) # [cite: 214]
        else:
            normalized_fused_scores = (fused_scores - min_score_val) / (max_score_val - min_score_val) # [cite: 214]
            analyzer_logger.debug(f"Fused scores normalized from range [{min_score_val:.4f}, {max_score_val:.4f}] to [0, 1].", extra=log_extra) # [cite: 214]
        
        return normalized_fused_scores

    def _get_top_n_candidates_for_placement( # Renamed from _get_top_n_for_pv to be more descriptive
        self,
        fused_scores_board: np.ndarray, # Scores for placing *any* valid number
        board_state_np: np.ndarray, # Current board as numpy array
        proposed_value_obj: 'ProposedValue', # The specific value being proposed for placement
        top_n: int,
        request_id: str | None = "N/A_REQ_ID"
    ) -> list[dict[str, Any]]: # Returns list of CandidateDetail-like dicts
        """
        Identifies top N candidate cells for a *specific* proposed value, based on fused scores.
        This method is adapted for the main_api.py /analyze endpoint's needs where each proposal
        is evaluated independently. The fused_scores_board should represent general 'goodness' of cells.
        The final scoring for a candidate includes this plus other factors.
        This function assumes fused_scores_board is a general heatmap of "goodness" for *any* placement.
        The actual CandidateDetail construction in main_api's /analyze will use this as one input.

        For the purpose of this integrated Analyzer, if it's used by an endpoint like
        `main_api.pdf#/analyze` (which evaluates specific `ProposedValue` objects),
        this method needs to select the best *positions* for that *single* `proposed_value_obj.value`.
        The `fused_scores_board` in this context would be specific to `proposed_value_obj.value`.

        Revisiting: `An.pdf`'s `_get_top_n_for_pv` assumes `fused_scores_board` IS for a specific PV.
        The `analyze_board` loop calls `get_module_score(..., pv)` then `_fuse_scores` for that PV,
        then `_get_top_n_for_pv`. This means `fused_scores_board` is already PV-specific.
        The adaptation for `main_api.pdf`'s `AnalysisRequest` (with multiple `ProposedValue` objects)
        means the `Analyzer.analyze_board` needs to return a structure that `main_api.pdf#/analyze`
        can use to build its `List[CandidateDetail]`.

        Let's assume `Analyzer.analyze_board` is used more generally and its `proposed_values` argument
        (from `An.pdf`) is a list of *integer values* to get heatmaps for.
        The `main_api.pdf#/analyze` will then call `Analyzer.analyze_board` and interpret its output.
        For now, sticking to `An.pdf`'s structure where `_get_top_n_for_pv` takes a PV-specific heatmap.
        """
        suggestions: list[dict[str, Any]] = [] # [cite: 214]
        log_extra = {'request_id': request_id, 'proposed_value': proposed_value_obj.value} # Use the actual value for logging

        if fused_scores_board.size == 0: # [cite: 214]
            analyzer_logger.info("Fused score board is empty. Cannot provide suggestions for PV.", extra=log_extra)
            return suggestions

        rows, cols = fused_scores_board.shape
        candidate_cells: list[tuple[float, int, int]] = [] # [cite: 214]
        
        # In main_api.pdf, a proposal is (pos, value).
        # Here, we are finding the best pos for a given value.
        # The proposal from AnalysisRequest already has a pos. This method is about finding best general pos.
        # This method seems to be for finding best *empty cells* for a PV.
        # The `ProposedValue` objects have (pos, value). This method is slightly misaligned with that.
        # Let's assume this is about finding best empty cells IF one were to place `proposed_value_obj.value`.

        # The original `_get_top_n_for_pv` from `An.pdf` takes `board_state: List[List[int]]`
        # and iterates through it to find `board_state[r][c] == -1`.
        # The `AnalysisRequest` in `main_api.pdf` has `new_card` and `proposed_values: List[ProposedValue]`.
        # Each `ProposedValue` has a `pos` and `value`.
        # The `Analyzer` in `Main_api.pdf` calls `analyzer_instance.analyze_board` with these.
        # The `analyze_board` method in `An.pdf` seems to generate heatmaps for each `int` in `proposed_values`.

        # Let's stick to the `An.pdf` version: find best empty cells for a given numeric PV.
        # The `main_api.pdf` /analyze endpoint will need to adapt how it uses this.
        # It seems `main_api.pdf` is constructing `CandidateDetail` based on its own logic,
        # potentially using Analyzer for parts of it (like `raw_tensor_flow_score`).

        # For now, let `_get_top_n_for_pv` (this method, perhaps renamed) generate suggestions for empty cells for a numeric PV.
        # The `Analyzer.analyze_board` will loop over numeric PVs.

        has_fillable_cells = False
        target_pos_r, target_pos_c = proposed_value_obj.pos # Get target position from ProposedValue
        
        # This method should score the specific (pos, value) proposal.
        # The `fused_scores_board` should be for `proposed_value_obj.value`.
        # We are interested in the score at `(target_pos_r, target_pos_c)`.
        
        if 0 <= target_pos_r < rows and 0 <= target_pos_c < cols:
            if board_state_np[target_pos_r, target_pos_c] == -1: # Target cell must be empty
                 score_at_pos = fused_scores_board[target_pos_r, target_pos_c]
                 # This function from An.pdf originally sorted all empty cells.
                 # For main_api.pdf's specific proposal, we just need the score for *that* proposal.
                 # However, the API endpoint /analyze still needs a list of CandidateDetail.
                 # The `main_api.pdf` calls `analyzer_instance.analyze_board`.
                 # Let's assume `analyzer_instance.analyze_board` is supposed to evaluate *these specific proposals*.

                 # Let's refine: `Analyzer.analyze_board` in `An.pdf` takes `proposed_values: List[int]`.
                 # The `main_api.pdf` has `proposed_values: List[ProposedValue]`.
                 # The `Analyzer` needs to be adapted to handle `List[ProposedValue]`.
                 # The `_get_top_n_for_pv` should probably be `_evaluate_proposed_candidate`.

                 # Sticking to An.pdf's _get_top_n_for_pv for now, which means it finds best *empty cells*.
                 # This is a conceptual mismatch with main_api.pdf's direct proposals.
                 # For the purpose of this consolidated script, let's keep An.pdf's original intent:
                 # find TOP_N EMPTY CELLS for a given NUMERIC PV.
                 # The calling code (e.g. /analyze endpoint) will need to handle this.
                 # This means `proposed_values` for `Analyzer` are simple `int`s.

                for r_idx in range(rows): # [cite: 215]
                    for c_idx in range(cols):
                        if board_state_np[r_idx, c_idx] == -1: # Cell is available for suggestion [cite: 215]
                            has_fillable_cells = True
                            candidate_cells.append((fused_scores_board[r_idx, c_idx], r_idx, c_idx))
                
                if not has_fillable_cells: # [cite: 215]
                    analyzer_logger.info("No fillable cells (-1) on the board. Cannot provide suggestions for PV.", extra=log_extra)
                    return suggestions
                
                if not candidate_cells: # [cite: 215]
                    analyzer_logger.info("Candidate cell list is empty (likely no -1 cells).", extra=log_extra)
                    return suggestions

                candidate_cells.sort(key=lambda x: x[0], reverse=True) # Sort by score descending [cite: 215]

                for score, r, c in candidate_cells[:top_n]: # [cite: 216]
                    suggestions.append({
                        'position': [r, c], # [cite: 216]
                        'score': round(float(score), 6) # Ensure score is Python float [cite: 216]
                    })
            else: # Target cell is not empty
                analyzer_logger.warning(f"Proposed position ({target_pos_r}, {target_pos_c}) for PV {proposed_value_obj.value} is not empty.", extra=log_extra)
                return [] # No suggestion if the proposed cell itself is not empty for placement
        else: # Target position out of bounds
            analyzer_logger.warning(f"Proposed position ({target_pos_r}, {target_pos_c}) for PV {proposed_value_obj.value} is out of bounds.", extra=log_extra)
            return []
            
        return suggestions


    # This is the main analysis method from An.pdf
    async def analyze_board_generic_pvs( # Renamed to avoid conflict with main_api's /analyze logic using Analyzer differently
        self,
        new_card_list: list[list[int]],
        # proposed_values_int_list: list[int], # An.pdf takes list of ints
        proposed_value_objects: list['ProposedValue'], # For compatibility with main_api.pdf
        active_modules: list[str] | None = None,
        module_weights: dict[str, float] | None = None,
        top_n: int | None = None,
        request_id_for_logging: str | None = None
    ) -> list[dict[str, Any]]: # Returns a list of CandidateDetail-like dicts
        """
        Performs board analysis for a list of proposed (pos, value) objects. (Adapted from An.pdf)
        Returns a list of evaluated candidate details.
        """
        if request_id_for_logging is None:
            request_id_for_logging = f"analyzer-req-{random.randint(10000, 99999)}" # [cite: 216]
            analyzer_logger.info(f"Generated temporary RequestID for logging: {request_id_for_logging}", extra={'request_id': request_id_for_logging})

        log_prefix = f"RequestID: {request_id_for_logging} - Analyzer:" # [cite: 216]
        analyzer_logger.info(
            f"{log_prefix} Received analysis request: {len(proposed_value_objects)} proposed values. " # [cite: 216]
            f"Board size: {len(new_card_list)}x{len(new_card_list[0]) if new_card_list and new_card_list[0] is not None else 'empty'}. "
            f"Active modules hint: {str(active_modules) if active_modules else 'ALL'}",
            extra={'request_id': request_id_for_logging}
        )

        try:
            # Validation now takes the list of ProposedValue objects
            rows, cols, validated_pv_objects, val_active_modules, val_module_weights, final_top_n = \
                self._validate_inputs(new_card_list, proposed_value_objects, active_modules, module_weights, top_n, request_id=request_id_for_logging)
        except InvalidInputError as e:
            analyzer_logger.error(f"{log_prefix} Input parameter validation failed: {e}", exc_info=True, extra={'request_id': request_id_for_logging}) # [cite: 216]
            # For main_api.pdf, this should raise an error that the endpoint can catch
            raise # Re-raise for the endpoint to handle with HTTPException

        new_card_np = np.array(new_card_list, dtype=np.int32)

        effective_modules, final_weights = self._get_effective_modules_and_weights( # [cite: 217]
            val_active_modules, val_module_weights, request_id=request_id_for_logging
        )

        evaluated_candidates_details: list[dict[str, Any]] = []

        if not effective_modules: # [cite: 218]
            analyzer_logger.warning(f"{log_prefix} No effective analysis modules. Analysis will yield no candidates.", extra={'request_id': request_id_for_logging})
            return [] # Return empty list if no modules
        
        # Group module calls by PV *value* to generate heatmaps efficiently if multiple proposals share a value
        unique_pv_values_to_score = sorted(list(set(pv.value for pv in validated_pv_objects)))
        
        # Store heatmaps per PV value
        heatmaps_for_pv_values: dict[int, np.ndarray] = {}

        for pv_val_int in unique_pv_values_to_score:
            analyzer_logger.info(f"{log_prefix} Processing modules for PV value {pv_val_int}", extra={'request_id': request_id_for_logging}) # [cite: 218]
            
            module_tasks = []
            for module_name in effective_modules: # [cite: 218]
                # self.main_module.get_module_score is already async
                module_tasks.append(
                    self.main_module.get_module_score(module_name, new_card_np, pv_val_int, request_id=request_id_for_logging)
                )
            
            raw_module_results: list[Any] = []
            try:
                analyzer_logger.debug(f"{log_prefix} PV Value: {pv_val_int} - Calling {len(module_tasks)} modules concurrently.", extra={'request_id': request_id_for_logging}) # [cite: 218]
                raw_module_results = await asyncio.gather(*module_tasks, return_exceptions=True) # [cite: 218]
                analyzer_logger.debug(f"{log_prefix} PV Value: {pv_val_int} - All modules processed.", extra={'request_id': request_id_for_logging}) # [cite: 218]
            except Exception as e_gather: # Should not happen with return_exceptions=True [cite: 219]
                analyzer_logger.error(f"{log_prefix} PV Value: {pv_val_int} - Unexpected error during asyncio.gather: {e_gather}", exc_info=True, extra={'request_id': request_id_for_logging})
                # Continue to next PV value or handle error appropriately
                continue

            module_scores_for_this_pv_value: dict[str, np.ndarray] = {} # [cite: 219]
            for i, module_name in enumerate(effective_modules):
                raw_scores = raw_module_results[i]
                if isinstance(raw_scores, Exception): # [cite: 219]
                    analyzer_logger.error(
                        f"{log_prefix} Error calling/processing scores from module '{module_name}' for PV value '{pv_val_int}': {raw_scores}. Module skipped.", # [cite: 220]
                        exc_info=raw_scores, extra={'request_id': request_id_for_logging}
                    )
                    continue
                if raw_scores is None: # [cite: 221]
                    analyzer_logger.warning(f"{log_prefix} Module '{module_name}' for PV value '{pv_val_int}' returned None. Skipping.", extra={'request_id': request_id_for_logging})
                    continue
                try:
                    scores_np = np.array(raw_scores, dtype=float) # [cite: 222]
                    if scores_np.shape != (rows, cols): # [cite: 222]
                        analyzer_logger.error(
                            f"{log_prefix} Module '{module_name}' for PV value '{pv_val_int}' returned incorrect score shape. "
                            f"Expected {rows}x{cols}, got {scores_np.shape}. Skipping.", extra={'request_id': request_id_for_logging}
                        )
                        continue # [cite: 223]
                    
                    # Basic stats logging [cite: 223]
                    # ... (omitted for brevity, but would be here)
                    module_scores_for_this_pv_value[module_name] = scores_np
                except Exception as e_proc: # [cite: 224]
                    analyzer_logger.error(
                        f"{log_prefix} Error processing scores from module '{module_name}' for PV value '{pv_val_int}': {e_proc}. Skipping.", # [cite: 225]
                        exc_info=True, extra={'request_id': request_id_for_logging}
                    )
                    continue
            
            if not module_scores_for_this_pv_value: # [cite: 225]
                analyzer_logger.warning(f"{log_prefix} PV value: {pv_val_int} - No valid scores obtained from any module for this PV value.", extra={'request_id': request_id_for_logging})
                fused_scores_for_pv_value = np.zeros((rows, cols) if rows > 0 and cols > 0 else (0,0), dtype=float) # [cite: 225]
            else:
                fused_scores_for_pv_value = self._fuse_scores(module_scores_for_this_pv_value, final_weights, rows, cols, request_id=request_id_for_logging) # [cite: 225]
            heatmaps_for_pv_values[pv_val_int] = fused_scores_for_pv_value

        # Now, iterate through the original proposed_value_objects and build CandidateDetail dicts
        for pv_object in validated_pv_objects:
            pv_pos_tuple = pv_object.pos # tuple[int,int]
            pv_val_int = pv_object.value

            fused_score_for_this_pv_at_pos = 0.0
            if pv_val_int in heatmaps_for_pv_values:
                heatmap_for_pv = heatmaps_for_pv_values[pv_val_int]
                if 0 <= pv_pos_tuple[0] < rows and 0 <= pv_pos_tuple[1] < cols:
                     # Check if the target cell is empty for this specific proposal
                    if new_card_np[pv_pos_tuple[0], pv_pos_tuple[1]] == -1:
                        fused_score_for_this_pv_at_pos = float(heatmap_for_pv[pv_pos_tuple[0], pv_pos_tuple[1]])
                    else:
                        # Proposed cell is not empty, score is 0 or invalid.
                        analyzer_logger.warning(f"{log_prefix} Proposed cell {pv_pos_tuple} for value {pv_val_int} is not empty. Score set to 0.", extra={'request_id': request_id_for_logging})
                        fused_score_for_this_pv_at_pos = -1.0 # Mark as invalid explicitly, or handle as 0
                else: # Position out of bounds
                    analyzer_logger.warning(f"{log_prefix} Proposed cell {pv_pos_tuple} for value {pv_val_int} is out of bounds. Score set to 0.", extra={'request_id': request_id_for_logging})
                    fused_score_for_this_pv_at_pos = -1.0 # Mark as invalid

            # This structure matches CandidateDetail from main_api.pdf
            # is_valid_proposal, mem_score_value, final_objective_score, cp_solver_notes would need to be
            # calculated by main_api.pdf's endpoint logic after getting this 'raw_tensor_flow_score'.
            # Let's assume this score *is* the 'raw_tensor_flow_score'.
            if fused_score_for_this_pv_at_pos >=0: # Only add valid proposals (score >=0)
                candidate_detail_dict = {
                    "pos": list(pv_pos_tuple), # Ensure it's a list for Pydantic model
                    "value": pv_val_int,
                    "raw_tensor_flow_score": round(fused_score_for_this_pv_at_pos, 6),
                    # These fields are from Main_api.pdf's CandidateDetail and would be filled by the caller
                    "is_valid_proposal": True, # Placeholder - caller should validate
                    "mem_score_value": 0.0, # Placeholder
                    "final_objective_score": 0.0, # Placeholder
                    "cp_solver_notes": None # Placeholder
                }
                evaluated_candidates_details.append(candidate_detail_dict)
        
        # Sort candidates by raw_tensor_flow_score (descending) before returning top_n
        evaluated_candidates_details.sort(key=lambda x: x["raw_tensor_flow_score"], reverse=True)
        
        analyzer_logger.info(f"{log_prefix} Analysis complete. Evaluated {len(evaluated_candidates_details)} candidates for specific proposals. Returning top {final_top_n}.", extra={'request_id': request_id_for_logging}) # [cite: 225]
        return evaluated_candidates_details[:final_top_n]


    # --- Visualization Methods (from An.pdf) ---
    def _setup_plot_figure(self, rows: int, cols: int, num_proposed_values: int) -> tuple[plt.Figure, plt.Axes, float]: # [cite: 229]
        """Sets up the Matplotlib figure and axes for visualization."""
        cell_size_inch = max(0.5, min(1.0, 10.0 / max(rows, cols, 1))) # Avoid division by zero [cite: 229]
        fig_width = max(cols * cell_size_inch, 6) # [cite: 229]
        fig_height = max(rows * cell_size_inch, 4) # [cite: 229]
        if num_proposed_values > 3: # [cite: 229]
            fig_width += 2  # Make space for legend
        fig, ax = plt.subplots(figsize=(fig_width, fig_height)) # [cite: 229]
        return fig, ax, cell_size_inch

    def _configure_plot_axes(self, ax: plt.Axes, rows: int, cols: int, cell_size_inch: float): # [cite: 229]
        """Configures the properties of the plot axes."""
        ax.set_xlim(-0.5, cols - 0.5) # [cite: 229]
        ax.set_ylim(rows - 0.5, -0.5)  # Inverted y-axis for matrix style [cite: 229]
        ax.set_xticks(np.arange(cols)) # [cite: 229]
        ax.set_yticks(np.arange(rows)) # [cite: 229]
        ax.set_xticklabels(np.arange(1, cols + 1), fontsize=max(6, cell_size_inch * 10)) # [cite: 229]
        ax.set_yticklabels(np.arange(1, rows + 1), fontsize=max(6, cell_size_inch * 10)) # [cite: 229]
        ax.xaxis.tick_top() # [cite: 229]
        ax.xaxis.set_label_position('top') # [cite: 229]
        ax.set_xlabel("Column (Col)", fontsize=max(7, cell_size_inch * 12)) # [cite: 229]
        ax.set_ylabel("Row (Row)", fontsize=max(7, cell_size_inch * 12)) # [cite: 229]
        ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5) # [cite: 229]
        ax.set_aspect('equal', adjustable='box') # [cite: 230]

    def _draw_heatmap(self, ax: plt.Axes, board_state: list[list[int]],
                      all_fused_scores_for_pvs: dict[Union[int, str], np.ndarray], # Union for pv key type
                      proposed_values_int_list: list[int]): # List of numeric PVs for heatmap [cite: 230]
        """Draws the heatmap of scores on the plot."""
        rows = len(board_state)
        cols = len(board_state[0]) if rows > 0 else 0
        if not (rows > 0 and cols > 0): return # [cite: 230]

        heatmap_data = np.full((rows, cols), np.nan) # [cite: 230]
        first_pv_for_heatmap: int | str | None = None
        if proposed_values_int_list and proposed_values_int_list[0] in all_fused_scores_for_pvs: # [cite: 230]
            first_pv_for_heatmap = proposed_values_int_list[0]
            scores_for_first_pv = all_fused_scores_for_pvs[first_pv_for_heatmap]
            if scores_for_first_pv.shape == (rows, cols): # [cite: 230]
                for r_idx in range(rows):
                    for c_idx in range(cols):
                        if board_state[r_idx][c_idx] == -1:
                            heatmap_data[r_idx, c_idx] = scores_for_first_pv[r_idx, c_idx]
            else:
                analyzer_logger.warning(f"Heatmap scores for PV {first_pv_for_heatmap} shape mismatch. Skipping heatmap.") # [cite: 230]
        
        if not np.all(np.isnan(heatmap_data)): # [cite: 230]
            cmap = plt.cm.viridis 
            cmap.set_bad(color='white', alpha=0.0)  # Transparent for NaN cells [cite: 230]
            ax.imshow(heatmap_data, cmap=cmap, alpha=0.6, aspect='auto', vmin=0, vmax=1) # [cite: 230]

    def _draw_suggestions_and_highlights(self, ax: plt.Axes,
                                         all_suggestions: dict[Union[int, str], list[dict[str, Any]]], # Key is PV (int or str)
                                         proposed_values_int_list: list[int], # List of numeric PVs
                                         top_n_suggestion_count: int
                                        ) -> dict[tuple[int, int], list[str]]: # [cite: 231]
        """Draws highlights and suggestion texts for top candidates."""
        suggestion_texts_on_cells: dict[tuple[int, int], list[str]] = {} # [cite: 231]
        cell_highlights: list[dict[str, Any]] = [] # [cite: 231]

        for pv_idx, pv_int in enumerate(proposed_values_int_list): # [cite: 231]
            pv_color = self.PV_COLORS[pv_idx % len(self.PV_COLORS)] # [cite: 231]
            if pv_int in all_suggestions:
                top_n_to_display_on_graph = min(top_n_suggestion_count, 3) # [cite: 231]
                for rank_idx, suggestion in enumerate(all_suggestions[pv_int][:top_n_to_display_on_graph]): # [cite: 231]
                    r, c = suggestion['position']
                    rank = rank_idx + 1
                    text_for_cell = f"{pv_int}(R{rank})" # [cite: 231]
                    if (r, c) not in suggestion_texts_on_cells: # [cite: 232]
                        suggestion_texts_on_cells[(r, c)] = []
                    suggestion_texts_on_cells[(r, c)].append(text_for_cell) # [cite: 232]

                    rect_line_width = 2.0 if rank == 1 else (1.5 if rank == 2 else 1.0) # [cite: 232]
                    cell_highlights.append({ # [cite: 232]
                        'coords': (c - 0.5, r - 0.5), 'width': 1, 'height': 1,
                        'linewidth': rect_line_width, 'edgecolor': pv_color,
                        'facecolor': mcolors.to_rgba(pv_color, alpha=0.10 if rank == 1 else 0.05)
                    })
        
        for highlight in cell_highlights: # [cite: 232]
            rect_params = {k: v for k, v in highlight.items() if k != 'coords'}
            rect = patches.Rectangle(xy=highlight['coords'], **rect_params)
            ax.add_patch(rect)
        return suggestion_texts_on_cells # [cite: 232]

    def _draw_board_texts(self, ax: plt.Axes, board_state: list[list[int]],
                          suggestion_texts_on_cells: dict[tuple[int, int], list[str]],
                          cell_size_inch: float): # [cite: 232]
        """Draws text values (existing numbers or suggestions) onto the board cells."""
        rows = len(board_state)
        cols = len(board_state[0]) if rows > 0 else 0
        if not (rows > 0 and cols > 0): return

        base_font_size = max(6, cell_size_inch * 10) # [cite: 232]
        for r_idx in range(rows): # [cite: 233]
            for c_idx in range(cols):
                cell_value = board_state[r_idx][c_idx]
                current_cell_texts = []
                if cell_value != -1: # [cite: 233]
                    current_cell_texts.append(str(cell_value))
                else:
                    if (r_idx, c_idx) in suggestion_texts_on_cells: # [cite: 233]
                        current_cell_texts.extend(suggestion_texts_on_cells[(r_idx, c_idx)])
                    else:
                        current_cell_texts.append(".")  # Placeholder for empty, sugg-less cells [cite: 233]
                
                final_display_text = "\n".join(current_cell_texts) # [cite: 233]
                num_lines = final_display_text.count('\n') + 1 # [cite: 233]
                dynamic_font_size = base_font_size / num_lines if num_lines > 1 else base_font_size # [cite: 233]
                
                # Further reduce if text is too wide for the cell (approximate) [cite: 234]
                avg_chars_per_line = len(final_display_text.replace("\n", "")) / num_lines if num_lines > 0 else 0
                width_factor = (cell_size_inch * 10) / (avg_chars_per_line + 1) if avg_chars_per_line > -1 else 1 # Avoid div by zero or too small [cite: 235]
                dynamic_font_size = max(4, dynamic_font_size * min(1.0, width_factor if width_factor > 0 else 1.0)) # [cite: 235]

                ax.text(c_idx, r_idx, final_display_text, # [cite: 235]
                        ha='center', va='center', fontsize=dynamic_font_size, color='black', wrap=True)

    def _add_legend_and_title(self, fig: plt.Figure, ax: plt.Axes,
                              proposed_values_int_list: list[int], # Numeric PV list
                              all_suggestions: dict[Union[int, str], list[dict[str, Any]]],
                              rows: int, cols: int, cell_size_inch: float): # [cite: 235]
        """Adds a legend and title to the plot."""
        pv_str = ", ".join(map(str, proposed_values_int_list)) if proposed_values_int_list else "None" # [cite: 235]
        title_str = f"Board Analysis ({rows}x{cols}) - Candidate Values: [{pv_str}]" # [cite: 235]
        if not any(sugg_list for sugg_list in all_suggestions.values()): # [cite: 235]
            title_str += "\n(No -1 cells on board or modules provided no valid suggestions)"
        
        fig.suptitle(title_str, fontsize=max(8, cell_size_inch * 14)) # [cite: 235]

        legend_elements = [] # [cite: 235]
        if proposed_values_int_list and any(s for pv_suggs in all_suggestions.values() for s in pv_suggs): # [cite: 235]
            added_pvs_to_legend = set() # [cite: 236]
            for pv_idx, pv_int in enumerate(proposed_values_int_list):
                if pv_int not in added_pvs_to_legend and any(s for s in all_suggestions.get(pv_int, [])): # [cite: 236]
                    color = self.PV_COLORS[pv_idx % len(self.PV_COLORS)]
                    legend_elements.append(patches.Patch(facecolor=color, edgecolor=color, label=f'Suggestions for PV {pv_int}')) # [cite: 237]
                    added_pvs_to_legend.add(pv_int) # [cite: 237]
        
        if legend_elements: # [cite: 237]
            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.03, 0.5),
                      fontsize=max(7, cell_size_inch * 10), title="Legend")
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust for legend [cite: 237]
        else:
            plt.tight_layout(rect=[0, 0, 1, 0.95]) # [cite: 237]

    def _generate_visualization(
        self,
        board_state: list[list[int]],
        proposed_values_int_list: list[int], # Numeric PV list
        all_suggestions: dict[Union[int, str], list[dict[str, Any]]], # PV (key) specific suggestions
        all_fused_scores_for_pvs: dict[Union[int, str], np.ndarray], # PV (key) specific heatmaps
        top_n_suggestion_count: int,
        request_id: str | None = "N/A_REQ_ID"
    ) -> str: # [cite: 238]
        """Generates a base64 encoded string of the board visualization."""
        analyzer_logger.debug("Generating visualization...", extra={'request_id': request_id}) # [cite: 238]
        rows = len(board_state)
        cols = len(board_state[0]) if rows > 0 else 0
        if rows == 0 or cols == 0: # [cite: 238]
            analyzer_logger.warning("Cannot generate visualization: board is empty.", extra={'request_id': request_id})
            return self._generate_error_visualization(0, 0, "Board is empty", request_id=request_id)

        try:
            fig, ax, cell_size_inch = self._setup_plot_figure(rows, cols, len(proposed_values_int_list)) # [cite: 238]
            self._current_cell_size_inch_for_dpi = cell_size_inch # For _fig_to_base64 DPI [cite: 238]
            self._configure_plot_axes(ax, rows, cols, cell_size_inch) # [cite: 238]
            self._draw_heatmap(ax, board_state, all_fused_scores_for_pvs, proposed_values_int_list) # [cite: 238]
            suggestion_texts = self._draw_suggestions_and_highlights(ax, all_suggestions, proposed_values_int_list, top_n_suggestion_count) # [cite: 238]
            self._draw_board_texts(ax, board_state, suggestion_texts, cell_size_inch) # [cite: 238]
            self._add_legend_and_title(fig, ax, proposed_values_int_list, all_suggestions, rows, cols, cell_size_inch) # [cite: 238]

            img_base64 = self._fig_to_base64(fig) # [cite: 238]
            plt.close(fig) # [cite: 238]
            if hasattr(self, '_current_cell_size_inch_for_dpi'): # [cite: 238]
                delattr(self, '_current_cell_size_inch_for_dpi') # Clean up temp attribute
            return img_base64
        except Exception as e_viz_detail:
            analyzer_logger.error(f"Detailed error during visualization generation: {e_viz_detail}", exc_info=True, extra={'request_id':request_id})
            plt.close(fig) # Ensure figure is closed on error
            if hasattr(self, '_current_cell_size_inch_for_dpi'):
                delattr(self, '_current_cell_size_inch_for_dpi')
            return self._generate_error_visualization(rows, cols, f"Visualization failed: {type(e_viz_detail).__name__}", request_id=request_id)


    def _generate_error_visualization(self, rows: int, cols: int, error_message: str, request_id: str | None = "N/A") -> str: # [cite: 227]
        """Generates a base64 encoded image displaying an error message."""
        log_extra = {'request_id': request_id}
        analyzer_logger.info(f"Generating error visualization: {error_message}", extra=log_extra)
        try:
            fig_width = max(cols * 0.5 if cols > 0 else 1, 5) # [cite: 227]
            fig_height = max(rows * 0.5 if rows > 0 else 1, 3) # [cite: 227]
            fig, ax = plt.subplots(figsize=(fig_width, fig_height)) # [cite: 227]
            ax.text(0.5, 0.5, f"Error:\n{error_message}", # [cite: 227]
                    ha='center', va='center', fontsize=10, color='red', wrap=True)
            ax.axis('off') # [cite: 227]
            img_base64 = self._fig_to_base64(fig) # [cite: 227]
            plt.close(fig) # [cite: 227]
            return img_base64
        except Exception as e:
            analyzer_logger.error("Failed to generate error visualization image itself: %s", e, exc_info=True, extra=log_extra) # [cite: 227]
            return "Error generating error visualization." # Fallback plain text error [cite: 228]

    def _fig_to_base64(self, fig: plt.Figure) -> str: # [cite: 239]
        """Converts a Matplotlib figure to a base64 encoded PNG string."""
        buf = io.BytesIO()
        try:
            current_cell_size_inch = getattr(self, '_current_cell_size_inch_for_dpi', 0.75) # [cite: 239]
            dpi = max(75, int(current_cell_size_inch * 120)) # Increased multiplier for quality [cite: 239]
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight') # [cite: 239]
        except Exception as e:
            analyzer_logger.error("fig.savefig failed: %s", e, exc_info=True) # [cite: 239]
            plt.close(fig) # Ensure figure is closed on error [cite: 239]
            raise VisualizationError(f"Failed to save figure to buffer: {e}") from e # [cite: 239]
        
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8') # [cite: 239]
        buf.close() # [cite: 240]
        return img_base64

# Instantiate Analyzer globally
analyzer_instance: Analyzer | None = None
try:
    analyzer_instance = Analyzer(main_module=brain_interface, default_top_n=3) # [cite: 5]
    logger.info("Global Analyzer instance created successfully.")
except InitializationError as e_init: # [cite: 5]
    logger.critical("CRITICAL_API_STARTUP_ERROR: Failed to initialize Analyzer: %s", e_init, exc_info=True)
except Exception as e_unexp: # [cite: 5]
    logger.critical("CRITICAL_API_STARTUP_ERROR: Unexpected error during Analyzer initialization: %s", e_unexp, exc_info=True)


# --- Pydantic Models for FastAPI ---

# Models from main_api.pdf
class HealthResponse(BaseModel): # [cite: 4]
    status: str
    message: str | None = None
    reason: str | None = None # [cite: 4]
    analyzer_status: str | None = None

class AnalyzeHealthStatus(BaseModel): # [cite: 4]
    status: str
    analysis_engine_version: str
    checks: dict[str, str]
    components: dict[str, str]

class CandidateDetail(BaseModel): # [cite: 4]
    pos: list[int] # Expect [row, col]
    value: int
    is_valid_proposal: bool # To be determined by endpoint logic
    raw_tensor_flow_score: float # This is what Analyzer provides primarily
    mem_score_value: float # Placeholder, from main_api.pdf
    final_objective_score: float # Placeholder, from main_api.pdf
    cp_solver_notes: str | None = None # Placeholder, from main_api.pdf

class AnalyzeSuccessResponse(BaseModel): # [cite: 4]
    request_id: str
    message: str
    grid_shape: tuple[int, ...]
    evaluated_candidates: list[CandidateDetail] # [cite: 5]

class AnalyzeErrorResponse(BaseModel): # [cite: 5]
    detail: str
    request_id: str | None = None

class ProposedValue(BaseModel): # [cite: 5]
    pos: tuple[int, int] # (row, col)
    value: int

class AnalysisRequest(BaseModel): # [cite: 5]
    new_card: list[list[int]]
    proposed_values: list[ProposedValue]
    active_modules: list[str] | None = None
    module_weights: dict[str, float] | None = None
    top_n: int | None = Field(None, gt=0) # Ensure top_n is positive if provided

# Models from main 2.pdf (for background tasks and module listing)
class GridDataBase(BaseModel): # [cite: 258]
    grid_data: list[list[Union[int, float]]] = Field(..., example=[[-1, 1.0, -1], [2, -1, 3.5], [-1, 4, -1]])

    @field_validator('grid_data') # [cite: 258]
    def validate_grid_data_structure(cls, v: list[list[Union[int, float]]]) -> list[list[Union[int, float]]]: # [cite: 258]
        if not v or not all(isinstance(row, list) for row in v) or not v[0]: # [cite: 258]
            raise ValueError("Grid data must be a non-empty list of non-empty lists.")
        num_cols = len(v[0]) # [cite: 258]
        if num_cols == 0: # [cite: 258]
            raise ValueError("Grid columns cannot be empty (first row is empty).")
        if not all(len(row) == num_cols for row in v): # [cite: 258]
            raise ValueError("All rows must have the same number of columns.")
        for r_idx, row in enumerate(v): # [cite: 258]
            for c_idx, cell_val in enumerate(row):
                if not isinstance(cell_val, (int, float)): # [cite: 259]
                    raise ValueError(f"Cell ({r_idx}, {c_idx}) type invalid: {type(cell_val)}. Must be number.")
        return v

class GridInput(GridDataBase): # [cite: 260]
    client_request_id: str | None = Field(None, description="Optional client-provided request ID for tracing.") # [cite: 260]

class BatchGridItem(GridDataBase): # [cite: 260]
    item_id: str = Field(description="Unique identifier for this item in the batch.") # [cite: 260]
    module_name: str = Field(description="Scoring module to use for this item.") # [cite: 260]

class BatchGridInput(BaseModel): # [cite: 260]
    grids: list[BatchGridItem] = Field(..., max_length=50)  # Limit batch size [cite: 260]
    client_request_id: str | None = Field(None, description="Optional client-provided request ID for the batch.") # [cite: 260]

class TaskAcceptedResponse(BaseModel): # [cite: 260]
    task_id: str # [cite: 261]
    status: str = "accepted" # [cite: 261]
    message: str # [cite: 261]
    client_request_id: str | None = None # [cite: 261]

class ModuleInfo(BaseModel): # [cite: 261]
    name: str # [cite: 261]
    description: str | None = "No description available." # [cite: 261]
    version: str | None = "N/A" # [cite: 262]
    # input_constraints: dict[str, Any] | None = None # [cite: 263] # Example if brain provided more details


# --- Mock CP Model (from main_api.pdf) ---
class MockCPModel: # [cite: 1]
    _version: str = "9.9.mock-2025"
    def CpModel(self):
        logger.info("[Placeholder] MockCPModel.CpModel() invoked.", extra={'request_id': 'N/A_cp_model'}) # [cite: 1]
        # In a real scenario, this would initialize an OR-Tools CP-SAT model
        pass
cp_model = MockCPModel() # [cite: 1]

# --- Placeholder for extreme_tensor_flow_score_detailed (from main_api.pdf) ---
# This would be a complex TensorFlow integration in reality.
def extreme_tensor_flow_score_detailed_placeholder(grid: np.ndarray, request_id_context: str) -> tuple[np.ndarray, list[list[dict[str, Any]]]]: # [cite: 1]
    """Placeholder for a detailed TensorFlow scoring function."""
    logger.info(f"[Placeholder] extreme_tensor_flow_score_detailed for request_id_context {request_id_context}, grid: {grid.shape}", extra={'request_id': request_id_context}) # [cite: 1]
    scores = np.random.rand(*grid.shape).astype(np.float32) * 10 # [cite: 1]
    contributions = [[{"rule": f"dummy_r{r}_c{c}", "value": np.random.random()} for c in range(grid.shape[1])] for r in range(grid.shape[0])] # [cite: 3]
    return scores, contributions


# --- API Key Authentication ---
api_key_query_auth = APIKeyQuery(name=settings.API_KEY_NAME, auto_error=False) # [cite: 256]
api_key_header_auth = APIKeyHeader(name=settings.API_KEY_NAME, auto_error=False) # [cite: 256]

async def get_api_key( # [cite: 257]
    key_query: str | None = Security(api_key_query_auth),
    key_header: str | None = Security(api_key_header_auth),
) -> str:
    """Validates API key from query or header."""
    if key_query == settings.API_KEY: # [cite: 257]
        return key_query
    if key_header == settings.API_KEY: # [cite: 258]
        return key_header
    raise HTTPException( # [cite: 258]
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key"
    )

# --- Rate Limiter (Simple In-Memory - For Demonstration) ---
# WARNING: Not suitable for multi-process/multi-worker deployments. Use Redis-backed (e.g. slowapi) for production. [cite: 264]
request_counts: dict[str, list[float]] = {} # [cite: 265]


# --- Helper Functions for Scoring Task (Background) ---
async def run_scoring_task( # [cite: 265]
    task_id: str,
    module_name: str,
    grid_data: list[list[Union[int, float]]], # From GridDataBase
    original_request_id: str, # Main request ID for logging linkage
    client_request_id: str | None = None
):
    """
    Performs the actual scoring in a background task. [cite: 266]
    Results might be stored in a DB or sent to a callback in a real application. [cite: 267]
    """
    ACTIVE_BACKGROUND_TASKS.inc() # [cite: 267]
    log_extra = {'request_id': original_request_id, 'task_id': task_id, 'module_name': module_name, 'client_request_id': client_request_id or "N/A"} # [cite: 267]
    logger.info("Background scoring task started.", extra=log_extra) # [cite: 267]

    try:
        np_grid = np.array(grid_data, dtype=np.int32) # Brain modules expect int32 typically
        if np_grid.size == 0: # [cite: 267]
            raise ValueError("Input grid is empty after numpy conversion.")

        start_time_task = time.monotonic() # [cite: 272]
        # Call the async get_module_score from brain_interface
        score_np_array = await brain_interface.get_module_score(module_name, np_grid, request_id=task_id) # [cite: 272]
        duration_task = time.monotonic() - start_time_task # [cite: 272]

        score_list_of_lists = score_np_array.tolist() # [cite: 272]
        result_message = f"Scoring successful for module {module_name}." # [cite: 273]
        logger.info(result_message + f" Duration: {duration_task:.4f}s", extra=log_extra) # [cite: 273]

        # Placeholder for result handling (e.g., save to DB, notify, callback)
        if settings.TASK_CALLBACK_URL_ENABLED and settings.TASK_CALLBACK_URL: # [cite: 274]
            logger.info(f"Simulating callback to {settings.TASK_CALLBACK_URL} with result.", extra=log_extra) # [cite: 274]
            # In a real app, use an HTTP client like httpx:
            # callback_payload = {"task_id": task_id, "status": "completed", "result": score_list_of_lists, "client_request_id": client_request_id}
            # async with httpx.AsyncClient() as client:
            #     await client.post(str(settings.TASK_CALLBACK_URL), json=callback_payload)
            pass

    except Exception as e:
        error_message = f"Error in background scoring task for module {module_name}: {str(e)}" # [cite: 273]
        logger.error(error_message, exc_info=True, extra=log_extra)
        # Placeholder for error handling
    finally:
        ACTIVE_BACKGROUND_TASKS.dec() # [cite: 273]
        logger.info("Background scoring task finished.", extra=log_extra)


# --- FastAPI Application Instance & Middlewares ---
app = FastAPI( # [cite: 273, 305]
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    # openapi_tags can be defined here if needed
)

# Add Prometheus Middleware (exposes /metrics automatically if new version, or add route) [cite: 303, 309]
app.add_middleware(PrometheusMiddleware)
# No need for app.add_route("/metrics", handle_metrics) if using modern starlette-prometheus [cite: 304]

@app.middleware("http") # [cite: 275, 305]
async def base_middleware(request: Request, call_next: Callable[[Request], Coroutine[Any, Any, Any]]) -> Any: # Type hint for call_next from [cite: 305]
    """
    Base middleware to handle request ID, rate limiting, security headers, and metrics.
    """
    # 1. Manage Request ID [cite: 275]
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4()) # [cite: 275, 305]
    request.state.request_id = request_id # For access in routes/dependencies [cite: 276, 305]
    log_extra_mw = {'request_id': request_id} # For logs within this middleware

    # 2. Simple In-Memory Rate Limiting (Basic, for demonstration) [cite: 279]
    client_ip = request.client.host if request.client else "unknown_client" # [cite: 280]
    current_time = time.time() # [cite: 280]
    
    # Clean up old timestamps for the IP
    if client_ip not in request_counts:
        request_counts[client_ip] = []
    
    request_counts[client_ip] = [t for t in request_counts[client_ip] if t > current_time - settings.RATE_LIMIT_WINDOW_SECONDS] # [cite: 280]

    if len(request_counts[client_ip]) >= settings.RATE_LIMIT_REQUESTS: # [cite: 280]
        logger.warning(f"Rate limit exceeded for IP: {client_ip}", extra=log_extra_mw)
        REQUEST_COUNT.labels(method=request.method, endpoint=str(request.url.path), status_code=429).inc() # [cite: 280]
        return JSONResponse( # [cite: 280]
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Too many requests, please try again later."}
        )
    request_counts[client_ip].append(current_time) # [cite: 280]

    # 3. Timing and Processing Request
    start_time_metric = time.monotonic() # [cite: 281]
    
    # Log request arrival (modern style from starter kit [cite: 309])
    logger.info(f"→ {request.method} {request.url.path} User-Agent: {request.headers.get('user-agent', 'N/A')}", extra=log_extra_mw)

    response = await call_next(request) # Process request [cite: 281, 306]
    duration_metric = time.monotonic() - start_time_metric # [cite: 281]

    # 4. Add Security Headers and Request ID to Response [cite: 281, 306]
    response.headers["X-Request-ID"] = request_id # [cite: 281, 306]
    response.headers["X-Content-Type-Options"] = "nosniff" # [cite: 281]
    response.headers["X-Frame-Options"] = "DENY" # [cite: 281]
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none';" # [cite: 281]
    if request.url.scheme == "https": # Only add HSTS if served over HTTPS [cite: 282]
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    # 5. Metrics Recording [cite: 282]
    REQUEST_COUNT.labels(method=request.method, endpoint=str(request.url.path), status_code=response.status_code).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=str(request.url.path)).observe(duration_metric)
    
    logger.info( # [cite: 283]
        f"← {request.method} {request.url.path} - Response: {response.status_code} - Duration: {duration_metric:.4f}s",
        extra=log_extra_mw
    )
    return response

# --- Global Exception Handler ---
@app.exception_handler(Exception) # [cite: 283]
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handles any unhandled exceptions globally."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4())) # [cite: 283]
    log_extra_exc = {'request_id': request_id}
    
    logger.error(f"Global unhandled exception: {exc}", exc_info=True, extra=log_extra_exc) # [cite: 283]
    
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    content = {
            "request_id": request_id,
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please contact support.",
            "detail": str(exc) if settings.LOG_LEVEL.upper() == "DEBUG" else None # Show details only in DEBUG [cite: 283]
        }

    if isinstance(exc, HTTPException): # If it's an HTTPException, use its status and detail
        status_code = exc.status_code
        content["error"] = "HTTP Exception"
        content["message"] = exc.detail
        content["detail"] = None # Detail is already in message for HTTPException

    REQUEST_COUNT.labels(method=request.method, endpoint=str(request.url.path), status_code=status_code).inc() # [cite: 283]
    
    return JSONResponse(status_code=status_code, content=content) # [cite: 283]


# --- API Endpoints ---

# Endpoints from main_api.pdf (adapted)
@app.get("/", tags=["Utilities"], summary="Root Path / Basic Health Ping") # [cite: 5]
async def read_root(request: Request): # [cite: 5]
    """Provides a welcome message and link to API documentation."""
    log_extra = {'request_id': request.state.request_id}
    logger.info("Root path '/' accessed.", extra=log_extra)
    return {
        "message": f"Welcome to {settings.APP_TITLE} v{settings.APP_VERSION}", # [cite: 6]
        "docs_url": str(request.url.replace(path="/docs")),
        "redoc_url": str(request.url.replace(path="/redoc")),
        "analyzer_status": "Initialized" if analyzer_instance else "Not Initialized"
    }

@app.get("/health", response_model=HealthResponse, tags=["Utilities"], summary="Simple Analyzer Health Check") # [cite: 6]
async def health_check_simple(request: Request): # [cite: 6]
    """Performs a simple health check of the Analyzer component."""
    request_id = request.state.request_id # [cite: 6]
    log_extra = {'request_id': request_id}

    if analyzer_instance is None: # [cite: 6]
        logger.warning("HEALTH_CHECK_SIMPLE /health: Failed - Analyzer not initialized.", extra=log_extra) # [cite: 7]
        return HealthResponse(status="unhealthy", reason="Analyzer core component not initialized.", analyzer_status="Not Initialized") # [cite: 8]
    
    logger.info("HEALTH_CHECK_SIMPLE /health: Successful - Analyzer is initialized.", extra=log_extra) # [cite: 8]
    return HealthResponse(status="ok", message="Analyzer API is running and Analyzer core is initialized.", analyzer_status="Initialized") # [cite: 8]

@app.get("/health/analyze", response_model=AnalyzeHealthStatus, tags=["Health & Monitoring"], summary="Detailed System Health Analysis") # [cite: 8]
async def health_analyze_detailed(request: Request): # [cite: 8]
    """Provides a detailed health analysis of system components including brain modules and dependencies."""
    request_id = request.state.request_id # [cite: 8]
    log_extra = {'request_id': request_id}
    logger.info("HEALTH_CHECK_DETAILED /health/analyze: Request received.", extra=log_extra) # [cite: 8]

    checks: dict[str, str] = {} # [cite: 8]
    overall_status: str = "UP" # [cite: 8]

    # Check brain modules (conceptual, from main_api.pdf)
    # EXTREME_MODULE_FUNCS_VEC and EXTREME_MODULE_WEIGHTS were global vars in main_api.pdf,
    # here we'd check against brain_interface.registered_modules
    if not brain_interface.registered_modules: # [cite: 9]
        checks["brain_module_funcs_load"] = "FAIL: No modules registered in brain_interface"
        overall_status = "DEGRADED" # [cite: 9]
    else:
        checks["brain_module_funcs_load"] = f"OK: {len(brain_interface.registered_modules)} modules registered" # [cite: 9]
    
    # Check MEM_PATH (placeholder from main_api.pdf)
    # This is now settings.MEM_PATH
    # mem_path_exists = await asyncio.to_thread(os.path.exists, settings.MEM_PATH) # If os.path is used
    # For this exercise, os.path is not imported. Assume direct check or this part becomes conceptual.
    # Let's simulate it.
    # if not mem_path_exists: # [cite: 10]
    #     checks["memory_file_exists"] = f"FAIL: {settings.MEM_PATH} not found"
    #     overall_status = "DEGRADED" # [cite: 10]
    # else:
    #     checks["memory_file_exists"] = "OK (Path exists)" # [cite: 10]
    checks["memory_file_exists"] = "SKIPPED (os.path not used for this version)"


    # Placeholder TensorFlow execution test [cite: 11]
    try:
        dummy_grid_data = [[-1, 1, 5, 0], [2, -1, 8, 3], [4, 6, -1, 7], [0, 0, 0, 0]]
        dummy_grid_np = np.array(dummy_grid_data, dtype=np.int32)
        # In main_api.pdf this was await run_in_threadpool(extreme_tensor_flow_score_detailed, ...)
        # Now we call the placeholder directly, assuming it could be async or wrapped.
        # For simplicity, call sync placeholder, in real async app, use asyncio.to_thread
        await asyncio.to_thread(extreme_tensor_flow_score_detailed_placeholder, dummy_grid_np, f"health_tf_{request_id}")
        checks["extreme_tf_execution_test"] = "OK (Placeholder Executed)" # [cite: 11]
    except Exception as e:
        checks["extreme_tf_execution_test"] = f"FAIL (Placeholder): {str(e)}" # [cite: 12]
        logger.error("HEALTH_ERROR /health/analyze: extreme_tf placeholder test FAIL.", exc_info=True, extra=log_extra) # [cite: 11]
        overall_status = "ERROR" # [cite: 12]

    # Placeholder CP Solver availability test [cite: 12]
    try:
        _ = cp_model.CpModel() # [cite: 12]
        checks["cp_solver_avail_test"] = "OK (MockCPModel Invoked)" # [cite: 12]
    except Exception as e:
        checks["cp_solver_avail_test"] = f"FAIL (MockCPModel): {str(e)}" # [cite: 12]
        logger.error("HEALTH_ERROR /health/analyze: CP Solver (Mock) test FAIL.", exc_info=True, extra=log_extra) # [cite: 12]
        overall_status = "ERROR" # [cite: 13]

    return AnalyzeHealthStatus( # [cite: 13]
        status=overall_status,
        analysis_engine_version=settings.ANALYZER_VERSION, # Using version from settings
        checks=checks,
        components={ # [cite: 13]
            "numpy_version": np.__version__,
            "ortools_version": getattr(cp_model, '_version_', "unknown"), # [cite: 13]
            "analyzer_type": "Extreme Logic Modules v2.5 (2025 Enhanced)"
        }
    )

@app.post("/analyze",
    response_model=AnalyzeSuccessResponse,
    responses={ # [cite: 13]
        400: {"model": AnalyzeErrorResponse, "description": "Invalid input data (client-side error)"},
        422: {"model": AnalyzeErrorResponse, "description": "Validation error in request data (unprocessable entity)"},
        500: {"model": AnalyzeErrorResponse, "description": "Internal server processing error"},
        503: {"model": AnalyzeErrorResponse, "description": "Service temporarily unavailable (e.g., Analyzer not initialized)"}
    },
    tags=["Analysis Engine vExtreme"],
    summary="Perform Extreme N-Dimensional Tensor Analysis"
)
async def analyze_board_main(
    payload: AnalysisRequest, # FastAPI uses Pydantic for request body
    request: Request,
    api_key: APIKey = Depends(get_api_key) # Secure this endpoint
): # [cite: 13]
    """
    Main analysis endpoint. Receives a board state and proposed values,
    then returns evaluated candidates.
    """
    request_id = request.state.request_id # [cite: 13]
    log_extra = {'request_id': request_id}

    logger.info( # [cite: 14]
        f"API_CALL /analyze: Grid: {len(payload.new_card)}x{len(payload.new_card[0]) if payload.new_card and payload.new_card[0] else 'empty'}. "
        f"Proposals: {len(payload.proposed_values)}.",
        extra=log_extra
    )

    if analyzer_instance is None: # [cite: 14]
        logger.error("API_ERROR /analyze: Analyzer instance not available.", extra=log_extra) # [cite: 15]
        raise HTTPException(status_code=503, detail="Analysis service is temporarily unavailable due to initialization failure.") # [cite: 15]

    if not payload.new_card or not payload.new_card[0]: # [cite: 15]
        logger.warning("API_VALIDATION_ERROR /analyze: Empty new_card received.", extra=log_extra) # [cite: 15]
        raise HTTPException(status_code=400, detail="Input 'new_card' cannot be empty or contain empty rows.") # [cite: 15]

    try:
        grid_np = np.array(payload.new_card, dtype=np.int32) # [cite: 15]
    except Exception as e_np_convert: # [cite: 15]
        logger.error("API_VALIDATION_ERROR /analyze: Failed to convert new_card to NumPy array.", exc_info=True, extra=log_extra) # [cite: 15]
        raise HTTPException(status_code=400, detail=f"Invalid data format in 'new_card': {str(e_np_convert)}") # [cite: 15]

    try:
        # The Analyzer.analyze_board_generic_pvs method is async
        analysis_result_list_of_dicts = await analyzer_instance.analyze_board_generic_pvs( # [cite: 16]
            new_card_list=payload.new_card,
            proposed_value_objects=payload.proposed_values, # Pass Pydantic models
            active_modules=payload.active_modules,
            module_weights=payload.module_weights,
            top_n=payload.top_n,
            request_id_for_logging=request_id
        )
        
        processed_candidates: list[CandidateDetail] = []
        if isinstance(analysis_result_list_of_dicts, list): # [cite: 16]
            for cand_data_dict in analysis_result_list_of_dicts:
                if isinstance(cand_data_dict, dict): # [cite: 16]
                    # Fill in other CandidateDetail fields if necessary or assume Analyzer provides them all
                    # For now, Analyzer returns dicts that mostly match CandidateDetail's raw_tensor_flow_score part
                    # Here we would potentially add more logic (mem_score, final_objective_score, cp_solver_notes)
                    # For this example, we directly cast, assuming the dict from analyzer is compatible enough
                    # Or, we construct CandidateDetail more carefully.
                    # Let's assume `cand_data_dict` has keys matching `CandidateDetail` or we map them.
                    # The current `analyze_board_generic_pvs` populates: pos, value, raw_tensor_flow_score.
                    # Other fields (is_valid_proposal, mem_score, final_objective, cp_notes) are placeholders.
                    # We need to ensure they are present for Pydantic validation or mark as optional.
                    # For now, they are placeholders in the dict returned by analyzer.

                    # Simulate determining is_valid_proposal (e.g., based on score or external check)
                    is_valid = cand_data_dict.get("raw_tensor_flow_score", 0.0) > 0.05 # Example validity
                    
                    # Simulate mem_score and final_objective_score
                    # These were placeholders in main_api.pdf as well.
                    # In a real system, these would come from other logic or the analyzer itself.
                    mem_score_val = round(random.uniform(0,5), 4) if is_valid else 0.0 # Mock
                    final_obj_score = round(cand_data_dict.get("raw_tensor_flow_score",0.0) + mem_score_val * 0.2, 4) if is_valid else 0.0 # Mock

                    processed_candidates.append(CandidateDetail(
                        pos=cand_data_dict["pos"],
                        value=cand_data_dict["value"],
                        is_valid_proposal=is_valid, # Set based on some logic
                        raw_tensor_flow_score=cand_data_dict["raw_tensor_flow_score"],
                        mem_score_value=mem_score_val, # Placeholder
                        final_objective_score=final_obj_score, # Placeholder
                        cp_solver_notes=None # Placeholder
                    ))
                else:
                    logger.warning(f"API_RESULT_WARN /analyze: Unexpected candidate data type: {type(cand_data_dict)}.", extra=log_extra) # [cite: 16]
        else: # [cite: 16]
            logger.warning(f"API_RESULT_WARN /analyze: Unexpected result type from analyzer: {type(analysis_result_list_of_dicts)}.", extra=log_extra)
            raise HTTPException(status_code=500, detail="Internal error: Unexpected analysis result format.") # [cite: 16]

        logger.info(f"API_SUCCESS /analyze: Analysis complete. Evaluated {len(processed_candidates)} candidates.", extra=log_extra) # [cite: 16]
        return AnalyzeSuccessResponse( # [cite: 17]
            request_id=request_id,
            message="Analysis successfully completed.",
            grid_shape=grid_np.shape,
            evaluated_candidates=processed_candidates
        )
    except InvalidInputError as e_val_analyzer: # From Analyzer's validation # [cite: 17]
        logger.warning(f"API_VALIDATION_ERROR /analyze: Invalid input from Analyzer: {e_val_analyzer}.", exc_info=True, extra=log_extra) # [cite: 17]
        raise HTTPException(status_code=422, detail=f"Invalid Input Parameters: {str(e_val_analyzer)}") # [cite: 17]
    except (ModuleNotFoundError, ModuleExecutionError, ModuleError, VisualizationError) as e_module_analyzer: # [cite: 17]
        logger.error(f"API_MODULE_ERROR /analyze: Analyzer module error: {e_module_analyzer}.", exc_info=True, extra=log_extra) # [cite: 17]
        raise HTTPException(status_code=500, detail=f"Module Error during analysis ({type(e_module_analyzer).__name__}): {str(e_module_analyzer)}") # [cite: 17]
    except Exception as e_unexpected_analyzer: # [cite: 17]
        logger.critical(f"API_UNEXPECTED_ERROR /analyze: Unexpected critical error: {e_unexpected_analyzer}.", exc_info=True, extra=log_extra) # [cite: 17]
        raise HTTPException(status_code=500, detail=f"Unexpected internal server error: {type(e_unexpected_analyzer).__name__} - {str(e_unexpected_analyzer)}") # [cite: 17]


# Endpoints from main 2.pdf (for background tasks and module listing)
@app.get("/modules", response_model=list[ModuleInfo], tags=["Modules"], summary="List all available scoring modules.") # [cite: 283]
async def list_available_modules_endpoint(request: Request, api_key: APIKey = Depends(get_api_key)): # [cite: 283]
    """Lists all scoring modules registered in the brain interface."""
    log_extra = {'request_id': request.state.request_id}
    logger.info("Listing available modules.", extra=log_extra)
    modules_info_list: list[ModuleInfo] = [] # [cite: 285]
    for module_name in brain_interface.registered_modules.keys(): # [cite: 285]
        try:
            details = brain_interface.get_module_details(module_name) # [cite: 285]
            modules_info_list.append(ModuleInfo( # [cite: 286]
                name=module_name,
                description=details.get('description'),
                version=details.get('version')
                # input_constraints=details.get('input_constraints') # Example if details were richer
            ))
        except Exception: # Fallback [cite: 286]
            modules_info_list.append(ModuleInfo(name=module_name))
    return modules_info_list

@app.get("/modules/{module_name}", response_model=ModuleInfo, tags=["Modules"], summary="Get details for a specific scoring module.") # [cite: 286]
async def get_module_info_endpoint(
    request: Request,
    module_name: str = Path(..., description="The name of the scoring module."), # [cite: 286]
    api_key: APIKey = Depends(get_api_key)
):
    """Provides detailed information for a specific scoring module."""
    log_extra = {'request_id': request.state.request_id, 'module_name': module_name}
    logger.info(f"Fetching details for module '{module_name}'.", extra=log_extra)
    if module_name not in brain_interface.registered_modules: # [cite: 287]
        logger.warning(f"Module '{module_name}' not found.", extra=log_extra)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Module '{module_name}' not found.") # [cite: 287]
    try:
        details = brain_interface.get_module_details(module_name) # [cite: 287]
        return ModuleInfo( # [cite: 287]
            name=module_name,
            description=details.get('description'),
            version=details.get('version')
        )
    except Exception: # Fallback
        return ModuleInfo(name=module_name)

@app.post("/score/{module_name}",
    response_model=TaskAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Scoring (Async)"],
    summary="Submit a single grid for scoring (background task)."
) # [cite: 287]
async def score_grid_background_endpoint( # [cite: 287]
    request: Request,
    payload: GridInput, # Uses GridInput from main 2.pdf [cite: 287]
    module_name: str = Path(..., description="The name of the scoring module to use."), # [cite: 287]
    background_tasks: BackgroundTasks = Depends(), # FastAPI injects this [cite: 287]
    api_key: APIKey = Depends(get_api_key)
):
    """
    Accepts a grid for scoring by a specified module. [cite: 288]
    The scoring is performed as a background task. [cite: 288]
    The API immediately returns a task ID. [cite: 289]
    """
    req_id = request.state.request_id # [cite: 289]
    client_req_id = payload.client_request_id # [cite: 289]
    task_id = str(uuid.uuid4()) # [cite: 289]
    log_extra = {'request_id': req_id, 'task_id': task_id, 'module_name': module_name, 'client_request_id': client_req_id or "N/A"} # [cite: 289]

    if module_name not in brain_interface.registered_modules: # [cite: 290]
        logger.warning(f"Module '{module_name}' not found for background task.", extra=log_extra) # [cite: 290]
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Module '{module_name}' not found.") # [cite: 290]

    background_tasks.add_task( # [cite: 290]
        run_scoring_task,
        task_id=task_id,
        module_name=module_name,
        grid_data=payload.grid_data,
        original_request_id=req_id,
        client_request_id=client_req_id
    )
    MODULE_USAGE_COUNT.labels(module_name=module_name).inc() # [cite: 290]
    logger.info("Grid scoring task enqueued.", extra=log_extra) # [cite: 290]
    return TaskAcceptedResponse( # [cite: 290]
        task_id=task_id,
        message=f"Scoring task for module '{module_name}' accepted and is being processed in the background.",
        client_request_id=client_req_id
    )

@app.post("/score/batch",
    response_model=list[TaskAcceptedResponse],
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Scoring (Async)"],
    summary="Submit multiple grids for batch scoring (background tasks)."
) # [cite: 290]
async def score_batch_grid_background_endpoint( # [cite: 290]
    request: Request,
    payload: BatchGridInput, # Uses BatchGridInput [cite: 290]
    background_tasks: BackgroundTasks = Depends(),
    api_key: APIKey = Depends(get_api_key)
):
    """Submits multiple grid scoring jobs, each processed as a background task."""
    req_id = request.state.request_id # [cite: 290]
    client_req_id = payload.client_request_id # [cite: 290]
    responses: list[TaskAcceptedResponse] = [] # [cite: 290]
    log_extra_batch = {'request_id': req_id, 'batch_size': len(payload.grids), 'client_request_id': client_req_id or "N/A"} # [cite: 290]
    logger.info("Batch grid scoring task received.", extra=log_extra_batch)

    for item in payload.grids: # [cite: 290]
        task_id = str(uuid.uuid4()) # [cite: 290]
        log_extra_item = {**log_extra_batch, 'task_id': task_id, 'item_id': item.item_id, 'module_name': item.module_name} # [cite: 290]

        if item.module_name not in brain_interface.registered_modules: # [cite: 291]
            logger.warning(f"Module '{item.module_name}' not found for batch item ID '{item.item_id}'. Skipping enqueue.", extra=log_extra_item) # [cite: 291]
            responses.append(TaskAcceptedResponse( # [cite: 295]
                task_id=f"error_invalid_module_{item.item_id}", # Special task_id [cite: 295]
                status="rejected", # [cite: 295]
                message=f"Module '{item.module_name}' for item_id '{item.item_id}' not found. Task not created.", # [cite: 295]
                client_request_id=client_req_id
            ))
            continue

        background_tasks.add_task( # [cite: 291]
            run_scoring_task,
            task_id=task_id,
            module_name=item.module_name,
            grid_data=item.grid_data,
            original_request_id=req_id,
            client_request_id=client_req_id
        )
        MODULE_USAGE_COUNT.labels(module_name=item.module_name).inc() # [cite: 291]
        responses.append(TaskAcceptedResponse( # [cite: 291]
            task_id=task_id,
            message=f"Scoring task for item_id '{item.item_id}' (module '{item.module_name}') accepted.",
            client_request_id=client_req_id
        ))
        logger.info("Batch item enqueued for scoring.", extra=log_extra_item) # [cite: 291]
    return responses


# --- Main Execution Block ---
if __name__ == "__main__": # [cite: 17, 301]
    import uvicorn
    log_extra_main = {'request_id': 'SYSTEM_MAIN'}
    logger.info(f"Starting {settings.APP_TITLE} v{settings.APP_VERSION} on {settings.APP_HOST}:{settings.APP_PORT}", extra=log_extra_main) # [cite: 301]
    logger.info(f"Default API Key: {settings.API_KEY[:4]}... (Ensure this is changed for production!)", extra=log_extra_main) # [cite: 301]
    logger.info(f"Rate Limiting: {settings.RATE_LIMIT_REQUESTS} requests per {settings.RATE_LIMIT_WINDOW_SECONDS} seconds (in-memory).", extra=log_extra_main) # [cite: 301]
    if settings.TASK_CALLBACK_URL_ENABLED: # [cite: 302]
        logger.info(f"Task callback enabled, will attempt to POST to: {settings.TASK_CALLBACK_URL}", extra=log_extra_main) # [cite: 302]
    else:
        logger.info("Task callback is disabled. Background task results will only be logged.", extra=log_extra_main) # [cite: 302]

    # For Uvicorn programmatic run, ensure log_config is passed or Uvicorn uses the configured root logger.
    # If Uvicorn has its own default logging config, it might override.
    # Passing None should make Uvicorn use the already configured root logger.
    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT, log_config=None)