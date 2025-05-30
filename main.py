
# 強化合併版 main9_optimized.py
# === Logging 設定 ===
import logging
import sys
from logging.config import dictConfig
from contextvars import ContextVar

request_id_ctx_var: ContextVar[str] = ContextVar("request_id", default="-")

class RequestIdLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx_var.get()
        return True

def setup_logging() -> None:
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "request_id": {
                "()": RequestIdLogFilter,
            }
        },
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(levelname)s - [%(request_id)s] %(name)s:%(lineno)d - %(message)s"
            }
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "default",
                "filters": ["request_id"]
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["default"]
        }
    }
    dictConfig(log_config)

setup_logging()

# === 設定管理 ===
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "MyFastAPIApp"
    enable_metrics: bool = True

    class Config:
        env_file = ".env"

settings = Settings()

# === Prometheus 整合 ===
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

def setup_metrics(app: FastAPI) -> None:
    Instrumentator().instrument(app).expose(app)


import sys
from contextvars import ContextVar

request_id_ctx_var: ContextVar[str] = ContextVar("request_id", default="-")

class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx_var.get()
        return True

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(request_id)s] - %(message)s",
        stream=sys.stdout,
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(RequestIdFilter())

# main.py - Part 1 of 3
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
from typing import (str | int | float | bool | list | dict, Callable, Coroutine, Dict, List,
                    Tuple, Union)

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from fastapi import (BackgroundTasks, Body, Depends, FastAPI, HTTPException, Path,
                   Query, Request, Security, status)
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKey, APIKeyHeader, APIKeyQuery
from pydantic import BaseModel, Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings
from prometheus_client import Counter as PrometheusCounter, Gauge, Histogram
from starlette_prometheus import PrometheusMiddleware

matplotlib.use('Agg')

# --- Application Settings ---
class Settings(BaseSettings):
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    APP_TITLE: str = "智慧評分系統 API (Extreme Edition) v2.5 - 全功能版"
    APP_DESCRIPTION: str = "提供基於進階N維張量運算與AI模組的盤面分析、評分建議、批次處理與背景任務的API服務 (2025 Enhanced - 全模組實作)。"
    APP_VERSION: str = "2.5.1" # Incremented version for full implementation
    ANALYZER_VERSION: str = "2.0.1-extreme"

    API_KEY: str = "YOUR_VERY_SECRET_API_KEY_FOR_2025_FULL"
    API_KEY_NAME: str = "X-API-KEY"

    RATE_LIMIT_REQUESTS: int = 200 # Increased for potentially heavier use
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    TASK_CALLBACK_URL_ENABLED: bool = False
    TASK_CALLBACK_URL: HttpUrl | None = None

    MEM_PATH: str = "data/persistent_memory.json"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

# --- Logging Configuration ---
class RequestContextLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, 'request_id'):
            record.request_id = 'N/A_context'
        return True

root_logger = logging.getLogger()
# Clear existing handlers (if any, useful for re-runs in some environments)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
root_logger.addFilter(RequestContextLogFilter())

logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - RequestID: %(request_id)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)

logger = logging.getLogger(__name__)
brain_logger = logging.getLogger("brain")
analyzer_logger = logging.getLogger("analyzer")

# --- Prometheus Metrics ---
REQUEST_COUNT = PrometheusCounter(
    "api_request_count", "Total API requests", ["method", "endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "API request latency", ["method", "endpoint"]
)
ACTIVE_BACKGROUND_TASKS = Gauge(
    "api_active_background_tasks", "Active background scoring tasks"
)
MODULE_USAGE_COUNT = PrometheusCounter(
    "api_module_usage_count", "Usage count per scoring module", ["module_name"]
)

# --- Brain Logic: Utility Classes ---
class MathUtils:
    def sigmoid(self, x: float, k: float = 1.0) -> float:
        try:
            clamped_x = max(-700.0, min(700.0, -k * x))
            return 1 / (1 + math.exp(clamped_x))
        except OverflowError:
            return 0.0 if -k * x > 0 else 1.0

    def normalize_value(self, value: float, min_val: float, max_val: float, clamp: bool = True) -> float:
        if math.isclose(max_val, min_val):
            if math.isclose(value, min_val): return 0.5
            elif value < min_val: return 0.0
            else: return 1.0
        normalized = (value - min_val) / (max_val - min_val)
        if clamp: return max(0.0, min(1.0, normalized))
        return normalized

    def manhattan_distance(self, p1: tuple[int, int], p2: tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def euclidean_distance(self, p1: tuple[int, int], p2: tuple[int, int]) -> float:
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def get_entropy(self, values: list[str | int | float | bool | list | dict]) -> float:
        if not values: return 0.0
        counts = Counter(values)
        total_count = len(values)
        entropy = 0.0
        for count_val in counts.values(): # Renamed 'count' to 'count_val'
            probability = count_val / total_count
            entropy -= probability * math.log2(probability)
        return entropy

class BoardAnalyzerUtils:
    def get_neighborhood_values(
        self, grid: np.ndarray, r: int, c: int, radius: int = 1,
        eight_connectivity: bool = True,
        val_func: Callable[[int], float | None] = lambda x_val: float(x_val) if x_val != -1 else None,
        include_center: bool = False
    ) -> list[float]:
        neighbors: list[float] = []
        rows, cols = grid.shape
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if not include_center and dr == 0 and dc == 0: continue
                if not eight_connectivity:
                    if radius == 1 and abs(dr) + abs(dc) != 1: continue
                    elif radius > 1 and abs(dr) + abs(dc) > radius: continue
                nr, nc_val = r + dr, c + dc # Renamed 'nc' to 'nc_val' to avoid conflict later
                if 0 <= nr < rows and 0 <= nc_val < cols:
                    processed_val = val_func(grid[nr, nc_val])
                    if processed_val is not None:
                        neighbors.append(processed_val)
        return neighbors

    def get_value_gradient_at_cell(
        self, grid: np.ndarray, r: int, c: int,
        val_func: Callable[[int], float] = lambda x_val: float(x_val) if x_val != -1 else 0.0
    ) -> tuple[float, float]:
        rows, cols = grid.shape
        def safe_val(r_in: int, c_in: int) -> float:
            if 0 <= r_in < rows and 0 <= c_in < cols:
                return val_func(grid[r_in, c_in])
            return 0.0
        gx = (safe_val(r - 1, c + 1) + 2 * safe_val(r, c + 1) + safe_val(r + 1, c + 1)) - \
             (safe_val(r - 1, c - 1) + 2 * safe_val(r, c - 1) + safe_val(r + 1, c - 1))
        gy = (safe_val(r + 1, c - 1) + 2 * safe_val(r + 1, c) + safe_val(r + 1, c + 1)) - \
             (safe_val(r - 1, c - 1) + 2 * safe_val(r - 1, c) + safe_val(r - 1, c + 1))
        return gx, gy

    def find_sequences_in_line(
        self, line: list[int], min_len: int = 3,
        check_arithmetic: bool = True, check_geometric: bool = False,
        allow_gaps: int = 0
    ) -> list[list[int]]:
        sequences: list[list[int]] = []
        n = len(line)
        if n < min_len: return sequences

        if check_arithmetic:
            for i in range(n):
                if line[i] == -1: continue
                for j in range(i + 1, n):
                    gaps_between_i_j = 0
                    processed_j = j
                    if line[j] == -1:
                        k = j
                        while k < n and line[k] == -1:
                            gaps_between_i_j += 1
                            k += 1
                        if k == n or gaps_between_i_j > allow_gaps: continue
                        processed_j = k
                    if line[processed_j] == -1: continue

                    diff = line[processed_j] - line[i]
                    # Original PDF logic for diff == 0 and line[i] != 0: break (line 4 page 4 & line 22 page 4)
                    # This means it excludes constant non-zero sequences.
                    # If we want to include them (e.g. [5,5,5] as arithmetic with diff 0):
                    # if diff == 0 and line[i] != 0: pass
                    # For this implementation, let's keep the exclusion for strict arithmetic.
                    if diff == 0 and line[i] != 0 and line[i] != -1: # Added line[i] != -1
                        # Check if a constant sequence of this value should be considered.
                        # For many puzzles, arithmetic implies change.
                        # If min_len is 1, a single number is trivially arithmetic.
                        # Let's assume for min_len >= 2, constant non-zero is not a target sequence.
                        # If we need [5,5,5] to be found, this condition needs adjustment or separate logic for constant sequences.
                        # The PDF's logic at "EXT_F10_Discontinuity_Vec" implies it looks for arithmetic sequences.
                        # The find_sequences_in_line in PDF seems to filter out constant sequences if line[i]!=0 and diff=0.
                        # Let's adhere to that:
                        if line[i] != 0 : # and diff == 0 (already established)
                           # This skips [5,5,5] but allows [0,0,0]
                           continue # Skip this j, try next j for a different diff from i

                    current_seq_values = [line[i], line[processed_j]]
                    last_val_in_seq = line[processed_j]
                    gaps_after_last_val = gaps_between_i_j

                    for l_extend in range(processed_j + 1, n):
                        if line[l_extend] == -1:
                            gaps_after_last_val += 1
                            if gaps_after_last_val > allow_gaps: break
                            continue
                        expected_next = last_val_in_seq + diff
                        if math.isclose(line[l_extend], expected_next):
                            current_seq_values.append(line[l_extend])
                            last_val_in_seq = line[l_extend]
                            gaps_after_last_val = 0
                        elif line[l_extend] != -1: break
                    if len(current_seq_values) >= min_len:
                        sequences.append(current_seq_values)
        
        if check_geometric: # Simplified geometric logic from PDF
            for i in range(n):
                if line[i] == -1 or line[i] == 0 : continue # Geometric sequences usually don't start/contain 0
                for j in range(i + 1, n):
                    gaps_between_i_j_geom = 0
                    processed_j_geom = j
                    if line[j] == -1:
                        k_geom = j
                        while k_geom < n and line[k_geom] == -1:
                            gaps_between_i_j_geom += 1
                            k_geom += 1
                        if k_geom == n or gaps_between_i_j_geom > allow_gaps: continue
                        processed_j_geom = k_geom
                    if line[processed_j_geom] == -1 or line[processed_j_geom] == 0: continue
                    
                    ratio_val: float | None = None
                    try:
                        if math.isclose(line[i], 0): continue # Should be caught
                        ratio_candidate = line[processed_j_geom] / line[i]
                        # PDF check: line[j] % line[i] !=0 and line[i] % line[j] != 0 and not math.isclose(line[j]/line[i], round(line[j]/line[i]))
                        # This was a complex check for integer-like ratios. For simplicity, allow float ratios.
                        ratio_val = ratio_candidate
                    except ZeroDivisionError: continue
                    if ratio_val is None: continue

                    if math.isclose(ratio_val, 1.0) and not math.isclose(line[i], line[processed_j_geom]): continue
                    
                    current_seq_values_geom = [line[i], line[processed_j_geom]]
                    last_val_in_seq_geom = line[processed_j_geom]
                    gaps_after_last_val_geom = gaps_between_i_j_geom

                    for l_extend_geom in range(processed_j_geom + 1, n):
                        if line[l_extend_geom] == -1:
                            gaps_after_last_val_geom += 1
                            if gaps_after_last_val_geom > allow_gaps: break
                            continue
                        if math.isclose(line[l_extend_geom], 0.0) and not math.isclose(last_val_in_seq_geom, 0.0): break

                        expected_next_float = float(last_val_in_seq_geom) * ratio_val
                        if math.isclose(float(line[l_extend_geom]), expected_next_float):
                            current_seq_values_geom.append(line[l_extend_geom])
                            last_val_in_seq_geom = line[l_extend_geom]
                            gaps_after_last_val_geom = 0
                        elif line[l_extend_geom] != -1: break
                    if len(current_seq_values_geom) >= min_len:
                        sequences.append(current_seq_values_geom)
        return sequences

    def get_card_max_value_from_grid_dimensions(self, grid_shape: tuple[int, int]) -> int:
        rows, cols = grid_shape
        if rows == 0 or cols == 0: return 0
        return rows * cols

    def get_all_possible_numbers_for_grid(self, grid_shape: tuple[int, int]) -> set[int]:
        max_val = self.get_card_max_value_from_grid_dimensions(grid_shape)
        if max_val == 0: return set()
        return set(range(1, max_val + 1))

    def get_legal_values_for_placement(self, grid: np.ndarray) -> set[int]:
        if grid.size == 0: return set()
        rows, cols = grid.shape
        all_possible = self.get_all_possible_numbers_for_grid((rows, cols))
        used_positive = set(int(v) for v in grid.flatten() if v != -1 and v > 0)
        return all_possible - used_positive

_math_utils = MathUtils()
_board_analyzer_utils = BoardAnalyzerUtils()

# --- Brain Logic: Scoring Module Implementations ---

# Module 1: EXT_A2_Weighted_Proximity_Vec
async def EXT_A2_Weighted_Proximity_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_A2") -> np.ndarray:
    brain_logger.debug("Executing EXT_A2_Weighted_Proximity_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    radius = 2
    value_weight_factor = 0.15
    distance_decay_factor = 1.8

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            proximity_score = 0.0
            for dr in range(-radius, radius + 1):
                for dc_val in range(-radius, radius + 1): # Renamed dc to dc_val
                    if dr == 0 and dc_val == 0: continue
                    nr, nc_val_neigh = r_idx + dr, c_idx + dc_val # Renamed nc to nc_val_neigh
                    if 0 <= nr < rows and 0 <= nc_val_neigh < cols and grid[nr, nc_val_neigh] != -1:
                        dist = _math_utils.manhattan_distance((r_idx, c_idx), (nr, nc_val_neigh))
                        if dist == 0: dist = 1
                        score_contribution = (grid[nr, nc_val_neigh] * value_weight_factor) / (dist ** distance_decay_factor)
                        proximity_score += score_contribution
            
            max_val_on_grid = _board_analyzer_utils.get_card_max_value_from_grid_dimensions((rows, cols))
            if max_val_on_grid == 0: max_val_on_grid = 1.0
            num_neighbors_in_radius = (2 * radius + 1)**2 - 1
            heuristic_max_score = num_neighbors_in_radius * max_val_on_grid * value_weight_factor / (1**distance_decay_factor)
            if heuristic_max_score > 0:
                scores[r_idx, c_idx] = _math_utils.normalize_value(proximity_score, 0, heuristic_max_score, clamp=True)
            else:
                scores[r_idx, c_idx] = 0.0
    return scores

# Module 2: EXT_M3_Local_Heterogeneity_Vec
async def EXT_M3_Local_Heterogeneity_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_M3") -> np.ndarray:
    brain_logger.debug("Executing EXT_M3_Local_Heterogeneity_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    radius = 1
    min_neighbors_for_robust_score = 2
    all_possible_values_in_game = _board_analyzer_utils.get_all_possible_numbers_for_grid(grid.shape)
    if not all_possible_values_in_game: return scores

    max_theoretical_entropy: float
    num_possible_values = len(all_possible_values_in_game)
    if num_possible_values > 1:
        max_theoretical_entropy = math.log2(num_possible_values)
    elif num_possible_values == 1: #Handles case where only one number is possible (e.g. 1x1 grid, only 1)
        max_theoretical_entropy = math.log2(2) # Avoid log2(1)=0, give some scale or use 0 if no diversity possible
    else: # No possible values (empty set)
        max_theoretical_entropy = 1.0 # Fallback, though should be caught by early exit

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            neighbor_values = _board_analyzer_utils.get_neighborhood_values(
                grid, r_idx, c_idx, radius=radius,
                val_func=lambda x_val: int(x_val) if x_val != -1 else None,
                include_center=False
            )
            if len(neighbor_values) < min_neighbors_for_robust_score:
                scores[r_idx, c_idx] = 0.0
                continue
            current_entropy = _math_utils.get_entropy(neighbor_values)
            if max_theoretical_entropy > 0:
                normalized_score = current_entropy / max_theoretical_entropy
                scores[r_idx, c_idx] = _math_utils.normalize_value(normalized_score, 0, 1, clamp=True)
            else:
                scores[r_idx, c_idx] = 0.0
    return scores

# Module 3: EXT_D3_Potential_Field_Vec
async def EXT_D3_Potential_Field_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_D3") -> np.ndarray:
    brain_logger.debug("Executing EXT_D3_Potential_Field_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    decay_exponent = 1.5
    max_influence_radius = 3
    max_possible_val_on_grid = _board_analyzer_utils.get_card_max_value_from_grid_dimensions((rows, cols))
    if max_possible_val_on_grid == 0: return scores

    num_cells_in_radius_approx = (2 * max_influence_radius + 1)**2 - 1
    heuristic_max_potential = num_cells_in_radius_approx * (max_possible_val_on_grid / (1**decay_exponent))
    if heuristic_max_potential == 0: heuristic_max_potential = 1.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            current_cell_potential = 0.0
            for nr in range(rows):
                for nc_val in range(cols): # Renamed nc
                    if grid[nr, nc_val] != -1:
                        num_val = grid[nr, nc_val]
                        if num_val <= 0: continue # Consider only positive charges
                        dist = _math_utils.manhattan_distance((r_idx, c_idx), (nr, nc_val))
                        if dist == 0: continue
                        if dist > max_influence_radius: continue
                        potential_contribution = num_val / (dist ** decay_exponent)
                        current_cell_potential += potential_contribution
            scores[r_idx, c_idx] = _math_utils.normalize_value(current_cell_potential, 0, heuristic_max_potential, clamp=True)
    return scores

# Module 4: EXT_F10_Discontinuity_Vec
async def EXT_F10_Discontinuity_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_F10") -> np.ndarray:
    brain_logger.debug("Executing EXT_F10_Discontinuity_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    legal_values_for_placement = _board_analyzer_utils.get_legal_values_for_placement(grid)
    if not legal_values_for_placement: return scores

    min_sequence_len_to_score = 3
    # Ensure heuristic_max_len is at least min_sequence_len_to_score for normalization
    heuristic_max_len = float(max(rows, cols, min_sequence_len_to_score))


    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_len_contribution_for_this_cell = 0.0

            for val_to_try in legal_values_for_placement:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = val_to_try
                current_val_max_len = 0.0

                # Check Row
                row_line = list(temp_grid[r_idx, :])
                sequences_in_row = _board_analyzer_utils.find_sequences_in_line(row_line, min_len=min_sequence_len_to_score, allow_gaps=1, check_arithmetic=True)
                for seq in sequences_in_row:
                    if val_to_try in seq: current_val_max_len = max(current_val_max_len, len(seq))
                
                # Check Column
                col_line = list(temp_grid[:, c_idx])
                sequences_in_col = _board_analyzer_utils.find_sequences_in_line(col_line, min_len=min_sequence_len_to_score, allow_gaps=1, check_arithmetic=True)
                for seq in sequences_in_col:
                    if val_to_try in seq: current_val_max_len = max(current_val_max_len, len(seq))

                # Check Main Diagonal
                diag1_line = list(np.diag(temp_grid, k=c_idx - r_idx))
                sequences_in_diag1 = _board_analyzer_utils.find_sequences_in_line(diag1_line, min_len=min_sequence_len_to_score, allow_gaps=1, check_arithmetic=True)
                for seq in sequences_in_diag1:
                    if val_to_try in seq: current_val_max_len = max(current_val_max_len, len(seq))

                # Check Anti-Diagonal
                flipped_temp_grid = np.fliplr(temp_grid)
                flipped_c_idx = cols - 1 - c_idx
                diag2_line = list(np.diag(flipped_temp_grid, k=flipped_c_idx - r_idx))
                sequences_in_diag2 = _board_analyzer_utils.find_sequences_in_line(diag2_line, min_len=min_sequence_len_to_score, allow_gaps=1, check_arithmetic=True)
                for seq in sequences_in_diag2:
                    if val_to_try in seq: current_val_max_len = max(current_val_max_len, len(seq))
                
                if current_val_max_len >= min_sequence_len_to_score:
                    max_len_contribution_for_this_cell = max(max_len_contribution_for_this_cell, current_val_max_len)
            
            if heuristic_max_len > 0:
                scores[r_idx, c_idx] = _math_utils.normalize_value(max_len_contribution_for_this_cell, 0, heuristic_max_len, clamp=True)
            else: # Should not happen if heuristic_max_len is correctly handled
                scores[r_idx, c_idx] = 0.0
    return scores

# Module 5: EXT_P7_Pathfinding_Value_Vec
async def EXT_P7_Pathfinding_Value_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_P7") -> np.ndarray:
    brain_logger.debug("Executing EXT_P7_Pathfinding_Value_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    legal_values_for_placement = _board_analyzer_utils.get_legal_values_for_placement(grid)
    if not legal_values_for_placement: return scores

    max_path_search_depth = 4
    path_value_decay_factor = 1.0
    max_possible_val_on_grid = _board_analyzer_utils.get_card_max_value_from_grid_dimensions((rows, cols))
    if max_possible_val_on_grid == 0: max_possible_val_on_grid = 1.0
    
    # Heuristic from PDF: ((2*max_path_search_depth + 1)**2 * max_possible_val_on_grid / (1**path_value_decay_factor))
    # This seems to overestimate by squaring (2*D+1) (area) vs linear connections
    # A simpler heuristic: max_val * (number of directions, e.g. 4 or 8) / (min_dist_decay)
    # For now, use PDF's heuristic, but note it might be high.
    heuristic_max_path_score = ((2 * max_path_search_depth + 1)**2 * max_possible_val_on_grid) / (1**path_value_decay_factor)
    if heuristic_max_path_score == 0: heuristic_max_path_score = 1.0

    for r_start in range(rows):
        for c_start in range(cols):
            if grid[r_start, c_start] != -1: continue
            max_score_for_this_cell = 0.0
            for val_to_try in legal_values_for_placement: # val_to_try is not used in path logic from PDF
                current_placement_path_score = 0.0
                q = deque([((r_start, c_start), 0)]) # ((r,c), path_len)
                visited_for_bfs = set([(r_start, c_start)])
                
                head_count = 0
                # Max steps: rows * cols is a reasonable limit for BFS on a grid
                max_bfs_steps = rows * cols 

                while q and head_count < max_bfs_steps :
                    head_count += 1
                    (curr_r, curr_c), path_len = q.popleft()
                    # Corrected BFS neighbor exploration (4-connectivity for paths)
                    for dr, dc_val in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # dc renamed to dc_val
                        next_r, next_c = curr_r + dr, curr_c + dc_val
                        if 0 <= next_r < rows and 0 <= next_c < cols:
                            if grid[next_r, next_c] != -1: # Found an existing number
                                reached_val = grid[next_r, next_c]
                                effective_path_len = path_len + 1 # Path to this number
                                current_placement_path_score += reached_val / (effective_path_len ** path_value_decay_factor)
                                # Do not add to queue or visited, this path segment ends here
                            elif (next_r, next_c) not in visited_for_bfs and \
                                 grid[next_r, next_c] == -1 and \
                                 path_len + 1 < max_path_search_depth: # Path can traverse other empty cells
                                visited_for_bfs.add((next_r, next_c))
                                q.append(((next_r, next_c), path_len + 1))
                
                if current_placement_path_score > max_score_for_this_cell:
                    max_score_for_this_cell = current_placement_path_score
            
            scores[r_start, c_start] = _math_utils.normalize_value(max_score_for_this_cell, 0, heuristic_max_path_score, clamp=True)
    return scores

# Module 6: EXT_R5_Resource_Control_Vec
async def EXT_R5_Resource_Control_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_R5") -> np.ndarray:
    brain_logger.debug("Executing EXT_R5_Resource_Control_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(_board_analyzer_utils.get_legal_values_for_placement(grid))
    max_possible_val_on_grid = _board_analyzer_utils.get_card_max_value_from_grid_dimensions((rows, cols))
    if max_possible_val_on_grid == 0: max_possible_val_on_grid = 1.0

    hypothetical_high_val_placed = 0.0
    if potential_numbers_to_place:
        hypothetical_high_val_placed = float(np.max(potential_numbers_to_place)) # Cast to float

    w_row, w_col, w_val = 0.3, 0.3, 0.4 # Weights

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            
            num_filled_in_row = np.count_nonzero(grid[r_idx, :] != -1)
            row_completeness_score = (num_filled_in_row + 1) / cols if cols > 0 else 0.0 # +1 for placing here

            num_filled_in_col = np.count_nonzero(grid[:, c_idx] != -1)
            col_completeness_score = (num_filled_in_col + 1) / rows if rows > 0 else 0.0 # +1 for placing here

            value_capture_score = 0.0
            if hypothetical_high_val_placed > 0 and max_possible_val_on_grid > 0 : # Ensure max_possible is not 0
                 # PDF used 1 as min_val for normalization. If hypothetical can be 0, use 0.
                 # Assume numbers are >= 1 for placement.
                value_capture_score = _math_utils.normalize_value(hypothetical_high_val_placed, 1, max_possible_val_on_grid, clamp=True)
            
            combined_score = (w_row * row_completeness_score +
                              w_col * col_completeness_score +
                              w_val * value_capture_score)
            # Since components are [0,1] and weights sum to 1, combined_score is also [0,1]
            # Normalizing again ensures it if weights change or for robustness.
            scores[r_idx, c_idx] = _math_utils.normalize_value(combined_score, 0, 1.0, clamp=True)
    return scores

# Module 7: EXT_GM1_Row_Control_Vec
async def EXT_GM1_Row_Control_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM1") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM1_Row_Control_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(_board_analyzer_utils.get_legal_values_for_placement(grid))
    avg_potential_num_to_place = 0.0
    if potential_numbers_to_place:
        avg_potential_num_to_place = float(np.mean(potential_numbers_to_place)) # Cast

    max_val_board = _board_analyzer_utils.get_card_max_value_from_grid_dimensions((rows, cols))
    if max_val_board == 0: max_val_board = 1.0

    w_density, w_sum, w_seq = 0.4, 0.3, 0.3

    for r_idx in range(rows):
        current_row_values_list = [val for val in grid[r_idx, :] if val != -1]
        num_filled_in_row = len(current_row_values_list)
        sum_current_row_values = sum(current_row_values_list)

        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue

            density_score = (num_filled_in_row + 1.0) / cols if cols > 0 else 0.0
            
            potential_row_sum = sum_current_row_values + avg_potential_num_to_place
            heuristic_max_row_sum = float(cols * max_val_board) # Ensure float
            sum_score = 0.0
            if heuristic_max_row_sum > 0:
                sum_score = _math_utils.normalize_value(potential_row_sum, 0, heuristic_max_row_sum, clamp=True)

            seq_score = 0.0
            temp_row_line = list(grid[r_idx, :])
            # PDF used avg_potential_num_to_place for temp placement.
            # This might be problematic if avg is float and sequences expect ints.
            # Let's cast it to int for sequence checking, or iterate through potential numbers.
            # For simplicity, casting to int(round(avg))
            val_for_seq_check = int(round(avg_potential_num_to_place)) if avg_potential_num_to_place > 0 else 0 # ensure non-negative or handle -1
            if val_for_seq_check in potential_numbers_to_place or not potential_numbers_to_place : # only if it's a plausible value or no values
                temp_row_line[c_idx] = val_for_seq_check if val_for_seq_check != -1 else 0 # use 0 if avg is ~-1
                
                # Check if placing avg_potential_num_to_place (or rounded int version) completes/extends a sequence.
                # The find_sequences_in_line takes list[int].
                sequences_found = _board_analyzer_utils.find_sequences_in_line(temp_row_line, min_len=3, allow_gaps=1, check_arithmetic=True)
                max_seq_len_with_avg = 0
                for seq in sequences_found:
                    # Check if the *placed value at c_idx* is part of the sequence
                    # This means the value we put (val_for_seq_check) should be in the found seq.
                    if val_for_seq_check in seq :
                         # And also that the original cell c_idx was involved.
                         # This is implicitly true as temp_row_line[c_idx] was changed.
                        max_seq_len_with_avg = max(max_seq_len_with_avg, len(seq))

                if max_seq_len_with_avg >= 3:
                    seq_score = _math_utils.normalize_value(float(max_seq_len_with_avg), 3, float(cols), clamp=True) # Normalize by row length
                elif max_seq_len_with_avg > 0: # Small bonus for contributing
                    seq_score = 0.25
            
            combined_score = (w_density * density_score +
                              w_sum * sum_score +
                              w_seq * seq_score)
            scores[r_idx, c_idx] = _math_utils.normalize_value(combined_score, 0, 1.0, clamp=True)
    return scores

# Module 8: EXT_GM2_Col_Flow_Vec
async def EXT_GM2_Col_Flow_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM2") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM2_Col_Flow_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(_board_analyzer_utils.get_legal_values_for_placement(grid))
    avg_potential_num_to_place = 0.0
    if potential_numbers_to_place:
        avg_potential_num_to_place = float(np.mean(potential_numbers_to_place))

    max_val_board = _board_analyzer_utils.get_card_max_value_from_grid_dimensions((rows, cols))
    if max_val_board == 0: max_val_board = 1.0

    w_density, w_sum, w_seq = 0.4, 0.3, 0.3

    for c_idx in range(cols):
        current_col_values_list = [val for val in grid[:, c_idx] if val != -1]
        num_filled_in_col = len(current_col_values_list)
        sum_current_col_values = sum(current_col_values_list)

        for r_idx in range(rows):
            if grid[r_idx, c_idx] != -1: continue

            density_score = (num_filled_in_col + 1.0) / rows if rows > 0 else 0.0
            
            potential_col_sum = sum_current_col_values + avg_potential_num_to_place
            heuristic_max_col_sum = float(rows * max_val_board)
            sum_score = 0.0
            if heuristic_max_col_sum > 0:
                sum_score = _math_utils.normalize_value(potential_col_sum, 0, heuristic_max_col_sum, clamp=True)

            seq_score = 0.0
            temp_col_line = list(grid[:, c_idx])
            val_for_seq_check = int(round(avg_potential_num_to_place)) if avg_potential_num_to_place > 0 else 0
            if val_for_seq_check in potential_numbers_to_place or not potential_numbers_to_place:
                temp_col_line[r_idx] = val_for_seq_check if val_for_seq_check != -1 else 0
                sequences_found = _board_analyzer_utils.find_sequences_in_line(temp_col_line, min_len=3, allow_gaps=1, check_arithmetic=True)
                max_seq_len_with_avg = 0
                for seq in sequences_found:
                    if val_for_seq_check in seq:
                        max_seq_len_with_avg = max(max_seq_len_with_avg, len(seq))
                
                if max_seq_len_with_avg >= 3:
                    seq_score = _math_utils.normalize_value(float(max_seq_len_with_avg), 3, float(rows), clamp=True) # Normalize by col length
                elif max_seq_len_with_avg > 0:
                    seq_score = 0.25
            
            combined_score = (w_density * density_score +
                              w_sum * sum_score +
                              w_seq * seq_score)
            scores[r_idx, c_idx] = _math_utils.normalize_value(combined_score, 0, 1.0, clamp=True)
    return scores

# Module 9: EXT_GM3_Adv_Connected_Comp_Vec (空格區域)
async def EXT_GM3_Adv_Connected_Comp_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM3") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM3_Adv_Connected_Comp_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    visited_overall = np.zeros_like(grid, dtype=bool) # Tracks visited cells for any component search

    for r_start in range(rows):
        for c_start in range(cols):
            if visited_overall[r_start, c_start] or grid[r_start, c_start] != -1: # Skip if visited or not empty
                continue

            component_cells: list[tuple[int, int]] = []
            q = deque([(r_start, c_start)])
            # visited_bfs_current_component = set([(r_start, c_start)]) # PDF had this, but visited_overall should be enough
            visited_overall[r_start, c_start] = True # Mark as globally visited

            current_component_size = 0
            while q:
                r_curr, c_curr = q.popleft()
                component_cells.append((r_curr, c_curr))
                current_component_size +=1

                for dr_bfs, dc_bfs in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # 4-connectivity
                    nr, nc_val = r_curr + dr_bfs, c_curr + dc_bfs # Renamed nc
                    if 0 <= nr < rows and 0 <= nc_val < cols and \
                       grid[nr, nc_val] == -1 and \
                       not visited_overall[nr, nc_val]:
                        visited_overall[nr, nc_val] = True
                        # visited_bfs_current_component.add((nr, nc_val)) # Not strictly needed if visited_overall is primary
                        q.append((nr, nc_val))
            
            area_size = float(current_component_size) # Was len(component_cells)
            total_cells = float(rows * cols)
            norm_area_size = 0.0
            if total_cells > 0:
                norm_area_size = _math_utils.normalize_value(area_size, 0, total_cells, clamp=True)
            
            for r_comp, c_comp in component_cells:
                scores[r_comp, c_comp] = norm_area_size
    return scores

# Module 10: EXT_GM4_Spatial_Auto_Corr_Vec
async def EXT_GM4_Spatial_Auto_Corr_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM4") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM4_Spatial_Auto_Corr_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers = list(_board_analyzer_utils.get_legal_values_for_placement(grid))
    hypothetical_val_to_place: float
    if potential_numbers:
        hypothetical_val_to_place = float(np.median(potential_numbers)) # PDF used median
    else:
        max_board_val = _board_analyzer_utils.get_card_max_value_from_grid_dimensions((rows, cols))
        hypothetical_val_to_place = (1.0 + float(max_board_val)) / 2.0 if max_board_val > 0 else 0.5

    max_val_on_grid_for_norm = float(_board_analyzer_utils.get_card_max_value_from_grid_dimensions((rows, cols)))
    if max_val_on_grid_for_norm == 0: max_val_on_grid_for_norm = 1.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue

            neighbor_values = _board_analyzer_utils.get_neighborhood_values(
                grid, r_idx, c_idx, radius=1, eight_connectivity=True,
                val_func=lambda x_val: float(x_val) if x_val != -1 else None, # PDF used x, not x_val
                include_center=False
            )
            if not neighbor_values:
                scores[r_idx, c_idx] = 0.5 # Neutral score if no neighbors
                continue
            
            mean_neighbors = float(np.mean(neighbor_values))
            diff_hypothetical_to_mean_neighbors = abs(hypothetical_val_to_place - mean_neighbors)
            
            # Score for positive autocorrelation: 1.0 - normalized_difference
            norm_diff = _math_utils.normalize_value(diff_hypothetical_to_mean_neighbors, 0, max_val_on_grid_for_norm, clamp=True)
            positive_autocorr_score = 1.0 - norm_diff
            scores[r_idx, c_idx] = positive_autocorr_score
    return scores
# main.py - Part 2 of 3 (Connects after Part 1)

# (Continuing Brain Logic: Scoring Module Implementations from Part 1)

# Module 11: EXT_GM5_Line_Completion_Vec
async def EXT_GM5_Line_Completion_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM5") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM5_Line_Completion_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0 or min(rows,cols) < 1: return scores # min(rows,cols) < 1 for any line

    potential_numbers_to_place = list(_board_analyzer_utils.get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores

    line_completion_score_map = {
        "identical_3": 0.6,
        "arithmetic_3_mend": 0.7, # X, p_val, Y
        "arithmetic_3_extend": 0.5, # p_val, X, Y or X, Y, p_val
        "arithmetic_3_mend_high_val": 0.9, # Added per PDF
    }
    max_board_val = _board_analyzer_utils.get_card_max_value_from_grid_dimensions((rows, cols))
    high_val_threshold = max_board_val * 0.7 if max_board_val > 0 else 10.0 # Ensure float

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_completion_score_for_cell = 0.0
            for p_val in potential_numbers_to_place:
                # Check 4 directions: H, V, Diag, Anti-Diag
                for dr, dc_val in [(0, 1), (1, 0), (1, 1), (1, -1)]: # Renamed dc
                    if dr == 0 and dc_val == 0: continue # Should not happen with this set

                    # Case 1: Mending a line -> N1 - p_val - N2
                    r_n1, c_n1 = r_idx - dr, c_idx - dc_val
                    r_n2, c_n2 = r_idx + dr, c_idx + dc_val
                    if 0 <= r_n1 < rows and 0 <= c_n1 < cols and \
                       0 <= r_n2 < rows and 0 <= c_n2 < cols:
                        val_n1, val_n2 = grid[r_n1, c_n1], grid[r_n2, c_n2]
                        if val_n1 != -1 and val_n2 != -1:
                            # Identical
                            if val_n1 == p_val and val_n2 == p_val:
                                max_completion_score_for_cell = max(max_completion_score_for_cell, line_completion_score_map["identical_3"])
                            # Arithmetic (mending, diff != 0)
                            if (val_n1 + val_n2) == 2 * p_val and abs(p_val - val_n1) > 0:
                                current_score = line_completion_score_map["arithmetic_3_mend"]
                                if (val_n1 + p_val + val_n2) / 3.0 > high_val_threshold: # Ensure float division
                                    current_score = max(current_score, line_completion_score_map.get("arithmetic_3_mend_high_val", current_score))
                                max_completion_score_for_cell = max(max_completion_score_for_cell, current_score)
                    
                    # Case 2: Extending a line -> p_val - N1 - N2
                    r_n1_ext1, c_n1_ext1 = r_idx + dr, c_idx + dc_val
                    r_n2_ext1, c_n2_ext1 = r_idx + 2 * dr, c_idx + 2 * dc_val
                    if 0 <= r_n1_ext1 < rows and 0 <= c_n1_ext1 < cols and \
                       0 <= r_n2_ext1 < rows and 0 <= c_n2_ext1 < cols:
                        val_n1_ext1, val_n2_ext1 = grid[r_n1_ext1, c_n1_ext1], grid[r_n2_ext1, c_n2_ext1]
                        if val_n1_ext1 != -1 and val_n2_ext1 != -1:
                            # Identical
                            if p_val == val_n1_ext1 and p_val == val_n2_ext1:
                                max_completion_score_for_cell = max(max_completion_score_for_cell, line_completion_score_map["identical_3"])
                            # Arithmetic (p_val, N1, N2 is arith, diff !=0)
                            if (p_val + val_n2_ext1) == 2 * val_n1_ext1 and abs(val_n1_ext1 - p_val) > 0 :
                                max_completion_score_for_cell = max(max_completion_score_for_cell, line_completion_score_map["arithmetic_3_extend"])

                    # Case 3: Extending a line -> N1 - N2 - p_val
                    r_n1_ext2, c_n1_ext2 = r_idx - 2 * dr, c_idx - 2 * dc_val
                    r_n2_ext2, c_n2_ext2 = r_idx - dr, c_idx - dc_val
                    if 0 <= r_n1_ext2 < rows and 0 <= c_n1_ext2 < cols and \
                       0 <= r_n2_ext2 < rows and 0 <= c_n2_ext2 < cols:
                        val_n1_ext2, val_n2_ext2 = grid[r_n1_ext2, c_n1_ext2], grid[r_n2_ext2, c_n2_ext2]
                        if val_n1_ext2 != -1 and val_n2_ext2 != -1:
                            # Identical
                            if val_n1_ext2 == val_n2_ext2 and val_n1_ext2 == p_val:
                                max_completion_score_for_cell = max(max_completion_score_for_cell, line_completion_score_map["identical_3"])
                            # Arithmetic (N1, N2, p_val is arith, diff !=0)
                            if (val_n1_ext2 + p_val) == 2 * val_n2_ext2 and abs(val_n2_ext2 - val_n1_ext2) > 0:
                                max_completion_score_for_cell = max(max_completion_score_for_cell, line_completion_score_map["arithmetic_3_extend"])
            
            scores[r_idx, c_idx] = _math_utils.normalize_value(max_completion_score_for_cell, 0, 1.0, clamp=True) # Scores are already ~0-1
    return scores

# Module 12: EXT_GM6_Symmetry_Potential_Vec
async def EXT_GM6_Symmetry_Potential_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM6") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM6_Symmetry_Potential_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(_board_analyzer_utils.get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores

    symmetry_scores_map = {
        "horizontal": 0.7, "vertical": 0.7, "point_center": 0.8,
        "main_diagonal": 0.6, "anti_diagonal": 0.6
    }
    if rows == cols: # More emphasis for square grids
        symmetry_scores_map["main_diagonal"] = 0.7
        symmetry_scores_map["anti_diagonal"] = 0.7

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_symmetry_score_for_cell = 0.0
            for p_val in potential_numbers_to_place:
                current_pval_max_sym = 0.0
                # Horizontal: (r, c) vs (r, cols-1-c)
                sr_h, sc_h = r_idx, cols - 1 - c_idx
                if sc_h != c_idx and 0 <= sr_h < rows and 0 <= sc_h < cols and grid[sr_h, sc_h] == p_val:
                    current_pval_max_sym = max(current_pval_max_sym, symmetry_scores_map["horizontal"])
                
                # Vertical: (r, c) vs (rows-1-r, c)
                sr_v, sc_v = rows - 1 - r_idx, c_idx
                if sr_v != r_idx and 0 <= sr_v < rows and 0 <= sc_v < cols and grid[sr_v, sc_v] == p_val:
                    current_pval_max_sym = max(current_pval_max_sym, symmetry_scores_map["vertical"])

                # Point (Center): (r, c) vs (rows-1-r, cols-1-c)
                sr_p, sc_p = rows - 1 - r_idx, cols - 1 - c_idx
                if (sr_p != r_idx or sc_p != c_idx) and \
                   0 <= sr_p < rows and 0 <= sc_p < cols and grid[sr_p, sc_p] == p_val: # Check it's not the same cell
                    current_pval_max_sym = max(current_pval_max_sym, symmetry_scores_map["point_center"])

                if rows == cols: # Diagonal symmetries mainly for square grids
                    # Main Diagonal: (r, c) vs (c, r)
                    sr_d1, sc_d1 = c_idx, r_idx
                    if (sr_d1 != r_idx or sc_d1 != c_idx) and \
                       0 <= sr_d1 < rows and 0 <= sc_d1 < cols and grid[sr_d1, sc_d1] == p_val:
                        current_pval_max_sym = max(current_pval_max_sym, symmetry_scores_map["main_diagonal"])
                    
                    # Anti-Diagonal: (r,c) vs (N-1-c, N-1-r) where N=rows=cols
                    # PDF: (cols-1-c_idx, rows-1-r_idx) which becomes (N-1-c, N-1-r) for (r,c)
                    # Check: if grid[r,c] compares with grid[ (N-1)-c, (N-1)-r ]
                    sr_d2, sc_d2 = (rows - 1) - c_idx, (cols - 1) - r_idx # Note: PDF uses (rows-1)-c_idx, (rows-1)-r_idx for (N-1)-c, (N-1)-r reflection.
                                                                      # This means element at (r,c) is symmetric to element at ( (N-1)-c, (N-1)-r ).
                    if (sr_d2 != r_idx or sc_d2 != c_idx) and \
                       0 <= sr_d2 < rows and 0 <= sc_d2 < cols and grid[sr_d2, sc_d2] == p_val:
                        current_pval_max_sym = max(current_pval_max_sym, symmetry_scores_map["anti_diagonal"])
                
                if current_pval_max_sym > max_symmetry_score_for_cell:
                    max_symmetry_score_for_cell = current_pval_max_sym
            
            scores[r_idx, c_idx] = _math_utils.normalize_value(max_symmetry_score_for_cell, 0, 1.0, clamp=True) # Max of map is ~0.8
    return scores

# Module 13: EXT_GM7_Numeric_Gaps_Vec
async def EXT_GM7_Numeric_Gaps_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM7") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM7_Numeric_Gaps_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(_board_analyzer_utils.get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores

    gap_fill_scores_map = {
        "arithmetic_1_gap_fill": 0.9,  # Fills X, p_val, X+2 (i.e. p_val = X+1)
        "arithmetic_generic_mend": 0.7, # Fills X, p_val, Y where X, p_val, Y is arithmetic
        "arithmetic_generic_extend": 0.5, # p_val, X, Y or X, Y, p_val is arithmetic
        "arithmetic_gap_fill_high_val": 0.95, # Added per PDF
    }
    max_board_val = _board_analyzer_utils.get_card_max_value_from_grid_dimensions((rows,cols))
    high_val_threshold = max_board_val * 0.7 if max_board_val > 0 else 10.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_cell_gap_score = 0.0
            for p_val in potential_numbers_to_place:
                # Iterate over 4 directions (H, V, Diag, Anti-Diag)
                for dr, dc_val in [(0, 1), (1, 0), (1, 1), (1, -1)]: # Renamed dc
                    # Case 1: p_val mends a gap: N1 - p_val - N2
                    r_n1, c_n1 = r_idx - dr, c_idx - dc_val
                    r_n2, c_n2 = r_idx + dr, c_idx + dc_val
                    if 0 <= r_n1 < rows and 0 <= c_n1 < cols and \
                       0 <= r_n2 < rows and 0 <= c_n2 < cols:
                        val_n1, val_n2 = grid[r_n1, c_n1], grid[r_n2, c_n2]
                        if val_n1 != -1 and val_n2 != -1:
                            # Specific check for arithmetic sequence with common difference 1
                            if val_n1 == p_val - 1 and val_n2 == p_val + 1:
                                current_score = gap_fill_scores_map["arithmetic_1_gap_fill"]
                                if (val_n1 + p_val + val_n2) / 3.0 > high_val_threshold:
                                    current_score = max(current_score, gap_fill_scores_map.get("arithmetic_gap_fill_high_val", current_score))
                                max_cell_gap_score = max(max_cell_gap_score, current_score)
                            # Generic arithmetic sequence check (d != 0)
                            elif (val_n1 + val_n2) == 2 * p_val and abs(p_val - val_n1) > 0:
                                max_cell_gap_score = max(max_cell_gap_score, gap_fill_scores_map["arithmetic_generic_mend"])
                    
                    # Case 2: p_val extends a sequence: p_val - N1 - N2
                    r_n1_ext1, c_n1_ext1 = r_idx + dr, c_idx + dc_val
                    r_n2_ext1, c_n2_ext1 = r_idx + 2 * dr, c_idx + 2 * dc_val
                    if 0 <= r_n1_ext1 < rows and 0 <= c_n1_ext1 < cols and \
                       0 <= r_n2_ext1 < rows and 0 <= c_n2_ext1 < cols:
                        val_n1_ext1, val_n2_ext1 = grid[r_n1_ext1, c_n1_ext1], grid[r_n2_ext1, c_n2_ext1]
                        if val_n1_ext1 != -1 and val_n2_ext1 != -1:
                             # Check for: p_val, val_n1_ext1, val_n2_ext1 is arithmetic (and diff != 0)
                            if (val_n1_ext1 - p_val) == (val_n2_ext1 - val_n1_ext1) and (val_n1_ext1 - p_val) != 0 :
                                max_cell_gap_score = max(max_cell_gap_score, gap_fill_scores_map["arithmetic_generic_extend"])
                    
                    # Case 3: p_val extends a sequence: N1 - N2 - p_val
                    r_n1_ext2, c_n1_ext2 = r_idx - 2 * dr, c_idx - 2 * dc_val
                    r_n2_ext2, c_n2_ext2 = r_idx - dr, c_idx - dc_val
                    if 0 <= r_n1_ext2 < rows and 0 <= c_n1_ext2 < cols and \
                       0 <= r_n2_ext2 < rows and 0 <= c_n2_ext2 < cols:
                        val_n1_ext2, val_n2_ext2 = grid[r_n1_ext2, c_n1_ext2], grid[r_n2_ext2, c_n2_ext2]
                        if val_n1_ext2 != -1 and val_n2_ext2 != -1:
                            # Check for: val_n1_ext2, val_n2_ext2, p_val is arithmetic (and diff != 0)
                            if (val_n2_ext2 - val_n1_ext2) == (p_val - val_n2_ext2) and (val_n2_ext2 - val_n1_ext2) != 0:
                                max_cell_gap_score = max(max_cell_gap_score, gap_fill_scores_map["arithmetic_generic_extend"])
            scores[r_idx, c_idx] = _math_utils.normalize_value(max_cell_gap_score, 0, 1.0, clamp=True) # Scores are ~0-1
    return scores

# Module 14: EXT_GM8_Edge_Affinity_Vec
async def EXT_GM8_Edge_Affinity_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM8") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM8_Edge_Affinity_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    affinity_mode = "prefer_edge" # "prefer_edge" or "avoid_edge"
    corner_bonus_prefer = 0.2
    corner_penalty_avoid = 0.2 # Not used if mode is prefer_edge

    # Max possible minimum distance to an edge (for a cell at the center)
    max_min_dist_to_edge_row = (rows - 1) // 2 if rows > 0 else 0
    max_min_dist_to_edge_col = (cols - 1) // 2 if cols > 0 else 0
    
    # overall_max_of_min_distances is the smallest of these two if board is not a line
    # If it's a line (e.g. 1xN or Nx1), one of them is 0.
    # Example: 5x5, center (2,2), dists (2,2,2,2), min_dist=2. max_min_dist_row=2, max_min_dist_col=2. overall=2.
    # Example: 5x3, center-ish (2,1), dists_r (2,2), dists_c (1,1). min_dist for (2,1) is 1.
    # max_min_dist_row=2, max_min_dist_col=1. overall=1.
    overall_max_of_min_distances = float(min(max_min_dist_to_edge_row, max_min_dist_to_edge_col))

    if rows <= 1 or cols <= 1: # If it's a line or a single cell
        overall_max_of_min_distances = 0.0 # All cells are on an edge
    
    # To avoid division by zero if all cells are effectively on edge (e.g. 2xN grid, max_min_dist can be 0)
    # Or if overall_max_of_min_distances is 0 for lines.
    # The normalization logic should handle min_dist / 0. If overall_max_of_min_distances is 0,
    # it means all cells have min_dist 0 to an edge.
    # A value of 0.5 was used in PDF to avoid div by zero; math_utils.normalize should handle it if max=min=0.
    if overall_max_of_min_distances == 0 and (rows > 1 and cols > 1): # e.g. 2x2 grid, max_min_dist = 0
         pass # It's fine, normalize_value handles it
    elif overall_max_of_min_distances == 0: # Covers 1xN, Nx1, 1x1, 2x2 (max_min=0)
        pass


    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue

            dist_to_top = r_idx
            dist_to_bottom = rows - 1 - r_idx
            dist_to_left = c_idx
            dist_to_right = cols - 1 - c_idx
            min_dist = float(min(dist_to_top, dist_to_bottom, dist_to_left, dist_to_right))

            is_corner = (r_idx == 0 or r_idx == rows - 1) and \
                        (c_idx == 0 or c_idx == cols - 1)
            
            current_score = 0.0
            # normalized_dist will be 0 if on edge, 1 if at overall_max_of_min_distances (center-most)
            # If overall_max_of_min_distances is 0, all cells are "on edge", so min_dist is 0, norm_dist should be 0 (or 0.5 by normalize_value if val=min=max=0)
            # MathUtils.normalize(0,0,0) -> 0.5. If min_dist=0, max_overall=0 then it's 0.5.
            # If min_dist=0, max_overall > 0 -> 0.
            # This logic is fine.
            normalized_dist = _math_utils.normalize_value(min_dist, 0, overall_max_of_min_distances, clamp=True)


            if affinity_mode == "prefer_edge":
                current_score = 1.0 - normalized_dist # Closer to edge (smaller dist) -> higher score
                if is_corner and min_dist == 0: # Check min_dist == 0 to ensure it's truly on edge for corner bonus
                    current_score += corner_bonus_prefer
            elif affinity_mode == "avoid_edge":
                current_score = normalized_dist # Further from edge (larger dist) -> higher score
                if is_corner and min_dist == 0:
                    current_score -= corner_penalty_avoid # Penalty for being a corner when avoiding edges
            
            # Clamp score to [0, 1] range, considering bonus/penalty
            final_score_range_min = 0.0 - corner_penalty_avoid if affinity_mode == "avoid_edge" else 0.0
            final_score_range_max = 1.0 + corner_bonus_prefer if affinity_mode == "prefer_edge" else 1.0
            scores[r_idx, c_idx] = _math_utils.normalize_value(current_score, final_score_range_min, final_score_range_max, clamp=True)
    return scores

# Module 15: EXT_GM9_Center_Control_Vec
async def EXT_GM9_Center_Control_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM9") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM9_Center_Control_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    affinity_mode = "prefer_center" # "prefer_center" or "avoid_center"
    center_r, center_c = (rows - 1) / 2.0, (cols - 1) / 2.0

    # Max distance is from a corner to the center. Using (0,0) as ref corner.
    max_dist_to_center = _math_utils.euclidean_distance((0.0, 0.0), (center_r, center_c))
    if max_dist_to_center == 0: # Handles 1x1 grid or if rows/cols are such that center is 0,0 (e.g. 1x1)
        # For a 1x1 grid, cell (0,0) is the center, dist is 0. max_dist is 0. norm will be 0.5 by normalize_value.
        # If we want score 1 for prefer_center in 1x1 grid:
        if rows == 1 and cols == 1:
            if affinity_mode == "prefer_center": scores[0,0] = 1.0
            else: scores[0,0] = 0.0 # For avoid_center, score 0
            return scores
        max_dist_to_center = 1.0 # Avoid division by zero for other small grids if calc is 0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            
            current_dist_to_center = _math_utils.euclidean_distance((float(r_idx), float(c_idx)), (center_r, center_c))
            normalized_dist = _math_utils.normalize_value(current_dist_to_center, 0, max_dist_to_center, clamp=True)
            
            current_score = 0.0
            if affinity_mode == "prefer_center":
                current_score = 1.0 - normalized_dist
            elif affinity_mode == "avoid_center":
                current_score = normalized_dist
            
            scores[r_idx, c_idx] = _math_utils.normalize_value(current_score, 0, 1.0, clamp=True) # Final clamp
    return scores

# Module 16: EXT_GM10_Blocking_Value_Vec
async def EXT_GM10_Blocking_Value_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM10") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM10_Blocking_Value_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(_board_analyzer_utils.get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores # No legal moves, so no risk/blocking to evaluate

    # Define undesirable sequences (length 3 for this example from PDF)
    # These are patterns that placing p_val *completes*.
    UNDESIRABLE_SEQUENCES_PATTERNS = [ # Patterns of 3
        ([1, 1, 1], "Avoid three 1s"),
        ([2, 2, 2], "Avoid three 2s"),
        # ([1,2,3], "Avoid short ascending sequence if contextually bad") # Example from PDF
    ]

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            
            # Score for this cell: max safety score achievable by placing any potential number
            # High score means placement AVOIDS completing undesirable patterns.
            max_safety_score_for_cell = 0.0 # Default low if all placements are bad or no options.
                                            # If potential_numbers_to_place is not empty, it should find some score.
            
            # If no potential numbers, this loop won't run, scores remain 0. Handled by early exit.

            for p_val in potential_numbers_to_place:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val
                
                completes_undesirable_pattern = False
                # Check lines of length 3 passing through (r_idx, c_idx)
                # Directions: H, V, Main Diag, Anti-Diag
                for dr_line, dc_line in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    # Check 3 possible alignments for a 3-cell window where p_val is involved
                    for offset in range(-2, 1): # Window starts at (r_idx + offset*dr, c_idx + offset*dc)
                        current_line_values: list[int] = []
                        line_coords_check: list[tuple[int,int]] = []
                        valid_line_segment = True
                        
                        for i_in_segment in range(3): # Check 3 cells in this window
                            check_r = r_idx + (offset + i_in_segment) * dr_line
                            check_c = c_idx + (offset + i_in_segment) * dc_line
                            
                            if 0 <= check_r < rows and 0 <= check_c < cols:
                                line_coords_check.append((check_r,check_c))
                                current_line_values.append(int(temp_grid[check_r, check_c])) # Ensure int for comparison
                            else:
                                valid_line_segment = False
                                break
                        
                        if valid_line_segment and len(current_line_values) == 3:
                            # Ensure the p_val at (r_idx, c_idx) is part of this specific 3-cell line segment
                            if (r_idx, c_idx) not in line_coords_check: # Should be covered by offset logic
                                continue

                            for undesirable_seq_pattern, _ in UNDESIRABLE_SEQUENCES_PATTERNS:
                                if current_line_values == undesirable_seq_pattern:
                                    completes_undesirable_pattern = True
                                    break
                        if completes_undesirable_pattern: break
                    if completes_undesirable_pattern: break
                
                # If it forms an undesirable pattern, score is low (e.g. 0.1). If not, high (e.g. 0.9).
                current_score_for_pval = 0.1 if completes_undesirable_pattern else 0.9
                if current_score_for_pval > max_safety_score_for_cell:
                    max_safety_score_for_cell = current_score_for_pval
            
            scores[r_idx, c_idx] = max_safety_score_for_cell
    return scores

# Module 17: EXT_GM11_Pair_Correlation_Vec
async def EXT_GM11_Pair_Correlation_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM11") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM11_Pair_Correlation_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(_board_analyzer_utils.get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores

    max_val_board_for_pair = _board_analyzer_utils.get_card_max_value_from_grid_dimensions((rows,cols))
    # Ensure max_val_board_for_pair is at least 1 to avoid issues with // 2 if it's 0
    if max_val_board_for_pair == 0 : max_val_board_for_pair = 1 


    FAVORABLE_PAIRS_SCORES: Dict[tuple[int,int], float] = { # (val_placed, existing_neighbor_val)
        (3, 7): 0.8, (7, 3): 0.8,
        (1, 2): 0.6, (2, 1): 0.6,
        (10, 20): 0.7, (20, 10): 0.7,
        (5, 10): 0.5, (10, 5): 0.5,
        # PDF example: (max(1, max_val // 2), max(1, max_val // 2) + 1): 0.4
        (max(1, max_val_board_for_pair // 2), max(1, max_val_board_for_pair // 2) + 1): 0.4,
        (max(1, max_val_board_for_pair // 2) + 1, max(1, max_val_board_for_pair // 2)): 0.4,
    }
    
    max_single_pair_score = 0.0
    if FAVORABLE_PAIRS_SCORES:
        max_single_pair_score = max(FAVORABLE_PAIRS_SCORES.values()) if FAVORABLE_PAIRS_SCORES else 0.0
    
    # Heuristic max: if all 8 neighbors form max-scoring pairs
    heuristic_max_total_pair_score = 8.0 * max_single_pair_score if max_single_pair_score > 0 else 1.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_accumulated_score_for_cell = 0.0
            for p_val in potential_numbers_to_place:
                current_pval_accumulated_score = 0.0
                # Check 8 neighbors
                for dr in [-1, 0, 1]:
                    for dc_val in [-1, 0, 1]: # Renamed dc
                        if dr == 0 and dc_val == 0: continue
                        nr, nc_val_neigh = r_idx + dr, c_idx + dc_val # Renamed nc
                        if 0 <= nr < rows and 0 <= nc_val_neigh < cols:
                            neighbor_val = grid[nr, nc_val_neigh]
                            if neighbor_val != -1: # If neighbor is an existing number
                                if (p_val, int(neighbor_val)) in FAVORABLE_PAIRS_SCORES:
                                    current_pval_accumulated_score += FAVORABLE_PAIRS_SCORES[(p_val, int(neighbor_val))]
                if current_pval_accumulated_score > max_accumulated_score_for_cell:
                    max_accumulated_score_for_cell = current_pval_accumulated_score
            
            scores[r_idx, c_idx] = _math_utils.normalize_value(max_accumulated_score_for_cell, 0, heuristic_max_total_pair_score, clamp=True)
    return scores

# Module 18: EXT_GM12_Island_Analysis_Vec (已填數字島嶼)
async def EXT_GM12_Island_Analysis_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM12") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM12_Island_Analysis_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float) # Empty cells get 0 from this module
    if rows == 0 or cols == 0: return scores

    visited_island_search = np.zeros_like(grid, dtype=bool)
    max_val_on_board = _board_analyzer_utils.get_card_max_value_from_grid_dimensions((rows, cols))
    if max_val_on_board == 0: max_val_on_board = 1.0 # Avoid div by zero if grid is 1x1 with 0 or empty

    w_size, w_compactness, w_avg_value = 0.4, 0.3, 0.3

    for r_start in range(rows):
        for c_start in range(cols):
            if visited_island_search[r_start, c_start]: continue # Already processed or marked as empty

            if grid[r_start, c_start] != -1 : # Found an unvisited number (start of a potential island)
                current_island_cells: list[tuple[int, int]] = []
                current_island_values: list[int] = []
                q = deque([(r_start, c_start)])
                visited_island_search[r_start, c_start] = True
                
                min_r_bbox, max_r_bbox = r_start, r_start
                min_c_bbox, max_c_bbox = c_start, c_start

                while q:
                    r_curr, c_curr = q.popleft()
                    current_island_cells.append((r_curr, c_curr))
                    current_island_values.append(int(grid[r_curr, c_curr]))
                    min_r_bbox, max_r_bbox = min(min_r_bbox, r_curr), max(max_r_bbox, r_curr)
                    min_c_bbox, max_c_bbox = min(min_c_bbox, c_curr), max(max_c_bbox, c_curr)

                    for dr, dc_val in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # 4-connectivity
                        nr, nc_val_neigh = r_curr + dr, c_curr + dc_val
                        if 0 <= nr < rows and 0 <= nc_val_neigh < cols and \
                           grid[nr, nc_val_neigh] != -1 and not visited_island_search[nr, nc_val_neigh]:
                            visited_island_search[nr, nc_val_neigh] = True
                            q.append((nr, nc_val_neigh))
                
                island_size = float(len(current_island_cells))
                avg_value = 0.0
                if island_size > 0 : # Avoid division by zero for empty island_values
                     avg_value = sum(current_island_values) / island_size if current_island_values else 0.0

                bbox_height = float(max_r_bbox - min_r_bbox + 1)
                bbox_width = float(max_c_bbox - min_c_bbox + 1)
                bbox_area = bbox_height * bbox_width
                compactness = 0.0
                if bbox_area > 0: # Avoid division by zero
                    compactness = island_size / bbox_area
                
                norm_size = _math_utils.normalize_value(island_size, 1, float(rows * cols), clamp=True) # min size is 1
                norm_compactness = _math_utils.normalize_value(compactness, 0, 1.0, clamp=True) # Already 0-1
                norm_avg_value = _math_utils.normalize_value(avg_value, 1, max_val_on_board, clamp=True) # Assume min value is 1

                island_score_val = (w_size * norm_size +
                                 w_compactness * norm_compactness +
                                 w_avg_value * norm_avg_value)
                final_island_score = _math_utils.normalize_value(island_score_val, 0, 1.0, clamp=True)

                for r_cell, c_cell in current_island_cells:
                    scores[r_cell, c_cell] = final_island_score
            
            elif grid[r_start, c_start] == -1: # Mark empty cells as visited (score remains 0)
                 visited_island_search[r_start, c_start] = True
                 # scores[r_start, c_start] is already 0.0
    return scores

# Module 19: EXT_GM13_Sequence_Diversity_Vec
async def EXT_GM13_Sequence_Diversity_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM13") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM13_Sequence_Diversity_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(_board_analyzer_utils.get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores

    short_sequence_len = 3
    # Heuristic from PDF: 8.0. Max distinct types of sequences a cell can start/mend/end
    # H, V, D1, D2 = 4 directions. For each, arithmetic or identical.
    # A cell can be start, middle, or end of a 3-length sequence.
    # This heuristic seems reasonable.
    heuristic_max_distinct_sequences = 8.0 

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_diversity_count_for_cell = 0
            for p_val in potential_numbers_to_place:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val
                found_sequence_signatures: set[tuple[str, tuple[int,int], int | float]] = set() # (type, direction_vector, characteristic_val)

                # Check in 4 directions (H, V, D1, D2)
                for dr_dir, dc_dir in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    # For each direction, check short_sequence_len possible alignments
                    for i_offset in range(short_sequence_len): # p_val is at index i_offset in the window
                        current_sequence_values_segment: list[int] = [] #
                        valid_segment = True
                        segment_coords: list[tuple[int,int]] = []

                        for k_seq in range(short_sequence_len):
                            check_r = r_idx + (k_seq - i_offset) * dr_dir
                            check_c = c_idx + (k_seq - i_offset) * dc_dir
                            segment_coords.append((check_r, check_c))
                            if 0 <= check_r < rows and 0 <= check_c < cols:
                                current_sequence_values_segment.append(int(temp_grid[check_r, check_c]))
                            else:
                                valid_segment = False
                                break
                        
                        if valid_segment and len(current_sequence_values_segment) == short_sequence_len:
                            # Ensure p_val (at r_idx, c_idx) is part of this current segment
                            if (r_idx, c_idx) not in segment_coords:
                                continue

                            s = current_sequence_values_segment
                            if all(val != -1 for val in s): # Ensure all are numbers
                                # Arithmetic (non-constant)
                                diff1, diff2 = s[1] - s[0], s[2] - s[1]
                                if diff1 == diff2 and diff1 != 0:
                                    found_sequence_signatures.add(("arithmetic", (dr_dir, dc_dir), diff1))
                                # Identical
                                if s[0] == s[1] and s[1] == s[2] and s[0]!=-1 : # Check s[0]!=-1 for identical
                                    found_sequence_signatures.add(("identical", (dr_dir, dc_dir), s[0]))
                
                current_pval_diversity_count = len(found_sequence_signatures)
                if current_pval_diversity_count > max_diversity_count_for_cell:
                    max_diversity_count_for_cell = current_pval_diversity_count
            
            scores[r_idx, c_idx] = _math_utils.normalize_value(float(max_diversity_count_for_cell), 0, heuristic_max_distinct_sequences, clamp=True)
    return scores
# main.py - Part 3 of 3 (Connects after Part 2)

# (Continuing Brain Logic: Scoring Module Implementations from Part 2)

# Module 20: EXT_GM14_Risk_Assessment_Vec
async def EXT_GM14_Risk_Assessment_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM14") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM14_Risk_Assessment_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    initial_potential_numbers = list(_board_analyzer_utils.get_legal_values_for_placement(grid))
    if not initial_potential_numbers:
        # If no initial moves, risk is max (score 0) or undefined.
        # Let's return 0 scores, as no placement is possible to assess.
        return scores

    # Max possible subsequent legal moves is roughly (rows*cols - 1) after 1 placement
    # or (rows*cols) if we consider current legal moves before any placement
    # The PDF normalizes by (rows*cols - 1) for subsequent moves.
    max_heuristic_flex = float(rows * cols -1) # After placing one number
    if max_heuristic_flex <=0 : max_heuristic_flex = 1.0 # Avoid div by zero for 1x1 or empty grid resulting in 0


    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            
            max_flexibility_score_for_cell = 0.0 # This actually tracks max number of subsequent moves

            # Check if (r_idx, c_idx) is a valid placement cell for *any* initial number
            # This loop iterates through values that *could* be placed in (r_idx, c_idx)
            # However, the PDF seems to imply that p_val here is from *initial_potential_numbers* for the *original* grid.
            # This means p_val is a candidate for *any* empty cell, not specifically (r_idx, c_idx).
            # The logic is: IF p_val is placed at (r_idx, c_idx), what's the flexibility?
            # This is correct.

            if not initial_potential_numbers : # If cell (r_idx,c_idx) makes no initial numbers placeable there, this loop is skipped.
                                              # This check is redundant due to outer check.
                scores[r_idx, c_idx] = 0.0 # Or some neutral value
                continue

            # Test placing each initially possible number into (r_idx, c_idx)
            # The PDF used initial_potential_numbers (legal values for original grid).
            # It should be: for each p_val that can be placed in (r_idx, c_idx) from initial_potential_numbers...
            # No, it's simpler: IF (r_idx,c_idx) is empty, try placing each of the game's general potential numbers there.
            # The current `initial_potential_numbers` is correct.
            
            for p_val_to_try_at_rc in initial_potential_numbers:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val_to_try_at_rc # Place this specific p_val here

                subsequent_legal_moves_count = len(_board_analyzer_utils.get_legal_values_for_placement(temp_grid))
                current_flexibility = float(subsequent_legal_moves_count)

                if current_flexibility > max_flexibility_score_for_cell:
                    max_flexibility_score_for_cell = current_flexibility
            
            scores[r_idx, c_idx] = _math_utils.normalize_value(max_flexibility_score_for_cell, 0, max_heuristic_flex, clamp=True)
    return scores

# Module 21: EXT_GM15_Information_Gain_Vec
async def EXT_GM15_Information_Gain_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM15") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM15_Information_Gain_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(_board_analyzer_utils.get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores

    initial_grid_values = [int(val) for val in grid.flatten()] # Includes -1
    entropy_before = _math_utils.get_entropy(initial_grid_values)

    # Max possible entropy change (reduction) is entropy_before. Or log2(num_symbols).
    # PDF uses log2(num_symbols) where num_symbols = R*C + 1 (for -1).
    num_symbols = rows * cols + 1 
    max_possible_entropy_change = math.log2(num_symbols) if num_symbols > 1 else 1.0
    if max_possible_entropy_change == 0: max_possible_entropy_change = 1.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            
            max_entropy_reduction_for_cell = -float('inf')
            for p_val in potential_numbers_to_place:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val
                temp_grid_values = [int(val) for val in temp_grid.flatten()]
                entropy_after = _math_utils.get_entropy(temp_grid_values)
                entropy_reduction = entropy_before - entropy_after # Higher is better
                if entropy_reduction > max_entropy_reduction_for_cell:
                    max_entropy_reduction_for_cell = entropy_reduction
            
            if max_entropy_reduction_for_cell == -float('inf'): # Should not happen if p_nums not empty
                max_entropy_reduction_for_cell = 0.0
            
            # Normalize: min reduction can be negative. Max is entropy_before.
            # PDF normalizes in range [0, max_possible_entropy_change] clamping negative reductions to 0.
            scores[r_idx, c_idx] = _math_utils.normalize_value(max_entropy_reduction_for_cell, 0, max_possible_entropy_change, clamp=True)
    return scores

# Module 22: EXT_GM16_Harmonic_Centrality_Vec
async def EXT_GM16_Harmonic_Centrality_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM16") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM16_Harmonic_Centrality_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0 or (rows * cols) <= 1: return scores # Needs > 1 cell

    # Max HC: sum of 1/1 for N-1 nodes = N-1
    max_hc_heuristic = float(rows * cols - 1)
    if max_hc_heuristic == 0: max_hc_heuristic = 1.0 # Avoid div by zero if only 1 cell (though caught above)

    for r_eval in range(rows):
        for c_eval in range(cols):
            if grid[r_eval, c_eval] != -1: continue # Only score empty cells

            current_harmonic_centrality = 0.0
            num_other_nodes = 0
            for r_other in range(rows):
                for c_other in range(cols):
                    if r_eval == r_other and c_eval == c_other: continue
                    
                    # PDF doesn't specify if other nodes must be empty or filled.
                    # Harmonic centrality usually considers all other nodes in the graph.
                    # Here, graph nodes are all cells.
                    dist = _math_utils.manhattan_distance((r_eval, c_eval), (r_other, c_other))
                    if dist > 0: # Should always be > 0 if not self
                        current_harmonic_centrality += 1.0 / dist
                    num_other_nodes +=1 # Count all other cells as nodes
            
            if num_other_nodes == 0: # Should not happen if (rows*cols) > 1
                scores[r_eval, c_eval] = 0.0
            else:
                scores[r_eval, c_eval] = _math_utils.normalize_value(current_harmonic_centrality, 0, max_hc_heuristic, clamp=True)
    return scores

# Module 23: EXT_GM17_Entropy_Minimization_Vec (Local Entropy)
async def EXT_GM17_Entropy_Minimization_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM17") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM17_Entropy_Minimization_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(_board_analyzer_utils.get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores

    radius_local = 1 # Local neighborhood radius
    num_cells_in_neighborhood = (2 * radius_local + 1)**2 # Including center
    # Max local entropy change: log2 of number of cells in neighborhood (if each cell is a unique symbol)
    # Or, if symbols are numbers + -1, then log2(num_possible_values_in_neighborhood)
    # PDF uses log2(num_cells_in_neighborhood), simpler.
    max_local_entropy_change = math.log2(num_cells_in_neighborhood) if num_cells_in_neighborhood > 1 else 1.0
    if max_local_entropy_change == 0: max_local_entropy_change = 1.0

    def val_func_for_local_entropy(x_val: int) -> int: return int(x_val) # Includes -1 as a symbol

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            
            # Entropy of local neighborhood *before* placement (with (r_idx, c_idx) as -1)
            # The get_neighborhood_values will use the original grid where (r,c) is -1.
            values_before_placement_local = _board_analyzer_utils.get_neighborhood_values(
                grid, r_idx, c_idx, radius=radius_local,
                val_func=val_func_for_local_entropy, include_center=True
            )
            entropy_before_local = _math_utils.get_entropy(values_before_placement_local)
            
            max_entropy_reduction_for_cell = -float('inf')
            for p_val in potential_numbers_to_place:
                temp_grid_local_place = grid.copy()
                temp_grid_local_place[r_idx, c_idx] = p_val
                
                values_after_placement_local = _board_analyzer_utils.get_neighborhood_values(
                    temp_grid_local_place, r_idx, c_idx, radius=radius_local,
                    val_func=val_func_for_local_entropy, include_center=True
                )
                entropy_after_local = _math_utils.get_entropy(values_after_placement_local)
                entropy_reduction = entropy_before_local - entropy_after_local
                if entropy_reduction > max_entropy_reduction_for_cell:
                    max_entropy_reduction_for_cell = entropy_reduction
            
            if max_entropy_reduction_for_cell == -float('inf'):
                max_entropy_reduction_for_cell = 0.0
            
            scores[r_idx, c_idx] = _math_utils.normalize_value(max_entropy_reduction_for_cell, 0, max_local_entropy_change, clamp=True)
    return scores

# Module 24: EXT_GM18_RL_Value_Est_Vec
async def EXT_GM18_RL_Value_Est_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM18") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM18_RL_Value_Est_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    potential_numbers_to_place = list(_board_analyzer_utils.get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores

    FEATURE_WEIGHTS = {
        "identical_3": 1.0, "arithmetic_3": 0.7,
        "board_density_factor": 0.2,
        "central_control_boost": 0.15, # Increased from PDF
        "edge_affinity_boost": 0.05
    }
    # Max heuristic: Sum of max possible scores from each feature
    # Line features: 4 directions * (ident_3 + arith_3), density: max 1, central/edge: max 1 each
    max_heuristic_feature_score = (4 * (FEATURE_WEIGHTS["identical_3"] + FEATURE_WEIGHTS["arithmetic_3"])) + \
                                  FEATURE_WEIGHTS["board_density_factor"] + \
                                  FEATURE_WEIGHTS["central_control_boost"] + \
                                  FEATURE_WEIGHTS["edge_affinity_boost"]
    if max_heuristic_feature_score == 0: max_heuristic_feature_score = 1.0


    center_r_rl, center_c_rl = (rows - 1) / 2.0, (cols - 1) / 2.0
    max_dist_to_center_rl = _math_utils.euclidean_distance((0.0,0.0), (center_r_rl, center_c_rl)) if not (rows==1 and cols==1) else 1.0
    if max_dist_to_center_rl == 0 : max_dist_to_center_rl = 1.0 # Avoid div by zero

    max_min_dist_to_edge_rl = float(min((rows - 1) // 2, (cols - 1) // 2))
    if max_min_dist_to_edge_rl == 0 : max_min_dist_to_edge_rl = 1.0 # Avoid div by zero for normalization

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_feature_score_for_cell = 0.0
            for p_val in potential_numbers_to_place:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val
                current_features_score = 0.0

                # Feature 1 & 2: Lines of 3
                for dr_line, dc_line in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    for offset in range(-2, 1):
                        line_vals: list[int] = []
                        is_valid_line = True
                        involved_pval = False
                        line_coords_temp: list[tuple[int,int]] = []
                        for i in range(3):
                            check_r, check_c = r_idx + (offset + i) * dr_line, c_idx + (offset + i) * dc_line
                            line_coords_temp.append((check_r,check_c))
                            if 0 <= check_r < rows and 0 <= check_c < cols:
                                line_vals.append(int(temp_grid[check_r, check_c]))
                            else:
                                is_valid_line = False; break
                        if (r_idx,c_idx) in line_coords_temp: involved_pval = True

                        if is_valid_line and involved_pval and len(line_vals) == 3 and all(v != -1 for v in line_vals):
                            s_line = line_vals
                            if s_line[0] == s_line[1] and s_line[1] == s_line[2]:
                                current_features_score += FEATURE_WEIGHTS["identical_3"]
                            elif (s_line[1] - s_line[0]) == (s_line[2] - s_line[1]) and (s_line[1] - s_line[0]) != 0:
                                current_features_score += FEATURE_WEIGHTS["arithmetic_3"]
                
                # Feature 3: Board density
                num_filled = np.count_nonzero(temp_grid != -1)
                density = num_filled / (rows * cols) if (rows * cols) > 0 else 0.0
                current_features_score += FEATURE_WEIGHTS["board_density_factor"] * density

                # Feature 4: Central control boost (if grid > 1x1)
                if rows > 1 or cols > 1: # Avoid for 1x1 or smaller
                    dist_to_center = _math_utils.euclidean_distance((float(r_idx), float(c_idx)), (center_r_rl, center_c_rl))
                    # score is higher if closer to center (1 - normalized_dist)
                    current_features_score += FEATURE_WEIGHTS["central_control_boost"] * \
                        (1.0 - _math_utils.normalize_value(dist_to_center, 0, max_dist_to_center_rl, clamp=True))
                
                # Feature 5: Edge affinity boost (if grid > 1x1)
                if rows > 1 or cols > 1: # Avoid for 1x1 or smaller
                    dist_to_edge = float(min(r_idx, rows - 1 - r_idx, c_idx, cols - 1 - c_idx))
                     # score is higher if closer to edge (1 - normalized_dist_from_edge)
                    current_features_score += FEATURE_WEIGHTS["edge_affinity_boost"] * \
                        (1.0 - _math_utils.normalize_value(dist_to_edge, 0, max_min_dist_to_edge_rl, clamp=True))

                if current_features_score > max_feature_score_for_cell:
                    max_feature_score_for_cell = current_features_score
            
            scores[r_idx, c_idx] = _math_utils.normalize_value(max_feature_score_for_cell, 0, max_heuristic_feature_score, clamp=True)
    return scores

# Module 25: EXT_GM19_Masked_Number_Skip_Pattern_Vec
async def EXT_GM19_Masked_Number_Skip_Pattern_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM19") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM19_Masked_Number_Skip_Pattern_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    revealed_numbers_info: list[dict[str, str | int | float | bool | list | dict]] = [
        {'value': int(grid[r, c]), 'r': r, 'c': c}
        for r in range(rows) for c in range(cols)
        if grid[r, c] != -1 and grid[r, c] > 0 # Assuming positive numbers
    ]
    if not revealed_numbers_info: return scores

    expected_max_number_on_card = _board_analyzer_utils.get_card_max_value_from_grid_dimensions((rows, cols))
    base_positions: dict[int, tuple[int, int]] = {} # value -> (expected_r, expected_c)
    for k_val in range(1, expected_max_number_on_card + 1):
        base_r = (k_val - 1) // cols
        base_c = (k_val - 1) % cols
        if base_r < rows: # Ensure base position is within grid dimensions
            base_positions[k_val] = (base_r, base_c)

    skip_vectors: dict[int, tuple[int, int]] = {} # value -> (delta_r, delta_c)
    for rn_info in revealed_numbers_info:
        val = rn_info['value']
        if val in base_positions:
            expected_r, expected_c = base_positions[val]
            skip_vectors[val] = (rn_info['r'] - expected_r, rn_info['c'] - expected_c)
    if not skip_vectors: return scores

    dominant_skip_patterns_strength: dict[tuple[int, int], float] = {} # (dr,dc) -> strength
    skip_vector_tuples_list = list(skip_vectors.values())
    if not skip_vector_tuples_list: return scores # Should be caught by `if not skip_vectors`

    counts = Counter(skip_vector_tuples_list)
    # PDF: min_occurrences = max(1, int(len_list * 0.05)) Adjusted from 2
    min_occurrences_for_pattern = max(1, int(len(skip_vector_tuples_list) * 0.05)) 

    for skip_vec_tuple, count_val in counts.most_common():
        if count_val >= min_occurrences_for_pattern:
            pattern_strength = _math_utils.normalize_value(float(count_val), float(min_occurrences_for_pattern), float(len(skip_vector_tuples_list)), clamp=True)
            dominant_skip_patterns_strength[skip_vec_tuple] = pattern_strength
        else: break # most_common is sorted
    if not dominant_skip_patterns_strength: return scores

    potential_numbers_to_place_set = _board_analyzer_utils.get_legal_values_for_placement(grid)
    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            cell_max_pattern_score = 0.0
            for p_val_test in potential_numbers_to_place_set:
                if p_val_test not in base_positions: continue
                base_r_test, base_c_test = base_positions[p_val_test]
                for current_skip_pattern, pattern_str_val in dominant_skip_patterns_strength.items(): # Renamed pattern_str
                    skip_dr, skip_dc = current_skip_pattern
                    predicted_r, predicted_c = base_r_test + skip_dr, base_c_test + skip_dc
                    if predicted_r == r_idx and predicted_c == c_idx:
                        current_score_fit = pattern_str_val
                        if current_score_fit > cell_max_pattern_score:
                            cell_max_pattern_score = current_score_fit
            scores[r_idx, c_idx] = cell_max_pattern_score
    return scores

# Module 26: EXT_GM20_Skip_Pattern_Confidence_Vec
async def EXT_GM20_Skip_Pattern_Confidence_Vec(grid: np.ndarray, request_id: str | None = "N/A_brain_GM20") -> np.ndarray:
    brain_logger.debug("Executing EXT_GM20_Skip_Pattern_Confidence_Vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    # --- Initial Pattern Analysis (simplified from GM19 logic) ---
    revealed_numbers_info_gm20: list[dict[str,str | int | float | bool | list | dict]] = []
    for r in range(rows):
        for c in range(cols):
            if grid[r,c] != -1 and grid[r,c] > 0:
                revealed_numbers_info_gm20.append({'value': int(grid[r,c]), 'r': r, 'c': c})
    if not revealed_numbers_info_gm20: return scores

    expected_max_num_gm20 = _board_analyzer_utils.get_card_max_value_from_grid_dimensions((rows,cols))
    base_pos_gm20: dict[int, tuple[int,int]] = {
        k: ((k-1)//cols, (k-1)%cols) for k in range(1, expected_max_num_gm20 + 1)
        if ((k-1)//cols) < rows # Ensure base_r is within grid
    }
    skip_vecs_initial_gm20: dict[int, tuple[int,int]] = {}
    for rn in revealed_numbers_info_gm20:
        val = rn['value']
        if val in base_pos_gm20:
            base_r_val, base_c_val = base_pos_gm20[val] # Renamed base_r, base_c
            skip_vecs_initial_gm20[val] = (rn['r'] - base_r_val, rn['c'] - base_c_val)
    
    dominant_patterns_details_gm20: list[dict[str,str | int | float | bool | list | dict]] = [] # List of {'skip':(dr,dc), 'values':[sorted_values], 'strength':float}
    if skip_vecs_initial_gm20:
        skip_tuples_list_gm20 = list(skip_vecs_initial_gm20.values())
        if not skip_tuples_list_gm20 : return scores # No skip vectors found
        counts_gm20 = Counter(skip_tuples_list_gm20)
        min_occ_gm20 = max(1, int(len(skip_tuples_list_gm20) * 0.05))
        for skip_v, count_v_val in counts_gm20.most_common(): # Renamed count_v
            if count_v_val >= min_occ_gm20:
                pattern_vals = sorted([val for val, sv_tuple in skip_vecs_initial_gm20.items() if sv_tuple == skip_v])
                p_strength = _math_utils.normalize_value(float(count_v_val), float(min_occ_gm20), float(len(skip_tuples_list_gm20)), clamp=True)
                dominant_patterns_details_gm20.append({'skip': skip_v, 'values': pattern_vals, 'strength': p_strength})
            else: break
    if not dominant_patterns_details_gm20: return scores
    # --- End Initial Pattern Analysis ---

    potential_nums_to_place_gm20 = _board_analyzer_utils.get_legal_values_for_placement(grid)
    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue
            max_confidence_score_for_cell_gm20 = 0.0
            for p_val_test in potential_nums_to_place_gm20:
                if p_val_test not in base_pos_gm20: continue
                base_r_t, base_c_t = base_pos_gm20[p_val_test]
                current_max_conf_for_pval = 0.0
                for pattern_detail in dominant_patterns_details_gm20:
                    pat_skip_dr, pat_skip_dc = pattern_detail['skip']
                    pat_existing_vals = pattern_detail['values'] # sorted list
                    pat_strength = pattern_detail['strength']

                    predicted_r_for_pval, predicted_c_for_pval = base_r_t + pat_skip_dr, base_c_t + pat_skip_dc
                    if predicted_r_for_pval == r_idx and predicted_c_for_pval == c_idx: # Geometrically fits
                        enhancement_factor = 0.5 # Base for geometric fit
                        if len(pat_existing_vals) >= 1: # Need at least one existing number to extend/mend
                            temp_sequence_with_pval = sorted(pat_existing_vals + [p_val_test])
                            if len(temp_sequence_with_pval) >= 2:
                                diffs_in_temp_seq = np.diff(temp_sequence_with_pval)
                                if len(diffs_in_temp_seq) > 0:
                                    is_arithmetic_now = len(set(diffs_in_temp_seq)) == 1 # All diffs same
                                    first_diff = diffs_in_temp_seq[0]
                                    if is_arithmetic_now and first_diff != 0: # Forms consistent non-constant arithmetic
                                        enhancement_factor += 0.4
                                        # Bonus for filling internal gap
                                        if min(pat_existing_vals) < p_val_test < max(pat_existing_vals):
                                            enhancement_factor += 0.1 
                        current_conf = pat_strength * enhancement_factor
                        if current_conf > current_max_conf_for_pval:
                            current_max_conf_for_pval = current_conf
                if current_max_conf_for_pval > max_confidence_score_for_cell_gm20:
                    max_confidence_score_for_cell_gm20 = current_max_conf_for_pval
            scores[r_idx, c_idx] = _math_utils.normalize_value(max_confidence_score_for_cell_gm20, 0, 1.0, clamp=True) # Max enh can be ~0.5+0.4+0.1 = 1.0
    return scores

# --- Brain Core Dispatch Area ---
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, str | None], Coroutine[str | int | float | bool | list | dict, str | int | float | bool | list | dict, np.ndarray]]] = {
    "EXT_A2_Weighted_Proximity_Vec": EXT_A2_Weighted_Proximity_Vec,
    "EXT_M3_Local_Heterogeneity_Vec": EXT_M3_Local_Heterogeneity_Vec,
    "EXT_D3_Potential_Field_Vec": EXT_D3_Potential_Field_Vec,
    "EXT_F10_Discontinuity_Vec": EXT_F10_Discontinuity_Vec,
    "EXT_P7_Pathfinding_Value_Vec": EXT_P7_Pathfinding_Value_Vec,
    "EXT_R5_Resource_Control_Vec": EXT_R5_Resource_Control_Vec,
    "EXT_GM1_Row_Control_Vec": EXT_GM1_Row_Control_Vec,
    "EXT_GM2_Col_Flow_Vec": EXT_GM2_Col_Flow_Vec,
    "EXT_GM3_Adv_Connected_Comp_Vec": EXT_GM3_Adv_Connected_Comp_Vec,
    "EXT_GM4_Spatial_Auto_Corr_Vec": EXT_GM4_Spatial_Auto_Corr_Vec,
    "EXT_GM5_Line_Completion_Vec": EXT_GM5_Line_Completion_Vec,
    "EXT_GM6_Symmetry_Potential_Vec": EXT_GM6_Symmetry_Potential_Vec,
    "EXT_GM7_Numeric_Gaps_Vec": EXT_GM7_Numeric_Gaps_Vec,
    "EXT_GM8_Edge_Affinity_Vec": EXT_GM8_Edge_Affinity_Vec,
    "EXT_GM9_Center_Control_Vec": EXT_GM9_Center_Control_Vec,
    "EXT_GM10_Blocking_Value_Vec": EXT_GM10_Blocking_Value_Vec,
    "EXT_GM11_Pair_Correlation_Vec": EXT_GM11_Pair_Correlation_Vec,
    "EXT_GM12_Island_Analysis_Vec": EXT_GM12_Island_Analysis_Vec,
    "EXT_GM13_Sequence_Diversity_Vec": EXT_GM13_Sequence_Diversity_Vec,
    "EXT_GM14_Risk_Assessment_Vec": EXT_GM14_Risk_Assessment_Vec,
    "EXT_GM15_Information_Gain_Vec": EXT_GM15_Information_Gain_Vec,
    "EXT_GM16_Harmonic_Centrality_Vec": EXT_GM16_Harmonic_Centrality_Vec,
    "EXT_GM17_Entropy_Minimization_Vec": EXT_GM17_Entropy_Minimization_Vec,
    "EXT_GM18_RL_Value_Est_Vec": EXT_GM18_RL_Value_Est_Vec,
    "EXT_GM19_Masked_Number_Skip_Pattern_Vec": EXT_GM19_Masked_Number_Skip_Pattern_Vec,
    "EXT_GM20_Skip_Pattern_Confidence_Vec": EXT_GM20_Skip_Pattern_Confidence_Vec,
}

async def get_module_score(module_name: str, grid: np.ndarray, pv_value_unused: int | None = None, request_id: str | None = "N/A_brain_dispatch") -> np.ndarray:
    if module_name not in REGISTERED_MODULES_BRAIN:
        brain_logger.error(f"Module {module_name} not found.", extra={'request_id': request_id})
        rows, cols = grid.shape
        return np.zeros((rows, cols), dtype=float)
    module_func = REGISTERED_MODULES_BRAIN[module_name]
    brain_logger.info(f"Executing brain module: {module_name} (PV {pv_value_unused} not directly used by module signature)", extra={'request_id': request_id})
    try:
        score_grid = await module_func(grid, request_id=request_id)
        if isinstance(score_grid, np.ndarray) and score_grid.shape == grid.shape:
            score_grid[grid != -1] = 0.0 # Ensure non-empty cells have zero score from placement modules
        else:
            brain_logger.warning(f"Module {module_name} returned bad score_grid. Shape: {getattr(score_grid, 'shape', 'N/A')}", extra={'request_id': request_id})
            rows,cols = grid.shape
            return np.zeros((rows,cols), dtype=float)
        return score_grid
    except Exception as e:
        brain_logger.error(f"Error in module {module_name}: {e}", exc_info=True, extra={'request_id': request_id})
        rows, cols = grid.shape
        return np.zeros((rows, cols), dtype=float)

def get_module_details(module_name: str) -> dict[str, str | int | float | bool | list | dict]:
    # Basic descriptions, can be expanded
    descriptions = {name: f"Scoring module: {name}" for name in REGISTERED_MODULES_BRAIN}
    return {
        "description": descriptions.get(module_name, "N/A"),
        "version": "2.0.0", # Generic version for all fully implemented modules
        "input_constraints": {"requires_empty_cells_for_score": True}
    }

class BrainInterface:
    def __init__(self):
        self.registered_modules = REGISTERED_MODULES_BRAIN
        self.get_module_score = get_module_score
        self.get_module_details = get_module_details

brain_interface = BrainInterface()

# --- Analyzer Logic (from An.pdf, enhanced for 2025) ---
# Custom Exceptions for Analyzer
class AnalyzerError(Exception): pass
class InitializationError(AnalyzerError): pass
class InvalidInputError(AnalyzerError): pass
class ModuleError(AnalyzerError): pass
class ModuleNotFoundError(ModuleError): pass
class ModuleExecutionError(ModuleError): pass
class VisualizationError(AnalyzerError): pass

class Analyzer:
    PV_COLORS = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    _current_cell_size_inch_for_dpi: float

    def __init__(self, main_module: BrainInterface, default_top_n: int = 3):
        if not hasattr(main_module, 'get_module_score') or not callable(main_module.get_module_score):
            raise InitializationError("main_module missing 'get_module_score'")
        if not hasattr(main_module, 'registered_modules') or not isinstance(main_module.registered_modules, dict):
            raise InitializationError("main_module missing 'registered_modules'")
        self.main_module = main_module
        self.default_top_n = default_top_n
        analyzer_logger.info(
            f"Analyzer initialized. default_top_n={default_top_n}. Modules: {len(main_module.registered_modules)}"
        )

    def _validate_inputs(
        self, new_card: list[list[int]], proposed_values_input: list['ProposedValue'],
        active_modules: list[str] | None, module_weights: dict[str, float] | None,
        top_n: int | None, request_id: str | None = "N/A"
    ) -> tuple[int, int, list['ProposedValue'], list[str] | None, dict[str, float] | None, int]:
        log_extra = {'request_id': request_id}
        if not new_card or not isinstance(new_card, list):
            raise InvalidInputError("Board (new_card) must be a non-empty list.")
        if not all(isinstance(row, list) for row in new_card):
            raise InvalidInputError("Each row in new_card must be a list.")
        
        rows = len(new_card)
        cols = len(new_card[0]) if rows > 0 and new_card[0] is not None else 0
        if rows > 0 and not all(len(row) == cols for row in new_card):
            raise InvalidInputError("Board must be rectangular.")
        if not all(isinstance(val, int) for row in new_card for val in row):
            raise InvalidInputError("All board values must be integers.")
        if not proposed_values_input or not isinstance(proposed_values_input, list):
            raise InvalidInputError("Proposed_values must be a non-empty list of ProposedValue objects.")
        if active_modules is not None and (not isinstance(active_modules, list) or not all(isinstance(m, str) for m in active_modules)):
            raise InvalidInputError("Active_modules must be a list of strings.")
        if module_weights is not None and (not isinstance(module_weights, dict) or not all(isinstance(k, str) and isinstance(v, (float, int)) for k,v in module_weights.items())):
            raise InvalidInputError("Module_weights must be a dict[str, float/int].")

        final_top_n = top_n if top_n is not None else self.default_top_n
        if not isinstance(final_top_n, int) or final_top_n <= 0:
            raise InvalidInputError(f"Top-N ({final_top_n}) must be a positive integer.")
        analyzer_logger.debug(f"Inputs validated. Board: {rows}x{cols}, Top_N: {final_top_n}", extra=log_extra)
        return rows, cols, proposed_values_input, active_modules, module_weights, final_top_n

    def _get_effective_modules_and_weights(
        self, requested_active_modules: list[str] | None,
        requested_module_weights: dict[str, float] | None, request_id: str | None = "N/A"
    ) -> tuple[list[str], dict[str, float]]:
        log_extra = {'request_id': request_id}
        registered = list(self.main_module.registered_modules.keys())
        effective_modules: list[str]
        if requested_active_modules is None:
            effective_modules = registered
            analyzer_logger.info("Using all registered modules.", extra=log_extra)
        else:
            effective_modules = [m for m in requested_active_modules if m in registered]
            ignored = [m for m in requested_active_modules if m not in registered]
            if ignored: analyzer_logger.warning(f"Ignored unregistered modules: {ignored}", extra=log_extra)
        if not effective_modules: analyzer_logger.warning("No effective modules selected.", extra=log_extra)

        final_weights = {name: 1.0 for name in effective_modules}
        if requested_module_weights:
            for name, weight in requested_module_weights.items():
                if name in final_weights: final_weights[name] = float(weight)
                else: analyzer_logger.warning(f"Weight for non-effective module '{name}' ignored.", extra=log_extra)
        analyzer_logger.info(f"Effective modules: {effective_modules}, Weights: {final_weights}", extra=log_extra)
        return effective_modules, final_weights

    def _fuse_scores(
        self, module_scores_map: dict[str, np.ndarray], module_weights_map: dict[str, float],
        rows: int, cols: int, request_id: str | None = "N/A"
    ) -> np.ndarray:
        log_extra = {'request_id': request_id}
        if rows == 0 or cols == 0: return np.array([[]]) if rows == 0 else np.empty((rows,0))
        fused = np.zeros((rows, cols), dtype=float)
        if not module_scores_map:
            analyzer_logger.warning("No module scores to fuse.", extra=log_extra)
            return fused
        
        for name, scores_arr in module_scores_map.items():
            weight = module_weights_map.get(name, 1.0)
            if not isinstance(scores_arr, np.ndarray) or scores_arr.shape != (rows,cols):
                analyzer_logger.error(f"Score format error for module {name}. Skipping.", extra=log_extra)
                continue
            fused += scores_arr * weight
        
        min_s, max_s = np.min(fused) if fused.size > 0 else 0.0, np.max(fused) if fused.size > 0 else 0.0
        if math.isclose(max_s, min_s):
            return np.zeros_like(fused) if math.isclose(min_s,0.0) else np.full_like(fused, 0.5 if min_s !=0 else 0.0) # if all same non-zero, 0.5
        norm_fused = (fused - min_s) / (max_s - min_s)
        analyzer_logger.debug(f"Scores fused and normalized. Range: [{min_s:.2f}, {max_s:.2f}] -> [0,1]", extra=log_extra)
        return norm_fused

    async def analyze_board_generic_pvs(
        self, new_card_list: list[list[int]], proposed_value_objects: list['ProposedValue'],
        active_modules: list[str] | None = None, module_weights: dict[str, float] | None = None,
        top_n: int | None = None, request_id_for_logging: str | None = None
    ) -> list[dict[str, str | int | float | bool | list | dict]]:
        req_id = request_id_for_logging or f"analyzer-{uuid.uuid4().hex[:8]}"
        log_extra = {'request_id': req_id}
        analyzer_logger.info(f"Analyzer starting for {len(proposed_value_objects)} proposals.", extra=log_extra)

        try:
            rows, cols, val_pv_objs, val_act_mods, val_mod_weights, final_top_n = \
                self._validate_inputs(new_card_list, proposed_value_objects, active_modules, module_weights, top_n, request_id=req_id)
        except InvalidInputError as e:
            analyzer_logger.error(f"Validation failed: {e}", exc_info=True, extra=log_extra)
            raise
        
        new_card_np = np.array(new_card_list, dtype=np.int32)
        eff_mods, final_w = self._get_effective_modules_and_weights(val_act_mods, val_mod_weights, request_id=req_id)
        
        evaluated_candidates: list[dict[str, str | int | float | bool | list | dict]] = []
        if not eff_mods:
            analyzer_logger.warning("No effective modules, returning empty candidates.", extra=log_extra)
            return []

        unique_pv_values = sorted(list(set(pv.value for pv in val_pv_objs)))
        heatmaps_cache: dict[int, np.ndarray] = {}

        for pv_val in unique_pv_values:
            analyzer_logger.debug(f"Generating heatmap for PV value: {pv_val}", extra=log_extra)
            mod_tasks = [self.main_module.get_module_score(mod_name, new_card_np, pv_val, request_id=req_id) for mod_name in eff_mods]
            raw_results = await asyncio.gather(*mod_tasks, return_exceptions=True)
            
            scores_for_pv_val: dict[str, np.ndarray] = {}
            for i, mod_name in enumerate(eff_mods):
                res = raw_results[i]
                if isinstance(res, Exception):
                    analyzer_logger.error(f"Module {mod_name} failed for PV {pv_val}: {res}", exc_info=res, extra=log_extra)
                    continue
                if res is None or not isinstance(res, np.ndarray) or res.shape != (rows,cols):
                    analyzer_logger.warning(f"Bad scores from {mod_name} for PV {pv_val}. Skipping.", extra=log_extra)
                    continue
                scores_for_pv_val[mod_name] = res
            
            if not scores_for_pv_val:
                heatmaps_cache[pv_val] = np.zeros((rows,cols), dtype=float) if rows > 0 and cols > 0 else np.array([[]])
                analyzer_logger.warning(f"No valid scores for PV {pv_val} to create heatmap.", extra=log_extra)
            else:
                heatmaps_cache[pv_val] = self._fuse_scores(scores_for_pv_val, final_w, rows, cols, request_id=req_id)

        for pv_obj in val_pv_objs:
            pos_r, pos_c = pv_obj.pos
            pv_actual_val = pv_obj.value
            raw_tf_score = 0.0
            is_valid = False

            if pv_actual_val in heatmaps_cache:
                heatmap = heatmaps_cache[pv_actual_val]
                if 0 <= pos_r < rows and 0 <= pos_c < cols:
                    if new_card_np[pos_r, pos_c] == -1: # Must be an empty cell
                        raw_tf_score = round(float(heatmap[pos_r, pos_c]), 6)
                        is_valid = True # Basic validity: proposed for an empty cell
                    else:
                        analyzer_logger.warning(f"Proposed cell {pv_obj.pos} for PV {pv_actual_val} is not empty.", extra=log_extra)
                else:
                     analyzer_logger.warning(f"Proposed cell {pv_obj.pos} for PV {pv_actual_val} is out of bounds.", extra=log_extra)
            
            # These are placeholders from main_api.pdf, to be filled by endpoint logic if needed
            mem_score_val = round(random.uniform(0,1) * raw_tf_score, 4) if is_valid else 0.0 # Mock
            final_obj_score = round(raw_tf_score + mem_score_val * 0.5, 4) if is_valid else 0.0 # Mock

            evaluated_candidates.append({
                "pos": list(pv_obj.pos), "value": pv_actual_val,
                "is_valid_proposal": is_valid, # Based on being an empty cell & having a score
                "raw_tensor_flow_score": raw_tf_score,
                "mem_score_value": mem_score_val,
                "final_objective_score": final_obj_score,
                "cp_solver_notes": None
            })
        
        evaluated_candidates.sort(key=lambda x: x["final_objective_score"], reverse=True) # Sort by final score
        analyzer_logger.info(f"Analysis complete. Returning {min(len(evaluated_candidates), final_top_n)} candidates.", extra=log_extra)
        return evaluated_candidates[:final_top_n]

    # Visualization methods (_setup_plot_figure, _configure_plot_axes, etc.) are identical to previous full response.
    # Omitting them here for brevity but they would be part of the Analyzer class.
    # Ensure _generate_visualization and _generate_error_visualization are present.
    def _setup_plot_figure(self, rows: int, cols: int, num_proposed_values: int) -> tuple[plt.Figure, plt.Axes, float]:
        cell_size_inch = max(0.5, min(1.0, 10.0 / max(rows, cols, 1)))
        fig_width = max(cols * cell_size_inch, 6); fig_height = max(rows * cell_size_inch, 4)
        if num_proposed_values > 3: fig_width += 2
        fig, ax = plt.subplots(figsize=(fig_width, fig_height)); return fig, ax, cell_size_inch
    def _configure_plot_axes(self, ax: plt.Axes, rows: int, cols: int, cell_size_inch: float):
        ax.set_xlim(-0.5, cols - 0.5); ax.set_ylim(rows - 0.5, -0.5)
        ax.set_xticks(np.arange(cols)); ax.set_yticks(np.arange(rows))
        ax.set_xticklabels(np.arange(1, cols + 1), fontsize=max(6, cell_size_inch * 10))
        ax.set_yticklabels(np.arange(1, rows + 1), fontsize=max(6, cell_size_inch * 10))
        ax.xaxis.tick_top(); ax.xaxis.set_label_position('top')
        ax.set_xlabel("Col", fontsize=max(7, cell_size_inch * 12)); ax.set_ylabel("Row", fontsize=max(7, cell_size_inch * 12))
        ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5); ax.set_aspect('equal', adjustable='box')
    def _draw_heatmap(self, ax: plt.Axes, board_state: list[list[int]], all_fused_scores_for_pvs: dict[int | str, np.ndarray], proposed_values_int_list: list[int]):
        rows = len(board_state); cols = len(board_state[0]) if rows > 0 else 0;  _log_extra = {'request_id': 'viz_heatmap'}
        if not (rows > 0 and cols > 0): return
        heatmap_data = np.full((rows, cols), np.nan)
        if proposed_values_int_list and proposed_values_int_list[0] in all_fused_scores_for_pvs:
            first_pv = proposed_values_int_list[0]; scores_pv = all_fused_scores_for_pvs[first_pv]
            if scores_pv.shape == (rows,cols):
                for r in range(rows):
                    for c in range(cols):
                        if board_state[r][c] == -1: heatmap_data[r,c] = scores_pv[r,c]
            else: analyzer_logger.warning(f"Heatmap shape mismatch for PV {first_pv}. Skipping.", extra=_log_extra)
        if not np.all(np.isnan(heatmap_data)):
            cmap = plt.cm.viridis; cmap.set_bad(color='white', alpha=0.0)
            ax.imshow(heatmap_data, cmap=cmap, alpha=0.6, aspect='auto', vmin=0, vmax=1)
    def _draw_suggestions_and_highlights(self, ax: plt.Axes, all_suggestions: dict[int | str, list[dict[str, str | int | float | bool | list | dict]]], proposed_values_int_list: list[int], top_n_suggestion_count: int ) -> dict[tuple[int, int], list[str]]:
        texts_on_cells: dict[tuple[int,int], list[str]] = {}; highlights: list[dict[str,str | int | float | bool | list | dict]] = []
        for pv_idx, pv_val in enumerate(proposed_values_int_list):
            color = self.PV_COLORS[pv_idx % len(self.PV_COLORS)]
            if pv_val in all_suggestions:
                for rank_idx, sugg in enumerate(all_suggestions[pv_val][:min(top_n_suggestion_count,3)]):
                    r,c = sugg['position']; rank = rank_idx + 1
                    txt = f"{pv_val}(R{rank})"; texts_on_cells.setdefault((r,c),[]).append(txt)
                    lw = 2.0 if rank==1 else (1.5 if rank==2 else 1.0)
                    highlights.append({'coords':(c-0.5,r-0.5),'width':1,'height':1,'lw':lw,'ec':color,'fc':mcolors.to_rgba(color,alpha=0.1 if rank==1 else 0.05)})
        for hl in highlights: ax.add_patch(patches.Rectangle(xy=hl['coords'],width=hl['width'],height=hl['height'],linewidth=hl['lw'],edgecolor=hl['ec'],facecolor=hl['fc']))
        return texts_on_cells
    def _draw_board_texts(self, ax: plt.Axes, board_state: list[list[int]], suggestion_texts_on_cells: dict[tuple[int, int], list[str]], cell_size_inch: float):
        rows = len(board_state); cols = len(board_state[0]) if rows > 0 else 0;
        if not (rows > 0 and cols > 0): return
        base_fs = max(6, cell_size_inch * 10)
        for r in range(rows):
            for c in range(cols):
                val = board_state[r][c]; cell_txts = []
                if val != -1: cell_txts.append(str(val))
                else:
                    if (r,c) in suggestion_texts_on_cells: cell_txts.extend(suggestion_texts_on_cells[(r,c)])
                    else: cell_txts.append(".")
                disp_txt = "\n".join(cell_txts); num_lines = disp_txt.count('\n')+1
                dyn_fs = base_fs / num_lines if num_lines > 1 else base_fs
                avg_chars = len(disp_txt.replace("\n","")) / num_lines if num_lines > 0 else 0
                width_f = (cell_size_inch * 10) / (avg_chars + 1) if avg_chars > -1 else 1
                dyn_fs = max(4, dyn_fs * min(1.0, width_f if width_f > 0 else 1.0))
                ax.text(c,r,disp_txt,ha='center',va='center',fontsize=dyn_fs,color='black',wrap=True)
    def _add_legend_and_title(self, fig: plt.Figure, ax: plt.Axes, proposed_values_int_list: list[int], all_suggestions: dict[int | str, list[dict[str, str | int | float | bool | list | dict]]], rows: int, cols: int, cell_size_inch: float):
        pv_s = ", ".join(map(str,proposed_values_int_list)) if proposed_values_int_list else "N/A"
        title = f"Board Analysis ({rows}x{cols}) - PVs: [{pv_s}]"
        if not any(s for sl in all_suggestions.values() for s in sl): title += "\n(No valid suggestions)"
        fig.suptitle(title, fontsize=max(8, cell_size_inch*14))
        leg_elements = []
        if proposed_values_int_list and any(s for sl in all_suggestions.values() for s in sl):
            added_pvs = set()
            for pv_idx, pv_val in enumerate(proposed_values_int_list):
                if pv_val not in added_pvs and any(s for s in all_suggestions.get(pv_val,[])):
                    color = self.PV_COLORS[pv_idx % len(self.PV_COLORS)]
                    leg_elements.append(patches.Patch(facecolor=color,edgecolor=color,label=f'PV {pv_val} Sugg.'))
                    added_pvs.add(pv_val)
        if leg_elements:
            ax.legend(handles=leg_elements, loc='center left', bbox_to_anchor=(1.03,0.5),fontsize=max(7,cell_size_inch*10),title="Legend")
            plt.tight_layout(rect=[0,0,0.9,0.95])
        else: plt.tight_layout(rect=[0,0,1,0.95])
    def _generate_visualization(self,board_state: list[list[int]],proposed_values_int_list: list[int],all_suggestions: dict[int | str, list[dict[str, str | int | float | bool | list | dict]]],all_fused_scores_for_pvs: dict[int | str, np.ndarray],top_n_suggestion_count: int,request_id: str | None = "N/A") -> str:
        _log_extra = {'request_id': request_id}; analyzer_logger.debug("Generating visualization...", extra=_log_extra)
        rows = len(board_state); cols = len(board_state[0]) if rows > 0 else 0
        if rows==0 or cols==0: return self._generate_error_visualization(0,0,"Board empty",request_id)
        fig, ax, cs_inch = self._setup_plot_figure(rows,cols,len(proposed_values_int_list))
        self._current_cell_size_inch_for_dpi = cs_inch # For DPI in _fig_to_base64
        try:
            self._configure_plot_axes(ax,rows,cols,cs_inch)
            self._draw_heatmap(ax,board_state,all_fused_scores_for_pvs,proposed_values_int_list)
            sugg_txts = self._draw_suggestions_and_highlights(ax,all_suggestions,proposed_values_int_list,top_n_suggestion_count)
            self._draw_board_texts(ax,board_state,sugg_txts,cs_inch)
            self._add_legend_and_title(fig,ax,proposed_values_int_list,all_suggestions,rows,cols,cs_inch)
            img_b64 = self._fig_to_base64(fig)
        except Exception as e_viz_detail:
            analyzer_logger.error(f"Viz detail error: {e_viz_detail}", exc_info=True, extra=_log_extra)
            img_b64 = self._generate_error_visualization(rows,cols,f"Viz fail: {type(e_viz_detail).__name__}",request_id)
        finally:
            plt.close(fig)
            if hasattr(self, '_current_cell_size_inch_for_dpi'): delattr(self, '_current_cell_size_inch_for_dpi')
        return img_b64
    def _generate_error_visualization(self, rows: int, cols: int, error_message: str, request_id: str | None = "N/A") -> str:
        _log_extra = {'request_id': request_id}; analyzer_logger.info(f"Generating error viz: {error_message}", extra=_log_extra)
        try:
            fig_w=max(cols*0.5 if cols>0 else 1,5); fig_h=max(rows*0.5 if rows>0 else 1,3)
            fig,ax=plt.subplots(figsize=(fig_w,fig_h))
            ax.text(0.5,0.5,f"Error:\n{error_message}",ha='center',va='center',fontsize=10,color='red',wrap=True)
            ax.axis('off'); img_b64=self._fig_to_base64(fig)
        except Exception as e_err_viz:
            analyzer_logger.error(f"Error viz itself failed: {e_err_viz}", exc_info=True, extra=_log_extra)
            return "Error in error viz" # Fallback text
        finally:
            if 'fig' in locals() and fig is not None: plt.close(fig) # Ensure fig is closed
        return img_b64
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        buf = io.BytesIO()
        try:
            cs_inch = getattr(self, '_current_cell_size_inch_for_dpi', 0.75)
            dpi = max(75, int(cs_inch * 120))
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        except Exception as e_save:
            analyzer_logger.error(f"fig.savefig failed: {e_save}", exc_info=True)
            raise VisualizationError(f"Savefig fail: {e_save}") from e_save
        finally: # Ensure buf is closed even if savefig fails, though not strictly needed for BytesIO if not written
            if 'fig' in locals() and fig is not None: plt.close(fig) # ensure figure is closed

        buf.seek(0); img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close(); return img_b64


analyzer_instance: Analyzer | None = None
try:
    analyzer_instance = Analyzer(main_module=brain_interface, default_top_n=3)
    logger.info("Global Analyzer instance created successfully (full brain module implementation).")
except InitializationError as e_init_analyzer:
    logger.critical("CRITICAL: Analyzer init failed: %s", e_init_analyzer, exc_info=True)
except Exception as e_unexp_analyzer:
    logger.critical("CRITICAL: Unexpected Analyzer init error: %s", e_unexp_analyzer, exc_info=True)

# --- Pydantic Models ---
class HealthResponse(BaseModel):
    status: str; message: str | None = None; reason: str | None = None; analyzer_status: str | None = None
class AnalyzeHealthStatus(BaseModel):
    status: str; analysis_engine_version: str; checks: dict[str, str]; components: dict[str, str]
class CandidateDetail(BaseModel):
    pos: list[int]; value: int; is_valid_proposal: bool; raw_tensor_flow_score: float
    mem_score_value: float; final_objective_score: float; cp_solver_notes: str | None = None
class AnalyzeSuccessResponse(BaseModel):
    request_id: str; message: str; grid_shape: tuple[int, ...]; evaluated_candidates: list[CandidateDetail]
class AnalyzeErrorResponse(BaseModel):
    detail: str; request_id: str | None = None
class ProposedValue(BaseModel):
    pos: tuple[int, int]; value: int
    @field_validator('pos')
    def pos_must_be_tuple_of_two_ints(cls, v):
        if not (isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], int) and isinstance(v[1], int)):
            raise ValueError("Position 'pos' must be a tuple of two integers (row, col).")
        return v
class AnalysisRequest(BaseModel):
    new_card: list[list[int]]; proposed_values: list[ProposedValue]
    active_modules: list[str] | None = None; module_weights: dict[str, float] | None = None
    top_n: int | None = Field(None, gt=0)
class GridDataBase(BaseModel): # For background tasks
    grid_data: list[list[int | float]] = Field(..., example=[[-1,1.0,-1],[2,-1,3.5]])
    @field_validator('grid_data')
    def validate_grid_data_bg(cls, v): # Renamed validator
        if not v or not all(isinstance(r,list) for r in v) or not v[0]: raise ValueError("Grid must be non-empty list of lists.")
        cols = len(v[0]);_ = [r for r in v if len(r)!=cols and (_ for _ in ()).throw(ValueError("Rows must have same cols."))]
        _ = [c for r in v for c in r if not isinstance(c,(int,float)) and (_ for _ in ()).throw(ValueError("Cells must be numbers."))]
        return v
class GridInput(GridDataBase): client_request_id: str | None = None
class BatchGridItem(GridDataBase): item_id: str; module_name: str
class BatchGridInput(BaseModel): grids: list[BatchGridItem] = Field(..., max_length=100); client_request_id: str | None = None # Increased batch size
class TaskAcceptedResponse(BaseModel): task_id: str; status: str="accepted"; message: str; client_request_id: str | None = None
class ModuleInfo(BaseModel): name: str; description: str|None="N/A"; version: str|None="N/A"

# --- Mock CP Model & TF Placeholder ---
class MockCPModel:
    _version: str = "9.9.mock-2025-full"
    def CpModel(self): logger.info("[Placeholder] MockCPModel.CpModel() invoked.", extra={'request_id':'N/A_cp'})
cp_model = MockCPModel()
def extreme_tensor_flow_score_detailed_placeholder(grid: np.ndarray, req_id_ctx: str) -> tuple[np.ndarray, list[list[dict[str,str | int | float | bool | list | dict]]]]:
    logger.info(f"[Placeholder] extreme_tf_score_detailed for {req_id_ctx}, grid: {grid.shape}", extra={'request_id':req_id_ctx})
    s = np.random.rand(*grid.shape).astype(np.float32)*10
    c = [[{"rule":f"dummy_r{r}_c{col}","value":np.random.random()} for col in range(grid.shape[1])] for r in range(grid.shape[0])]
    return s, c

# --- API Auth & Rate Limiter Dict ---
api_key_query_auth = APIKeyQuery(name=settings.API_KEY_NAME, auto_error=False)
api_key_header_auth = APIKeyHeader(name=settings.API_KEY_NAME, auto_error=False)
async def get_api_key(key_query:str|None=Security(api_key_query_auth),key_header:str|None=Security(api_key_header_auth))->str:
    if key_query==settings.API_KEY: return key_query
    if key_header==settings.API_KEY: return key_header
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API Key")
request_counts_rl: dict[str, list[float]] = {} # Renamed from request_counts

# --- Background Task Runner ---
async def run_scoring_task(task_id:str, module_name:str, grid_data:list[list[int | float]], orig_req_id:str, client_req_id:str|None=None):
    ACTIVE_BACKGROUND_TASKS.inc(); log_ex = {'request_id':orig_req_id,'task_id':task_id,'module_name':module_name,'client_req_id':client_req_id or "N/A"}
    logger.info("BG task started.", extra=log_ex)
    try:
        np_grid = np.array(grid_data, dtype=np.int32)
        if np_grid.size==0: raise ValueError("Empty grid for BG task.")
        start_t = time.monotonic()
        score_arr = await brain_interface.get_module_score(module_name, np_grid, request_id=task_id)
        dur_t = time.monotonic()-start_t; score_list = score_arr.tolist()
        logger.info(f"BG scoring success. Dur: {dur_t:.4f}s", extra=log_ex)
        if settings.TASK_CALLBACK_URL_ENABLED and settings.TASK_CALLBACK_URL:
            logger.info(f"Simulating callback to {settings.TASK_CALLBACK_URL}", extra=log_ex) # httpx call here
    except Exception as e_bg: logger.error(f"Error in BG task: {e_bg}", exc_info=True, extra=log_ex)
    finally: ACTIVE_BACKGROUND_TASKS.dec(); logger.info("BG task finished.", extra=log_ex)

# --- FastAPI App & Middleware ---
app = FastAPI(title=settings.APP_TITLE, description=settings.APP_DESCRIPTION, version=settings.APP_VERSION)
app.add_middleware(PrometheusMiddleware)

@app.middleware("http")
async def base_middleware_full(request: Request, call_next: Callable[[Request], Coroutine[str | int | float | bool | list | dict,str | int | float | bool | list | dict,str | int | float | bool | list | dict]]) -> str | int | float | bool | list | dict:
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4()); request.state.request_id = req_id
    log_ex_mw = {'request_id': req_id}
    client_ip = request.client.host if request.client else "unknown"; current_t = time.time()
    request_counts_rl.setdefault(client_ip, [])
    request_counts_rl[client_ip] = [t for t in request_counts_rl[client_ip] if t > current_t - settings.RATE_LIMIT_WINDOW_SECONDS]
    if len(request_counts_rl[client_ip]) >= settings.RATE_LIMIT_REQUESTS:
        logger.warning(f"Rate limit hit for IP: {client_ip}", extra=log_ex_mw)
        REQUEST_COUNT.labels(method=request.method,endpoint=str(request.url.path),status_code=429).inc()
        return JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS,content={"detail":"Rate limit exceeded."})
    request_counts_rl[client_ip].append(current_t)
    
    start_tm = time.monotonic()
    logger.info(f"→ {request.method} {request.url.path} Agent: {request.headers.get('user-agent','N/A')}", extra=log_ex_mw)
    response = await call_next(request); dur_m = time.monotonic() - start_tm
    response.headers["X-Request-ID"]=req_id; response.headers["X-Content-Type-Options"]="nosniff"; response.headers["X-Frame-Options"]="DENY"
    response.headers["Content-Security-Policy"]="default-src 'none'; frame-ancestors 'none';"
    if request.url.scheme=="https": response.headers["Strict-Transport-Security"]="max-age=31536000; includeSubDomains"
    REQUEST_COUNT.labels(method=request.method,endpoint=str(request.url.path),status_code=response.status_code).inc()
    REQUEST_LATENCY.labels(method=request.method,endpoint=str(request.url.path)).observe(dur_m)
    logger.info(f"← {request.method} {request.url.path} Status: {response.status_code} Dur: {dur_m:.4f}s", extra=log_ex_mw)
    return response

@app.exception_handler(Exception)
async def global_exception_handler_full(request: Request, exc: Exception) -> JSONResponse:
    req_id = getattr(request.state, 'request_id', str(uuid.uuid4())); log_ex_exc = {'request_id': req_id}
    logger.error(f"Global unhandled exception: {exc}", exc_info=True, extra=log_ex_exc)
    s_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    content = {"request_id":req_id,"error":"Internal Server Error","message":"Unexpected error.","detail":str(exc) if settings.LOG_LEVEL=="DEBUG" else None}
    if isinstance(exc, HTTPException): s_code=exc.status_code; content["error"]="HTTP Exception"; content["message"]=exc.detail; content["detail"]=None
    REQUEST_COUNT.labels(method=request.method,endpoint=str(request.url.path),status_code=s_code).inc()
    return JSONResponse(status_code=s_code, content=content)

# --- API Endpoints ---
@app.get("/", tags=["Utilities"], summary="Root / Basic Health")
async def read_root_full(request: Request):
    logger.info("Root path '/' accessed.", extra={'request_id': request.state.request_id})
    return {"app":settings.APP_TITLE,"version":settings.APP_VERSION,"docs":str(request.url.replace(path="/docs")), "analyzer": "Initialized" if analyzer_instance else "Not Initialized"}

@app.get("/health", response_model=HealthResponse, tags=["Utilities"], summary="Analyzer Health Check")
async def health_check_simple_full(request: Request):
    req_id = request.state.request_id
    if analyzer_instance is None:
        logger.warning("Health check: Analyzer not init.", extra={'request_id': req_id})
        return HealthResponse(status="unhealthy",reason="Analyzer not initialized",analyzer_status="Not Initialized")
    logger.info("Health check: Analyzer OK.", extra={'request_id': req_id})
    return HealthResponse(status="ok",message="Analyzer API OK",analyzer_status="Initialized")

@app.get("/health/analyze", response_model=AnalyzeHealthStatus, tags=["Health & Monitoring"], summary="Detailed System Health")
async def health_analyze_detailed_full(request: Request):
    req_id = request.state.request_id; log_ex = {'request_id': req_id}; logger.info("Detailed health check.", extra=log_ex)
    checks:dict[str,str]={}; overall="UP"
    checks["brain_modules"] = f"OK: {len(brain_interface.registered_modules)} modules" if brain_interface.registered_modules else "FAIL: No modules"
    if not brain_interface.registered_modules: overall="DEGRADED"
    checks["mem_path"] = "SKIPPED (os.path not used)" # Simplified
    try:
        dummy_grid = np.array([[-1,1],[1,-1]],dtype=np.int32)
        await asyncio.to_thread(extreme_tensor_flow_score_detailed_placeholder, dummy_grid, f"health_tf_{req_id}")
        checks["tf_exec_test"] = "OK (Placeholder)"
    except Exception as e_tf: checks["tf_exec_test"]=f"FAIL: {e_tf}"; logger.error("TF health test fail.",exc_info=True,extra=log_ex); overall="ERROR"
    try: _=cp_model.CpModel(); checks["cp_solver_test"]="OK (Mock)"
    except Exception as e_cp: checks["cp_solver_test"]=f"FAIL: {e_cp}"; logger.error("CP health test fail.",exc_info=True,extra=log_ex); overall="ERROR"
    return AnalyzeHealthStatus(status=overall,analysis_engine_version=settings.ANALYZER_VERSION,checks=checks,
        components={"numpy":np.__version__,"ortools_mock":getattr(cp_model,'_version_',"N/A"),"analyzer_type":"Extreme Logic v2.5 Full"})

@app.post("/analyze", response_model=AnalyzeSuccessResponse, tags=["Analysis Engine vExtreme"], summary="Perform Tensor Analysis")
async def analyze_board_main_full(payload:AnalysisRequest, request:Request, api_key:APIKey=Depends(get_api_key)):
    req_id = request.state.request_id; log_ex = {'request_id':req_id}
    logger.info(f"Analyze API call. Grid: {len(payload.new_card)}x{len(payload.new_card[0]) if payload.new_card and payload.new_card[0] else 'N/A'}. Proposals: {len(payload.proposed_values)}", extra=log_ex)
    if analyzer_instance is None: logger.error("Analyzer not available for /analyze.", extra=log_ex); raise HTTPException(status_code=503,detail="Analysis service unavailable.")
    if not payload.new_card or not payload.new_card[0]: raise HTTPException(status_code=400,detail="new_card cannot be empty.")
    try: grid_np = np.array(payload.new_card, dtype=np.int32)
    except Exception as e_np: raise HTTPException(status_code=400,detail=f"Invalid new_card format: {e_np}")
    try:
        analysis_results = await analyzer_instance.analyze_board_generic_pvs(
            payload.new_card, payload.proposed_values, payload.active_modules,
            payload.module_weights, payload.top_n, request_id_for_logging=req_id
        )
        # analysis_results is already List[Dict] matching CandidateDetail structure (mostly)
        processed_candidates = [CandidateDetail(**cand) for cand in analysis_results] # Direct cast
        logger.info(f"Analysis success. Candidates: {len(processed_candidates)}", extra=log_ex)
        return AnalyzeSuccessResponse(request_id=req_id,message="Analysis successful.",grid_shape=grid_np.shape,evaluated_candidates=processed_candidates)
    except InvalidInputError as e_val: raise HTTPException(status_code=422,detail=f"Invalid params: {e_val}")
    except (ModuleError, VisualizationError) as e_mod: logger.error(f"Module/Viz error: {e_mod}",exc_info=True,extra=log_ex); raise HTTPException(status_code=500,detail=f"Module/Viz error: {e_mod}")
    except Exception as e_gen: logger.critical(f"Unexpected /analyze error: {e_gen}",exc_info=True,extra=log_ex); raise HTTPException(status_code=500,detail=f"Unexpected: {e_gen}")

@app.get("/modules", response_model=list[ModuleInfo], tags=["Modules"], summary="List available modules")
async def list_modules_full(request:Request, api_key:APIKey=Depends(get_api_key)):
    logger.info("Listing modules.", extra={'request_id':request.state.request_id})
    return [ModuleInfo(name=name, **brain_interface.get_module_details(name)) for name in brain_interface.registered_modules.keys()]

@app.get("/modules/{module_name}",response_model=ModuleInfo, tags=["Modules"], summary="Get module details")
async def get_module_info_full(request:Request, module_name:str=Path(...,description="Module name"), api_key:APIKey=Depends(get_api_key)):
    log_ex={'request_id':request.state.request_id,'module_name':module_name}; logger.info("Getting module info.",extra=log_ex)
    if module_name not in brain_interface.registered_modules: raise HTTPException(status_code=404,detail=f"Module '{module_name}' not found.")
    return ModuleInfo(name=module_name, **brain_interface.get_module_details(module_name))

@app.post("/score/{module_name}", response_model=TaskAcceptedResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Scoring (Async)"], summary="Submit grid for background scoring")
async def score_grid_bg_full(request:Request, payload:GridInput, module_name:str=Path(...,description="Module to use"), bg_tasks:BackgroundTasks=Depends(), api_key:APIKey=Depends(get_api_key)):
    req_id=request.state.request_id; client_req_id=payload.client_request_id; task_id=str(uuid.uuid4())
    log_ex={'request_id':req_id,'task_id':task_id,'module_name':module_name,'client_req_id':client_req_id or "N/A"}
    if module_name not in brain_interface.registered_modules: logger.warning("Module not found for BG task.",extra=log_ex); raise HTTPException(status_code=404,detail=f"Module '{module_name}' not found.")
    bg_tasks.add_task(run_scoring_task,task_id,module_name,payload.grid_data,req_id,client_req_id)
    MODULE_USAGE_COUNT.labels(module_name=module_name).inc(); logger.info("BG task enqueued.",extra=log_ex)
    return TaskAcceptedResponse(task_id=task_id,message=f"Task for module '{module_name}' accepted.",client_request_id=client_req_id)

@app.post("/score/batch", response_model=list[TaskAcceptedResponse], status_code=status.HTTP_202_ACCEPTED, tags=["Scoring (Async)"], summary="Submit batch grids for background scoring")
async def score_batch_bg_full(request:Request, payload:BatchGridInput, bg_tasks:BackgroundTasks=Depends(), api_key:APIKey=Depends(get_api_key)):
    req_id=request.state.request_id; client_req_id=payload.client_request_id; responses:list[TaskAcceptedResponse]=[]
    log_ex_batch={'request_id':req_id,'batch_size':len(payload.grids),'client_req_id':client_req_id or "N/A"}; logger.info("Batch BG task received.",extra=log_ex_batch)
    for item in payload.grids:
        task_id=str(uuid.uuid4()); log_ex_item={**log_ex_batch,'task_id':task_id,'item_id':item.item_id,'module_name':item.module_name}
        if item.module_name not in brain_interface.registered_modules:
            logger.warning(f"Module {item.module_name} for item {item.item_id} not found.",extra=log_ex_item)
            responses.append(TaskAcceptedResponse(task_id=f"err_mod_{item.item_id}",status="rejected",message=f"Module {item.module_name} for {item.item_id} not found.",client_request_id=client_req_id))
            continue
        bg_tasks.add_task(run_scoring_task,task_id,item.module_name,item.grid_data,req_id,client_req_id)
        MODULE_USAGE_COUNT.labels(module_name=item.module_name).inc()
        responses.append(TaskAcceptedResponse(task_id=task_id,message=f"Task for item {item.item_id} ('{item.module_name}') accepted.",client_request_id=client_req_id))
        logger.info("Batch item enqueued.",extra=log_ex_item)
    return responses

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    log_ex_main = {'request_id':'SYSTEM_MAIN'}
    logger.info(f"Starting {settings.APP_TITLE} v{settings.APP_VERSION} on {settings.APP_HOST}:{settings.APP_PORT}", extra=log_ex_main)
    logger.info(f"API Key (first 4 chars): {settings.API_KEY[:4]}...", extra=log_ex_main)
    if settings.TASK_CALLBACK_URL_ENABLED: logger.info(f"Task callback ON: {settings.TASK_CALLBACK_URL}", extra=log_ex_main)
    else: logger.info("Task callback OFF.", extra=log_ex_main)
    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT, log_config=None)