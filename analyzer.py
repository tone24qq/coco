"""
analyzer.py: Core Analysis Logic Layer.

This module is responsible for orchestrating the analysis using modules from brain.py.
"""
import asyncio
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from brain import REGISTERED_MODULES_BRAIN, BrainModuleFn

logger = logging.getLogger(__name__)


class AnalysisResultDetail(BaseModel):
    module_name: str
    score_grid: list[list[float]] | None = Field(None, description="Per-cell scores from this module")
    error: str | None = Field(None, description="Error message if module execution failed")


class OverallAnalysisResult(BaseModel):
    request_id: str
    total_modules_processed: int
    module_results: list[AnalysisResultDetail]
    # Example: could also have a combined_score_grid or overall_assessment
    # combined_score_grid: list[list[float]] | None = None
    # overall_panel_score: float | None = None


async def run_module_async(
    module_name: str,
    module_func: BrainModuleFn,
    grid: np.ndarray,
    request_id: str | None,
) -> AnalysisResultDetail:
    """Helper to run a single module, potentially in a thread for CPU-bound tasks."""
    try:
        # For CPU-bound numpy operations, true parallelism needs multiprocessing.
        # asyncio.to_thread is good for blocking I/O or libraries that release GIL.
        # Here, numpy operations might release GIL for some parts, but can be CPU intensive.
        # If modules are very heavy, ProcessPoolExecutor via FastAPI's run_in_threadpool
        # (which can wrap a ProcessPoolExecutor) or a dedicated task queue (Celery/RQ)
        # would be better for a production system to not block the main FastAPI event loop.
        # For this structure, we'll call them directly or via to_thread if they were I/O bound.
        # Given they are numpy based, direct call in an async request handler means they run
        # synchronously within that handler's thread execution slice.
        # If these modules become very slow, consider:
        # score_grid_np = await asyncio.to_thread(module_func, grid, request_id)

        logger.debug(f"Starting module {module_name}", extra={"request_id": request_id})
        score_grid_np = module_func(grid, request_id) # Direct call
        logger.debug(f"Finished module {module_name}", extra={"request_id": request_id})

        return AnalysisResultDetail(
            module_name=module_name,
            score_grid=score_grid_np.tolist() if score_grid_np is not None else None,
        )
    except Exception as e: # pragma: no cover
        logger.error(
            f"Error executing module {module_name} in analyzer: {e}",
            exc_info=True,
            extra={"request_id": request_id},
        )
        return AnalysisResultDetail(module_name=module_name, error=str(e))


async def analyze_grid(
    grid_data: np.ndarray, request_id: str
) -> OverallAnalysisResult:
    """
    Analyzes the grid using all registered brain modules.
    Calculates scores and aggregates results.
    """
    logger.info(
        "Starting grid analysis in analyzer",
        extra={
            "request_id": request_id,
            "grid_shape": grid_data.shape,
            "num_modules": len(REGISTERED_MODULES_BRAIN),
        },
    )

    module_tasks = []
    for module_name, module_func in REGISTERED_MODULES_BRAIN.items():
        module_tasks.append(
            run_module_async(module_name, module_func, grid_data.copy(), request_id) # Pass copy of grid
        )
    
    # Run modules concurrently if they were truly async or using to_thread for blocking IO
    # For CPU-bound tasks like these, asyncio.gather doesn't provide parallelism on its own,
    # they will run sequentially in the available thread if not offloaded.
    # If modules were I/O bound and used `await asyncio.to_thread`, then `gather` is beneficial.
    results: list[AnalysisResultDetail] = await asyncio.gather(*module_tasks)

    # --- Example of further aggregation (can be expanded) ---
    # combined_scores = np.zeros_like(grid_data, dtype=float)
    # successful_modules = 0
    # for res in results:
    #     if res.score_grid and res.error is None:
    #         try:
    #             combined_scores += np.array(res.score_grid)
    #             successful_modules +=1
    #         except ValueError as ve: # pragma: no cover
    #             logger.warning(f"Could not add scores from {res.module_name} due to shape mismatch or error: {ve}",
    #                            extra={"request_id": request_id})

    # if successful_modules > 0:
    #     final_combined_grid = (combined_scores / successful_modules).tolist()
    # else: # pragma: no cover
    #     final_combined_grid = np.zeros_like(grid_data, dtype=float).tolist()

    logger.info(
        "Finished grid analysis in analyzer",
        extra={"request_id": request_id, "modules_processed": len(results)},
    )

    return OverallAnalysisResult(
        request_id=request_id,
        total_modules_processed=len(results),
        module_results=results,
        # combined_score_grid=final_combined_grid, # Example
        # overall_panel_score=float(np.mean(combined_scores)) if successful_modules > 0 else 0.0 # Example
    )