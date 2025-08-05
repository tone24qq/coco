"""Asynchronous memory loader using background tasks and memmap.

Implements plan A from the design document: load hot shapes in a
background task during FastAPI startup while allowing lazy loading for
other shapes on demand.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Directory containing ``{rows}x{cols}_keys.npy`` and ``..._vals.npy`` files.
DATA_DIR = Path(os.environ.get("MEMORY_DATA_DIR", "data_archives"))

# Global in-memory cache mapping ``(rows, cols)`` to ``(keys, values)``.
MEMORY_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

# Locks to avoid duplicate loads for the same shape.
LOCKS: Dict[Tuple[int, int], asyncio.Lock] = {}

logger = logging.getLogger(__name__)


def _file_path(rows: int, cols: int, kind: str) -> Path:
    """Return path for memory `kind` ("keys" or "vals")."""

    return DATA_DIR / f"{rows}x{cols}_{kind}.npy"


def _load_shape(rows: int, cols: int) -> None:
    """Load memory arrays for `(rows, cols)` into :data:`MEMORY_CACHE`.

    The arrays are memory-mapped to avoid reading the whole file into RAM.
    """

    k_path = _file_path(rows, cols, "keys")
    v_path = _file_path(rows, cols, "vals")
    keys = np.load(k_path, mmap_mode="r")
    vals = np.load(v_path, mmap_mode="r")
    MEMORY_CACHE[(rows, cols)] = (keys, vals)
    logger.info("✅ memory %dx%d loaded", rows, cols)


async def ensure_loaded(rows: int, cols: int) -> None:
    """Ensure memory for `(rows, cols)` is available in the cache."""

    if (rows, cols) in MEMORY_CACHE:
        return
    lock = LOCKS.setdefault((rows, cols), asyncio.Lock())
    async with lock:
        if (rows, cols) not in MEMORY_CACHE:
            try:
                await asyncio.to_thread(_load_shape, rows, cols)
            except FileNotFoundError:
                logger.warning("memory files missing for %dx%d", rows, cols)


async def preload_hot_shapes(hot_shapes: list[tuple[int, int]]) -> None:
    """Spawn background tasks to load commonly used shapes."""

    for rows, cols in hot_shapes:
        asyncio.create_task(ensure_loaded(rows, cols))
