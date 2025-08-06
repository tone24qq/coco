"""Asynchronous memory loader using background tasks and memmap.

Implements plan A from the design document: load hot shapes in a
background task during FastAPI startup while allowing lazy loading for
other shapes on demand.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Directory containing sharded memory files such as
# ``{rows}x{cols}_keys_p0.npy`` / ``{rows}x{cols}_vals_p0.npy``.
DATA_DIR = Path(os.environ.get("MEMORY_DATA_DIR", "data_archives"))

# Cache mapping ``(rows, cols)`` to ``(keys, values, targets, boards)``.
MEMORY_CACHE: Dict[
    Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
] = {}

# Locks to avoid duplicate loads for the same shape.
LOCKS: Dict[Tuple[int, int], asyncio.Lock] = {}

logger = logging.getLogger(__name__)

_PART_RE = re.compile(r"_p(\d+)\.npy$")


def _glob_paths(rows: int, cols: int, kind: str) -> List[Path]:
    """Return all shard paths for ``kind`` sorted by part number."""

    pat = DATA_DIR / f"{rows}x{cols}_{kind}_p*.npy"
    paths = pat.parent.glob(pat.name)
    return sorted(paths, key=lambda p: int(_PART_RE.search(p.name).group(1)))


def _load_shape(rows: int, cols: int) -> None:
    """Load memory arrays for ``(rows, cols)`` and cache them.

    Sharded files with pattern ``{rows}x{cols}_{kind}_p*.npy`` are
    concatenated in order of part number. Missing ``targets`` or ``boards``
    shards yield empty arrays.
    """

    k_paths = _glob_paths(rows, cols, "keys")
    v_paths = _glob_paths(rows, cols, "vals")
    if not k_paths or not v_paths:
        raise FileNotFoundError(f"missing keys/vals shards for {rows}x{cols}")

    keys = np.concatenate([np.load(p, mmap_mode="r") for p in k_paths], axis=0)
    vals = np.concatenate([np.load(p, mmap_mode="r") for p in v_paths], axis=0)

    t_paths = _glob_paths(rows, cols, "targets")
    if t_paths:
        targets = np.concatenate([np.load(p, mmap_mode="r") for p in t_paths], axis=0)
    else:
        targets = np.empty((0,), dtype=np.int16)

    b_paths = _glob_paths(rows, cols, "boards")
    if b_paths:
        boards = np.concatenate([np.load(p, mmap_mode="r") for p in b_paths], axis=0)
    else:
        boards = np.empty((0, rows * cols), dtype=np.int8)

    MEMORY_CACHE[(rows, cols)] = (keys, vals, targets, boards)
    logger.info("✅ memory %dx%d loaded parts=%d", rows, cols, len(k_paths))


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
