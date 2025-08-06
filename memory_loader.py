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
    """
    將 (rows, cols) 的 keys / vals / targets 透過 mem-map 掛進快取。
    """
    k_path = _file_path(rows, cols, "keys")
    v_path = _file_path(rows, cols, "vals")
    t_path = _file_path(rows, cols, "targets")      # ← 新增 targets 路徑

    # 1️⃣  mmap keys / vals（fp16）
    keys = np.load(k_path, mmap_mode="r")
    vals = np.load(v_path, mmap_mode="r")

    # 2️⃣  mmap targets（如果檔案存在）
    targets = None
    if t_path.exists():
        targets = np.load(t_path, mmap_mode="r")
        import app                                   # 避免循環 import
        app.memory_targets[(rows, cols)] = targets   # ← 放進全域快取

    # 3️⃣  放進 MEMORY_CACHE 供 predict 使用
    MEMORY_CACHE[(rows, cols)] = (keys, vals)

    logger.info(
        "✅ memory %dx%d loaded (targets=%s)",
        rows,
        cols,
        "yes" if targets is not None else "no",
    )

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
