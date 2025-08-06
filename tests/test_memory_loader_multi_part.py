import asyncio
from pathlib import Path

import numpy as np

import memory_loader


def test_memory_loader_multi_part(tmp_path: Path) -> None:
    rows, cols = 2, 3
    parts = [2, 3, 1]
    offset = 0
    for idx, n in enumerate(parts):
        k = np.full((n, 2), idx, dtype=np.float32)
        v = np.full((n, cols * rows // 2), idx, dtype=np.float32)
        t = np.arange(offset, offset + n, dtype=np.int16)
        np.save(tmp_path / f"{rows}x{cols}_keys_p{idx}.npy", k)
        np.save(tmp_path / f"{rows}x{cols}_vals_p{idx}.npy", v)
        np.save(tmp_path / f"{rows}x{cols}_targets_p{idx}.npy", t)
        offset += n
    memory_loader.DATA_DIR = tmp_path
    memory_loader.MEMORY_CACHE.clear()
    asyncio.run(memory_loader.ensure_loaded(rows, cols))
    keys, vals, targets, boards = memory_loader.MEMORY_CACHE[(rows, cols)]
    assert keys.shape[0] == sum(parts)
    assert vals.shape[0] == sum(parts)
    assert targets.shape[0] == sum(parts)
    assert boards.shape == (0, rows * cols)
