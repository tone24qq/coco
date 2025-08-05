import asyncio
from pathlib import Path

import numpy as np

import memory_loader


def test_ensure_loaded(tmp_path: Path) -> None:
    keys = np.ones((2, 3), dtype=np.float32)
    vals = np.ones((2, 2), dtype=np.float32)
    np.save(tmp_path / "2x3_keys.npy", keys)
    np.save(tmp_path / "2x3_vals.npy", vals)
    memory_loader.DATA_DIR = tmp_path
    memory_loader.MEMORY_CACHE.clear()
    asyncio.run(memory_loader.ensure_loaded(2, 3))
    assert (2, 3) in memory_loader.MEMORY_CACHE
    loaded_keys, loaded_vals = memory_loader.MEMORY_CACHE[(2, 3)]
    assert loaded_keys.shape == keys.shape
    assert loaded_vals.shape == vals.shape


def test_preload_hot_shapes(tmp_path: Path) -> None:
    keys = np.zeros((1, 4), dtype=np.float32)
    vals = np.zeros((1, 2), dtype=np.float32)
    np.save(tmp_path / "1x4_keys.npy", keys)
    np.save(tmp_path / "1x4_vals.npy", vals)
    memory_loader.DATA_DIR = tmp_path
    memory_loader.MEMORY_CACHE.clear()

    async def run() -> None:
        await memory_loader.preload_hot_shapes([(1, 4)])
        # give background task a chance to run
        for _ in range(10):
            if (1, 4) in memory_loader.MEMORY_CACHE:
                break
            await asyncio.sleep(0.01)

    asyncio.run(run())
    assert (1, 4) in memory_loader.MEMORY_CACHE
