import importlib
from pathlib import Path

import numpy as np


def test_build_memories_creates_npz(monkeypatch, capsys):
    monkeypatch.setenv("MEMORY_SAMPLE_LIMIT", "2")

    for p in Path("data_archives").glob("*_memory.npz"):
        p.unlink()

    build_memories = importlib.import_module("build_memories")
    importlib.reload(build_memories)
    build_memories.main()

    out = capsys.readouterr().out
    assert "已快取 4x5" in out
    assert "已快取 8x10" in out

    path_4x5 = Path("data_archives/4x5_memory.npz")
    path_8x10 = Path("data_archives/8x10_memory.npz")
    assert path_4x5.is_file()
    assert path_8x10.is_file()

    data = np.load(path_4x5)
    assert "keys" in data and "values" in data
