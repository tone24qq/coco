import json
from pathlib import Path

import numpy as np
import orjson
from fastapi.testclient import TestClient

# 準備 8x10.json（取前幾筆樣本）
json_file = Path("data_archives/8x10.json")
if not json_file.exists():
    items = []
    with Path("data_archives/8x10.jsonl").open("rb") as f:
        for _ in range(3):
            line = f.readline()
            if not line:
                break
            items.append(orjson.loads(line))
    json.dump(items, json_file.open("w", encoding="utf-8"))

from app import (  # noqa: E402
    BLANK_VALUE,
    _preload_memories,
    app,
    memories,
    memory_files,
)

_preload_memories()
client = TestClient(app)


def _check_shape(rows: int, cols: int) -> None:
    data = json.load(open(f"data_archives/{rows}x{cols}.json", "r", encoding="utf-8"))
    board = np.array(data[0]["board"], dtype=int)
    board.flat[:3] = BLANK_VALUE
    target = int(data[0]["target"])
    resp = client.post("/predict", json={"board": board.tolist(), "target": target})
    assert resp.status_code == 200
    res = resp.json()
    blank_count = int(np.sum(board == BLANK_VALUE))
    assert len(res) == min(3, blank_count)


def test_preloaded_memories_multi_shape() -> None:
    assert (4, 5) in memories and (4, 5) not in memory_files
    assert (8, 10) in memories and (8, 10) not in memory_files
    _check_shape(4, 5)
    _check_shape(8, 10)
