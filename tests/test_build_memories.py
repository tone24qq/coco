import json

import numpy as np

import build_memories


def test_build_memories_supports_json_and_jsonl(tmp_path):
    base = tmp_path / "data_archives"
    base.mkdir()
    # Create small JSON archive
    data_json = [{"board": [[1, 2], [3, 4]], "target": 1}]
    (base / "2x2.json").write_text(json.dumps(data_json), encoding="utf-8")
    # Create small JSONL archive
    data_jsonl = [{"board": [[1, -1], [2, 3]], "target": 2}]
    with (base / "2x3.jsonl").open("wb") as f:
        for obj in data_jsonl:
            f.write(json.dumps(obj).encode("utf-8") + b"\n")
    count = build_memories.build_all_memories(base)
    assert count == 2
    npz1 = base / "2x2_memory.npz"
    npz2 = base / "2x3_memory.npz"
    assert npz1.exists() and npz2.exists()
    keys = np.load(npz1)["keys"]
    assert keys.shape[0] == len(data_json)
