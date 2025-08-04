"""Generate memory bank NPZ files from JSON/JSONL archives.

This script scans the ``data_archives`` directory for files named
``{rows}x{cols}.json`` or ``{rows}x{cols}.jsonl``. For each file it reads all
records, builds a memory bank using :func:`agents.memory_agent.build_memory`,
then writes ``{rows}x{cols}_memory.npz``. Chinese logs report the number of
samples processed for each shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import orjson

from agents.memory_agent import build_memory as build_memory_agent
from app import MEMORY_SAMPLE_LIMIT, _create_model

BASE = Path("data_archives")


def _iter_data_files() -> Dict[str, Path]:
    """Return mapping from shape (e.g. ``"4x5"``) to data file path.

    ``.json`` files take precedence over ``.jsonl`` if both exist.
    """
    files: Dict[str, Path] = {}
    for path in BASE.glob("*x*.jsonl"):
        files[path.stem] = path
    for path in BASE.glob("*x*.json"):
        files[path.stem] = path  # prefer json over jsonl
    return files


def main() -> None:
    files = _iter_data_files()
    if not files:
        print("未找到任何資料檔")
        return
    for shape, path in sorted(files.items()):
        rows, cols = map(int, shape.split("x"))
        limit = MEMORY_SAMPLE_LIMIT or 0
        if path.suffix == ".jsonl":
            data = []
            with path.open("rb") as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    data.append(orjson.loads(line))
        else:
            data = json.load(path.open("r", encoding="utf-8"))
            if limit and len(data) > limit:
                data = data[:limit]
        print(f"尺寸 {shape}：讀取 {len(data)} 筆樣本")
        model = _create_model(rows, cols)
        if hasattr(model, "eval"):
            model.eval()
        samples = [(np.array(e["board"], dtype=int), int(e["target"])) for e in data]
        keys, values = build_memory_agent(samples, model)
        out = BASE / f"{shape}_memory.npz"
        np.savez_compressed(out, keys=keys, values=values)
        print(f"已快取 {shape}：{len(samples)} 筆 → {out}")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
