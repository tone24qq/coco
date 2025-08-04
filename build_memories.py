"""Utility script to precompute memory caches for all data archives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import orjson

from agents.memory_agent import build_memory as build_memory_agent
from app import _create_model

BASE = Path("data_archives")


def build_all_memories(base: Path = BASE, shapes: set[str] | None = None) -> int:
    """Scan ``base`` for JSON/JSONL archives and build memory caches.

    Parameters
    ----------
    base:
        Directory containing ``*x*.json`` or ``*x*.jsonl`` files.
    shapes:
        Optional set of shape strings (e.g. ``{"4x5"}``) to restrict which
        archives are processed.

    Returns
    -------
    int
        Number of memory files generated.
    """
    archives: dict[str, Path] = {}
    for path in list(base.glob("*x*.jsonl")) + list(base.glob("*x*.json")):
        archives[path.stem] = path

    count = 0
    for shape, path in archives.items():
        if shapes is not None and shape not in shapes:
            continue
        rows, cols = map(int, shape.split("x"))
        if path.suffix == ".jsonl":
            data = [orjson.loads(line) for line in path.open("rb")]
        else:
            data = json.load(path.open("r", encoding="utf-8"))
        model = _create_model(rows, cols)
        if hasattr(model, "eval"):
            model.eval()
        samples = [(np.array(e["board"], dtype=int), int(e["target"])) for e in data]
        keys, values = build_memory_agent(samples, model)
        out = base / f"{shape}_memory.npz"
        np.savez_compressed(out, keys=keys, values=values)
        print(f"已快取 {shape}：共 {len(samples)} 筆 → {out}")
        count += 1
    return count


if __name__ == "__main__":
    build_all_memories()
