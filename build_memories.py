"""Build pre-computed memory banks for all JSON/JSONL archives.

The script scans ``data_archives`` for files named ``{rows}x{cols}.json`` or
``{rows}x{cols}.jsonl`` and generates a corresponding compressed ``npz`` memory
file ``{rows}x{cols}_memory.npz`` for each unique shape.  It prints Chinese log
messages showing how many samples are processed for each size.

The number of samples loaded from each archive can be limited by setting the
environment variable ``MEMORY_SAMPLE_LIMIT``.  When specified, at most that many
records are read from the archive.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import orjson

from agents.memory_agent import build_memory as build_memory_agent
from app import _create_model

BASE = Path("data_archives")


def _iter_archives() -> list[tuple[str, Path]]:
    """Return mapping of shape ``"{rows}x{cols}"`` to archive path.

    If both ``.json`` and ``.jsonl`` exist for the same shape, the ``.jsonl``
    file takes precedence.
    """

    mapping: dict[str, Path] = {}
    for p in BASE.glob("*x*.json"):
        mapping[p.stem] = p
    for p in BASE.glob("*x*.jsonl"):
        mapping[p.stem] = p  # prefer jsonl if both exist
    return sorted(mapping.items())


def _load_records(path: Path, limit: int | None) -> list[dict[str, object]]:
    """Load records from ``path`` respecting ``limit`` if provided."""

    records: list[dict[str, object]] = []
    if path.suffix == ".jsonl":
        with path.open("rb") as fh:
            for idx, line in enumerate(fh):
                if limit is not None and idx >= limit:
                    break
                records.append(orjson.loads(line))
    else:
        records = json.load(path.open("r", encoding="utf-8"))
        if limit is not None and len(records) > limit:
            records = records[:limit]
    return records


def main() -> None:
    """Entry-point that builds memories for all archives under ``BASE``."""

    limit_env = os.getenv("MEMORY_SAMPLE_LIMIT")
    limit = int(limit_env) if limit_env and limit_env.isdigit() else None

    for shape, path in _iter_archives():
        rows, cols = map(int, shape.split("x"))
        data = _load_records(path, limit)
        print(f"讀取 {shape}：共 {len(data)} 筆樣本")

        model = _create_model(rows, cols)
        if hasattr(model, "eval"):
            model.eval()

        samples = [(np.array(e["board"], dtype=int), int(e["target"])) for e in data]
        keys, values = build_memory_agent(samples, model)

        out = BASE / f"{shape}_memory.npz"
        np.savez_compressed(out, keys=keys, values=values)
        print(f"已快取 {shape}：{len(samples)} 筆樣本 → {out}")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
