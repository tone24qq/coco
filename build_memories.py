#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""預先掃描 data_archives/*.json(.l)，生成 {rows}x{cols}_memory.npz。

讀取每個檔案的筆數並寫出對應的記憶庫快取，過程中會打印
中文日誌方便觀察流程。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import orjson

from agents.memory_agent import build_memory as build_memory_agent
from app import _create_model

try:  # 進度條：若未安裝 tqdm，仍可正常運作
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm missing

    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


BASE = Path("data_archives")
MEM_LIMIT = int(os.getenv("MEMORY_SAMPLE_LIMIT", "0")) or None


def _load_data(path: Path) -> list[dict]:
    """讀取 ``path`` 指定的 JSON 或 JSONL 檔案。"""
    if path.suffix == ".jsonl":
        records = []
        with path.open("rb") as f:
            for i, line in enumerate(f):
                if MEM_LIMIT and i >= MEM_LIMIT:
                    break
                records.append(orjson.loads(line))
        return records
    data = json.load(path.open("r", encoding="utf-8"))
    return data[:MEM_LIMIT] if MEM_LIMIT else data


def main() -> None:
    """掃描資料目錄，建立對應的記憶庫快取。"""
    mapping: Dict[str, Path] = {}
    for p in BASE.glob("*x*.json"):
        mapping[p.stem] = p
    for p in BASE.glob("*x*.jsonl"):
        mapping[p.stem] = p

    for shape, path in sorted(mapping.items()):
        rows, cols = map(int, shape.split("x"))
        data = _load_data(path)
        print(f"讀取 {shape}：共 {len(data)} 筆樣本")

        samples = []
        targets = []
        for e in tqdm(
            data,
            desc=f"建構樣本 {shape}",
            unit="筆",
        ):  # 中文註解：顯示樣本生成進度條
            samples.append((np.array(e["board"], dtype=int), int(e["target"])))
            targets.append(int(e["target"]))

        model = _create_model(rows, cols)
        if hasattr(model, "eval"):
            model.eval()
        keys, values = build_memory_agent(samples, model)

        out = BASE / f"{shape}_memory.npz"
        np.savez_compressed(out, keys=keys, values=values, targets=np.array(targets))
        print(f"已快取 {shape}：{len(samples)} 筆 → {out}")


if __name__ == "__main__":
    main()
