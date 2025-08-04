#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
預先掃描 data_archives/*.json(.l)，生成 {shape}_memory[_part{{i}}].npz，
單檔大小限制在 100MB 以內。
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import orjson

from agents.memory_agent import build_memory as build_memory_agent
from app import _create_model

BASE = Path("data_archives")
MEM_LIMIT = int(os.getenv("MEMORY_SAMPLE_LIMIT", "0")) or None
CHUNK_SIZE = 100 * 1024 * 1024  # 100MB per file


def _load_data(path: Path) -> List[dict]:
    """讀取 JSON 或 JSONL 資料，並套用 MEM_LIMIT。"""
    if path.suffix == ".jsonl":
        records: List[dict] = []
        with path.open("rb") as f:
            for i, line in enumerate(f):
                if MEM_LIMIT and i >= MEM_LIMIT:
                    break
                records.append(orjson.loads(line))
        return records
    data = json.load(path.open("r", encoding="utf-8"))
    return data[:MEM_LIMIT] if MEM_LIMIT else data


def main() -> None:
    # 掃描所有資料檔
    mapping: Dict[str, Path] = {}
    for p in BASE.glob("*x*.json"):
        mapping[p.stem] = p
    for p in BASE.glob("*x*.jsonl"):
        mapping[p.stem] = p

    for shape, path in sorted(mapping.items()):
        rows, cols = map(int, shape.split("x"))
        data = _load_data(path)
        print(f"讀取 {shape}：共 {len(data)} 筆樣本")

        # 準備樣本與目標
        samples = []
        targets = []
        for e in data:
            samples.append((np.array(e["board"], dtype=int), int(e["target"])))
            targets.append(int(e["target"]))

        # 建立模型並產生記憶索引
        model = _create_model(rows, cols)
        if hasattr(model, "eval"):
            model.eval()
        keys, values = build_memory_agent(samples, model)
        targets_arr = np.array(targets, dtype=int)

        # 計算總位元組
        total_bytes = keys.nbytes + values.nbytes + targets_arr.nbytes
        num_parts = (total_bytes + CHUNK_SIZE - 1) // CHUNK_SIZE

        if num_parts <= 1:
            out = BASE / f"{shape}_memory.npz"
            np.savez_compressed(out, keys=keys, values=values, targets=targets_arr)
            print(f"已快取 {shape}：{len(samples)} 筆，共 {total_bytes/1024/1024:.2f} MB → {out}")
        else:
            # 每筆樣本平均位元組
            bytes_per_sample = total_bytes / len(samples)
            # 每段包含的樣本數
            per_chunk = max(1, int(CHUNK_SIZE / bytes_per_sample))
            for i in range(num_parts):
                start = i * per_chunk
                end = len(samples) if i == num_parts - 1 else (i + 1) * per_chunk
                k_chunk = keys[start:end]
                v_chunk = values[start:end]
                t_chunk = targets_arr[start:end]
                out = BASE / f"{shape}_memory_part{i+1}.npz"
                np.savez_compressed(out, keys=k_chunk, values=v_chunk, targets=t_chunk)
                size_mb = (k_chunk.nbytes + v_chunk.nbytes + t_chunk.nbytes) / 1024 / 1024
                print(
                    f"已快取 {shape} 分段 {i+1}/{num_parts}："
                    f"{end-start} 筆，共 {size_mb:.2f} MB → {out}"
                )


if __name__ == "__main__":
    main()
