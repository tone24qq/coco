from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Sample:
    sample_id: str
    grid: np.ndarray
    answer_row: int
    answer_col: int
    shape: str
    source: str
    order_index: Optional[int] = None
    answer_value: Optional[int] = None


@dataclass
class DataAudit:
    total_samples: int
    valid_samples: int
    invalid_samples: int
    shape_distribution: Dict[str, int]
    missing_count_distribution: Dict[str, int]
    invalid_reasons: Dict[str, int]
    coordinate_mode: str


def discover_data_files(repo_root: Path) -> List[Path]:
    patterns = ["*.json", "*.jsonl", "*.csv", "*.parquet"]
    candidates: List[Path] = []
    for folder in [repo_root, repo_root / "data", repo_root / "samples", repo_root / "samples" / "data"]:
        if not folder.exists():
            continue
        for pat in patterns:
            for item in folder.rglob(pat):
                if ".venv" in item.parts or ".git" in item.parts:
                    continue
                candidates.append(item)
    return sorted(set(candidates))


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "samples" in obj and isinstance(obj["samples"], list):
            return obj["samples"]
        raise ValueError(f"Unsupported JSON shape in {path}")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported file format: {path}")


def _extract_grid(rec: Dict[str, Any]) -> np.ndarray:
    grid = rec.get("grid")
    if grid is None:
        raise ValueError("missing_grid")
    arr = np.array(grid)
    if arr.ndim != 2:
        raise ValueError("invalid_grid_dim")
    return arr.astype(int)


def _detect_coord_mode(records: List[Dict[str, Any]], grids: List[np.ndarray]) -> str:
    modes = []
    for mode in ("zero", "one"):
        ok = 0
        for rec, grid in zip(records, grids):
            r = rec.get("answer_row")
            c = rec.get("answer_col")
            if r is None or c is None:
                continue
            rr = int(r) - (1 if mode == "one" else 0)
            cc = int(c) - (1 if mode == "one" else 0)
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                ok += 1
        modes.append((mode, ok))
    modes.sort(key=lambda x: x[1], reverse=True)
    return modes[0][0]


def load_and_validate(data_path: Path) -> Tuple[List[Sample], DataAudit]:
    records = _load_records(data_path)
    if not records:
        raise ValueError("No records found")

    grids: List[np.ndarray] = []
    for rec in records:
        grids.append(_extract_grid(rec))

    coord_mode = _detect_coord_mode(records, grids)
    invalid_reasons: Dict[str, int] = {}
    samples: List[Sample] = []
    shape_dist: Dict[str, int] = {}
    miss_dist: Dict[str, int] = {}

    for idx, (rec, grid) in enumerate(zip(records, grids)):
        reason = None
        r_raw = rec.get("answer_row")
        c_raw = rec.get("answer_col")
        if r_raw is None or c_raw is None:
            reason = "missing_answer_coord"
        else:
            row = int(r_raw) - (1 if coord_mode == "one" else 0)
            col = int(c_raw) - (1 if coord_mode == "one" else 0)
            if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
                reason = "answer_out_of_bound"
            elif grid[row, col] != -1:
                reason = "answer_not_missing_cell"

        shape = f"{grid.shape[0]}x{grid.shape[1]}"
        shape_dist[shape] = shape_dist.get(shape, 0) + 1
        miss = int(np.sum(grid == -1))
        miss_dist[str(miss)] = miss_dist.get(str(miss), 0) + 1

        if reason:
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            continue

        sample = Sample(
            sample_id=str(rec.get("sample_id", f"sample_{idx}")),
            grid=grid,
            answer_row=row,
            answer_col=col,
            shape=shape,
            source=str(rec.get("source", data_path.name)),
            order_index=int(rec["order_index"]) if rec.get("order_index") is not None else None,
            answer_value=int(rec["answer_value"]) if rec.get("answer_value") is not None else None,
        )
        samples.append(sample)

    audit = DataAudit(
        total_samples=len(records),
        valid_samples=len(samples),
        invalid_samples=len(records) - len(samples),
        shape_distribution=shape_dist,
        missing_count_distribution=miss_dist,
        invalid_reasons=invalid_reasons,
        coordinate_mode=coord_mode,
    )

    if audit.valid_samples == 0:
        raise ValueError(f"Fail-fast: no valid samples. reasons={audit.invalid_reasons}")

    return samples, audit


def write_data_audit(audit: DataAudit, output_path: Path) -> None:
    payload = {
        "total_samples": audit.total_samples,
        "valid_samples": audit.valid_samples,
        "invalid_samples": audit.invalid_samples,
        "shape_distribution": audit.shape_distribution,
        "missing_count_distribution": audit.missing_count_distribution,
        "invalid_reasons": audit.invalid_reasons,
        "coordinate_mode": audit.coordinate_mode,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
