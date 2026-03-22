from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.io_utils import list_shards, read_manifest, safe_read_table, safe_write_table
from src.modeling import load_ranking_dataset
from src.utils import DataContractError


def _force_sharded(df: pd.DataFrame, path: Path) -> Path:
    for limit in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
        out = safe_write_table(df, path, max_file_mb=limit, preferred_format="csv")
        if out.is_dir():
            return out
    raise AssertionError("failed to produce sharded output for test")


def _ranking_frame(issue_count: int = 4) -> pd.DataFrame:
    rows: list[dict[str, int | str | float]] = []
    for i in range(issue_count):
        issue = f"202601{i:04d}"
        for n in range(1, 81):
            rows.append(
                {
                    "issue": issue,
                    "draw_date": "2026-01-01",
                    "candidate_number": n,
                    "label": 1 if n <= 20 else 0,
                    "group_id": i,
                    "f1": float(n),
                }
            )
    return pd.DataFrame(rows)


def test_safe_write_table_shards_and_manifest(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "issue": [f"i{i}" for i in range(5000)],
            "draw_date": ["2026-01-01"] * 5000,
            "numbers": [json.dumps(list(range(1, 21)), ensure_ascii=False)] * 5000,
            "day_issue_index": [1] * 5000,
        }
    )
    out = safe_write_table(
        df,
        tmp_path / "history_processed.csv",
        max_file_mb=0.01,
        preferred_format="csv",
        producer_script="test",
    )
    assert out.is_dir()
    manifest = read_manifest(out / "manifest.json")
    assert manifest["shard_count"] >= 2
    shards = list_shards(out)
    assert len(shards) == manifest["shard_count"]
    assert all(s.stat().st_size < 100 * 1024 * 1024 for s in shards)

    read_back = safe_read_table(out)
    assert len(read_back) == len(df)


def test_safe_read_table_manifest_missing_shard_fail_fast(tmp_path: Path) -> None:
    df = _ranking_frame(40)
    out = _force_sharded(df, tmp_path / "ranking_dataset.csv")
    manifest = out / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    first = out / payload["shards"][0]
    first.unlink()
    with pytest.raises(DataContractError):
        safe_read_table(manifest)


def test_mainline_modeling_can_read_sharded_dataset(tmp_path: Path) -> None:
    df = _ranking_frame(5)
    out = safe_write_table(df, tmp_path / "ranking_dataset.csv", max_file_mb=95, preferred_format="csv")
    loaded = load_ranking_dataset(out)
    assert len(loaded) == len(df)
    assert loaded["issue"].nunique() == 5
