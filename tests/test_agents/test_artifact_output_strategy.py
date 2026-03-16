from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import src.io.canonical_dataset as canonical_module
import src.io.raw_resolver as raw_resolver
from src.io.artifact_guard import write_parquet_with_size_guard
from src.io.canonical_dataset import build_canonical_dataset


def test_size_guard_shards_only_in_export_mode(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "issue": list(range(1, 4000)),
            "draw_date": ["2026-01-01"] * 3999,
            "numbers": [json.dumps(list(range(1, 21)), ensure_ascii=False)] * 3999,
        }
    )
    out_path = tmp_path / "big.parquet"
    result, summary = write_parquet_with_size_guard(
        df,
        output_path=out_path,
        artifact_mode="export",
        size_threshold_mib=0,
        shard_rows=500,
    )
    assert summary["sharded"] is True
    assert result.sharded is True
    assert result.manifest_path is not None
    assert Path(result.manifest_path).exists()


def test_canonical_runtime_writes_parquet_and_disables_csv(
    tmp_path: Path, monkeypatch
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    sample_csv = raw_dir / "賓果賓果_2026.csv"
    sample_csv.write_text(
        "期別,開獎日期,連莊球,猜大小,猜單雙,"
        + ",".join([f"獎號{i}" for i in range(1, 21)])
        + "\n"
        + "115000001,2026-01-01,3,大,單,"
        + ",".join([str(i) for i in range(1, 21)])
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(raw_resolver, "ROOT", tmp_path)
    monkeypatch.setattr(
        canonical_module, "CANONICAL_PARQUET", tmp_path / "canonical.parquet"
    )
    monkeypatch.setattr(canonical_module, "CANONICAL_CSV", tmp_path / "canonical.csv")
    monkeypatch.setattr(canonical_module, "AUDIT_JSON", tmp_path / "audit.json")

    df, audit = build_canonical_dataset(raw_dir=raw_dir, artifact_mode="runtime")

    assert len(df) == 1
    assert (tmp_path / "canonical.parquet").exists()
    assert not (tmp_path / "canonical.csv").exists()
    assert audit["output_format"] in {"parquet", "parquet_dataset"}
