from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

SOFT_THRESHOLD_MB = 95
HARD_LIMIT_MB = 100


class SafeIOError(ValueError):
    pass


@dataclass
class SafeWriteConfig:
    max_file_mb: int = HARD_LIMIT_MB
    soft_threshold_mb: int = SOFT_THRESHOLD_MB
    producer_script: str = "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bytes_to_mb(size: int) -> float:
    return float(size) / (1024.0 * 1024.0)


def _assert_under_limit(path: Path, max_file_mb: int) -> None:
    size_mb = _bytes_to_mb(path.stat().st_size)
    if size_mb >= float(max_file_mb):
        raise SafeIOError(f"file exceeds hard limit {max_file_mb}MB: {path} ({size_mb:.2f}MB)")


def _manifest_payload(
    *,
    fmt: str,
    shards: Sequence[Dict[str, Any]],
    row_count: int,
    columns: Sequence[str],
    producer_script: str,
) -> Dict[str, Any]:
    return {
        "format": fmt,
        "shard_count": len(shards),
        "shards": list(shards),
        "row_count": int(row_count),
        "columns": list(columns),
        "created_at": _now_iso(),
        "producer_script": producer_script,
    }


def _write_manifest(dataset_dir: Path, payload: Dict[str, Any], max_file_mb: int) -> Path:
    manifest = dataset_dir / "manifest.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _assert_under_limit(manifest, max_file_mb)
    return manifest


def write_dataframe_safe(
    df: pd.DataFrame,
    output: Path,
    fmt: str,
    config: SafeWriteConfig,
    shard_rows: int = 0,
) -> Dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt not in {"parquet", "csv", "csv.gz", "jsonl"}:
        raise SafeIOError(f"unsupported format: {fmt}")

    should_shard = shard_rows > 0 or len(df) > 0
    if not should_shard:
        raise SafeIOError("empty dataframe is not allowed for safe write")

    def _write_single(path: Path, frame: pd.DataFrame) -> None:
        if fmt == "parquet":
            frame.to_parquet(path, index=False)
        elif fmt == "csv":
            frame.to_csv(path, index=False)
        elif fmt == "csv.gz":
            frame.to_csv(path, index=False, compression="gzip")
        else:
            frame.to_json(path, orient="records", lines=True, force_ascii=False)

    tmp_single = output
    _write_single(tmp_single, df)
    size_mb = _bytes_to_mb(tmp_single.stat().st_size)
    if size_mb < float(config.soft_threshold_mb):
        _assert_under_limit(tmp_single, config.max_file_mb)
        return {
            "type": "file",
            "path": str(tmp_single),
            "size_mb": size_mb,
            "row_count": int(len(df)),
            "columns": list(df.columns),
        }

    if output.suffix:
        dataset_dir = output.with_suffix("")
    else:
        dataset_dir = output
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if shard_rows <= 0:
        # rough estimate to keep each shard under soft threshold
        shard_rows = max(1, int(len(df) * (config.soft_threshold_mb / max(size_mb, 1e-6) * 0.9)))

    shards: List[Dict[str, Any]] = []
    for start in range(0, len(df), shard_rows):
        shard_idx = start // shard_rows
        suffix = {
            "parquet": ".parquet",
            "csv": ".csv",
            "csv.gz": ".csv.gz",
            "jsonl": ".jsonl",
        }[fmt]
        shard_path = dataset_dir / f"shard_{shard_idx:05d}{suffix}"
        part = df.iloc[start : start + shard_rows]
        _write_single(shard_path, part)
        _assert_under_limit(shard_path, config.max_file_mb)
        shards.append({"path": str(shard_path), "rows": int(len(part))})

    payload = _manifest_payload(
        fmt=fmt,
        shards=shards,
        row_count=len(df),
        columns=df.columns,
        producer_script=config.producer_script,
    )
    _write_manifest(dataset_dir, payload, config.max_file_mb)

    # remove single file if it is too large and sharded output exists
    if tmp_single.exists() and tmp_single.is_file() and tmp_single.parent == output.parent and tmp_single == output:
        tmp_single.unlink(missing_ok=True)

    return {
        "type": "dataset_dir",
        "path": str(dataset_dir),
        "manifest": str(dataset_dir / "manifest.json"),
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "shard_count": len(shards),
    }


def write_jsonl_records_safe(
    rows: Iterable[Dict[str, Any]],
    output: Path,
    config: SafeWriteConfig,
    shard_rows: int = 50000,
) -> Dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    df = pd.DataFrame(rows_list)
    return write_dataframe_safe(df, output, "jsonl", config=config, shard_rows=shard_rows)


def write_dataframe_chunks_safe(
    chunks: Iterable[pd.DataFrame],
    output: Path,
    fmt: str,
    config: SafeWriteConfig,
    on_chunk: Any = None,
) -> Dict[str, Any]:
    if fmt not in {"parquet", "csv", "csv.gz", "jsonl"}:
        raise SafeIOError(f"unsupported format: {fmt}")
    output.mkdir(parents=True, exist_ok=True)
    shards: List[Dict[str, Any]] = []
    total_rows = 0
    all_columns: List[str] = []

    def _write(path: Path, frame: pd.DataFrame) -> None:
        if fmt == "parquet":
            frame.to_parquet(path, index=False)
        elif fmt == "csv":
            frame.to_csv(path, index=False)
        elif fmt == "csv.gz":
            frame.to_csv(path, index=False, compression="gzip")
        else:
            frame.to_json(path, orient="records", lines=True, force_ascii=False)

    suffix = {"parquet": ".parquet", "csv": ".csv", "csv.gz": ".csv.gz", "jsonl": ".jsonl"}[fmt]
    for idx, chunk in enumerate(chunks):
        if chunk is None or chunk.empty:
            continue
        if on_chunk is not None:
            on_chunk(chunk)
        for c in chunk.columns:
            if c not in all_columns:
                all_columns.append(c)
        shard_path = output / f"shard_{idx:05d}{suffix}"
        _write(shard_path, chunk)
        _assert_under_limit(shard_path, config.max_file_mb)
        rows = int(len(chunk))
        total_rows += rows
        shards.append({"path": str(shard_path), "rows": rows})

    if not shards:
        raise SafeIOError("no chunk data to write")

    payload = _manifest_payload(
        fmt=fmt,
        shards=shards,
        row_count=total_rows,
        columns=all_columns,
        producer_script=config.producer_script,
    )
    manifest = _write_manifest(output, payload, config.max_file_mb)
    return {
        "type": "dataset_dir",
        "path": str(output),
        "manifest": str(manifest),
        "row_count": int(total_rows),
        "columns": all_columns,
        "shard_count": len(shards),
    }


def read_dataset_auto(path: Path) -> pd.DataFrame:
    if path.is_file():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffixes[-2:] == [".csv", ".gz"]:
            return pd.read_csv(path, compression="gzip")
        if path.suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        raise SafeIOError(f"unsupported file format: {path}")

    manifest = path / "manifest.json"
    if not manifest.exists():
        raise SafeIOError(f"dataset dir missing manifest.json: {path}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    shards = data.get("shards", [])
    fmt = data.get("format")
    if not shards:
        raise SafeIOError(f"manifest has no shards: {manifest}")
    frames: List[pd.DataFrame] = []
    for item in shards:
        p = Path(item["path"])
        if fmt == "parquet":
            frames.append(pd.read_parquet(p))
        elif fmt == "csv":
            frames.append(pd.read_csv(p))
        elif fmt == "csv.gz":
            frames.append(pd.read_csv(p, compression="gzip"))
        elif fmt == "jsonl":
            frames.append(pd.read_json(p, lines=True))
        else:
            raise SafeIOError(f"unsupported manifest format: {fmt}")
    return pd.concat(frames, ignore_index=True)
