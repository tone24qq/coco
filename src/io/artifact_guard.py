from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_SIZE_THRESHOLD_MIB = 95


@dataclass(frozen=True)
class ParquetWriteResult:
    mode: str
    path: str
    format: str
    compression: str
    size_bytes: int
    sharded: bool
    manifest_path: str | None = None


def _file_size(path: Path) -> int:
    return int(path.stat().st_size) if path.exists() else 0


def _source_hash(df: pd.DataFrame) -> str:
    if df.empty:
        return hashlib.sha256(b"empty").hexdigest()
    keys = ["issue", "draw_date", "numbers"]
    cols = [c for c in keys if c in df.columns]
    if not cols:
        payload = f"rows={len(df)}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    sample = df[cols].head(2000).to_json(force_ascii=False, orient="records")
    payload = f"rows={len(df)}|sample={sample}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def benchmark_parquet_codecs(
    df: pd.DataFrame,
    *,
    tmp_dir: Path,
    codecs: tuple[str, ...] = ("zstd", "snappy"),
    sample_rows: int = 200_000,
) -> dict[str, dict[str, float]]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    sample = df.head(min(sample_rows, len(df))).copy()
    out: dict[str, dict[str, float]] = {}
    for codec in codecs:
        probe_path = tmp_dir / f"codec_probe_{codec}.parquet"
        start_write = time.perf_counter()
        sample.to_parquet(probe_path, index=False, compression=codec)
        write_ms = (time.perf_counter() - start_write) * 1000

        start_read = time.perf_counter()
        _ = pd.read_parquet(probe_path)
        read_ms = (time.perf_counter() - start_read) * 1000

        out[codec] = {
            "size_bytes": float(_file_size(probe_path)),
            "write_ms": float(write_ms),
            "read_ms": float(read_ms),
        }
        probe_path.unlink(missing_ok=True)
    return out


def choose_parquet_codec(bench: dict[str, dict[str, float]]) -> str:
    if not bench:
        return "zstd"

    codecs = list(bench.keys())
    if "zstd" not in codecs:
        return codecs[0]
    if "snappy" not in codecs:
        return "zstd"

    zstd = bench["zstd"]
    snappy = bench["snappy"]

    if (
        zstd["size_bytes"] <= snappy["size_bytes"] * 0.85
        and zstd["read_ms"] <= snappy["read_ms"] * 1.4
    ):
        return "zstd"
    return "snappy"


def write_parquet_with_size_guard(
    df: pd.DataFrame,
    *,
    output_path: Path,
    artifact_mode: str = "runtime",
    preferred_codec: str = "zstd",
    size_threshold_mib: int = DEFAULT_SIZE_THRESHOLD_MIB,
    shard_rows: int = 250_000,
) -> tuple[ParquetWriteResult, dict]:
    if artifact_mode not in {"runtime", "export"}:
        raise ValueError("artifact_mode must be runtime or export")

    threshold_bytes = int(size_threshold_mib * 1024 * 1024)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    codec_order = [preferred_codec, "zstd", "snappy"]
    codec_order = [
        c
        for i, c in enumerate(codec_order)
        if c and c in {"zstd", "snappy"} and c not in codec_order[:i]
    ]

    used_codec = codec_order[0] if codec_order else "zstd"
    for codec in codec_order:
        try:
            df.to_parquet(output_path, index=False, compression=codec)
            used_codec = codec
            break
        except (ImportError, ValueError, OSError):
            continue

    size_bytes = _file_size(output_path)
    summary = {
        "artifact_mode": artifact_mode,
        "size_threshold_bytes": threshold_bytes,
        "output_size_bytes": size_bytes,
    }

    if artifact_mode == "export" and size_bytes > threshold_bytes:
        # deterministic sharding for export/share only
        output_path.unlink(missing_ok=True)
        shard_dir = output_path.with_suffix("")
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_paths: list[str] = []
        total = len(df)
        for idx, start in enumerate(range(0, total, shard_rows)):
            end = min(start + shard_rows, total)
            shard = df.iloc[start:end].copy()
            shard_path = shard_dir / f"part-{idx:05d}.parquet"
            shard.to_parquet(shard_path, index=False, compression=used_codec)
            shard_paths.append(str(shard_path))

        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": int(len(df)),
            "schema": {c: str(t) for c, t in df.dtypes.items()},
            "compression": used_codec,
            "shards": [
                {"path": p, "size_bytes": _file_size(Path(p))}
                for p in sorted(shard_paths)
            ],
            "source_hash": _source_hash(df),
        }
        manifest_path = shard_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        result = ParquetWriteResult(
            mode=artifact_mode,
            path=str(shard_dir),
            format="parquet_dataset",
            compression=used_codec,
            size_bytes=sum(int(x["size_bytes"]) for x in manifest["shards"]),
            sharded=True,
            manifest_path=str(manifest_path),
        )
        summary["sharded"] = True
        summary["manifest_path"] = str(manifest_path)
        return result, summary

    result = ParquetWriteResult(
        mode=artifact_mode,
        path=str(output_path),
        format="parquet",
        compression=used_codec,
        size_bytes=size_bytes,
        sharded=False,
    )
    summary["sharded"] = False
    return result, summary


def is_over_threshold(
    path: Path, threshold_mib: int = DEFAULT_SIZE_THRESHOLD_MIB
) -> bool:
    threshold_bytes = int(threshold_mib * 1024 * 1024)
    return _file_size(path) > threshold_bytes
