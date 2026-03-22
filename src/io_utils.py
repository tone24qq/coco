from __future__ import annotations

import gzip
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import DataContractError

HARD_LIMIT_MB = 100.0
DEFAULT_MAX_FILE_MB = 95.0


def _mb_to_bytes(size_mb: float) -> int:
    return int(size_mb * 1024 * 1024)


def _hard_limit_bytes() -> int:
    return _mb_to_bytes(HARD_LIMIT_MB)


def _ensure_hard_limit(path: Path) -> None:
    if path.exists() and path.stat().st_size >= _hard_limit_bytes():
        raise DataContractError(f"hard file size limit reached (>=100MB): {path}")


def estimate_df_size_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / (1024 * 1024))


def downcast_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_integer_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="integer")
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="float")
    return out


def optimize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_object_dtype(out[col]):
            continue
        non_null = out[col].dropna()
        if non_null.empty:
            continue
        cardinality = non_null.nunique(dropna=True)
        ratio = cardinality / max(1, len(non_null))
        if ratio <= 0.2:
            out[col] = out[col].astype("category")
    return out


def _dataset_dir_for_output(output_path: Path) -> Path:
    suffix = output_path.suffix
    if suffix:
        return output_path.with_suffix(".dataset")
    return output_path


def write_manifest(
    manifest_path: Path,
    *,
    format_name: str,
    compression: str | None,
    shards: list[Path],
    columns: list[str],
    row_count: int,
    producer_script: str,
    base_path: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": format_name,
        "compression": compression,
        "shard_count": len(shards),
        "shards": [p.name for p in shards],
        "columns": columns,
        "row_count": int(row_count),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "producer_script": producer_script,
        "base_path": str(base_path),
        "schema": [],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _ensure_hard_limit(manifest_path)
    return payload


def read_manifest(path: Path) -> dict[str, Any]:
    target = path
    if target.is_dir():
        target = target / "manifest.json"
    if not target.exists():
        raise DataContractError(f"manifest not found: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    for key in ["format", "shard_count", "shards", "columns", "row_count"]:
        if key not in payload:
            raise DataContractError(f"manifest missing key: {key}")
    if int(payload["shard_count"]) != len(payload["shards"]):
        raise DataContractError("manifest shard_count does not match shard list")
    return payload


def _write_single(df: pd.DataFrame, path: Path, format_name: str, compression: str | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if format_name == "parquet":
        kwargs: dict[str, Any] = {"index": False}
        if compression:
            kwargs["compression"] = compression
        try:
            df.to_parquet(path, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise DataContractError("failed to write parquet; install parquet engine") from exc
    elif format_name == "csv":
        df.to_csv(path, index=False, compression=compression)
    else:
        raise DataContractError(f"unsupported format: {format_name}")
    return path


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _format_choice(preferred_format: str) -> tuple[str, str | None, str]:
    if preferred_format.lower() == "parquet":
        return "parquet", "snappy", ".parquet"
    return "csv", "gzip", ".csv.gz"


def _iterative_shard_count(df: pd.DataFrame, max_file_mb: float) -> int:
    est = max(1.0, estimate_df_size_mb(df))
    return max(2, int(math.ceil(est / max(1.0, max_file_mb * 0.6))))


def safe_write_table(
    df: pd.DataFrame,
    output_path: Path | str,
    *,
    max_file_mb: float = DEFAULT_MAX_FILE_MB,
    preferred_format: str = "parquet",
    producer_script: str = "unknown",
) -> Path:
    if df.empty:
        raise DataContractError("table output is empty")
    if max_file_mb >= HARD_LIMIT_MB:
        raise DataContractError("max_file_mb must be less than 100MB hard limit")

    out = Path(output_path)
    opt = optimize_object_columns(downcast_numeric_dtypes(df))
    format_name, compression, ext = _format_choice(preferred_format)

    single_path = out
    if format_name == "parquet" and single_path.suffix != ".parquet":
        single_path = single_path.with_suffix(".parquet")
    if format_name == "csv" and not single_path.name.endswith(".csv.gz"):
        single_path = single_path.with_suffix(".csv.gz")

    try:
        _write_single(opt, single_path, format_name, compression)
        file_mb = _size(single_path) / (1024 * 1024)
        _ensure_hard_limit(single_path)
        if file_mb <= max_file_mb:
            return single_path
    except DataContractError:
        if format_name == "parquet":
            format_name, compression, ext = _format_choice("csv")
            single_path = out.with_suffix(".csv.gz")
            _write_single(opt, single_path, format_name, compression)
            file_mb = _size(single_path) / (1024 * 1024)
            _ensure_hard_limit(single_path)
            if file_mb <= max_file_mb:
                return single_path
        else:
            raise

    if single_path.exists():
        single_path.unlink()

    dataset_dir = _dataset_dir_for_output(out)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    shard_count = _iterative_shard_count(opt, max_file_mb)
    while True:
        shard_paths: list[Path] = []
        for p in dataset_dir.glob(f"shard-*{ext}"):
            p.unlink()

        rows_per = max(1, int(math.ceil(len(opt) / shard_count)))
        for i in range(shard_count):
            start = i * rows_per
            if start >= len(opt):
                break
            end = min((i + 1) * rows_per, len(opt))
            shard_df = opt.iloc[start:end]
            shard_path = dataset_dir / f"shard-{i:05d}{ext}"
            _write_single(shard_df, shard_path, format_name, compression)
            _ensure_hard_limit(shard_path)
            shard_paths.append(shard_path)

        oversize = [p for p in shard_paths if _size(p) > _mb_to_bytes(max_file_mb)]
        if not oversize:
            write_manifest(
                dataset_dir / "manifest.json",
                format_name=format_name,
                compression=compression,
                shards=shard_paths,
                columns=list(opt.columns),
                row_count=len(opt),
                producer_script=producer_script,
                base_path=dataset_dir,
            )
            return dataset_dir
        shard_count *= 2
        if shard_count > len(opt):
            raise DataContractError("unable to shard table under max_file_mb")


def safe_read_table(path: Path | str) -> pd.DataFrame:
    target = Path(path)
    if target.is_dir():
        manifest_path = target / "manifest.json"
        if manifest_path.exists():
            payload = read_manifest(manifest_path)
            shards = [target / n for n in payload["shards"]]
            for s in shards:
                if not s.exists():
                    raise DataContractError(f"manifest shard missing: {s}")
            return _read_shards(shards, payload["format"])
        parquet_files = sorted(target.glob("*.parquet"))
        if parquet_files:
            return pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
        raise DataContractError(f"unsupported dataset directory: {target}")

    if target.name == "manifest.json":
        payload = read_manifest(target)
        base = target.parent
        shards = [base / n for n in payload["shards"]]
        for s in shards:
            if not s.exists():
                raise DataContractError(f"manifest shard missing: {s}")
        return _read_shards(shards, payload["format"])

    if target.exists():
        if target.suffix == ".csv":
            return pd.read_csv(target)
        if target.name.endswith(".csv.gz"):
            return pd.read_csv(target, compression="gzip")
        if target.suffix == ".parquet":
            return pd.read_parquet(target)
        raise DataContractError(f"unsupported table input format: {target}")

    legacy_parts = sorted(target.parent.glob(f"{target.stem}.part*{target.suffix}"))
    if legacy_parts:
        return pd.concat([pd.read_csv(p) for p in legacy_parts], ignore_index=True)

    fallback_dataset = target.with_suffix(".dataset")
    if fallback_dataset.exists():
        return safe_read_table(fallback_dataset)

    fallback_parquet = target.with_suffix(".parquet")
    if fallback_parquet.exists():
        return pd.read_parquet(fallback_parquet)

    fallback_csv_gz = target.with_suffix(".csv.gz")
    if fallback_csv_gz.exists():
        return pd.read_csv(fallback_csv_gz, compression="gzip")

    raise DataContractError(f"table input not found: {target}")


def _read_shards(shards: list[Path], format_name: str) -> pd.DataFrame:
    if format_name == "parquet":
        return pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    if format_name == "csv":
        return pd.concat([pd.read_csv(p, compression="gzip") for p in shards], ignore_index=True)
    raise DataContractError(f"unsupported shard format: {format_name}")


def write_json_gz_if_needed(
    data: Any,
    output_path: Path | str,
    *,
    max_file_mb: float = DEFAULT_MAX_FILE_MB,
    producer_script: str = "unknown",
) -> Path:
    out = Path(output_path)
    if max_file_mb >= HARD_LIMIT_MB:
        raise DataContractError("max_file_mb must be less than 100MB hard limit")

    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out if out.suffix == ".json" else out.with_suffix(".json")
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    _ensure_hard_limit(json_path)
    if _size(json_path) <= _mb_to_bytes(max_file_mb):
        return json_path

    json_path.unlink(missing_ok=True)

    gz_path = out if out.name.endswith(".json.gz") else out.with_suffix(".json.gz")
    with gzip.open(gz_path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(data, ensure_ascii=False))
    _ensure_hard_limit(gz_path)
    if _size(gz_path) <= _mb_to_bytes(max_file_mb):
        return gz_path

    gz_path.unlink(missing_ok=True)

    if not isinstance(data, list):
        raise DataContractError(
            "json payload exceeds max size and cannot be sharded (payload must be list)"
        )

    dataset_dir = _dataset_dir_for_output(out)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    shard_count = 2
    while True:
        shard_paths: list[Path] = []
        for p in dataset_dir.glob("shard-*.json.gz"):
            p.unlink()
        per = max(1, int(math.ceil(len(data) / shard_count)))
        for i in range(shard_count):
            start = i * per
            if start >= len(data):
                break
            end = min((i + 1) * per, len(data))
            shard = dataset_dir / f"shard-{i:05d}.json.gz"
            with gzip.open(shard, "wt", encoding="utf-8") as fh:
                fh.write(json.dumps(data[start:end], ensure_ascii=False))
            _ensure_hard_limit(shard)
            shard_paths.append(shard)
        oversize = [p for p in shard_paths if _size(p) > _mb_to_bytes(max_file_mb)]
        if not oversize:
            write_manifest(
                dataset_dir / "manifest.json",
                format_name="json",
                compression="gzip",
                shards=shard_paths,
                columns=[],
                row_count=len(data),
                producer_script=producer_script,
                base_path=dataset_dir,
            )
            return dataset_dir
        shard_count *= 2
        if shard_count > len(data):
            raise DataContractError("unable to shard json payload under max_file_mb")


def list_shards(path_or_dir: Path | str) -> list[Path]:
    base = Path(path_or_dir)
    if base.is_dir() and (base / "manifest.json").exists():
        payload = read_manifest(base / "manifest.json")
        return [base / n for n in payload["shards"]]
    if base.name == "manifest.json":
        payload = read_manifest(base)
        return [base.parent / n for n in payload["shards"]]
    return [base]
