from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = DEFAULT_RAW_DIR / "raw_manifest.json"

YEAR_NAME_PATTERN = re.compile(r"賓果賓果_(\d{4})\.csv$")
RANGE_NAME_PATTERN = re.compile(r"賓果賓果_(\d{4})(\d{2})_(\d{4})(\d{2})\.csv$")
UNICODE_TOKEN_PATTERN = re.compile(r"#U([0-9a-fA-F]{4})")


@dataclass
class RawFileMeta:
    path: str
    original_filename: str
    normalized_filename: str
    source_type: str
    year_start: int
    year_end: int
    month_start: int
    month_end: int


def decode_unicode_tokens(name: str) -> str:
    def _replace(m: re.Match[str]) -> str:
        return chr(int(m.group(1), 16))

    return UNICODE_TOKEN_PATTERN.sub(_replace, name)


def maybe_normalize_encoded_filename(path: Path) -> Path:
    decoded_name = decode_unicode_tokens(path.name)
    if decoded_name == path.name:
        return path
    target = path.with_name(decoded_name)
    if not target.exists():
        path.rename(target)
    return target


def _extract_meta(path: Path) -> RawFileMeta | None:
    base = path.name
    m1 = YEAR_NAME_PATTERN.match(base)
    if m1:
        y = int(m1.group(1))
        return RawFileMeta(
            path=str(path),
            original_filename=base,
            normalized_filename=f"賓果賓果_{y}.csv",
            source_type="local_csv",
            year_start=y,
            year_end=y,
            month_start=1,
            month_end=12,
        )
    m2 = RANGE_NAME_PATTERN.match(base)
    if m2:
        ys, ms, ye, me = map(int, m2.groups())
        return RawFileMeta(
            path=str(path),
            original_filename=base,
            normalized_filename=base,
            source_type="local_csv",
            year_start=ys,
            year_end=ye,
            month_start=ms,
            month_end=me,
        )
    return None


def scan_local_raw_csvs(raw_dir: Path = DEFAULT_RAW_DIR) -> list[RawFileMeta]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    metas: list[RawFileMeta] = []
    for path in sorted(raw_dir.glob("*.csv")):
        normalized_path = maybe_normalize_encoded_filename(path)
        meta = _extract_meta(normalized_path)
        if meta is not None:
            metas.append(meta)

    # fallback: legacy root-level csv files
    for path in sorted(ROOT.glob("賓果賓果_*.csv")):
        target = raw_dir / path.name
        if not target.exists():
            target.write_bytes(path.read_bytes())
        meta = _extract_meta(target)
        if meta is not None:
            metas.append(meta)

    # dedupe by normalized path
    uniq: dict[str, RawFileMeta] = {m.path: m for m in metas}
    return list(
        sorted(uniq.values(), key=lambda x: (x.year_start, x.month_start, x.path))
    )


def _count_csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return 0


def build_raw_manifest(raw_dir: Path = DEFAULT_RAW_DIR) -> dict:
    entries = scan_local_raw_csvs(raw_dir=raw_dir)
    years = set()
    for e in entries:
        years.update(range(e.year_start, e.year_end + 1))
    missing_years = [y for y in range(2008, 2027) if y not in years]
    detailed_entries = []
    total_rows = 0
    for e in entries:
        row_count = _count_csv_rows(Path(e.path))
        total_rows += row_count
        item = asdict(e)
        item["row_count"] = int(row_count)
        detailed_entries.append(item)

    payload = {
        "raw_dir": str(raw_dir),
        "file_count": len(entries),
        "total_rows": int(total_rows),
        "entries": detailed_entries,
        "coverage_year_start": min(years) if years else None,
        "coverage_year_end": max(years) if years else None,
        "missing_years": missing_years,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def load_or_build_manifest(raw_dir: Path = DEFAULT_RAW_DIR) -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return build_raw_manifest(raw_dir=raw_dir)


def load_manifest_if_exists() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def resolve_raw_csv_paths(
    years: Iterable[int] | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> list[Path]:
    manifest = load_or_build_manifest(raw_dir=raw_dir)
    entries = manifest.get("entries", [])
    paths = [Path(entry["path"]) for entry in entries]
    if years is None:
        return sorted(set(paths))

    wanted = set(int(y) for y in years)
    out: list[Path] = []
    for entry in entries:
        covered = set(range(int(entry["year_start"]), int(entry["year_end"]) + 1))
        if covered & wanted:
            out.append(Path(entry["path"]))
    out_unique = sorted(set(out))
    covered_years: set[int] = set()
    for p in out_unique:
        meta = _extract_meta(p)
        if meta:
            covered_years.update(range(meta.year_start, meta.year_end + 1))
    missing = sorted(wanted - covered_years)
    if missing:
        raise FileNotFoundError(f"missing years in local manifest: {missing}")
    return out_unique


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"missing required columns: {candidates}")


def read_raw_csv_to_standard_df(path: Path) -> pd.DataFrame:
    header_df = pd.read_csv(path, nrows=0, low_memory=False)
    existing_cols = set(header_df.columns)
    draw_number_candidates = [
        *[f"獎號{i}" for i in range(1, 21)],
        *[f"n{i}" for i in range(1, 21)],
    ]
    usecols = [
        c
        for c in [
            "期別",
            "issue",
            "開獎日期",
            "draw_date",
            "連莊球",
            "猜大小",
            "猜單雙",
        ]
        + draw_number_candidates
        if c in existing_cols
    ]
    dtype = {
        col: "string"
        for col in ["期別", "issue", "猜大小", "猜單雙", "開獎日期", "draw_date"]
        if col in existing_cols
    }
    df = pd.read_csv(path, low_memory=False, usecols=usecols, dtype=dtype)
    issue_col = _pick_col(df, ["期別", "issue"])
    date_col = _pick_col(df, ["開獎日期", "draw_date"])

    number_cols = [f"獎號{i}" for i in range(1, 21)]
    if not all(c in df.columns for c in number_cols):
        number_cols = [f"n{i}" for i in range(1, 21)]
    if not all(c in df.columns for c in number_cols):
        raise KeyError("missing draw number columns")

    out = pd.DataFrame()
    out["issue"] = pd.to_numeric(df[issue_col], errors="coerce").astype("Int64")
    out["draw_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    num_df = df[number_cols].apply(pd.to_numeric, errors="coerce")
    out["numbers"] = num_df.apply(
        lambda row: json.dumps(
            sorted([int(v) for v in row.tolist() if pd.notna(v)]), ensure_ascii=False
        ),
        axis=1,
    )
    out["numbers_draw_order"] = num_df.apply(
        lambda row: json.dumps(
            [int(v) for v in row.tolist() if pd.notna(v)], ensure_ascii=False
        ),
        axis=1,
    )
    out["draw_time"] = None
    out["consecutive_count"] = pd.to_numeric(df.get("連莊球", None), errors="coerce")
    out["size"] = df.get("猜大小", None)
    out["odd_even"] = df.get("猜單雙", None)
    out["source"] = "local_csv"
    out["source_priority"] = 1
    out["raw_file"] = path.name
    out = out.dropna(subset=["issue", "draw_date"]).copy()
    out["issue"] = out["issue"].astype(int)
    return out
