from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from src.utils import DataContractError, DrawRecord, ensure_numbers, parse_date


_REQUIRED_COLUMNS = {"issue", "draw_date", "numbers", "day_issue_index"}


def resolve_processed_source_files(input_path: Path) -> list[Path]:
    if input_path.exists():
        return [input_path]
    parts = sorted(input_path.parent.glob(f"{input_path.stem}.part*{input_path.suffix}"))
    if parts:
        return parts
    raise DataContractError("processed history missing; build processed history before deploy")


def _iter_processed_rows(paths: list[Path]) -> Iterator[dict[str, str]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            cols = set(reader.fieldnames or [])
            if not _REQUIRED_COLUMNS.issubset(cols):
                raise DataContractError(f"processed CSV missing columns: {_REQUIRED_COLUMNS}")
            for row in reader:
                yield row


def build_runtime_history_artifact(processed_csv_or_shards: Path, output_dir: Path) -> Path:
    source_files = resolve_processed_source_files(processed_csv_or_shards)

    row_count = 0
    issue_width = 16
    for row in _iter_processed_rows(source_files):
        row_count += 1
        issue_width = max(issue_width, len(str(row["issue"])))

    if row_count == 0:
        raise DataContractError("processed history exists but contains no rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    numbers_path = output_dir / "numbers.npy"
    issue_path = output_dir / "issue.npy"
    draw_date_path = output_dir / "draw_date_ordinal.npy"
    day_idx_path = output_dir / "day_issue_index.npy"

    numbers = np.lib.format.open_memmap(numbers_path, mode="w+", dtype=np.uint8, shape=(row_count, 20))
    issues = np.lib.format.open_memmap(issue_path, mode="w+", dtype=f"<U{issue_width}", shape=(row_count,))
    draw_ord = np.lib.format.open_memmap(draw_date_path, mode="w+", dtype=np.int32, shape=(row_count,))
    day_idx = np.lib.format.open_memmap(day_idx_path, mode="w+", dtype=np.uint16, shape=(row_count,))

    i = 0
    for row in _iter_processed_rows(source_files):
        issues[i] = str(row["issue"])
        draw_ord[i] = parse_date(str(row["draw_date"])).toordinal()
        numbers[i] = np.array(ensure_numbers(json.loads(str(row["numbers"]))), dtype=np.uint8)
        day_idx[i] = int(row["day_issue_index"])
        i += 1

    meta = {
        "row_count": row_count,
        "issue_dtype": f"<U{issue_width}",
        "source_files": [str(p) for p in source_files],
        "schema_version": 1,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


@dataclass(frozen=True)
class HistoryView(Sequence[DrawRecord]):
    store: "HistoryStore"
    start: int
    stop: int
    step: int

    def __len__(self) -> int:
        return len(range(self.start, self.stop, self.step))

    def __getitem__(self, idx: int | slice) -> DrawRecord | "HistoryView":
        rng = range(self.start, self.stop, self.step)
        if isinstance(idx, slice):
            sub = rng[idx]
            return HistoryView(self.store, sub.start, sub.stop, sub.step)
        return self.store[rng[idx]]


class HistoryStore(Sequence[DrawRecord]):
    def __init__(self, root: Path):
        meta_path = root / "meta.json"
        if not meta_path.exists():
            raise DataContractError(f"runtime history meta missing: {meta_path}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._numbers = np.load(root / "numbers.npy", mmap_mode="r")
        self._issues = np.load(root / "issue.npy", mmap_mode="r")
        self._draw_ord = np.load(root / "draw_date_ordinal.npy", mmap_mode="r")
        self._day_idx = np.load(root / "day_issue_index.npy", mmap_mode="r")

    def __len__(self) -> int:
        return int(self._numbers.shape[0])

    def __getitem__(self, idx: int | slice) -> DrawRecord | HistoryView:
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return HistoryView(self, start, stop, step)
        i = int(idx)
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            raise IndexError(i)
        return DrawRecord(
            issue=str(self._issues[i]),
            draw_date=date.fromordinal(int(self._draw_ord[i])),
            numbers=tuple(int(x) for x in self._numbers[i]),
            day_issue_index=int(self._day_idx[i]),
        )


def load_runtime_history_store(output_dir: Path) -> HistoryStore:
    return HistoryStore(output_dir)


def runtime_history_ready(output_dir: Path) -> bool:
    required = ["meta.json", "numbers.npy", "issue.npy", "draw_date_ordinal.npy", "day_issue_index.npy"]
    return all((output_dir / x).exists() for x in required)


def artifact_matches_source(output_dir: Path, source_files: list[Path]) -> bool:
    meta_path = output_dir / "meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    got = [str(Path(p)) for p in meta.get("source_files", [])]
    expected = [str(p) for p in source_files]
    return got == expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_runtime_history_artifact(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
