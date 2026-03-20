from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RawFileEntry:
    path: str
    size_bytes: int


def discover_raw_files(raw_dirs: list[Path] | None = None) -> list[Path]:
    dirs = raw_dirs or [Path("data/raw"), Path("raw")]
    files: list[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.csv")):
            if p.is_file():
                files.append(p)
    # stable unique order
    seen: set[str] = set()
    out: list[Path] = []
    for p in files:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def build_raw_manifest(raw_dirs: list[Path] | None = None, output_path: Path = Path("reports/raw_manifest.json")) -> dict:
    files = discover_raw_files(raw_dirs)
    manifest = {
        "detected_files": [str(p) for p in files],
        "file_count": len(files),
        "files": [{"path": str(p), "size_bytes": p.stat().st_size} for p in files],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
