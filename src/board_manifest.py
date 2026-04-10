from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal

SizeClass = Literal["20", "80", "120", "160"]


@dataclass
class ManifestEntry:
    sample_id: str
    size_class: SizeClass
    image_paths: List[str]
    page_count: int
    source_folder: str
    valid: bool
    invalid_reason: str | None = None


@dataclass
class ManifestAudit:
    total_images: int
    total_samples: int
    valid_samples: int
    invalid_samples: int
    invalid_reasons: Dict[str, int]


PAGE_PATTERN = re.compile(r"^(?P<sample_id>.+?)_頁面_(?P<page>[0-9]+)$")


def build_multisize_manifest(repo_root: Path) -> tuple[List[ManifestEntry], ManifestAudit]:
    entries: List[ManifestEntry] = []
    invalid_reasons: Dict[str, int] = {}
    total_images = 0

    for size in ("20", "80", "160"):
        folder = repo_root / "gogo" / size
        if not folder.exists():
            continue
        for img in sorted(folder.rglob("*.jpg")):
            total_images += 1
            sample_id = img.stem
            entries.append(
                ManifestEntry(
                    sample_id=sample_id,
                    size_class=size,  # type: ignore[arg-type]
                    image_paths=[str(img)],
                    page_count=1,
                    source_folder=str(img.parent),
                    valid=True,
                )
            )

    folder120 = repo_root / "gogo" / "120"
    buckets: Dict[str, Dict[int, Path]] = {}
    if folder120.exists():
        for img in sorted(folder120.rglob("*.jpg")):
            total_images += 1
            m = PAGE_PATTERN.match(img.stem)
            if not m:
                invalid_reasons["120_naming_mismatch"] = invalid_reasons.get("120_naming_mismatch", 0) + 1
                entries.append(
                    ManifestEntry(
                        sample_id=img.stem,
                        size_class="120",
                        image_paths=[str(img)],
                        page_count=1,
                        source_folder=str(img.parent),
                        valid=False,
                        invalid_reason="120_naming_mismatch",
                    )
                )
                continue
            sid = m.group("sample_id")
            page = int(m.group("page"))
            if sid not in buckets:
                buckets[sid] = {}
            if page in buckets[sid]:
                invalid_reasons["120_duplicate_page"] = invalid_reasons.get("120_duplicate_page", 0) + 1
            buckets[sid][page] = img

    for sid, page_map in sorted(buckets.items()):
        pages = sorted(page_map.keys())
        paths = [str(page_map[p]) for p in pages]
        valid = pages == [1, 2]
        reason = None
        if not valid:
            if pages == [1] or pages == [2]:
                reason = "120_missing_page"
            else:
                reason = "120_invalid_page_set"
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
        entries.append(
            ManifestEntry(
                sample_id=sid,
                size_class="120",
                image_paths=paths,
                page_count=len(paths),
                source_folder=str(folder120),
                valid=valid,
                invalid_reason=reason,
            )
        )

    audit = ManifestAudit(
        total_images=total_images,
        total_samples=len(entries),
        valid_samples=sum(1 for e in entries if e.valid),
        invalid_samples=sum(1 for e in entries if not e.valid),
        invalid_reasons=invalid_reasons,
    )
    return entries, audit


def write_manifest(entries: List[ManifestEntry], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        __import__("json").dumps([asdict(e) for e in entries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_manifest_audit(audit: ManifestAudit, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(__import__("json").dumps(asdict(audit), indent=2, ensure_ascii=False), encoding="utf-8")
