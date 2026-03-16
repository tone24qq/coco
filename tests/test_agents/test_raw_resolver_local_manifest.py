from __future__ import annotations

import json
from pathlib import Path

from src.io.canonical_dataset import build_canonical_dataset
from src.io.raw_resolver import build_raw_manifest, decode_unicode_tokens


def test_decode_unicode_tokens() -> None:
    raw = "#U8cd3#U679c#U8cd3#U679c_2023.csv"
    assert decode_unicode_tokens(raw) == "賓果賓果_2023.csv"


def test_manifest_and_canonical_build() -> None:
    manifest = build_raw_manifest()
    assert manifest["file_count"] >= 1
    assert "missing_years" in manifest

    df, audit = build_canonical_dataset()
    assert not df.empty
    assert audit["canonical_rows"] == len(df)
    assert Path("data/raw/raw_manifest.json").exists()
    payload = json.loads(Path("data/raw/raw_manifest.json").read_text(encoding="utf-8"))
    assert "entries" in payload
