from __future__ import annotations

import json
from pathlib import Path

from src.multi_size_data_loader import _load_cached_artifacts, load_multisize_samples


def test_cache_hit_records_metadata(tmp_path: Path) -> None:
    parsed = tmp_path / "parsed_boards.json"
    audit = tmp_path / "data_audit.json"
    pending = tmp_path / "pending.json"
    parsed.write_text(
        json.dumps(
            [
                {
                    "sample_id": "x1",
                    "size_class": "80",
                    "grid": [[1, 2], [3, 4]],
                    "shape": "2x2",
                    "metadata": {"parse_confidence": 1.0},
                }
            ]
        )
    )
    audit.write_text(
        json.dumps(
            {"manifest_audit": {"total_samples": 1}, "parse_counts_by_size": {"20": {}, "80": {}, "120": {}}}
        )
    )
    pending.write_text("[]")

    cfg = {
        "data": {"parsed_boards_output": str(parsed), "repo_root": str(tmp_path)},
        "reports": {"data_audit": str(audit), "pending_annotations": str(pending)},
        "parser": {"use_cached_first": True, "min_confidence": 0.1},
    }
    artifacts = _load_cached_artifacts(cfg)
    assert artifacts is not None
    assert artifacts.cache_info["used_cache"] is True


def test_cache_fallback_when_disabled(tmp_path: Path) -> None:
    cfg = {
        "data": {"parsed_boards_output": str(tmp_path / "x.json"), "repo_root": str(tmp_path)},
        "reports": {
            "data_audit": str(tmp_path / "a.json"),
            "pending_annotations": str(tmp_path / "p.json"),
            "manifest": str(tmp_path / "m.json"),
            "parse_audit": str(tmp_path / "pa.json"),
            "pending_review": str(tmp_path / "pr.json"),
        },
        "parser": {"use_cached_first": False, "min_confidence": 0.1},
    }
    # no gogo data; parse path returns zero samples but should still expose fallback parse metadata
    artifacts = load_multisize_samples(cfg)
    assert artifacts.cache_info["used_cache"] is False
    assert artifacts.cache_info["fallback_parse"] is True
