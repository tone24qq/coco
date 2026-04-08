from pathlib import Path

import yaml

from src.board_manifest import ManifestEntry
from src.multi_size_data_loader import (
    _apply_manual_manifest_overrides,
    load_multisize_samples,
)


def test_multisize_loader_builds_audit() -> None:
    cfg = yaml.safe_load(
        Path("configs/multisize_masking_eval.yaml").read_text(encoding="utf-8")
    )
    cfg["parser"]["max_samples_per_size"] = 1
    artifacts = load_multisize_samples(cfg)
    assert artifacts.audit["manifest_audit"]["total_samples"] > 0
    assert set(artifacts.audit["parse_counts_by_size"].keys()) == {"20", "80", "120"}


def test_apply_manual_manifest_overrides(tmp_path: Path) -> None:
    manifest = [
        ManifestEntry(
            sample_id="sid",
            size_class="20",
            image_paths=["a.jpg"],
            page_count=1,
            source_folder="gogo/20",
            valid=True,
        )
    ]
    p = tmp_path / "manual_manifest.json"
    p.write_text(
        '[{"sample_id":"sid","size_class":"20","manual_grid":"m.json","override":"o.json"}]',
        encoding="utf-8",
    )
    updated = _apply_manual_manifest_overrides(manifest, str(p))
    assert updated[0].manual_grid == "m.json"
    assert updated[0].override == "o.json"
