from pathlib import Path

import yaml

from src.multi_size_data_loader import load_multisize_samples


def test_multisize_loader_builds_audit() -> None:
    cfg = yaml.safe_load(
        Path("configs/multisize_masking_eval.yaml").read_text(encoding="utf-8")
    )
    cfg["parser"]["max_samples_per_size"] = 1
    artifacts = load_multisize_samples(cfg)
    assert artifacts.audit["manifest_audit"]["total_samples"] > 0
    assert set(artifacts.audit["parse_counts_by_size"].keys()) == {"20", "80", "120"}
