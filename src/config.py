"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    with path.open("r", encoding="utf8") as f:
        cfg = yaml.safe_load(f)
    if "import" in cfg:
        base = load_config(path.parent / cfg["import"])
        base.update({k: v for k, v in cfg.items() if k != "import"})
        return base
    return cfg
