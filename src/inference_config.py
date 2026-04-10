from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml


DEFAULT_CONFIG_PATH = Path("configs/inference.yaml")


def load_module_weights(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, float]:
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    modules = data.get("modules", {})
    enabled = modules.get("enabled", {})
    weights = modules.get("weights", {})

    active_weights: Dict[str, float] = {}
    for name, is_enabled in enabled.items():
        if not is_enabled:
            continue
        if name not in weights:
            raise ValueError(f"Missing weight for enabled module: {name}")
        weight_value = float(weights[name])
        if weight_value < 0:
            raise ValueError(f"Weight must be non-negative: {name}")
        active_weights[name] = weight_value

    if not active_weights:
        raise ValueError("At least one scoring module must be enabled")

    total = sum(active_weights.values())
    if total <= 0:
        raise ValueError("Enabled module weights must sum to a positive value")

    return {k: v / total for k, v in active_weights.items()}


def load_module_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Dict[str, object]]:
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    modules = data.get("modules", {})
    settings = modules.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("modules.settings must be a mapping")
    return settings
