from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path("configs/inference.yaml")


def _load_raw_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_module_weights(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, float]:
    data = _load_raw_config(config_path)

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
    data = _load_raw_config(config_path)
    modules = data.get("modules", {})
    settings = modules.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("modules.settings must be a mapping")
    return settings


def load_aggregator_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    data = _load_raw_config(config_path)
    modules = data.get("modules", {})
    aggregator = modules.get("aggregator", {})
    if not isinstance(aggregator, dict):
        raise ValueError("modules.aggregator must be a mapping")
    return aggregator


def load_joint_assignment_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    data = _load_raw_config(config_path)
    cfg = data.get("joint_assignment", {})
    if not isinstance(cfg, dict):
        raise ValueError("joint_assignment must be a mapping")
    return cfg


def load_fast_path_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    data = _load_raw_config(config_path)
    cfg = data.get("fast_path", {})
    if not isinstance(cfg, dict):
        raise ValueError("fast_path must be a mapping")
    return cfg


def load_trained_ranker_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    data = _load_raw_config(config_path)
    cfg = data.get("trained_ranker", {})
    if not isinstance(cfg, dict):
        raise ValueError("trained_ranker must be a mapping")
    return cfg
