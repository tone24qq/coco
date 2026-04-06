from __future__ import annotations

from typing import Dict


def fuse_scores(features: Dict[str, float], weights: Dict[str, float]) -> float:
    return float(sum(weights.get(k, 0.0) * v for k, v in features.items()))
