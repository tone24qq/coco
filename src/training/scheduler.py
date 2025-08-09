"""Learning rate scheduler utilities."""

from __future__ import annotations

import math
from typing import Iterator


def cosine_with_warmup(
    base_lr: float, warmup: int, total_steps: int
) -> Iterator[float]:
    """Yield learning rates following cosine decay with warmup."""
    for step in range(total_steps):
        if step < warmup:
            yield base_lr * (step + 1) / warmup
        else:
            progress = (step - warmup) / max(1, total_steps - warmup)
            yield base_lr * 0.5 * (1 + math.cos(math.pi * progress))
