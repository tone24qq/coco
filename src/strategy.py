from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StrategyConfig:
    strategy_version: str = "v1-ranking-mainline"
    dedup_top3: bool = True


def apply_top3_group_dedup(numbers: list[int]) -> list[int]:
    if len(numbers) <= 1:
        return numbers
    selected: list[int] = []
    for n in numbers:
        if len(selected) >= 3:
            break
        if any(abs(n - s) <= 1 for s in selected):
            continue
        selected.append(n)
    for n in numbers:
        if len(selected) >= 3:
            break
        if n not in selected:
            selected.append(n)
    return selected
