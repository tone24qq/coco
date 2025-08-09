"""Dataset utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

import torch


class GridDataset(torch.utils.data.Dataset):
    """Simple dataset reading JSONL files of full boards."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open("r", encoding="utf8") as f:
            self.data = [json.loads(line)["board"] for line in f]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        board = torch.tensor(self.data[idx], dtype=torch.long)
        return board


def iter_boards(path: str | Path) -> Iterator[List[List[int]]]:
    """Iterate boards from a JSONL file."""
    with Path(path).open("r", encoding="utf8") as f:
        for line in f:
            yield json.loads(line)["board"]
