"""2D relative positional encodings."""

from __future__ import annotations

import torch
from torch import nn


class RelPos2D(nn.Module):
    """Factorised 2D relative positional encoding."""

    def __init__(self, max_rows: int, max_cols: int, dim: int) -> None:
        super().__init__()
        self.row = nn.Embedding(2 * max_rows, dim)
        self.col = nn.Embedding(2 * max_cols, dim)

    def forward(self, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        r = self.row(rows)
        c = self.col(cols)
        return r + c
