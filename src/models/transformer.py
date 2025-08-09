"""Minimal Transformer block used for testing."""

from __future__ import annotations

import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dim_ff: int = 128) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            x, x, x, key_padding_mask=~attn_mask if attn_mask is not None else None
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x
