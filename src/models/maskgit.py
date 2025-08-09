"""Decoder-only Transformer for masked token modeling."""

from __future__ import annotations

import torch
from torch import nn

from .transformer import TransformerBlock


class MaskGit(nn.Module):
    """Simple decoder-only network producing logits over tokens."""

    def __init__(
        self, vocab_size: int, d_model: int, n_head: int, num_layers: int
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size + 1)

    def forward(
        self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.ln(x)
        return self.head(x)
