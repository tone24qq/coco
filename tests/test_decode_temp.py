import numpy as np
import torch
from torch import nn

from src.data.augment import mask_board
from src.inference.decode_temp import iterative_decode_temp


class StubModel(nn.Module):
    def __init__(self, target: torch.Tensor, vocab_size: int = 10) -> None:
        super().__init__()
        self.register_buffer("target", target.flatten())
        self.vocab_size = vocab_size

    def forward(self, tokens: torch.Tensor, attn_mask=None):  # type: ignore[override]
        B, M = tokens.shape
        logits = torch.zeros(B, M, self.vocab_size)
        for i, t in enumerate(self.target):
            logits[:, i, t] = 10.0
        return logits


def test_iterative_decode_temp_fills_grid() -> None:
    board = torch.arange(1, 10).view(3, 3)
    masked, _ = mask_board(board, ratio=0.5, rng=np.random.default_rng(0))
    tokens = masked.flatten().unsqueeze(0)
    attn = torch.ones_like(tokens, dtype=torch.bool)
    model = StubModel(board)
    out = iterative_decode_temp(
        model,
        tokens,
        attn,
        N=board.numel(),
        temperature=0.8,
        topk=5,
    )
    assert torch.equal(out.view_as(board), board)
