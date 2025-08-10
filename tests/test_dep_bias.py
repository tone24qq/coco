import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.dep_bias import apply_dep_bias
from src.training.train import evaluate


def test_apply_dep_bias_row_col_hist():
    # Grid: [[1, 2], [3, 4]] with position (0,1) masked
    tokens = torch.tensor([[1, 0, 3, 4]])
    target = torch.tensor([[1, 2, 3, 4]])
    rows = torch.tensor([2])
    cols = torch.tensor([2])
    N = torch.tensor([4])
    logits = torch.zeros(1, 4, 6)

    apply_dep_bias(logits, tokens, target, rows, cols, N, dep_alpha=1.0)

    bias = math.log(0.5)
    # Masked position index 1 should receive bias on numbers 1 and 4
    assert torch.isclose(logits[0, 1, 1], torch.tensor(bias))
    assert torch.isclose(logits[0, 1, 4], torch.tensor(bias))
    # Other numbers receive log(eps) bias
    other = math.log(1e-6)
    assert torch.isclose(logits[0, 1, 0], torch.tensor(other))
    assert torch.isclose(logits[0, 1, 2], torch.tensor(other))


def test_evaluate_dep_bias_applied():
    class DummyModel(nn.Module):
        def forward(self, tokens, attn):  # type: ignore[override]
            B, L = tokens.shape
            return torch.zeros(B, L, 6)

    sample = {
        "tokens": torch.tensor([1, 0, 3, 4], dtype=torch.long),
        "target": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        "attn_mask": torch.ones(4, dtype=torch.long),
        "N": torch.tensor(4, dtype=torch.long),
        "rows": torch.tensor(2, dtype=torch.long),
        "cols": torch.tensor(2, dtype=torch.long),
    }

    def collate(batch):
        return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}

    loader = DataLoader([sample], batch_size=1, collate_fn=collate)
    model = DummyModel()
    no_bias = evaluate(model, loader, "cpu")
    with_bias = evaluate(model, loader, "cpu", use_dep_bias=True, dep_alpha=1.0)
    assert with_bias["loss"] > no_bias["loss"]
