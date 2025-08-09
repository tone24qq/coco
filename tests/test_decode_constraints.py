import torch

from src.inference.model_loader import load_model
from src.models import constraints
from src.models.vocab import masked_logits_clip


def test_uniqueness_projection():
    model = load_model("weights/best.ckpt")
    tokens = torch.zeros(1, 4, dtype=torch.long)
    attn = torch.ones_like(tokens, dtype=torch.bool)
    logits = model(tokens, attn)
    logits = masked_logits_clip(logits, 4)
    out = constraints.uniqueness_projection(
        logits, torch.ones_like(tokens, dtype=torch.bool), 4
    )
    grid = out.view(2, 2).tolist()
    vals = sorted(v for row in grid for v in row)
    assert vals == [1, 2, 3, 4]
    assert all(len(set(row)) == 2 for row in grid)
    assert all(len({grid[r][c] for r in range(2)}) == 2 for c in range(2))
