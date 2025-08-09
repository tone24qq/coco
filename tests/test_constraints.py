import torch

from src.models.constraints import uniqueness_projection


def test_uniqueness_projection_assigns_unique_numbers() -> None:
    logits = torch.zeros(1, 4, 6)
    logits[:, :, 1:5] = 1.0  # uniform probabilities over 1..4
    mask = torch.ones(1, 4, dtype=torch.bool)
    out = uniqueness_projection(logits, mask, N=4)
    assert sorted(out[0].tolist()) == [1, 2, 3, 4]
