import torch

from train import uniqueness_soft_penalty


def test_uniqueness_soft_penalty():
    logits = torch.zeros(1, 4, 4)
    tokens = torch.tensor([[1, -1, 2, -1]])
    loss = uniqueness_soft_penalty(logits, tokens, lam=1.0)
    assert torch.isclose(loss, torch.tensor(0.25))

    tokens2 = torch.full((1, 4), -1, dtype=torch.long)
    loss2 = uniqueness_soft_penalty(logits, tokens2, lam=1.0)
    assert torch.isclose(loss2, torch.tensor(0.0))
