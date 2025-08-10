import torch

from train import SizePriorCache, apply_nbr3x3_bias, apply_size_bias


def test_apply_size_bias() -> None:
    V = 10
    logits = torch.zeros(1, 4, V)
    tokens = torch.tensor([[0, 2, 3, 4]])
    target = torch.tensor([[1, 2, 3, 4]])
    rows = torch.tensor([2])
    cols = torch.tensor([2])
    N = torch.tensor([5])

    cache = SizePriorCache(vocab_max=9, alpha=0.5)
    cache.update_batch(rows, cols, target, N)
    apply_size_bias(logits, tokens, target, rows, cols, N, cache, beta=1.0)

    H = cache.get(2, 2, logits.device)
    prior = H[: N.item() + 1, 0, 0]
    expected = torch.full((V,), -1e9)
    log_prior = torch.log(prior + 1e-6)
    expected[: log_prior.numel()] = log_prior
    assert torch.allclose(logits[0, 0], expected)
    assert torch.all(logits[0, 1:].eq(0))


def test_apply_nbr3x3_bias() -> None:
    V = 10
    tokens = torch.tensor([[1, 2, 3, 4, 0, 1, 2, 3, 4]])
    target = torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3, 4]])
    rows = torch.tensor([3])
    cols = torch.tensor([3])
    N = torch.tensor([5])
    logits = torch.zeros(1, 9, V)

    apply_nbr3x3_bias(logits, tokens, target, rows, cols, N, beta=1.0)

    grid = tokens.view(3, 3)
    patch = grid[0:3, 0:3].reshape(-1)
    hist = torch.bincount(patch, minlength=min(V - 1, N.item()) + 1)
    hist[0] = 0
    prior = hist.float()
    prior = prior / prior.sum()
    log_prior = torch.log(prior + 1e-6)
    expected = torch.full((V,), -1e9)
    expected[: log_prior.numel()] = log_prior
    assert torch.allclose(logits[0, 4], expected)
    mask = torch.arange(9) != 4
    assert torch.all(logits[0, mask] == 0)
