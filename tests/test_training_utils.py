import pytest

from utils.training import EarlyStopping, is_zero_loss, masked_topk_accuracy

torch = pytest.importorskip("torch")


def test_masked_topk_accuracy_basic() -> None:
    logits = torch.tensor([[[0.1, 0.9, 0.0], [0.2, 0.3, 0.5]]])
    targets = torch.tensor([[1, 2]])
    mask = torch.tensor([[True, True]])
    metrics = masked_topk_accuracy(logits, targets, mask, topk=(1, 2))
    assert metrics["top1"] == 0.5
    assert metrics["top2"] == 1.0


def test_early_stopping_restore() -> None:
    model = torch.nn.Linear(1, 1)
    es = EarlyStopping(patience=1, restore_best_weights=True)
    initial = model.weight.clone()
    assert not es.step(0.5, model)
    model.weight.data.add_(1.0)
    assert es.step(0.6, model)
    assert torch.allclose(model.weight, initial)


def test_is_zero_loss() -> None:
    assert is_zero_loss(0.0)
    assert is_zero_loss(5e-05)
    assert not is_zero_loss(1e-3)
