import numpy as np
import pytest

from utils import load_heatmap
from utils.training import EarlyStopping, masked_topk_accuracy

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


def test_load_heatmap(tmp_path) -> None:
    arr = np.arange(9, dtype=np.float32).reshape(3, 3)
    path = tmp_path / "heatmap_mid_3x3.npy"
    np.save(path, arr)
    hm = load_heatmap(3, 3, target=7, directory=str(tmp_path))
    assert hm.shape == (3, 3)
