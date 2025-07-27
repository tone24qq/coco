import numpy as np
import pytest

from dataset import MASK_TOKEN_ID, ScratchCardDataset

torch = pytest.importorskip("torch")


def test_dataset_mask_ratio() -> None:
    data = [(np.arange(12).reshape(3, 4), 7)]
    torch.manual_seed(0)  # ensure deterministic mask
    ds = ScratchCardDataset(data, mask_ratio=0.6)
    item = ds[0]
    mask = item["mask"].numpy()
    assert mask.mean() == pytest.approx(0.6, rel=0.3)
    assert (item["input_vals"][mask] == MASK_TOKEN_ID).all()
    assert item["target"].item() == 7
