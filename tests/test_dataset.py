import numpy as np
import pytest

from dataset import BLANK_VALUE, MASK_TOKEN_ID  # isort: split
from dataset import ScratchCardDataset, validate_board

torch = pytest.importorskip("torch")


def test_dataset_mask_ratio() -> None:
    data = [(np.arange(1, 13).reshape(3, 4), 7)]
    torch.manual_seed(0)  # ensure deterministic mask
    ds = ScratchCardDataset(data, mask_ratio=0.6)
    item = ds[0]
    mask = item["mask"].numpy()
    assert mask.mean() == pytest.approx(0.6, rel=0.3)
    assert (item["input_vals"][mask] == MASK_TOKEN_ID).all()
    assert item["target"].item() == 7


def test_dataset_validation_duplicate() -> None:
    board = np.array([[1, 2], [2, BLANK_VALUE]])
    with pytest.raises(ValueError):
        ScratchCardDataset([(board, 1)])


def test_dataset_validation_range() -> None:
    board = np.array([[0, 1], [2, BLANK_VALUE]])
    with pytest.raises(ValueError):
        validate_board(board)
