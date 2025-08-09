import torch

from src.training.datasets.pad_collate import pad_collate


def test_pad_collate_pads_and_stacks():
    sample1 = {
        "tokens": torch.tensor([1, 2, 3], dtype=torch.long),
        "target": torch.tensor([1, 2, 3], dtype=torch.long),
        "attn_mask": torch.ones(3, dtype=torch.bool),
        "rows": 1,
        "cols": 3,
        "N": 3,
    }
    sample2 = {
        "tokens": torch.tensor([4, 5], dtype=torch.long),
        "target": torch.tensor([4, 5], dtype=torch.long),
        "attn_mask": torch.ones(2, dtype=torch.bool),
        "rows": 1,
        "cols": 2,
        "N": 2,
    }
    batch = pad_collate([sample1, sample2])
    assert batch["tokens"].shape == (2, 3)
    assert batch["target"].shape == (2, 3)
    assert batch["attn_mask"].shape == (2, 3)
    # second sample padded with zeros/False at last position
    assert batch["tokens"][1, 2] == 0
    assert batch["target"][1, 2] == 0
    assert batch["attn_mask"][1, 2].item() == 0
    assert torch.equal(batch["rows"], torch.tensor([1, 1]))
    assert torch.equal(batch["cols"], torch.tensor([3, 2]))
    assert torch.equal(batch["N"], torch.tensor([3, 2]))
