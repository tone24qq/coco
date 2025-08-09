import json

from src.training.datasets import JsonBoardsDataset, MaskConfig, collate_batch


def test_json_boards_dataset_mask(tmp_path):
    data = {"boards": [{"grid": [[1, 2], [3, 4]]}]}
    path = tmp_path / "train.json"
    path.write_text(json.dumps(data))
    ds = JsonBoardsDataset(
        path, mask_cfg=MaskConfig(min_ratio=0.5, max_ratio=0.5, line_block_prob=0.0)
    )
    sample = ds[0]
    assert sample["tokens"].numel() == 4
    assert (sample["tokens"] == 0).any()
    assert (sample["target"] > 0).all()
    collated = collate_batch([sample])
    assert collated["tokens"].shape == (1, 4)


def test_json_boards_dataset_target_mask(tmp_path):
    data = {"boards": [{"grid": [[1, 2], [3, 4]], "target": 3}]}
    path = tmp_path / "train.json"
    path.write_text(json.dumps(data))
    ds = JsonBoardsDataset(path, mask_target=True, seed=0)
    sample = ds[0]
    tokens = sample["tokens"].view(2, 2)
    assert tokens[1, 0].item() == 0


def test_json_boards_dataset_target_random(tmp_path):
    data = {"boards": [{"grid": [[1, 2], [3, 4]]}]}
    path = tmp_path / "train.json"
    path.write_text(json.dumps(data))
    ds = JsonBoardsDataset(path, mask_target=True, seed=0)
    sample = ds[0]
    assert sample["tokens"].tolist().count(0) == 1
