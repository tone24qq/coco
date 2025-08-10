import pytest
import torch

from src.inference.topk import compute_topk_positions


def test_compute_topk_positions_basic():
    probs = torch.zeros(4, 6)
    probs[:, :] = 0.0
    # Set probabilities for number 2 at different positions
    probs[0, 2] = 0.8
    probs[1, 2] = 0.1
    probs[2, 2] = 0.4
    probs[3, 2] = 0.05
    holes = torch.tensor([True, True, False, True])  # holes at 0,1,3

    topk = compute_topk_positions(probs, holes, query_num=2, k=3, cols=2)
    expected = [
        {"row": 0, "col": 0, "prob": 0.8421052632},
        {"row": 0, "col": 1, "prob": 0.1052631579},
        {"row": 1, "col": 1, "prob": 0.0526315789},
    ]
    for item, exp in zip(topk, expected):
        assert item["row"] == exp["row"]
        assert item["col"] == exp["col"]
        assert item["prob"] == pytest.approx(exp["prob"], rel=1e-6)


def test_existing_zero_not_hole():
    probs = torch.zeros(2, 3)
    probs[0, 1] = 0.9
    probs[1, 1] = 0.1
    holes = torch.tensor([True, False])
    topk = compute_topk_positions(probs, holes, query_num=1, k=2, cols=1)
    assert len(topk) == 1
    assert topk[0]["row"] == 0 and topk[0]["col"] == 0
    assert topk[0]["prob"] == pytest.approx(1.0)
