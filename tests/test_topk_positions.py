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
    tokens = torch.tensor([0, 0, 3, 0])  # holes at 0,1,3

    topk = compute_topk_positions(probs, tokens, query_num=2, k=3, cols=2)
    expected = [
        {"row": 0, "col": 0, "prob": 0.8 / 0.95},
        {"row": 0, "col": 1, "prob": 0.1 / 0.95},
        {"row": 1, "col": 1, "prob": 0.05 / 0.95},
    ]
    for item, exp in zip(topk, expected):
        assert item["row"] == exp["row"]
        assert item["col"] == exp["col"]
        assert item["prob"] == pytest.approx(exp["prob"], rel=1e-6)
    assert pytest.approx(sum(i["prob"] for i in topk), rel=1e-6) == 1.0


def test_compute_topk_positions_skips_filled_cells():
    probs = torch.zeros(4, 6)
    probs[:, :] = 0.0
    probs[0, 2] = 0.8
    probs[1, 2] = 0.1
    probs[2, 2] = 0.4
    probs[3, 2] = 0.05
    tokens = torch.tensor([1, 0, 3, 0])  # filled at idx0, holes at 1 and 3

    topk = compute_topk_positions(probs, tokens, query_num=2, k=3, cols=2)
    assert len(topk) == 2
    assert {(t["row"], t["col"]) for t in topk} == {(0, 1), (1, 1)}
    assert pytest.approx(sum(i["prob"] for i in topk), rel=1e-6) == 1.0
