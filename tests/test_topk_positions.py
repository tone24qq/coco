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
        {"row": 0, "col": 0, "prob": 0.507884},
        {"row": 0, "col": 1, "prob": 0.252235},
        {"row": 1, "col": 1, "prob": 0.239881},
    ]
    for item, exp in zip(topk, expected):
        assert item["row"] == exp["row"]
        assert item["col"] == exp["col"]
        assert item["prob"] == pytest.approx(exp["prob"], rel=1e-3)
