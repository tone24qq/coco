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
    holes = torch.tensor([True, True, False, True])

    topk = compute_topk_positions(probs, holes, query_num=2, k=3, cols=2)
    expected = [
        {"row": 0, "col": 0, "prob": 0.8},
        {"row": 0, "col": 1, "prob": 0.1},
        {"row": 1, "col": 1, "prob": 0.05},
    ]
    for item, exp in zip(topk, expected):
        assert item["row"] == exp["row"]
        assert item["col"] == exp["col"]
        assert item["prob"] == pytest.approx(exp["prob"], rel=1e-3)


def _expected_from_seed(seed: int, cols: int) -> list[dict[str, float]]:
    g = torch.Generator().manual_seed(seed)
    noise = torch.rand(4, generator=g)
    vals, order = torch.sort(noise, descending=True, stable=True)
    expected = []
    for v, idx in zip(vals[:2].tolist(), order[:2].tolist()):
        r, c = divmod(idx, cols)
        expected.append({"row": r, "col": c, "prob": v * 1e-6})
    return expected


def test_compute_topk_positions_tie_breaker():
    probs = torch.zeros(4, 6)
    holes = torch.ones(4, dtype=torch.bool)

    topk1 = compute_topk_positions(probs, holes, query_num=1, k=2, cols=2)
    topk2 = compute_topk_positions(probs, holes, query_num=2, k=2, cols=2)

    expected1 = _expected_from_seed(1, cols=2)
    expected2 = _expected_from_seed(2, cols=2)

    for item, exp in zip(topk1, expected1):
        assert item["row"] == exp["row"]
        assert item["col"] == exp["col"]
        assert item["prob"] == pytest.approx(exp["prob"], rel=1e-6)

    for item, exp in zip(topk2, expected2):
        assert item["row"] == exp["row"]
        assert item["col"] == exp["col"]
        assert item["prob"] == pytest.approx(exp["prob"], rel=1e-6)
