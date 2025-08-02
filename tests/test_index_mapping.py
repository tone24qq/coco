import random
from pathlib import Path

import numpy as np
import pytest

from agents.met_agent import predict
from dataset import BLANK_VALUE, MASK_TOKEN_ID
from model import DynamicMET

torch = pytest.importorskip("torch")


def test_classifier_output_dim() -> None:
    model = DynamicMET(num_fields=80, rows=8, cols=10)
    assert model.classifier.out_features == 81


def test_index_semantics() -> None:
    model = DynamicMET(num_fields=80, rows=8, cols=10)
    board = np.full((8, 10), BLANK_VALUE, dtype=int)
    board[3, 5] = 17
    x = np.where(board == BLANK_VALUE, MASK_TOKEN_ID, board)
    y = np.where(board == BLANK_VALUE, MASK_TOKEN_ID, board)
    inp = torch.as_tensor(x.reshape(1, -1), dtype=torch.long)
    logits = model(inp)
    assert logits.shape == (1, 80, 81)
    dist = logits[0, 3 * 10 + 5]
    assert dist.shape[0] == 81
    assert y.flatten()[3 * 10 + 5] == 17


def test_deterministic_topk() -> None:
    seed = 20250802
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    board = np.full((8, 10), BLANK_VALUE, dtype=int)
    board[0, 0] = 1
    model = DynamicMET(num_fields=80, rows=8, cols=10)
    results = [predict(board.copy(), target=1, model=model) for _ in range(10)]
    for r in results[1:]:
        assert r == results[0]


def test_no_forbidden_keywords() -> None:
    forbidden = (
        "alpha",
        "heatmap",
        "prior",
        "temperature",
        "label_bias",
        "position_bias",
        "fuse_predictions_with_heatmap",
    )
    paths = [Path("agents/met_agent.py"), Path("app.py")]
    text = "\n".join(p.read_text().lower() for p in paths)
    for kw in forbidden:
        assert kw not in text
