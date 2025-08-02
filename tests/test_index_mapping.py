import numpy as np
import pytest

from dataset import BLANK_VALUE, MASK_TOKEN_ID
from model import DynamicMET

torch = pytest.importorskip("torch")


def test_model_output_dim_and_mapping():
    model = DynamicMET(num_fields=80, num_values=81, rows=8, cols=10)
    assert model.classifier.out_features == 81

    board = np.full((8, 10), BLANK_VALUE, dtype=int)
    board[3, 5] = 17
    x = np.where(board == BLANK_VALUE, MASK_TOKEN_ID, board).astype(np.int64)
    y = np.where(board == BLANK_VALUE, MASK_TOKEN_ID, board).astype(np.int64)
    assert y[3, 5] == 17

    inp = torch.from_numpy(x.flatten()).unsqueeze(0)
    logits = model(inp)
    assert logits.shape == (1, 80, 81)
    dist = logits[0, 3 * 10 + 5]
    assert dist.shape[0] == 81
    # bias to force argmax at 17 to smoke-test index semantics
    model.classifier.bias.data.zero_()
    model.classifier.bias.data[17] = 1.0
    logits2 = model(inp)
    dist2 = logits2[0, 3 * 10 + 5]
    assert int(dist2.argmax().item()) == 17


def test_no_prior_keywords():
    banned = ["alpha", "heatmap", "prior", "temperature", "label_bias", "position_bias"]
    for path in ["agents/met_agent.py", "app.py"]:
        text = open(path, encoding="utf-8").read().lower()
        for word in banned:
            assert word not in text
