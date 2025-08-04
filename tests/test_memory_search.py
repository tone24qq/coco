import json
from pathlib import Path

import numpy as np

import app


def test_find_similar_and_filter_by_target():
    model = app._create_model(4, 5)
    if hasattr(model, "eval"):
        model.eval()
    app.models[(4, 5)] = model
    app._load_memory_for_shape(4, 5)

    data = json.load(Path("data_archives/4x5.json").open("r", encoding="utf-8"))[0]
    board = np.array(data["board"], dtype=int)
    target = int(data["target"])

    sims = app.find_similar(4, 5, board, target, k=2)
    assert len(sims) == 2
    assert sims[0]["target"] == target
    assert sims[0]["distance"] >= 0.0

    indices = app.filter_by_target(4, 5, target)
    assert indices, "filter_by_target should return non-empty list"
    targets = app.memory_targets[(4, 5)]
    for idx in indices:
        assert targets[idx] == target
