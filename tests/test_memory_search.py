import asyncio
import json
import os
from pathlib import Path

import numpy as np

import app
import memory_loader
from dataset import BLANK_VALUE


def test_find_similar_and_filter_by_target():
    model = app._create_model(4, 5)
    if hasattr(model, "eval"):
        model.eval()
    app.models[(4, 5)] = model
    app._load_memory_for_shape(4, 5)

    data = json.load(Path("data_archives/4x5.json").open("r", encoding="utf-8"))[0]
    board = np.array(data["board"], dtype=int)
    target = int(data["target"])
    # HNSW path
    sims = app.find_similar(4, 5, board, target, k=2)
    assert len(sims) == 2
    assert sims[0]["distance"] >= 0.0

    # Fallback path: remove index then search again
    app.hnsw_indices.pop((4, 5), None)
    sims_fb = app.find_similar(4, 5, board, target, k=5)
    assert len(sims_fb) == 5
    assert any(s["target"] == target for s in sims_fb)

    # Combine filter_by_target + manual distance ranking
    indices = app.filter_by_target(4, 5, target)
    assert indices, "filter_by_target should return non-empty list"
    keys, _ = app.memories[(4, 5)]
    q, _ = app.build_memory_agent([(board, target)], model)
    dists = np.linalg.norm(keys[indices] - q[0], axis=1)
    best_idx = indices[int(np.argmin(dists))]
    assert any(s["sample_idx"] == best_idx for s in sims_fb if s["target"] == target)


def test_predict_memory_fusion_prefers_memory():
    os.environ["MEMORY_ALPHA"] = "0"
    model = app._create_model(4, 5)
    if hasattr(model, "eval"):
        model.eval()
    app.models[(4, 5)] = model

    data = json.load(Path("data_archives/4x5.json").open("r", encoding="utf-8"))[0]
    board = np.array(data["board"], dtype=int)
    target = int(data["target"])
    # 建立至少一個空格供預測
    flat = board.flatten()
    bidx = 0 if flat[0] != target else 1
    flat[bidx] = BLANK_VALUE
    board = flat.reshape(board.shape)
    mask_pos = np.where(board.flatten() == BLANK_VALUE)[0]
    assert mask_pos.size > 0
    bidx = int(mask_pos[0])

    memory_loader.MEMORY_CACHE.clear()
    keys = np.zeros((1, 256), dtype=np.float32)
    values = np.zeros((1, board.size), dtype=np.float32)
    values[0, bidx] = 1.0
    memory_loader.MEMORY_CACHE[(4, 5)] = (keys, values)
    app.memory_targets[(4, 5)] = np.array([target], dtype=np.int64)

    req = app.PredictRequest(board=board.tolist(), target=target)
    preds = asyncio.run(app.predict(req))
    assert preds[0].idx == bidx
