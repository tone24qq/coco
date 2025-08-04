import logging

import numpy as np
import orjson

from agents.memory_agent import predict_stream
from dataset import BLANK_VALUE


class DummyModel:
    """Minimal model to track invocation counts."""

    def __init__(self) -> None:
        self.hidden_calls = 0
        self.forward_calls = 0

    def get_hidden_state(self, x: np.ndarray) -> np.ndarray:  # noqa: D401
        self.hidden_calls += 1
        return np.ones(4, dtype=float)

    def forward(self, x: np.ndarray) -> np.ndarray:  # noqa: D401
        self.forward_calls += 1
        n = x.shape[0]
        return np.ones((1, n, 9), dtype=float)


def test_predict_stream_sampling(tmp_path, caplog) -> None:
    board = np.array([[BLANK_VALUE, 2], [3, 4]], dtype=int)
    target = 1
    data = [{"board": board.tolist(), "target": target} for _ in range(6)]
    jsonl = tmp_path / "mem.jsonl"
    with jsonl.open("wb") as f:
        for obj in data:
            f.write(orjson.dumps(obj) + b"\n")

    model = DummyModel()
    with caplog.at_level(logging.INFO):
        predict_stream(
            board, target=target, model=model, jsonl_path=jsonl, k_neighbors=1
        )

    assert model.hidden_calls == 2  # embeddings + query
    assert model.forward_calls == 2  # query + neighbor
