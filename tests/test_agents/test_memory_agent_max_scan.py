import numpy as np
import orjson

from agents.memory_agent import online_topk_from_jsonl
from dataset import BLANK_VALUE


class DummyModel:
    """Model returning constant embeddings for testing."""

    def get_hidden_state(self, x: np.ndarray) -> np.ndarray:  # noqa: D401
        if x.ndim == 1:
            return np.ones(4, dtype=float)
        return np.ones((x.shape[0], 4), dtype=float)


def test_online_topk_respects_max_scan(tmp_path, monkeypatch) -> None:
    board = np.array([[BLANK_VALUE, 2], [3, 4]], dtype=int)
    data = [{"board": board.tolist(), "target": 1} for _ in range(3)]
    jsonl = tmp_path / "mem.jsonl"
    with jsonl.open("wb") as f:
        for obj in data:
            f.write(orjson.dumps(obj) + b"\n")

    monkeypatch.setenv("MEMORY_MAX_SCAN", "1")

    model = DummyModel()
    res = online_topk_from_jsonl(jsonl, board, model, top_k=5)
    assert len(res) == 1
    assert res[0][0] == 0
