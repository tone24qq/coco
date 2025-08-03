import json
from pathlib import Path

import numpy as np
import orjson

from agents.memory_agent import build_memory, predict, predict_stream
from dataset import BLANK_VALUE


class DummyModel:
    def __init__(self, num_cells: int, hidden_dim: int, seed: int = 0) -> None:
        self.num_cells = num_cells
        self.hidden_dim = hidden_dim
        np.random.seed(seed)

    def forward(self, board_flat: np.ndarray) -> np.ndarray:
        return np.random.rand(self.num_cells, self.num_cells)

    def get_hidden_state(self, board_flat: np.ndarray) -> np.ndarray:
        return np.random.rand(self.hidden_dim)


class StrictModel(DummyModel):
    def forward(self, board_flat: np.ndarray) -> np.ndarray:
        assert np.all(board_flat >= 0)
        return super().forward(board_flat)

    def get_hidden_state(self, board_flat: np.ndarray) -> np.ndarray:
        assert np.all(board_flat >= 0)
        return super().get_hidden_state(board_flat)


def load_samples_for_shape(rows: int, cols: int, base_dir: str = "data_archives"):
    file_path = Path(base_dir) / f"{rows}x{cols}.json"
    data = json.load(open(file_path, "r", encoding="utf-8"))
    samples = []
    for entry in data:
        board = np.array(entry["board"], dtype=int)
        target = int(entry["target"])
        samples.append((board, target))
    return samples


def _write_jsonl(samples, path: Path) -> None:
    with path.open("wb") as f:
        for board, target in samples:
            obj = {"board": board.tolist(), "target": int(target)}
            f.write(orjson.dumps(obj))
            f.write(b"\n")


def test_memory_agent_predict_top3() -> None:
    rows, cols = 4, 5
    samples = load_samples_for_shape(rows, cols)
    model = DummyModel(num_cells=rows * cols, hidden_dim=8)
    memory_keys, memory_values = build_memory(samples, model)
    board, target = samples[0]
    res = predict(
        board.copy(),
        target=target,
        model=model,
        memory_keys=memory_keys,
        memory_values=memory_values,
        topk=3,
        query_index=0,
    )
    blank_count = np.sum(board == -1)
    assert len(res) == min(3, blank_count)
    scores = [item["score"] for item in res]
    assert scores == sorted(scores, reverse=True)
    for item in res:
        r0, c0 = item["row"] - 1, item["col"] - 1
        assert board[r0, c0] == -1


def test_memory_agent_predict_stream() -> None:
    rows, cols = 4, 5
    samples = load_samples_for_shape(rows, cols)
    jsonl_path = Path("data_archives/4x5.jsonl")
    if not jsonl_path.exists():
        _write_jsonl(samples[:50], jsonl_path)
    model = DummyModel(num_cells=rows * cols, hidden_dim=8)
    board, target = samples[0]
    res = predict_stream(
        board.copy(),
        target=target,
        model=model,
        jsonl_path=jsonl_path,
        topk=3,
    )
    blank_count = np.sum(board == -1)
    assert len(res) == min(3, blank_count)
    scores = [item["score"] for item in res]
    assert scores == sorted(scores, reverse=True)
    for item in res:
        r0, c0 = item["row"] - 1, item["col"] - 1
        assert board[r0, c0] == -1


def test_predict_stream_masks_blank(tmp_path: Path) -> None:
    rows, cols = 4, 5
    samples = load_samples_for_shape(rows, cols)
    board, target = samples[0]
    board = board.copy()
    board[0, 0] = BLANK_VALUE
    jsonl_path = tmp_path / "sample.jsonl"
    with jsonl_path.open("wb") as f:
        obj = {"board": board.tolist(), "target": int(target)}
        f.write(orjson.dumps(obj))
        f.write(b"\n")
    model = StrictModel(num_cells=rows * cols, hidden_dim=8)
    res = predict_stream(
        board.copy(),
        target=target,
        model=model,
        jsonl_path=jsonl_path,
        topk=3,
    )
    assert isinstance(res, list)
