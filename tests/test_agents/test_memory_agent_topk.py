# isort: skip_file
import numpy as np
import orjson
import pytest

from agents.memory_agent import online_topk_from_directory, online_topk_from_jsonl


class IdentityModel:
    """Model whose embedding equals the flattened board."""

    def get_hidden_state(self, x: np.ndarray) -> np.ndarray:  # noqa: D401
        return x.astype(float)


def _write_jsonl(path, data) -> None:
    with open(path, "wb") as f:
        for obj in data:
            f.write(orjson.dumps(obj) + b"\n")


def test_online_topk_jsonl_and_directory(tmp_path) -> None:
    boards = [
        np.array([[1, 2], [3, 4]], dtype=int),
        np.array([[1, 2], [3, 5]], dtype=int),
        np.array([[4, 3], [2, 1]], dtype=int),
    ]
    data = [{"id": i, "board": b.tolist()} for i, b in enumerate(boards)]

    jsonl = tmp_path / "mem.jsonl"
    _write_jsonl(jsonl, data)

    model = IdentityModel()
    query = boards[0]

    res = online_topk_from_jsonl(jsonl, query, model, top_k=2, batch_size=2)
    assert res[0][0] == 0
    assert len(res) == 2

    dir_path = tmp_path / "dir"
    dir_path.mkdir()
    _write_jsonl(dir_path / "a.jsonl", data[:2])
    _write_jsonl(dir_path / "b.jsonl", data[2:])

    res_dir = online_topk_from_directory(dir_path, query, model, top_k=1, batch_size=2)
    assert res_dir[0][0] == 0
    assert res_dir[0][1] == pytest.approx(1.0)
