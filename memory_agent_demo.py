"""Demo script for the memory-based agent."""

import json
from pathlib import Path

import numpy as np

from agents.memory_agent import build_memory, predict


class DummyModel:
    """Minimal model used for demonstration purposes."""

    def __init__(self, num_cells: int, hidden_dim: int, seed: int = 42) -> None:
        self.num_cells = num_cells
        self.hidden_dim = hidden_dim
        np.random.seed(seed)

    def forward(self, board_flat: np.ndarray) -> np.ndarray:
        return np.random.rand(self.num_cells, self.num_cells)

    def get_hidden_state(self, board_flat: np.ndarray) -> np.ndarray:
        return np.random.rand(self.hidden_dim)


def load_samples_for_shape(rows: int, cols: int, base_dir: str = "data_archives"):
    file_path = Path(base_dir) / f"{rows}x{cols}.json"
    if not file_path.is_file():
        raise FileNotFoundError(f"找不到：{file_path}")
    data = json.load(open(file_path, "r", encoding="utf-8"))
    samples = []
    for entry in data:
        board = np.array(entry["board"], dtype=int)
        target = int(entry["target"])
        samples.append((board, target))
    print(f"✅ 載入 {rows}×{cols} 樣本共 {len(samples)} 筆")
    return samples


def main() -> None:
    rows, cols = 4, 5
    samples = load_samples_for_shape(rows, cols)
    model = DummyModel(num_cells=rows * cols, hidden_dim=8)
    memory_keys, memory_values = build_memory(samples, model)
    print(f"✅ 記憶庫建置完畢：keys {memory_keys.shape}, values {memory_values.shape}")
    board, target = samples[0]
    results = predict(
        board.copy(),
        target=target,
        model=model,
        memory_keys=memory_keys,
        memory_values=memory_values,
        topk=3,
        query_index=0,
    )
    print("🏆 最終 Top-3 候選格子:", results)


if __name__ == "__main__":
    main()
