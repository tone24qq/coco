from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset

MASK_TOKEN = 0


def _validate_full_grid(grid: List[List[int]]) -> Tuple[int, int, int]:
    if not isinstance(grid, list) or not grid:
        raise ValueError("grid must be a non-empty list")
    rows, cols = len(grid), len(grid[0])
    if any(not isinstance(r, list) or len(r) != cols for r in grid):
        raise ValueError("grid must be rectangular")
    N = rows * cols
    flat = [v for row in grid for v in row]
    if any(not isinstance(v, int) or not (1 <= v <= N) for v in flat):
        raise ValueError(f"values must be 1..{N}")
    return rows, cols, N


def _validate_partial_grid(grid: List[List[int]]) -> Tuple[int, int, int]:
    if not isinstance(grid, list) or not grid:
        raise ValueError("grid must be a non-empty list")
    rows, cols = len(grid), len(grid[0])
    if any(not isinstance(r, list) or len(r) != cols for r in grid):
        raise ValueError("grid must be rectangular")
    N = rows * cols
    flat = [v for row in grid for v in row]
    if any(not isinstance(v, int) or (v != -1 and not (1 <= v <= N)) for v in flat):
        raise ValueError(f"values must be -1 or 1..{N}")
    return rows, cols, N


@dataclass
class MaskConfig:
    min_ratio: float = 0.15
    max_ratio: float = 0.6
    line_block_prob: float = 0.3


class JsonBoardsDataset(Dataset):
    """Training dataset loading full grids and masking them."""

    def __init__(
        self, json_path: str | Path, mask_cfg: MaskConfig | None = None, seed: int = 42
    ) -> None:
        self.path = Path(json_path)
        data = json.loads(self.path.read_text())
        self.boards = [b["grid"] for b in data["boards"]]
        self.meta = [_validate_full_grid(g) for g in self.boards]
        self.mask_cfg = mask_cfg or MaskConfig()
        random.seed(seed)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.boards)

    def _mask_grid(self, grid: List[List[int]]) -> List[List[int]]:
        rows, cols = len(grid), len(grid[0])
        total = rows * cols
        ratio = random.uniform(self.mask_cfg.min_ratio, self.mask_cfg.max_ratio)
        num_mask = max(1, int(total * ratio))
        masked = [row[:] for row in grid]

        if random.random() < self.mask_cfg.line_block_prob:
            if random.random() < 0.5:
                r = random.randrange(rows)
                for c in range(cols):
                    masked[r][c] = MASK_TOKEN
            else:
                c = random.randrange(cols)
                for r in range(rows):
                    masked[r][c] = MASK_TOKEN

        flat_idx = list(range(total))
        random.shuffle(flat_idx)
        count = 0
        for idx in flat_idx:
            r, c = divmod(idx, cols)
            if masked[r][c] != MASK_TOKEN:
                masked[r][c] = MASK_TOKEN
                count += 1
                if count >= num_mask:
                    break
        return masked

    def __getitem__(self, i: int) -> Dict[str, Any]:
        grid = self.boards[i]
        rows, cols, N = self.meta[i]
        masked = self._mask_grid(grid)

        target = torch.tensor([v for row in grid for v in row], dtype=torch.long)
        tokens = torch.tensor([v for row in masked for v in row], dtype=torch.long)
        attn = torch.ones_like(tokens, dtype=torch.bool)

        return {
            "tokens": tokens,
            "target": target,
            "attn_mask": attn,
            "rows": rows,
            "cols": cols,
            "N": N,
        }


def collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    tokens = torch.stack([s["tokens"] for s in samples])
    target = torch.stack([s["target"] for s in samples])
    attn = torch.stack([s["attn_mask"] for s in samples])
    rows = torch.tensor([s["rows"] for s in samples], dtype=torch.long)
    cols = torch.tensor([s["cols"] for s in samples], dtype=torch.long)
    N = torch.tensor([s["N"] for s in samples], dtype=torch.long)
    return {
        "tokens": tokens,
        "target": target,
        "attn_mask": attn,
        "rows": rows,
        "cols": cols,
        "N": N,
    }
