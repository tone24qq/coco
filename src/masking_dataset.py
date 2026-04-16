from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from src.whole_board_features import (
    compute_board_state_features,
    compute_candidate_delta_features,
    merge_feature_layers,
)


@dataclass
class MaskingConfig:
    ratios: Sequence[float]
    masks_per_ratio: int


def _rng_for_group(board_id: str, mask_ratio: float, mask_index: int) -> random.Random:
    key = f"{board_id}|{mask_ratio:.2f}|{mask_index}".encode("utf-8")
    seed = int(hashlib.sha256(key).hexdigest()[:16], 16)
    return random.Random(seed)


def _find_pos(grid: List[List[int]], target_number: int) -> Tuple[int, int]:
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v == target_number:
                return r, c
    raise ValueError(f"target_number {target_number} not found")


def create_masked_board(grid: List[List[int]], ratio: float, rng: random.Random) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])
    total = rows * cols
    mask_count = max(1, int(round(total * ratio)))
    cells = list(range(total))
    rng.shuffle(cells)
    mask_set = set(cells[:mask_count])
    out = [row[:] for row in grid]
    for idx in mask_set:
        r, c = divmod(idx, cols)
        out[r][c] = -1
    return out


def build_rows_for_group(
    board_row: Dict[str, object],
    masked_board: List[List[int]],
    group_id: str,
    mask_ratio: float,
    target_number: int,
) -> List[Dict[str, object]]:
    full_grid = board_row["grid"]
    rows = int(board_row["rows"])
    cols = int(board_row["cols"])
    source_type = str(board_row.get("source_type", "real"))
    true_r, true_c = _find_pos(full_grid, target_number)

    board_feats = compute_board_state_features(masked_board, target_number)
    rows_out: List[Dict[str, object]] = []
    for r in range(rows):
        for c in range(cols):
            cand_delta = compute_candidate_delta_features(masked_board, target_number, r, c, board_feats)
            feature_layer = merge_feature_layers(board_feats, cand_delta)
            record: Dict[str, object] = {
                "group_id": group_id,
                "lineage_id": str(board_row["board_id"]),
                "board_id": str(board_row["board_id"]),
                "source_type": source_type,
                "rows": rows,
                "cols": cols,
                "size_class": str(board_row["size_class"]),
                "mask_ratio": float(mask_ratio),
                "target_number": int(target_number),
                "cand_row": int(r + 1),
                "cand_col": int(c + 1),
                "label": int((r, c) == (true_r, true_c)),
                "is_feasible": int(masked_board[r][c] == -1),
                **feature_layer,
            }
            rows_out.append(record)
    return rows_out


def build_masked_ranking_dataset(
    boards: Iterable[Dict[str, object]],
    config: MaskingConfig,
) -> pd.DataFrame:
    rows_out: List[Dict[str, object]] = []
    for board in boards:
        grid = board.get("grid")
        if not grid:
            continue
        max_value = int(board["rows"]) * int(board["cols"])
        for ratio in config.ratios:
            for mask_idx in range(config.masks_per_ratio):
                rng = _rng_for_group(str(board["board_id"]), float(ratio), mask_idx)
                masked = create_masked_board(grid, float(ratio), rng)
                for target in range(1, max_value + 1):
                    group_id = (
                        f"{board['board_id']}::r{int(ratio * 100)}::m{mask_idx:03d}::t{target:03d}"
                    )
                    rows_out.extend(build_rows_for_group(board, masked, group_id, float(ratio), target))
    return pd.DataFrame(rows_out)


def write_rank_dataset(df: pd.DataFrame, out_path: Path, shard_rows: int = 0) -> List[str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if shard_rows <= 0 or len(df) <= shard_rows:
        df.to_parquet(out_path, index=False)
        return [str(out_path)]

    shard_dir = out_path.with_suffix("")
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[str] = []
    for i in range(0, len(df), shard_rows):
        shard_path = shard_dir / f"shard_{i // shard_rows:05d}.parquet"
        df.iloc[i : i + shard_rows].to_parquet(shard_path, index=False)
        manifest.append(str(shard_path))
    (shard_dir / "manifest.json").write_text(
        json.dumps({"files": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
