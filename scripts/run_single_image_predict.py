#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.board_export import grid_to_text, write_board_csv, write_board_json, write_overlay  # noqa: E402
from src.board_structurer import structure_board  # noqa: E402
from src.grid_detector import detect_grid  # noqa: E402
from src.masking_eval.candidate_scoring import legal_candidates, score_candidate  # noqa: E402
from src.masking_eval.modules import BASE_MODULES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--target-row", type=int, required=True)
    parser.add_argument("--target-col", type=int, required=True)
    parser.add_argument("--config", default="configs/board_parse.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cannot_read_image: {args.image}")
    det = detect_grid(gray)
    result = structure_board(sample_id=Path(args.image).stem, image_path=args.image, detection=det)

    write_board_json(result, Path(cfg["outputs"]["json"]))
    write_board_csv(result, Path(cfg["outputs"]["csv"]))
    write_overlay(result, det, args.image, Path(cfg["outputs"]["overlay"]))

    grid = np.array([[v if v is not None else -1 for v in row] for row in result.grid], dtype=int)
    tr, tc = args.target_row, args.target_col
    if not (0 <= tr < grid.shape[0] and 0 <= tc < grid.shape[1]):
        raise ValueError("target out of range")
    if grid[tr, tc] != -1:
        grid[tr, tc] = -1

    if result.parse_confidence < float(cfg["parser"]["min_confidence"]):
        low = {"status": "parse_confidence_too_low", "parse_confidence": result.parse_confidence}
        Path("reports/predict_summary.json").write_text(json.dumps(low, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(low, ensure_ascii=False))
        return

    candidates = legal_candidates(grid)
    weights = {m: 1.0 for m in BASE_MODULES}
    scored = []
    for cand in candidates:
        feats = score_candidate(grid, (tr, tc), cand, heatmap_prior=None, modules=BASE_MODULES)
        scored.append({"candidate": cand, "score": float(sum(feats[k] * weights[k] for k in feats))})
    scored.sort(key=lambda x: x["score"], reverse=True)

    out = {
        "shape": result.shape,
        "parse_confidence": result.parse_confidence,
        "target_cell": [tr, tc],
        "top1": scored[:1],
        "top3": scored[:3],
        "top5": scored[:5],
        "grid_preview": grid_to_text(result.grid),
    }
    Path("reports/predict_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
