#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.board_export import grid_to_text, write_board_csv, write_board_json, write_overlay  # noqa: E402
from src.board_structurer import structure_board  # noqa: E402
from src.grid_detector import detect_grid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
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

    print(grid_to_text(result.grid))
    print(json.dumps({
        "shape": result.shape,
        "parse_confidence": result.parse_confidence,
        "low_confidence_cells": len(result.low_confidence_cells),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
