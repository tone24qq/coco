#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.parse_board_image import parse_image_hybrid  # noqa: E402
from src.board_export import grid_to_text  # noqa: E402
from src.masking_eval.candidate_scoring import (
    legal_candidates,
    score_candidate,
)  # noqa: E402
from src.masking_eval.modules import BASE_MODULES  # noqa: E402
from src.number_position_predictor import predict_number_positions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--size-class", choices=["20", "80", "120"], default=None)
    parser.add_argument("--manual-grid", default=None)
    parser.add_argument("--override", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--target-row", type=int)
    parser.add_argument("--target-col", type=int)
    parser.add_argument("--query-number", type=int)
    args = parser.parse_args()

    mode_a = args.target_row is not None or args.target_col is not None
    mode_b = args.query_number is not None
    if mode_a and mode_b:
        raise ValueError("mode_conflict")
    if not mode_a and not mode_b:
        raise ValueError("mode_required")
    if mode_a and (args.target_row is None or args.target_col is None):
        raise ValueError("target_row_col_required")

    payload = parse_image_hybrid(args)
    if not payload.get("contract_passed"):
        print(
            json.dumps(
                {
                    "status": "reject_prediction",
                    "reason": payload.get("status"),
                    "source_mode": payload.get("source_mode"),
                    "contract_passed": False,
                },
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)

    if mode_b:
        query = predict_number_positions(
            grid=payload["grid"],
            query_number=int(args.query_number),
            missing_values=payload.get("missing_values", []),
            low_confidence_cells=payload.get("low_confidence_cells", []),
            black_cells=payload.get("black_cells", []),
            manual_override_cells={
                (x["row"] + 1, x["col"] + 1)
                for x in payload.get("parse_diagnostics", {}).get("override_audit", [])
                if "row" in x and "col" in x
            },
        )
        out = {
            "mode": "query_number_position",
            "source_mode": payload.get("source_mode"),
            "contract_passed": payload.get("contract_passed"),
            "grid_preview": grid_to_text(payload["grid"]),
            **query,
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    grid = np.array(
        [[v if v is not None else -1 for v in row] for row in payload["grid"]],
        dtype=int,
    )
    tr, tc = int(args.target_row), int(args.target_col)
    if not (0 <= tr < grid.shape[0] and 0 <= tc < grid.shape[1]):
        raise ValueError("target_out_of_range")
    if grid[tr, tc] != -1:
        grid[tr, tc] = -1
    candidates = legal_candidates(grid)
    weights = {m: 1.0 for m in BASE_MODULES}
    scored = []
    for cand in candidates:
        feats = score_candidate(
            grid, (tr, tc), cand, heatmap_prior=None, modules=BASE_MODULES
        )
        scored.append(
            {
                "candidate": cand,
                "score": float(sum(feats[k] * weights[k] for k in feats)),
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)

    out = {
        "mode": "target_cell_digit",
        "source_mode": payload.get("source_mode"),
        "contract_passed": payload.get("contract_passed"),
        "grid_preview": grid_to_text(payload["grid"]),
        "top1": scored[:1],
        "top3": scored[:3],
        "top5": scored[:5],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
