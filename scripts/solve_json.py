#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.inference.decode_temp import iterative_decode_temp
from src.inference.model_loader import load_model


def main() -> None:  # pragma: no cover - CLI
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("in_json")
    p.add_argument("out_json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--fill_ratio", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--topp", type=float, default=None)
    args = p.parse_args()

    data = json.loads(Path(args.in_json).read_text())
    boards = data["boards"]
    model = load_model(args.ckpt, device=args.device)

    outputs = []
    for b in boards:
        grid = b["grid"]
        rows, cols = len(grid), len(grid[0])
        N = rows * cols
        flat = []
        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v == -1:
                    v = 0
                flat.append(v)
        tokens = torch.tensor(flat, dtype=torch.long, device=args.device).unsqueeze(0)
        attn = torch.ones_like(tokens, dtype=torch.bool)
        out = iterative_decode_temp(
            model,
            tokens,
            attn,
            N,
            steps=args.steps,
            fill_ratio=args.fill_ratio,
            temperature=args.temperature,
            topk=args.topk,
            topp=args.topp,
        )
        solved = out.view(rows, cols).tolist()
        outputs.append({"grid": solved})

    Path(args.out_json).write_text(
        json.dumps({"boards": outputs}, ensure_ascii=False, indent=2)
    )
    print(f"wrote -> {args.out_json}")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
