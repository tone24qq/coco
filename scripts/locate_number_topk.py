#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 功能：在含 -1 的盤面上查詢「指定數字」在空格中的 Top-K 位置與機率。

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.inference.model_loader import load_model
from src.inference.topk import compute_topk_positions
from src.models.vocab import masked_logits_clip


def main() -> None:  # pragma: no cover - CLI
    p = argparse.ArgumentParser(description="查詢指定數字在空格中的 Top-K 位置與機率")
    p.add_argument("ckpt", help="模型權重檔路徑")
    p.add_argument("in_json", help="輸入 JSON（boards:[{grid:[[...], ...]}]）")
    p.add_argument("out_json", help="輸出 JSON")
    p.add_argument("--query_num", type=int, required=True, help="要評分的數字（1..N）")
    p.add_argument("--topk_pos", type=int, default=3, help="回傳前 K 個最可能位置")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("【載入資料】讀取輸入 JSON ...")
    data = json.loads(Path(args.in_json).read_text())
    boards = data["boards"]

    print(f"【載入模型】路徑: {args.ckpt}  裝置: {args.device}")
    model = load_model(args.ckpt, device=args.device)

    outputs = []
    for bi, b in enumerate(boards):
        grid = b["grid"]
        rows, cols = len(grid), len(grid[0])
        N = rows * cols
        print(f"\n=== 盤面 {bi} ===")
        print(f"尺寸：{rows}x{cols}（N={N}）；-1 視為空格")

        flat = []
        holes = []
        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v == -1:
                    flat.append(0)
                    holes.append(True)
                else:
                    flat.append(v)
                    holes.append(False)
        tokens = torch.tensor(flat, dtype=torch.long, device=args.device).unsqueeze(0)
        hole_mask = torch.tensor(holes, dtype=torch.bool, device=args.device)
        attn = torch.ones_like(tokens, dtype=torch.bool)

        print(f"【查詢】評分指定數字：{args.query_num}（僅在空格位置）")
        with torch.no_grad():
            logits = model(tokens, attn)
            logits = masked_logits_clip(logits, N)
            probs = torch.softmax(logits, dim=-1)[0]
            topk = compute_topk_positions(
                probs, hole_mask, args.query_num, args.topk_pos, cols
            )
        print("【TopK 位置（row, col, prob）】")
        for item in topk:
            print(f"  -> ({item['row']}, {item['col']}) 機率={item['prob']:.6f}")

        outputs.append({"query_num": int(args.query_num), "topk_positions": topk})

    Path(args.out_json).write_text(
        json.dumps({"boards": outputs}, ensure_ascii=False, indent=2)
    )
    print(f"\n【輸出完成】寫入 -> {args.out_json}")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
