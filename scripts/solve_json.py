#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 功能：讀取含 -1 的盤面（-1 視為空格=MASK=0），查詢「指定數字」在空格中的 Top-K 位置與機率，
#      並同場輸出完整解（使用既有 iterative_decode_temp）。全程印出中文 log。

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.inference.decode_temp import iterative_decode_temp
from src.inference.model_loader import load_model
from src.inference.topk import compute_topk_positions
from src.models.vocab import masked_logits_clip


def main() -> None:  # pragma: no cover - CLI
    p = argparse.ArgumentParser(
        description="根據盤面查詢指定數字在空格中的 Top-K 位置與機率（含完整解）"
    )
    p.add_argument("ckpt", help="模型權重檔路徑")
    p.add_argument("in_json", help="輸入 JSON（boards:[{grid:[[...], ...]}]）")
    p.add_argument("out_json", help="輸出 JSON")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=8, help="迭代填格步數")
    p.add_argument("--fill_ratio", type=float, default=0.3, help="每步填入比例（0~1）")
    p.add_argument("--temperature", type=float, default=1.0, help="取樣溫度")
    p.add_argument("--topk", type=int, default=None, help="解碼時的 top-k（可選）")
    p.add_argument("--topp", type=float, default=None, help="解碼時的 top-p（可選）")
    p.add_argument("--query_num", type=int, default=None, help="要評分的數字（1..N）")
    p.add_argument("--topk_pos", type=int, default=3, help="回傳前 K 個最可能位置")
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
                holes.append(v == -1)
                if v == -1:
                    v = 0
                flat.append(v)
        tokens = torch.tensor(flat, dtype=torch.long, device=args.device).unsqueeze(0)
        hole_mask = torch.tensor(holes, dtype=torch.bool, device=args.device)
        attn = torch.ones_like(tokens, dtype=torch.bool)

        query_topk = None
        if args.query_num is not None:
            if 1 <= args.query_num <= N:
                print(f"【查詢】評分指定數字：{args.query_num}（僅在空格位置）")
                with torch.no_grad():
                    logits = model(tokens, attn)
                    logits = masked_logits_clip(logits, N)
                    probs = torch.softmax(logits, dim=-1)[0]
                    query_topk = compute_topk_positions(
                        probs, hole_mask, args.query_num, args.topk_pos, cols
                    )
                if query_topk:
                    print("【TopK 位置（row, col, prob）】")
                    for item in query_topk:
                        print(
                            f"  -> ({item['row']}, {item['col']}) 機率={item['prob']:.6f}"
                        )
                else:
                    print("（沒有空格，略過查詢）")
            else:
                print(f"（query_num={args.query_num} 超出範圍 1..{N}，略過查詢）")

        print("【解碼】開始迭代填格（可能使用 temperature/topk/topp） ...")
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
        print("【完成】本盤面已解出。")

        item = {"grid": solved}
        if query_topk is not None:
            item["query_num"] = int(args.query_num)
            item["topk_positions"] = query_topk
        outputs.append(item)

    Path(args.out_json).write_text(
        json.dumps({"boards": outputs}, ensure_ascii=False, indent=2)
    )
    print(f"\n【輸出完成】寫入 -> {args.out_json}")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
