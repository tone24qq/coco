#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_rf.py

加载已训练好的 RandomForest 模型，对单个遮蔽盘面做 Top-k 位置预测。
"""

import argparse
import json
import os

import joblib
import numpy as np


def extract_features(board: np.ndarray, r: int, c: int):
    """同训练时的特征函数，保证特征一致性。"""
    rows, cols = board.shape
    feats = []

    # 位置 & 尺寸
    feats += [r, c, rows, cols]

    # 全盘已知格统计
    known = board[board >= 0]
    feats += [
        known.size,
        float(known.mean()) if known.size else 0.0,
        float(known.std()) if known.size else 0.0,
    ]

    # 所在行统计
    row_vals = board[r, :]
    row_known = row_vals[row_vals >= 0]
    feats += [
        row_known.size,
        float(row_known.mean()) if row_known.size else 0.0,
        float(row_known.std()) if row_known.size else 0.0,
    ]

    # 所在列统计
    col_vals = board[:, c]
    col_known = col_vals[col_vals >= 0]
    feats += [
        col_known.size,
        float(col_known.mean()) if col_known.size else 0.0,
        float(col_known.std()) if col_known.size else 0.0,
    ]

    # 3x3 邻域依赖（中心含自身）
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                feats.append(int(board[rr, cc]))
            else:
                feats.append(-1)

    return np.array(feats, dtype=float)


def predict_top_k(model_path, board_path, k):
    # 1. 加载模型
    rf = joblib.load(model_path)

    # 2. 读取输入盘面 JSON
    with open(board_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    board = np.array(data["board"], dtype=int)
    target = int(data["target"])

    rows, cols = board.shape
    # 3. 对所有空格（-1）做特征并预测概率
    feats_list = []
    coords = []
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == -1:
                feats_list.append(extract_features(board, r, c))
                coords.append((r, c))

    X = np.vstack(feats_list)
    probs = rf.predict_proba(X)  # shape = (n_blanks, n_classes)
    # 找到 target 在 classes_ 中的索引
    try:
        idx = list(rf.classes_).index(target)
    except ValueError as exc:
        raise RuntimeError(f"模型不包含目标数字 {target}") from exc

    target_probs = probs[:, idx]

    # 4. 取 Top-k
    top_idx = np.argsort(target_probs)[-k:][::-1]
    results = []
    for i in top_idx:
        r, c = coords[i]
        results.append(
            {"r": int(r), "c": int(c), "prob": float(round(target_probs[i], 4))}
        )

    # 5. 返回结构
    return {"rows": rows, "cols": cols, "target": target, "predictions": results}


def main():
    p = argparse.ArgumentParser(description="用 RandomForest 已训练模型做盘面预测")
    p.add_argument("--model", required=True, help="模型文件路径，如 models/4x5.pkl")
    p.add_argument("--input", required=True, help="输入 JSON 文件，格式见 README")
    p.add_argument("--k", type=int, default=3, help="Top-k 候选数")
    p.add_argument("--output", required=True, help="输出 JSON 路径")
    args = p.parse_args()

    res = predict_top_k(args.model, args.input, args.k)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"预测结果已写入 {args.output}")


if __name__ == "__main__":
    main()
