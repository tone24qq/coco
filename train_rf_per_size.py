#!/usr/bin/env python3
# train_rf_per_size.py

import argparse
import json
import os
import zipfile
from typing import List, Tuple

import numpy as np


def extract_features(board: np.ndarray, r: int, c: int) -> List[float]:
    """
    对位置 (r,c) 提取特征：
    - r, c, rows, cols
    - 全盘已知值的 count, mean, std
    - 同行已知值的 count, mean, std
    - 同列已知值的 count, mean, std
    - 3×3 邻域（包括自身，越界用 -1 填充）
    """
    rows, cols = board.shape
    feats: List[float] = []

    # 位置和尺寸
    feats += [r, c, rows, cols]

    # 全盘统计
    known = board[board >= 0]
    feats += [
        float(known.size),
        float(known.mean()) if known.size else 0.0,
        float(known.std()) if known.size else 0.0,
    ]

    # 行统计
    row_vals = board[r, :]
    row_known = row_vals[row_vals >= 0]
    feats += [
        float(row_known.size),
        float(row_known.mean()) if row_known.size else 0.0,
        float(row_known.std()) if row_known.size else 0.0,
    ]

    # 列统计
    col_vals = board[:, c]
    col_known = col_vals[col_vals >= 0]
    feats += [
        float(col_known.size),
        float(col_known.mean()) if col_known.size else 0.0,
        float(col_known.std()) if col_known.size else 0.0,
    ]

    # 3×3 邻域
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                feats.append(float(board[rr, cc]))
            else:
                feats.append(-1.0)
    return feats


def extract_all_features(
    boards: List[List[List[int]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 boards（每个 board 是一个 rows×cols 的二维列表）转换为特征矩阵 X
    和标签向量 y。
    对每个位置都当作“缺失”来提取一次特征，并将原始值作为 y。
    """
    X_list: List[List[float]] = []
    y_list: List[int] = []

    for b in boards:
        board = np.array(b, dtype=int)
        rows, cols = board.shape
        # 对每个位置循环
        for r in range(rows):
            for c in range(cols):
                true_val = int(board[r, c])
                # 暂时置为 -1，模拟缺失
                board[r, c] = -1
                feats = extract_features(board, r, c)
                X_list.append(feats)
                y_list.append(true_val)
                # 恢复
                board[r, c] = true_val

    # 转为 numpy，并降精度
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.uint16)
    return X, y


def process_zip(zip_path: str, out_dir: str) -> None:
    """
    逐个打开 zip，读取其中所有 *.json，按文件名 (e.g. 4x5.json) 
    确定尺寸，提取特征并保存为 compressed .npz
    """
    zf = zipfile.ZipFile(zip_path)
    for fname in zf.namelist():
        if not fname.lower().endswith(".json"):
            continue

        base = os.path.splitext(os.path.basename(fname))[0]
        try:
            rows, cols = map(int, base.lower().split("x"))
        except ValueError:
            print(f"跳过未识别尺寸文件: {fname}")
            continue

        print(f"\n→ 处理尺寸 {rows}x{cols}，文件 {fname}")
        data = json.loads(zf.read(fname).decode("utf-8"))
        if not isinstance(data, list):
            print(f"  警告: {fname} 中不是列表，跳过")
            continue

        X, y = extract_all_features(data)
        print(f"  特征矩阵: X={X.shape}, y={y.shape}")

        # 确保输出目录
        size_dir = os.path.join(out_dir, f"{rows}x{cols}")
        os.makedirs(size_dir, exist_ok=True)
        save_path = os.path.join(size_dir, f"{base}_features.npz")

        # 保存为压缩 npz
        np.savez_compressed(save_path, X=X, y=y)
        print(f"  已保存特征: {save_path}")


def main():
    p = argparse.ArgumentParser(
        description="从多个 ZIP 中提取各 board 尺寸的特征，生成压缩的 .npz 文件"
    )
    p.add_argument(
        "--zip",
        required=True,
        help="单个 ZIP 文件路径，或包含多个 ZIP 的目录",
    )
    p.add_argument(
        "--outdir",
        default="features",
        help="保存 .npz 的输出目录（默认 features/）",
    )
    args = p.parse_args()

    zip_arg = args.zip
    out_dir = args.outdir
    if os.path.isdir(zip_arg):
        zip_files = sorted(
            os.path.join(zip_arg, fn)
            for fn in os.listdir(zip_arg)
            if fn.lower().endswith(".zip")
        )
    else:
        zip_files = [zip_arg]

    for z in zip_files:
        print(f"\n====== 处理 ZIP: {z} ======")
        process_zip(z, out_dir)


if __name__ == "__main__":
    main()
