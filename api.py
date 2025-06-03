# convert_json_samples.py

import os
import json
import time
import re
import numpy as np

# 直接从 new_module3 里 import REGISTERED_MODULES_BRAIN
from new_module3 import REGISTERED_MODULES_BRAIN
from analyzer11 import collect_all_scores, normalize_tensor, fuse_scores

# 输入的原始卡片 JSON 文件夹
JSON_FOLDER = "json_samples"          
# 转换后输出的记忆样本文件
OUTPUT_PATH = "memory_data/all.json"  

# 确保 memory_data 文件夹存在
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 用来收集所有样本的列表
all_samples = []


def clean_cell(x):
    """
    把单元格的原始值 (可能是 int、也可能是 str) “清洗”成 int：
    - 如果就是 int，直接返回；
    - 如果是 str，取其中第一串数字 (如 "O\n8" → "8")，否则返回 -1。
    - 其它情况都当 -1（空格）。
    """
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        m = re.search(r"\d+", x)
        if m:
            return int(m.group(0))
    return -1


def load_and_clean(json_path: str) -> np.ndarray:
    """
    1) 读入原始卡片 JSON（里面需含 "grid" 字段）；
    2) 如果 JSON 里有 "rows"/"cols"，就以它为准，否则用 len(grid) / len(grid[0])；
    3) 对每一行：
       - 如果长度不等于 declared_cols，就当作 footer 跳过；
       - 用 clean_cell() 把每个单元格清洗成 int 或 -1；
       - 如果整行都是 -1，就当作 footer 跳过；
       - 剩下的行才存到 `cleaned` 列表里；
    4) 返回一个 numpy 数组；如果清洗后没有任何行，就返回 shape=(0,0)。
    """
    print(f"正在读取并清洗：{json_path}")
    data = json.load(open(json_path, "r", encoding="utf-8"))

    raw_grid = data.get("grid", None)
    if raw_grid is None:
        print(f"  → 这个 JSON 没有 'grid' 字段，跳过。")
        return np.empty((0, 0), dtype=int)

    # 如果有 rows/cols 就用它们，否则自动推断
    declared_rows = data.get("rows", None)
    declared_cols = data.get("cols", None)
    if declared_rows is None or declared_cols is None:
        declared_rows = len(raw_grid)
        declared_cols = len(raw_grid[0]) if raw_grid and isinstance(raw_grid[0], list) else 0

    cleaned = []
    for row in raw_grid:
        if not isinstance(row, list):
            # 如果这一行不是 list，就当 footer 跳过
            continue
        cleaned_row = [clean_cell(cell) for cell in row]
        if len(cleaned_row) != declared_cols:
            # 长度不对，跳过
            continue
        # 如果整行都是 -1，也跳过
        if all(val == -1 for val in cleaned_row):
            continue
        cleaned.append(cleaned_row)

    arr = np.array(cleaned, dtype=int) if cleaned else np.empty((0, 0), dtype=int)
    print(f"  → 清洗后矩阵 shape = {arr.shape}")
    return arr


def process_card(arr: np.ndarray, source_name: str) -> list[dict]:
    """
    对传进来的“干净卡片矩阵” arr (二维 numpy int array)，
    依次把每个正整数当作 target 遮成 -1 → 收集各模块 raw_scores → normalize + fuse → 生成一笔记忆样本 dict。
    返回该张卡所生成的所有样本列表 (List[dict])。
    """
    if arr.size == 0:
        print(f"  → 来自 {source_name} 的卡片数组为空，跳过。")
        return []

    samples = []
    rows, cols = arr.shape

    # arr.flatten() 里所有 >0 的，是我们要轮流遮掉当 target 的“已知数字”
    values = sorted(set(v for v in arr.flatten() if v > 0))
    print(f"  → {source_name} 有 {len(values)} 个已知数值 (前几个示例：{values[:5]})")

    for target in values:
        positions = np.argwhere(arr == target)
        if positions.shape[0] == 0:
            continue
        r, c = int(positions[0][0]), int(positions[0][1])

        # 复制一份矩阵，把 (r,c) 这个 target 暂时设为 -1（遮蔽）
        grid_masked = arr.copy()
        grid_masked[r, c] = -1

        # 1) collect_all_scores → 得到 (num_mod, rows, cols) 的 tensor
        tensor = collect_all_scores(grid_masked, request_id="convert")
        if tensor.size == 0:
            # 如果所有模块都没分数，就跳过这次 target
            continue

        # 2) raw_scores：取每个模块在 (r,c) 的分
        raw_scores = {}
        for i, name in enumerate(REGISTERED_MODULES_BRAIN.keys()):
            raw_scores[name] = float(tensor[i, r, c])

        # 3) normalize + fuse，得到 fused_score
        tensor_norm = normalize_tensor(tensor, method="minmax")
        fused_matrix = fuse_scores(tensor_norm, weights=None)
        fused_score = float(fused_matrix[r, c])

        # 4) 把一笔样本打包起来
        samples.append({
            "grid": grid_masked.tolist(),   # 2D list
            "target": target,
            "true_pos": [r, c],
            "scores": raw_scores,
            "fused_score": fused_score,
            "timestamp": int(time.time())
        })

    return samples


def main():
    all_samples.clear()

    # 逐个读取 json_samples/ 下所有 .json
    for fn in os.listdir(JSON_FOLDER):
        if not fn.lower().endswith(".json"):
            continue
        fullpath = os.path.join(JSON_FOLDER, fn)

        arr = load_and_clean(fullpath)
        if arr is None or arr.size == 0:
            # load_and_clean 遇到无 grid 或清洗后空，就返回 shape=(0,0) 的 arr
            continue

        # 对这张卡做样本生成
        card_samples = process_card(arr, fn)
        all_samples.extend(card_samples)
        print(f"{fn} → 生成 {len(card_samples)} 个样本")

    print("==========================")
    print(f"合计生成 {len(all_samples)} 个样本，准备写入 {OUTPUT_PATH}")

    # 最后把所有样本一次写入 memory_data/all.json
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print("写入完成。")


if __name__ == "__main__":
    main()