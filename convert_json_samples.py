# convert_json_samples.py
"""
脚本功能：把原始卡片 JSON（放在 json_samples/ 目录下）逐张清洗并拆解样本，
输出“记忆样本”到 memory_data/all.json，每个样本带 grid, target, true_pos, scores, fused_score, timestamp 等字段。

使用方法：
1. 确保本脚本位于 project_root 目录下。
2. 在 project_root 同级创建文件夹：
     - json_samples/      ← 放 sample1_8x10.json, sample2_8x10.json, … 
     - memory_data/       ← 可以先空着
3. 运行： python convert_json_samples.py
4. 脚本会把所有示例卡拆成样本并输出到 memory_data/all.json。
"""

import os
import json
import time
import re
import numpy as np

# 从 new_module3.py 中直接 import REGISTERED_MODULES_BRAIN
from new_module3 import REGISTERED_MODULES_BRAIN
# 从 analyzer11.py 中 import 必要函数
from analyzer11 import collect_all_scores, normalize_tensor, fuse_scores

# 1) 原始卡片 JSON 存放目录
JSON_FOLDER = "json_samples"

# 2) 转换后输出的记忆样本文件路径
OUTPUT_PATH = "memory_data/all.json"

# 确保 memory_data 目录存在
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 用来存放所有拆解出来的“遮蔽样本”的列表
all_samples = []


def clean_cell(x):
    """
    把单元格里的原始值（可能是 int、也可能是 str）清洗成整数或 -1：
      - 如果就是 int，直接返回该 int；
      - 如果是 str，尝试用正则提取第一个连续数字 (例："O\n8" → "8")，否则返回 -1；
      - 其它情况都返回 -1。
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
    1) 读取一个原始卡片 JSON（必须包含 "grid" 字段，grid 是 2D 列表）；
    2) 如果 JSON 里有 "rows"/"cols" 字段，则以它们为准；否则用 len(grid) / len(grid[0])；
    3) 对每一行：
         - 如果不是 list，跳过（视为 Footer）；
         - 用 clean_cell() 把每个单元格清洗成 int 或 -1；
         - 如果这一行长度不等于 declared_cols，也跳过（Footer）；
         - 如果这一行全部都是 -1，也跳过（Footer）；
         - 否则把这一行加入 cleaned 列表；
    4) 返回一个 numpy 二维数组 cleaned_arr；如果 cleaned 为空，则返回 shape=(0,0) 的空数组。
    """
    print(f"正在读取并清洗：{json_path}")
    data = json.load(open(json_path, "r", encoding="utf-8"))

    raw_grid = data.get("grid", None)
    if raw_grid is None:
        print(f"  → 该 JSON 没有 'grid' 字段，跳过此文件。")
        return np.empty((0, 0), dtype=int)

    # 如果 JSON 指定了 rows, cols，就用它们；否则尝试自动推
    declared_rows = data.get("rows", None)
    declared_cols = data.get("cols", None)
    if declared_rows is None or declared_cols is None:
        if isinstance(raw_grid, list) and len(raw_grid) > 0 and isinstance(raw_grid[0], list):
            declared_rows = len(raw_grid)
            declared_cols = len(raw_grid[0])
        else:
            declared_rows, declared_cols = 0, 0

    cleaned = []
    for row in raw_grid:
        if not isinstance(row, list):
            # 如果这一行不是 list，则视为 Footer，跳过
            continue
        cleaned_row = [clean_cell(cell) for cell in row]
        if len(cleaned_row) != declared_cols:
            # 长度不一致，也跳过
            continue
        # 如果整行全是 -1，也当 Footer 跳过
        if all(val == -1 for val in cleaned_row):
            continue
        cleaned.append(cleaned_row)

    # 转成 numpy array
    arr = np.array(cleaned, dtype=int) if cleaned else np.empty((0, 0), dtype=int)
    print(f"  → 清洗后矩阵 shape = {arr.shape}")
    return arr


def process_card(arr: np.ndarray, source_name: str) -> list[dict]:
    """
    给一个“已清洗”的卡片阵列 arr (shape = (rows, cols)，元素要么正整数，要么 -1)，
    对 arr 里每个 >0 的值当作 target：
      a) 找到 target 在 arr 中的位置 (r,c)，复制 arr 为 grid_masked，把 (r,c) 设为 -1；
      b) 调用 collect_all_scores(grid_masked) → 得到 tensor (形状 (num_mod, rows, cols))；
      c) 从 tensor 中采集 raw_scores：每个模块在 (r,c) 的值 → raw_scores[name]；
      d) normalize_tensor + fuse_scores → 得到 fused_matrix；取 fused_matrix[r,c] 作为 fused_score；
      e) 生成一条 dict 样本：
         {
           "grid": grid_masked.tolist(),
           "target": target,
           "true_pos": [r, c],
           "scores": raw_scores,
           "fused_score": fused_score,
           "timestamp": 当前时间戳（int）
         }
    返回该卡生成的所有样本列表。
    """
    if arr.size == 0:
        print(f"  → 来源 {source_name} 的卡片为空，跳过。")
        return []

    samples = []
    rows, cols = arr.shape

    # 找出所有 >0 的值，作为要轮流遮蔽的 target 列表
    values = sorted(set(v for v in arr.flatten() if v > 0))
    print(f"  → {source_name} 有 {len(values)} 个已知数值 (示例前 5 个：{values[:5]})")
    for target in values:
        positions = np.argwhere(arr == target)
        if positions.shape[0] == 0:
            continue
        r, c = int(positions[0][0]), int(positions[0][1])

        # 复制一份矩阵，把 (r,c) 这个 target 改成 -1（遮蔽）
        grid_masked = arr.copy()
        grid_masked[r, c] = -1

        # 1) collect_all_scores + 返回 tensor (num_mod, rows, cols)
        tensor = collect_all_scores(grid_masked, request_id="convert")
        if tensor.size == 0:
            # 如果没有任何模块返回分数，就跳过
            continue

        # 2) 从 tensor 提取 raw_scores：各模块在 (r,c) 的评分
        raw_scores = {}
        for i, name in enumerate(REGISTERED_MODULES_BRAIN.keys()):
            raw_scores[name] = float(tensor[i, r, c])

        # 3) normalize + fuse，得到 fused_matrix (rows, cols)，取 (r,c) 作为 fused_score
        tensor_norm = normalize_tensor(tensor, method="minmax")
        fused_matrix = fuse_scores(tensor_norm, weights=None)
        fused_score = float(fused_matrix[r, c])

        # 4) 打包成样本 dict
        samples.append({
            "grid": grid_masked.tolist(),
            "target": target,
            "true_pos": [r, c],
            "scores": raw_scores,
            "fused_score": fused_score,
            "timestamp": int(time.time())
        })

    return samples


def main():
    all_samples.clear()

    # 遍历 json_samples/ 下所有 .json 文件
    for fn in os.listdir(JSON_FOLDER):
        if not fn.lower().endswith(".json"):
            continue
        fullpath = os.path.join(JSON_FOLDER, fn)

        # 先加载并清洗
        arr = load_and_clean(fullpath)
        if arr is None or arr.size == 0:
            # 如果返回空，就跳过这张卡
            continue

        # 对卡做样本生成
        card_samples = process_card(arr, fn)
        all_samples.extend(card_samples)
        print(f"{fn} → 生成 {len(card_samples)} 个样本")

    print("==========================")
    print(f"合计生成 {len(all_samples)} 个样本，准备写入 {OUTPUT_PATH}")

    # 最后把所有样本写到 memory_data/all.json，并自动把 np.int64 / np.float 转成 Python 基本类型
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            all_samples,
            f,
            ensure_ascii=False,
            indent=2,
            default=lambda o: o.item() if hasattr(o, "item") else str(o)
        )
    print("写入完成。")


if __name__ == "__main__":
    main()