# split_converted_excel.py
"""
说明：把 converted_excel_samples.json 中每个工作表拆出来，
存成独立的 JSON 文件到 json_samples/ 目录，格式需包含最外层 "grid"。
"""

import os
import json

# 这两个常量就是你的输入/输出
INPUT_PATH = "converted_excel_samples.json"
OUTPUT_DIR = "json_samples"

# 确保输出目录一定存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 先加载这个大合并的 JSON
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# data 结构类似：
# {
#   "8x10-10x12.xlsx": {
#     "Sheet1": { "rows":..., "cols":..., "grid": [...] },
#     "Sheet2": { "rows":..., "cols":..., "grid": [...] }
#   },
#   "4x5樣本.xlsx": {
#     "Sheet1": { "rows":..., "cols":..., "grid": [...] }
#   },
#   ...
# }

# 2. 遍历每个 Excel 文件名（如 "8x10-10x12.xlsx"）和它的每个 sheet
for excel_filename, sheets_dict in data.items():
    # 去掉 .xlsx，形成基础名字
    base_name = os.path.splitext(excel_filename)[0]  # 例如 "8x10-10x12"

    # sheets_dict 是一个 dict，key=sheet 名称，value=该 sheet 本身的数据
    for sheet_name, sheet_content in sheets_dict.items():
        # 该 sheet_content 应该有 "grid" 这个 key
        if "grid" not in sheet_content:
            # 如果没有 grid，就跳过
            continue

        # 3. 构造输出文件名，比如 "8x10-10x12_Sheet1.json"
        out_filename = f"{base_name}_{sheet_name}.json"
        out_path = os.path.join(OUTPUT_DIR, out_filename)

        # 4. 把这个 sheet_content（包含 rows, cols, grid）直接写成 单独的 JSON
        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(sheet_content, out_f, ensure_ascii=False, indent=2)

        print(f"已输出：{out_path}")