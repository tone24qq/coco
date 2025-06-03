# excel_to_json_samples.py

"""
腳本功能：
  - 扫描 'excel_files/' 資料夾下所有 .xlsx 檔案
對每個 Excel 檔的每個工作表：
  1) 讀取整張表，禁止自動把第一列當 header，所有欄位以純文字處理
  2) 空白或 NaN 直接視為 -1
  3) 每個 cell 若為 str：
     - 先把 'O' → '0'，'I' → '1'
     - 再用正則挑出所有数字字符，如果符合純数字，轉成 int；否則填 -1
  4) 若為 int 或 float（非空）則保留 int(x)
  5) 其餘情況都 -1
  6) 統計最大列長度（cols），把短列自動補成長度一致
  7) 產生 JSON：
     {
       "rows": R,
       "cols": C,
       "grid": [...清洗後的 R×C 二維整數陣列...]
     }
  8) 輸出到 'json_samples/{ExcelBaseName}_{SheetName}.json'

使用方式：
  1. 把要轉的 .xlsx 都放進 'excel_files/' 資料夾
  2. 執行 python excel_to_json_samples.py
  3. 看 'json_samples/' 資料夾底下是否生成了對應的 JSON 檔
"""

import os
import re
import json
import pandas as pd

# -------------------------------------------------------------
#  1) 常數：輸入/輸出資料夾
# -------------------------------------------------------------
EXCEL_DIR = "excel_files"
OUTPUT_DIR = "json_samples"

# 確保輸出資料夾存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
#  2) 清洗單元格函式 clean_cell()
#     規則：
#       - 如果是 int、float（且非 NaN） → int()
#       - 如果是 str：
#           先把 'O'→'0', 'I'→'1'
#           用正則挑出所有數字字符，若全部字符都是數字，就轉 int；否則 -1
#       - 其他情況（比如 None、NaN、list、dict） → -1
# -------------------------------------------------------------
def clean_cell(x) -> int:
    # float 也可能是 NaN
    if isinstance(x, (int, float)):
        try:
            # pandas 讀出 NaN 會被視為 float("nan")
            if pd.isna(x):
                return -1
        except Exception:
            pass
        try:
            return int(x)
        except Exception:
            return -1

    if isinstance(x, str):
        # 先把 'O'→'0', 'I'→'1'
        s = x.replace("O", "0").replace("I", "1").strip()
        # 用正則抓出連續數字
        m = re.search(r"^\d+$", s)
        if m:
            return int(s)
        # 如果整串不全是數字，就挑出所有數字，若挑出的結果是空，回 -1
        digits = "".join(re.findall(r"\d+", s))
        if digits != "":
            return int(digits)
        return -1
    # 其他類型，都當 -1
    return -1

# -------------------------------------------------------------
#  3) 主要入口
# -------------------------------------------------------------
def main():
    # 列出 excel_files/ 底下所有 .xlsx、.xls 檔
    files = [f for f in os.listdir(EXCEL_DIR)
             if f.lower().endswith((".xlsx", ".xls"))]

    if not files:
        print(f"❌ 沒有在 '{EXCEL_DIR}' 找到任何 Excel 檔（.xlsx 或 .xls）。")
        return

    # 逐一處理每個 Excel 檔
    for filename in files:
        base_name, _ = os.path.splitext(filename)
        full_path = os.path.join(EXCEL_DIR, filename)
        print(f"\n▶ 正在處理：{full_path}")

        try:
            xls = pd.ExcelFile(full_path)
        except Exception as e:
            print(f"  ❌ 無法讀取 {filename}：{e}")
            continue

        # 處理這個檔案的每個工作表
        for sheet_name in xls.sheet_names:
            try:
                # 不要 header，全部當成 raw data 讀，dtype=object
                df = pd.read_excel(full_path,
                                   sheet_name=sheet_name,
                                   header=None,
                                   dtype=object)
            except Exception as e:
                print(f"  ❌ 無法讀取工作表 '{sheet_name}'：{e}")
                continue

            raw = df.values.tolist()
            rows = len(raw)
            # 計算這張表最大列長度
            cols = max(len(r) for r in raw) if rows > 0 else 0

            # 如果整個 sheet 沒任何數據，跳過
            if rows == 0 or cols == 0:
                print(f"  ⚠ 工作表 '{sheet_name}' 是空的，跳過。")
                continue

            # 清洗後的二維 int 陣列
            grid = []
            for row in raw:
                # 補齊到 cols
                extended = list(row) + ["" for _ in range(cols - len(row))]
                cleaned_row = [clean_cell(cell) for cell in extended]
                grid.append(cleaned_row)

            # 打包成一個 dict
            out_dict = {
                "rows": rows,
                "cols": cols,
                "grid": grid
            }

            # 輸出檔名 = "{ExcelBaseName}_{SheetName}.json"
            safe_sheet = sheet_name.replace(" ", "_").replace("/", "_")
            out_filename = f"{base_name}_{safe_sheet}.json"
            out_path = os.path.join(OUTPUT_DIR, out_filename)

            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out_dict, f, ensure_ascii=False, indent=2)
                print(f"  ✅ 已輸出：{out_path}")
            except Exception as e:
                print(f"  ❌ 寫 JSON 失敗：{e}")

    print("\n🎉 全部完成，請到 'json_samples/' 檢查輸出結果。")

if __name__ == "__main__":
    import pandas as pd
    main()