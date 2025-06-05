
"""convert_excel.py (pandas 版)
依據 Excel 讀取與預處理條例 1–5，將 .xlsx/.xls 轉成標準樣本 JSON
* 所有欄位以字串讀取
* 空白格 -> ""
* O→0, I→1, 全形數字→半形；純數字轉 int，其餘 -> -1
* 行、列索引皆 1-base
* 輸出到 data/samples/*.json
"""
import json, uuid, pathlib, re, sys
import pandas as pd
import numpy as np

BASE = pathlib.Path(__file__).resolve().parent.parent
SRC  = BASE / "data" / "excels"
DST  = BASE / "data" / "samples"
DST.mkdir(parents=True, exist_ok=True)

TRANS = {"O":"0","I":"1", **{chr(ord('０')+i):str(i) for i in range(10)}}
TRANS_MAP = str.maketrans(TRANS)
FLOAT_INT_RE = re.compile(r"^\d+\.0+$")

def clean_cell(cell):
    if cell == "" or cell is None:
        return ""
    if isinstance(cell, (int, np.integer)):
        return int(cell)
    if isinstance(cell, (float, np.floating)):
        return int(cell) if cell.is_integer() else -1
    s = str(cell).strip().translate(TRANS_MAP).upper()
    if FLOAT_INT_RE.match(s):
        s = s.split('.')[0]
    return int(s) if s.isdigit() else -1

def looks_like_header(row):
    non_num = sum(not str(c).strip().isdigit() for c in row)
    return non_num >= len(row)*0.5

excels = list(SRC.glob('*.xls*'))
if not excels:
    print(f'❗ 未找到 Excel 檔，請放入 {SRC}')
    sys.exit(1)

for excel in excels:
    xl = pd.ExcelFile(excel, engine='openpyxl')
    for sheet in xl.sheet_names:
        df_raw = xl.parse(sheet, header=None, dtype=str, keep_default_na=False)
        if len(df_raw) and looks_like_header(df_raw.iloc[0]):
            df_raw = df_raw.iloc[1:].reset_index(drop=True)
        df_clean = df_raw.applymap(clean_cell)
        df_clean.index = np.arange(1, len(df_clean)+1)
        df_clean.columns = np.arange(1, len(df_clean.columns)+1)
        sample = {
            'grid': df_clean.values.tolist(),
            'target': -1,
            'answer': [-1,-1],
            'size': f"{df_clean.shape[0]}x{df_clean.shape[1]}",
            'source': f"{excel.name}:{sheet}"
        }
        out = DST / f"{uuid.uuid4().hex}.json"
        json.dump(sample, open(out,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
        print(f'✓ 轉檔 {excel.name}:{sheet} -> {out.name}')
print('✓ 全部轉檔完成')
