import json, uuid, pathlib, re, sys
import pandas as pd
import numpy as np

# ---------------- 路徑 ----------------
BASE = pathlib.Path(__file__).resolve().parent.parent
SRC  = BASE / "data" / "excels"        # Excel 來源
DST  = BASE / "data" / "samples"       # JSON 目的
DST.mkdir(parents=True, exist_ok=True)

# ---------------- 清洗 ----------------
TRANS = {
    "O": "0", "I": "1",
    **{chr(ord("０") + i): str(i) for i in range(10)}
}
TRANS_MAP = str.maketrans(TRANS)
FLOAT_INT_RE = re.compile(r"^\d+\.0+$")   # 113.0 → 113

def clean_cell(cell):
    """清洗每格內容，回傳 int 或 -1"""
    if cell == "" or cell is None:
        return -1
    if isinstance(cell, (int, np.integer)):
        return int(cell)
    if isinstance(cell, (float, np.floating)):
        return int(cell) if cell.is_integer() else -1
    s = str(cell).strip().translate(TRANS_MAP).upper()
    if FLOAT_INT_RE.match(s):
        s = s.split(".")[0]
    return int(s) if s.isdigit() else -1

def looks_like_header(row):
    """判斷首列是否為說明列（半數以上非純數字）"""
    non_num = sum(not str(c).strip().isdigit() for c in row)
    return non_num >= len(row) * 0.5

# ---------------- 轉檔 ----------------
excels = list(SRC.glob("*.xls*"))
if not excels:
    print(f"❗ 未找到 Excel 檔，請放入 {SRC}")
    sys.exit(1)

for f in excels:
    xl = pd.ExcelFile(f, engine="openpyxl")
    for sheet in xl.sheet_names:
        df_raw = xl.parse(sheet, header=None, dtype=str, keep_default_na=False)

        if len(df_raw) and looks_like_header(df_raw.iloc[0]):
            df_raw = df_raw.iloc[1:].reset_index(drop=True)

        df_clean = df_raw.applymap(clean_cell)

        # 刪除空白列（所有值皆為 -1）
        df_clean = df_clean.loc[~(df_clean == -1).all(axis=1)]

        df_clean.index = np.arange(1, len(df_clean) + 1)
        df_clean.columns = np.arange(1, len(df_clean.columns) + 1)

        print(f"\n=== {f.name}:{sheet} ({df_clean.shape[0]}x{df_clean.shape[1]}) ===")
        col_w = max(4, df_clean.astype(str).applymap(len).max().max() + 1)
        header = "    " + " ".join(f"{i:>{col_w}}" for i in df_clean.columns)
        print(header)
        for r in df_clean.itertuples():
            row_str = " ".join(f"{str(c):>{col_w}}" if c != -1 else ".".rjust(col_w)
                               for c in r[1:])
            print(f"{r.Index:>3} {row_str}")
        print("-" * len(header))

        sample = {
            "grid": df_clean.values.tolist(),
            "target": -1,
            "answer": [-1, -1],
            "size": f"{df_clean.shape[0]}x{df_clean.shape[1]}",
            "source": f"{f.name}:{sheet}"
        }
        out = DST / f"{uuid.uuid4().hex}.json"
        with open(out, "w", encoding="utf-8") as fout:
            json.dump(sample, fout, ensure_ascii=False, indent=2)

print("\n✔ 轉檔完成，共產生", len(list(DST.glob('*.json'))), "個樣本")
