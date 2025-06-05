import json, uuid, pathlib, openpyxl, itertools, sys

# Base project root
BASE = pathlib.Path(__file__).resolve().parent.parent
SRC  = BASE / "data" / "excels"
DST  = BASE / "data" / "samples"
DST.mkdir(parents=True, exist_ok=True)

class FormatError(Exception):
    pass

# ------------------ 視覺輸出 ------------------
def print_grid(rows):
    """Pretty print with 1‑based row/col headers."""
    if not rows: 
        return
    col_w = max(4, max(len(str(c)) for r in rows for c in r if c != "") + 1)
    header = "    " + " ".join(f"{i:>{col_w}}" for i in range(1, len(rows[0]) + 1))
    print(header)
    for idx, row in enumerate(rows, start=1):
        body = " ".join(f"{str(c):>{col_w}}" if c != "" else " .".rjust(col_w) for c in row)
        print(f"{idx:>3} {body}")
    print("-" * len(header))

# ------------------ 資料清洗 ------------------
TRANS = str.maketrans({"O": "0", "I": "1"})

def clean_cell(val):
    """遵守清洗規則：空→"", O/I 轉數字, 其餘非法清空"""
    if val is None:
        return ""
    val = str(val).strip().upper().translate(TRANS)
    if val.isdigit():
        return int(val)
    return ""

# ------------------ 主流程 ------------------
def excel_to_samples(xlsx):
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        rows = []
        expected = None
        for r in ws.iter_rows(values_only=True):
            row = [clean_cell(c) for c in r]
            if expected is None:
                expected = len(row)
            if len(row) != expected:
                raise FormatError(f"{xlsx.name}:{sheet} 行列長度不一致【格式失真】")
            rows.append(row)

        # 視覺確認
        print(f"\n=== {xlsx.name}:{sheet} ({len(rows)}x{expected}) ===")
        print_grid(rows)

        sample = {
            "grid": rows,
            "target": -1,
            "answer": [-1, -1],  # 1‑based row/col; -1 表示未標
            "size": f"{len(rows)}x{expected}",
            "source": f"{xlsx.name}:{sheet}"
        }
        out = DST / f"{uuid.uuid4().hex}.json"
        json.dump(sample, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

if not list(SRC.glob("*.xls*")):
    print(f"❗ 未找到 Excel 檔，請放入 {SRC}")
    sys.exit(1)

for f in SRC.glob("*.xls*"):
    excel_to_samples(f)

print("✓ 轉檔完成，共生成", len(list(DST.glob("*.json"))), "個樣本")
