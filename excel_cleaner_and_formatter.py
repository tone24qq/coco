import json
from pathlib import Path
from typing import Any, Dict, List

from openpyxl import load_workbook

def clean_cell(val: Any) -> int:
    """
    清洗單元格內容：
      - None 或空白 → -1
      - O→0, I→1
      - 僅保留數字 → int；否則 -1
    """
    if val is None:
        return -1
    s = str(val).strip().upper().replace("O", "0").replace("I", "1")
    digits = ''.join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else -1

def clean_sheet(ws) -> List[List[int]]:
    """逐格清洗整個工作表並轉為 list[list[int]] 結構"""
    cleaned = []
    for row in ws.iter_rows(values_only=True):
        cleaned_row = [clean_cell(cell) for cell in row]
        cleaned.append(cleaned_row)
    return cleaned

def main() -> None:
    samples_dir = Path("samples")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    all_cleaned_data = {}

    for xlsx in samples_dir.glob("*.xlsx"):
        wb = load_workbook(xlsx, data_only=True)
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            cleaned_grid = clean_sheet(ws)
            all_cleaned_data[f"{xlsx.name}:{sheet_name}"] = cleaned_grid

    output_file = output_dir / "cleaned_grids.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(all_cleaned_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
