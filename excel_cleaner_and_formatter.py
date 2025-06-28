import json
from pathlib import Path
from typing import Any, Dict, List
from openpyxl import load_workbook

def clean_cell_dynamic(val: Any, max_valid: int) -> int:
    """
    清洗單元格：自動依據表格大小設定合法範圍。
    支援 O→0, I/L→1, B→8。
    """
    if val is None:
        return -1
    s = str(val).strip().upper()
    s = s.replace("O", "0").replace("I", "1").replace("L", "1").replace("B", "8")
    digits = ''.join(ch for ch in s if ch.isdigit())
    if not digits:
        return -1
    num = int(digits)
    return num if 1 <= num <= max_valid else -1

def clean_sheet(ws) -> (List[List[int]], int):
    rows, cols = ws.max_row, ws.max_column
    max_valid = rows * cols
    cleaned = [
        [clean_cell_dynamic(cell, max_valid) for cell in row]
        for row in ws.iter_rows(values_only=True)
    ]
    return cleaned, max_valid

def flatten(grid: List[List[int]]) -> List[int]:
    return [n for row in grid for n in row if n > 0]

def analyze_numbers(numbers: List[int], max_valid: int) -> Dict[str, Any]:
    from collections import Counter
    counter = Counter(numbers)
    duplicates = {k: v for k, v in counter.items() if v > 1}
    missing = [i for i in range(1, max_valid + 1) if i not in counter]
    return {"total": len(numbers), "unique": len(counter), "duplicates": duplicates, "missing": missing}

def main():
    samples_dir = Path("samples")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    result = {}

    for xlsx in samples_dir.glob("*.xlsx"):
        wb = load_workbook(xlsx, data_only=True)
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            cleaned_grid, max_valid = clean_sheet(ws)
            numbers = flatten(cleaned_grid)
            stats = analyze_numbers(numbers, max_valid)
            result[f"{xlsx.name}:{sheet_name}"] = {
                "max_valid": max_valid,
                "cleaned_grid": cleaned_grid,
                "stats": stats
            }

    with (output_dir / "cleaned_output.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("✅ 清洗完成，結果輸出至 output/cleaned_output.json")

if __name__ == "__main__":
    main()
