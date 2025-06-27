import json
from pathlib import Path
from typing import Dict, List

from openpyxl import load_workbook


def clean_cell(val) -> int:
    if val is None:
        return -1
    s = str(val).strip().upper()
    if not s:
        return -1
    s = s.replace("O", "0").replace("I", "1")
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else -1


def process_sheet(ws) -> List[List[int]]:
    rows = ws.max_row
    cols = ws.max_column
    data: List[List[int]] = []
    for r in range(1, rows + 1):
        row = []
        for c in range(1, cols + 1):
            row.append(clean_cell(ws.cell(r, c).value))
        data.append(row)
    return data


def main() -> None:
    samples_dir = Path("samples")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    result: Dict[str, List[List[int]]] = {}

    for xlsx in samples_dir.glob("*.xlsx"):
        wb = load_workbook(xlsx, data_only=True)
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            cleaned = process_sheet(ws)
            key = f"{xlsx.name}::{sheet}"
            result[key] = cleaned
            rows = len(cleaned)
            cols = len(cleaned[0]) if cleaned else 0
            csv_path = output_dir / f"{xlsx.stem}__{sheet}_{rows}x{cols}.csv"
            lines = ["," + ",".join(str(i) for i in range(1, cols + 1))]
            for idx, row in enumerate(cleaned, 1):
                lines.append(str(idx) + "," + ",".join(str(v) for v in row))
            csv_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = output_dir / "cleaned_data.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
