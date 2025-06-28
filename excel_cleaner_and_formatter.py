import json
from pathlib import Path
from typing import Dict, Any

from openpyxl import load_workbook

def clean_cell(val: Any) -> int:
    """
    清洗单元格内容：
      - None 或 空白 → -1
      - O→0, I→1
      - 提取所有数字字符并拼接成 int；否则 -1
    """
    if val is None:
        return -1
    s = str(val).strip().upper()
    if not s:
        return -1
    s = s.replace("O", "0").replace("I", "1")
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else -1

def main() -> None:
    samples_dir = Path("samples")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    counts: Dict[int, int] = {}
    total = 0

    # 扫描所有 Excel
    for xlsx in samples_dir.glob("*.xlsx"):
        wb = load_workbook(xlsx, data_only=True)
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            # 动态计算此 Sheet 合理数字范围
            rows, cols = ws.max_row, ws.max_column
            max_num = rows * cols
            # 逐行逐列读取并统计
            for row in ws.iter_rows(values_only=True):
                for val in row:
                    num = clean_cell(val)
                    # 只保留 1..max_num 的合法数字
                    if 1 <= num <= max_num:
                        counts[num] = counts.get(num, 0) + 1
                        total += 1

    # 归一化为先验概率
    if total > 0:
        priors = {num: cnt / total for num, cnt in counts.items()}
    else:
        priors = {}

    # 写入 JSON
    json_path = output_dir / "cleaned_data.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(priors, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()