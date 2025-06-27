import json
from pathlib import Path
from typing import Dict, List, Any

from openpyxl import load_workbook


def clean_cell(val: Any) -> int:
    """把单元格内容清洗成数字，不合法返回 -1。"""
    if val is None:
        return -1
    s = str(val).strip().upper()
    if not s:
        return -1
    # 字符替换：O→0, I→1
    s = s.replace("O", "0").replace("I", "1")
    # 提取其中所有数字字符
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else -1


def main() -> None:
    samples_dir = Path("samples")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 统计所有 clean_cell 后非 -1 数字的出现次数
    counts: Dict[int, int] = {}
    total = 0

    for xlsx in samples_dir.glob("*.xlsx"):
        wb = load_workbook(xlsx, data_only=True)
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            for row in ws.iter_rows(values_only=True):
                for val in row:
                    num = clean_cell(val)
                    if num != -1:
                        counts[num] = counts.get(num, 0) + 1
                        total += 1

    # 归一化成先验概率分布
    priors: Dict[int, float] = {
        num: cnt / total for num, cnt in counts.items()
    } if total > 0 else {}

    # 写入 JSON
    json_path = output_dir / "cleaned_data.json"
    json_path.write_text(
        json.dumps(priors, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()