# flake8: noqa
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


REAL_FULL_BOARDS = [
    {"board_id": "98151744-7875", "rows": 10, "cols": 16, "size_class": "10x16", "is_real": True, "source": "user_screenshot", "grid": [[94, 40, 115, 47, 72, 21, 149, 50, 119, 121, 82, 2, 32, 92, 142, 27], [120, 129, 77, 143, 112, 38, 39, 64, 152, 67, 43, 31, 86, 17, 144, 49], [74, 8, 160, 100, 75, 9, 146, 1, 11, 65, 12, 78, 4, 73, 99, 14], [105, 139, 110, 3, 80, 68, 7, 35, 125, 57, 18, 137, 15, 45, 102, 157], [5, 66, 126, 159, 88, 104, 103, 24, 132, 89, 59, 96, 147, 111, 29, 109], [6, 41, 42, 114, 63, 116, 150, 117, 155, 46, 56, 158, 52, 19, 124, 55], [60, 136, 13, 106, 54, 16, 10, 122, 26, 85, 79, 48, 58, 53, 81, 51], [131, 69, 156, 61, 134, 145, 25, 34, 140, 154, 62, 20, 30, 123, 93, 44], [90, 108, 135, 28, 70, 84, 91, 148, 76, 141, 83, 71, 151, 130, 23, 37], [33, 113, 133, 95, 98, 107, 22, 118, 153, 97, 138, 87, 128, 36, 127, 101]]},
    {"board_id": "98151744-7876", "rows": 10, "cols": 16, "size_class": "10x16", "is_real": True, "source": "user_screenshot", "grid": [[63, 10, 102, 111, 46, 2, 64, 76, 90, 21, 54, 26, 24, 132, 5, 93], [44, 34, 42, 146, 85, 51, 62, 39, 129, 35, 94, 73, 56, 52, 31, 8], [136, 154, 140, 75, 121, 100, 115, 45, 53, 59, 152, 74, 17, 149, 71, 78], [4, 70, 49, 72, 107, 47, 160, 38, 12, 23, 1, 55, 91, 148, 98, 141], [133, 106, 128, 143, 18, 109, 82, 37, 97, 69, 79, 130, 19, 139, 20, 40], [155, 16, 95, 68, 25, 22, 92, 99, 3, 67, 80, 124, 159, 15, 89, 36], [137, 13, 118, 50, 60, 28, 113, 112, 104, 126, 30, 120, 153, 87, 88, 138], [144, 151, 66, 150, 110, 134, 157, 61, 123, 103, 81, 84, 58, 41, 9, 114], [101, 127, 108, 119, 33, 77, 131, 86, 11, 57, 32, 158, 83, 65, 156, 7], [135, 48, 147, 6, 14, 125, 116, 96, 43, 117, 105, 145, 29, 142, 27, 122]]},
    {"board_id": "10074622", "rows": 10, "cols": 8, "size_class": "10x8", "is_real": True, "source": "user_screenshot", "grid": [[48, 47, 23, 61, 35, 14, 19, 67], [13, 60, 65, 80, 62, 5, 36, 41], [57, 52, 26, 76, 56, 15, 21, 29], [63, 18, 4, 79, 40, 10, 42, 3], [28, 8, 44, 33, 30, 46, 53, 75], [38, 71, 70, 72, 64, 27, 32, 31], [17, 50, 69, 43, 22, 74, 58, 37], [25, 9, 11, 20, 6, 66, 34, 2], [16, 77, 45, 39, 68, 12, 78, 24], [73, 55, 51, 59, 1, 49, 54, 7]]},
    {"board_id": "10135426", "rows": 8, "cols": 5, "size_class": "8x5", "is_real": True, "source": "user_screenshot", "grid": [[26, 4, 21, 22, 17], [40, 31, 10, 8, 34], [38, 15, 23, 37, 9], [25, 30, 36, 35, 3], [5, 27, 39, 1, 14], [19, 32, 13, 12, 2], [16, 18, 20, 28, 24], [33, 11, 7, 6, 29]]},
    {"board_id": "10135427", "rows": 8, "cols": 5, "size_class": "8x5", "is_real": True, "source": "user_screenshot", "grid": [[26, 1, 30, 22, 2], [6, 37, 17, 27, 20], [21, 32, 11, 33, 15], [36, 39, 7, 12, 24], [19, 8, 38, 31, 23], [28, 25, 13, 40, 34], [3, 14, 29, 35, 5], [10, 4, 9, 18, 16]]},
    {"board_id": "10135428", "rows": 8, "cols": 5, "size_class": "8x5", "is_real": True, "source": "user_screenshot", "grid": [[36, 25, 8, 34, 20], [32, 2, 15, 30, 1], [38, 21, 3, 37, 26], [7, 24, 5, 22, 18], [17, 9, 35, 29, 39], [4, 11, 13, 28, 14], [40, 19, 10, 27, 31], [6, 23, 16, 12, 33]]},
    {"board_id": "10135429", "rows": 8, "cols": 5, "size_class": "8x5", "is_real": True, "source": "user_screenshot", "grid": [[17, 39, 31, 20, 34], [13, 29, 37, 4, 19], [28, 40, 33, 23, 15], [16, 6, 3, 11, 30], [26, 2, 10, 8, 22], [36, 32, 27, 25, 9], [35, 38, 12, 18, 5], [24, 7, 14, 21, 1]]},
    {"board_id": "11203237", "rows": 10, "cols": 6, "size_class": "10x6", "is_real": True, "source": "user_screenshot", "grid": [[2, 25, 55, 23, 24, 49], [48, 56, 35, 12, 20, 28], [31, 4, 45, 39, 33, 8], [6, 22, 47, 40, 14, 43], [52, 13, 38, 19, 11, 10], [37, 36, 46, 54, 1, 7], [9, 53, 59, 5, 27, 16], [57, 42, 32, 17, 50, 34], [51, 15, 58, 18, 44, 21], [30, 3, 29, 26, 60, 41]]},
    {"board_id": "11203238", "rows": 10, "cols": 6, "size_class": "10x6", "is_real": True, "source": "user_screenshot", "grid": [[48, 49, 60, 28, 6, 40], [58, 35, 43, 5, 14, 2], [39, 33, 34, 32, 37, 20], [12, 55, 26, 10, 36, 4], [45, 27, 51, 29, 57, 22], [38, 3, 18, 13, 17, 52], [46, 8, 21, 25, 42, 7], [19, 31, 59, 24, 53, 56], [47, 23, 16, 41, 11, 54], [15, 44, 1, 50, 30, 9]]},
    {"board_id": "11203239", "rows": 10, "cols": 6, "size_class": "10x6", "is_real": True, "source": "user_screenshot", "grid": [[3, 20, 22, 40, 46, 52], [60, 10, 42, 13, 50, 43], [24, 28, 25, 1, 38, 4], [58, 2, 8, 37, 45, 44], [30, 41, 15, 18, 19, 11], [32, 7, 31, 39, 47, 6], [35, 27, 53, 54, 36, 51], [34, 55, 57, 5, 49, 16], [12, 23, 33, 21, 14, 9], [17, 56, 26, 29, 59, 48]]},
]

PARTIAL_REAL_BOARD_IDS = ["10135430", "10135431", "11203236", "11203241", "11203243"]
ADDITIONAL_REAL_SOURCE_FILES = [
    "刮刮卡製作 1.xlsx",
    "NG230516014_頁面_1.jpg.xlsx",
    "NG230516012_頁面_2.jpg.xlsx",
    "NG230516009_頁面_1.jpg.xlsx",
    "NL230516023.jpg.xlsx",
]


def validate_grid(board_id: str, rows: int, cols: int, grid: List[List[Any]]) -> None:
    if len(grid) != rows:
        raise ValueError(f"{board_id}: grid rows mismatch")
    if any(len(row) != cols for row in grid):
        raise ValueError(f"{board_id}: grid cols mismatch")
    flat = [int(v) for row in grid for v in row]
    expected = list(range(1, rows * cols + 1))
    if sorted(flat) != expected:
        raise ValueError(f"{board_id}: grid must be permutation 1..{rows * cols}")


def _sheet_to_grid(df: pd.DataFrame) -> Optional[List[List[int]]]:
    cleaned = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if cleaned.empty:
        return None
    values = cleaned.values.tolist()
    grid: List[List[int]] = []
    for row in values:
        out_row: List[int] = []
        for v in row:
            if pd.isna(v):
                return None
            if isinstance(v, float) and not float(v).is_integer():
                return None
            out_row.append(int(v))
        grid.append(out_row)
    if not grid or not grid[0]:
        return None
    if any(len(row) != len(grid[0]) for row in grid):
        return None
    return grid


def try_import_board_from_xlsx(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    sheets = pd.read_excel(path, sheet_name=None, header=None)
    out: List[Dict[str, Any]] = []
    for sheet_name, df in sheets.items():
        grid = _sheet_to_grid(df)
        if grid is None:
            continue
        rows, cols = len(grid), len(grid[0])
        board_id = f"{path.stem}:{sheet_name}"
        validate_grid(board_id, rows, cols, grid)
        out.append(
            {
                "board_id": board_id,
                "rows": rows,
                "cols": cols,
                "size_class": f"{rows}x{cols}",
                "grid": grid,
                "source": "xlsx_import",
                "source_file": path.name,
                "issue_id": path.stem,
                "group_id": board_id,
                "is_real": True,
                "source_type": "real",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/full_boards/full_board_corpus.jsonl")
    parser.add_argument("--audit", default="reports/full_board_corpus_audit.json")
    parser.add_argument("--partial-meta", default="data/full_boards/partial_real_board_metadata.jsonl")
    args = parser.parse_args()

    out_path = Path(args.output)
    audit_path = Path(args.audit)
    partial_path = Path(args.partial_meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    audit: Dict[str, Any] = {"status": "ok", "errors": [], "imported_files": [], "counts": {}}

    seen_ids = set()
    corpus_rows: List[Dict[str, Any]] = []
    try:
        for idx, row in enumerate(REAL_FULL_BOARDS):
            validate_grid(row["board_id"], row["rows"], row["cols"], row["grid"])
            if row["board_id"] in seen_ids:
                raise ValueError(f"duplicate board_id: {row['board_id']}")
            seen_ids.add(row["board_id"])
            corpus_rows.append(
                {
                    "board_id": row["board_id"],
                    "rows": row["rows"],
                    "cols": row["cols"],
                    "size_class": row["size_class"],
                    "grid": row["grid"],
                    "source": row.get("source", "manual"),
                    "source_file": "embedded_REAL_FULL_BOARDS",
                    "issue_id": row["board_id"],
                    "group_id": row["board_id"],
                    "is_real": True,
                    "source_type": "real",
                    "order_index": idx,
                }
            )

        for file_name in ADDITIONAL_REAL_SOURCE_FILES:
            imported = try_import_board_from_xlsx(Path(file_name))
            audit["imported_files"].append({"file": file_name, "records": len(imported)})
            for rec in imported:
                if rec["board_id"] in seen_ids:
                    raise ValueError(f"duplicate board_id after import: {rec['board_id']}")
                seen_ids.add(rec["board_id"])
                rec["order_index"] = len(corpus_rows)
                corpus_rows.append(rec)

        with out_path.open("w", encoding="utf-8") as fh:
            for rec in corpus_rows:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        with partial_path.open("w", encoding="utf-8") as fh:
            for bid in PARTIAL_REAL_BOARD_IDS:
                fh.write(json.dumps({"board_id": bid, "is_real": True, "is_full_board": False}, ensure_ascii=False) + "\n")

        counts: Dict[str, int] = {}
        for rec in corpus_rows:
            counts[rec["size_class"]] = counts.get(rec["size_class"], 0) + 1
        audit["counts"] = counts
        audit["full_board_count"] = len(corpus_rows)
        audit["partial_only_count"] = len(PARTIAL_REAL_BOARD_IDS)
        audit["status"] = "ok"
    except Exception as exc:
        audit["status"] = "error"
        audit["errors"].append(str(exc))
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        raise

    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(corpus_rows)} full boards -> {out_path}")


if __name__ == "__main__":
    main()
