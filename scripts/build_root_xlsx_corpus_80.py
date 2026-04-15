from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from openpyxl import load_workbook


ALLOWED_SHAPES: Tuple[Tuple[int, int], ...] = ((10, 8), (8, 10))


@dataclass
class SheetAudit:
    file: str
    sheet: str
    rows: int
    cols: int
    status: str
    format_loss: bool
    message: str
    found_windows: List[Dict[str, Any]]


def _read_sheet_as_strings(ws) -> List[List[str]]:
    """
    Strict Excel restore:
    - no header inference
    - no dtype inference
    - no NA coercion
    - preserve full NxM via max_row/max_column
    - blank cell => ""
    """
    max_row = int(ws.max_row or 0)
    max_col = int(ws.max_column or 0)
    table: List[List[str]] = []
    for row in ws.iter_rows(min_row=1, max_row=max_row, min_col=1, max_col=max_col):
        restored = [str(cell.value) if cell.value is not None else "" for cell in row]
        table.append(restored)
    return table


def _validate_rectangular(table: Sequence[Sequence[str]]) -> Tuple[bool, str]:
    if not isinstance(table, list):
        return False, "table is not list"
    if not table:
        return True, "empty table"
    expected = len(table[0])
    for idx, row in enumerate(table, start=1):
        if not isinstance(row, list):
            return False, f"row {idx} is not list"
        if len(row) != expected:
            return False, f"row {idx} length mismatch: expected={expected}, got={len(row)}"
    return True, "ok"


def _normalize_cell(cell: str) -> str:
    """
    Strict string pipeline.
    - preserve blank as ""
    - trim outer whitespace
    - normalize common OCR confusions O->0, I->1
    - invalid token remains non-digit string and will be treated as blank/invalid later
    """
    s = cell.strip()
    if s == "":
        return ""
    s = s.replace("Ｏ", "0").replace("Ｉ", "1")
    s = s.replace("O", "0").replace("I", "1")
    return s


def _cell_to_int_or_blank(cell: str) -> Tuple[bool, int | None]:
    """
    Returns:
      (True, int) for pure integer token
      (True, None) for blank
      (False, None) for invalid token
    """
    s = _normalize_cell(cell)
    if s == "":
        return True, None
    if re.fullmatch(r"\d+", s):
        return True, int(s)
    return False, None


def _render_table_box(table: Sequence[Sequence[str]], max_rows: int = 30, max_cols: int = 20) -> str:
    """
    Visual preview with row/col labels, preserving blanks.
    """
    rows = len(table)
    cols = len(table[0]) if rows else 0
    shown_rows = min(rows, max_rows)
    shown_cols = min(cols, max_cols)

    widths = [2] * shown_cols
    for c in range(shown_cols):
        widths[c] = max(
            2,
            len(f"C{c+1}"),
            max((len(table[r][c]) for r in range(shown_rows)), default=0),
        )

    header = "     " + " ".join(f"C{c+1}".rjust(widths[c]) for c in range(shown_cols))
    lines = [header]
    for r in range(shown_rows):
        body = " ".join((table[r][c] if table[r][c] != "" else "·").rjust(widths[c]) for c in range(shown_cols))
        lines.append(f"R{r+1:<3} {body}")

    if rows > shown_rows or cols > shown_cols:
        lines.append(f"... truncated preview: full_shape={rows}x{cols}, shown={shown_rows}x{shown_cols}")
    return "\n".join(lines)


def _window_to_int_grid(window: Sequence[Sequence[str]]) -> Tuple[bool, List[List[int]] | None, str]:
    grid: List[List[int]] = []
    seen: List[int] = []
    for row in window:
        out_row: List[int] = []
        for cell in row:
            ok, value = _cell_to_int_or_blank(cell)
            if not ok:
                return False, None, "invalid token in window"
            if value is None:
                return False, None, "blank cell in window"
            out_row.append(value)
            seen.append(value)
        grid.append(out_row)

    n = len(grid) * len(grid[0])
    if sorted(seen) != list(range(1, n + 1)):
        return False, None, f"window is not permutation 1..{n}"
    return True, grid, "ok"


def _scan_windows(table: Sequence[Sequence[str]], file_name: str, sheet_name: str) -> List[Dict[str, Any]]:
    rows = len(table)
    cols = len(table[0]) if rows else 0
    found: List[Dict[str, Any]] = []
    dedup: set[Tuple[int, int, Tuple[int, ...]]] = set()

    for shape_rows, shape_cols in ALLOWED_SHAPES:
        if rows < shape_rows or cols < shape_cols:
            continue
        for r0 in range(rows - shape_rows + 1):
            for c0 in range(cols - shape_cols + 1):
                window = [list(table[r][c0:c0 + shape_cols]) for r in range(r0, r0 + shape_rows)]
                ok, grid, _ = _window_to_int_grid(window)
                if not ok or grid is None:
                    continue
                flat = tuple(v for row in grid for v in row)
                key = (shape_rows, shape_cols, flat)
                if key in dedup:
                    continue
                dedup.add(key)
                found.append(
                    {
                        "board_id": f"{Path(file_name).stem}:{sheet_name}:{r0}:{c0}",
                        "rows": shape_rows,
                        "cols": shape_cols,
                        "size_class": f"{shape_rows}x{shape_cols}",
                        "grid": grid,
                        "source": "root_xlsx_window_scan",
                        "source_file": file_name,
                        "issue_id": Path(file_name).stem,
                        "group_id": f"{Path(file_name).stem}:{sheet_name}:{r0}:{c0}",
                        "is_real": True,
                        "source_type": "real",
                        "window_top_left_1_based": [r0 + 1, c0 + 1],
                    }
                )
    return found


def _iter_xlsx_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        yield path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=".")
    parser.add_argument("--output", default="data/full_boards/full_board_corpus_80.jsonl")
    parser.add_argument("--audit", default="reports/full_board_corpus_80_audit.json")
    parser.add_argument("--preview-dir", default="reports/root_xlsx_previews")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    audit_path = Path(args.audit)
    preview_dir = Path(args.preview_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    corpus: List[Dict[str, Any]] = []
    audits: List[SheetAudit] = []
    seen_board_ids: set[str] = set()

    for xlsx_path in _iter_xlsx_files(input_dir):
        try:
            wb = load_workbook(xlsx_path, data_only=True)
        except Exception as exc:
            audits.append(
                SheetAudit(
                    file=xlsx_path.name,
                    sheet="",
                    rows=0,
                    cols=0,
                    status="read_error",
                    format_loss=True,
                    message=str(exc),
                    found_windows=[],
                )
            )
            continue

        for ws in wb.worksheets:
            table = _read_sheet_as_strings(ws)
            rows = len(table)
            cols = len(table[0]) if rows else 0

            preview_txt = _render_table_box(table)
            preview_name = f"{xlsx_path.stem}__{ws.title}.txt".replace("/", "_").replace("\\", "_")
            (preview_dir / preview_name).write_text(preview_txt, encoding="utf-8")

            ok_rect, rect_msg = _validate_rectangular(table)
            if not ok_rect:
                audits.append(
                    SheetAudit(
                        file=xlsx_path.name,
                        sheet=ws.title,
                        rows=rows,
                        cols=cols,
                        status="format_loss",
                        format_loss=True,
                        message=f"【格式失真】{rect_msg}",
                        found_windows=[],
                    )
                )
                continue

            found_windows = _scan_windows(table, xlsx_path.name, ws.title)

            for rec in found_windows:
                board_id = str(rec["board_id"])
                if board_id in seen_board_ids:
                    continue
                seen_board_ids.add(board_id)
                corpus.append(rec)

            audits.append(
                SheetAudit(
                    file=xlsx_path.name,
                    sheet=ws.title,
                    rows=rows,
                    cols=cols,
                    status="ok" if found_windows else "no_80_board_found",
                    format_loss=False,
                    message="ok" if found_windows else "sheet restored successfully, but no 8x10/10x8 permutation-1..80 window found",
                    found_windows=[
                        {
                            "board_id": rec["board_id"],
                            "size_class": rec["size_class"],
                            "window_top_left_1_based": rec["window_top_left_1_based"],
                        }
                        for rec in found_windows
                    ],
                )
            )

    with output_path.open("w", encoding="utf-8") as fh:
        for rec in corpus:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    size_counts: Dict[str, int] = {}
    for rec in corpus:
        size_counts[rec["size_class"]] = size_counts.get(rec["size_class"], 0) + 1

    audit_payload = {
        "status": "ok",
        "output": str(output_path),
        "preview_dir": str(preview_dir),
        "board_count": len(corpus),
        "size_counts": size_counts,
        "sheets": [asdict(a) for a in audits],
    }
    audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"board_count": len(corpus), "size_counts": size_counts, "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()