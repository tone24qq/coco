from __future__ import annotations

import argparse
import json
import math
import re
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


def _iter_xlsx_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.glob("*.xlsx")):
        if not path.name.startswith("~$"):
            yield path


def _fill_merged_ranges(ws) -> None:
    """
    把 merged cell 的左上角值灌回整個 merged range，避免其餘格讀到空白。
    """
    merged_ranges = list(ws.merged_cells.ranges)
    for merged in merged_ranges:
        min_col, min_row, max_col, max_row = merged.bounds
        top_left_value = ws.cell(min_row, min_col).value
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                cell = ws.cell(r, c)
                if cell.value is None:
                    cell.value = top_left_value


def _read_sheet_as_strings(ws) -> List[List[str]]:
    max_row = int(ws.max_row or 0)
    max_col = int(ws.max_column or 0)
    table: List[List[str]] = []
    for row in ws.iter_rows(min_row=1, max_row=max_row, min_col=1, max_col=max_col):
        table.append([str(cell.value) if cell.value is not None else "" for cell in row])
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


def _normalize_digits(s: str) -> str:
    trans = str.maketrans("０１２３４５６７８９．－＋，", "0123456789.-+,")
    return s.translate(trans)


def _normalize_cell(cell: str) -> str:
    s = str(cell).strip()
    if s == "":
        return ""
    s = _normalize_digits(s)
    s = s.replace("Ｏ", "0").replace("Ｉ", "1")
    s = s.replace("O", "0").replace("I", "1")
    s = s.replace("\n", "").replace("\r", "")
    return s.strip()


def _cell_to_int_or_blank(cell: str) -> Tuple[bool, Optional[int]]:
    """
    回傳:
      (True, int)   => 可解析為整數
      (True, None)  => 空白
      (False, None) => 無法解析
    """
    s = _normalize_cell(cell)
    if s == "":
        return True, None

    # 純整數
    if re.fullmatch(r"[+-]?\d+", s):
        return True, int(s)

    # 1.0 / 12.000 這種
    if re.fullmatch(r"[+-]?\d+\.\d+", s):
        try:
            f = float(s)
            if math.isfinite(f) and f.is_integer():
                return True, int(f)
        except ValueError:
            pass

    # 去掉逗號後再試一次
    s2 = s.replace(",", "")
    if re.fullmatch(r"[+-]?\d+", s2):
        return True, int(s2)
    if re.fullmatch(r"[+-]?\d+\.\d+", s2):
        try:
            f = float(s2)
            if math.isfinite(f) and f.is_integer():
                return True, int(f)
        except ValueError:
            pass

    return False, None


def _is_nonblank(cell: str) -> bool:
    return _normalize_cell(cell) != ""


def _render_table_box(table: Sequence[Sequence[str]], max_rows: int = 30, max_cols: int = 20) -> str:
    rows = len(table)
    cols = len(table[0]) if rows else 0
    shown_rows = min(rows, max_rows)
    shown_cols = min(cols, max_cols)

    widths = [2] * shown_cols
    for c in range(shown_cols):
        widths[c] = max(
            2,
            len(f"C{c+1}"),
            max((len(str(table[r][c])) for r in range(shown_rows)), default=0),
        )

    header = "     " + " ".join(f"C{c+1}".rjust(widths[c]) for c in range(shown_cols))
    lines = [header]
    for r in range(shown_rows):
        body = " ".join((table[r][c] if str(table[r][c]) != "" else "·").rjust(widths[c]) for c in range(shown_cols))
        lines.append(f"R{r+1:<3} {body}")

    if rows > shown_rows or cols > shown_cols:
        lines.append(f"... truncated preview: full_shape={rows}x{cols}, shown={shown_rows}x{shown_cols}")
    return "\n".join(lines)


def _find_nonblank_components(table: Sequence[Sequence[str]]) -> List[Tuple[int, int, int, int]]:
    """
    找連續非空白區塊，回傳 (r1, c1, r2, c2)，0-based, 含端點
    """
    rows = len(table)
    cols = len(table[0]) if rows else 0
    if rows == 0 or cols == 0:
        return []

    visited = [[False] * cols for _ in range(rows)]
    comps: List[Tuple[int, int, int, int]] = []

    for r in range(rows):
        for c in range(cols):
            if visited[r][c] or not _is_nonblank(table[r][c]):
                continue

            q = deque([(r, c)])
            visited[r][c] = True
            min_r = max_r = r
            min_c = max_c = c

            while q:
                cr, cc = q.popleft()
                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)

                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and _is_nonblank(table[nr][nc]):
                        visited[nr][nc] = True
                        q.append((nr, nc))

            comps.append((min_r, min_c, max_r, max_c))

    return comps


def _crop(table: Sequence[Sequence[str]], r1: int, c1: int, r2: int, c2: int) -> List[List[str]]:
    return [list(row[c1:c2 + 1]) for row in table[r1:r2 + 1]]


def _trim_empty_borders(block: Sequence[Sequence[str]]) -> Tuple[List[List[str]], int, int]:
    """
    去掉區塊外圍全空白列/欄
    回傳: trimmed_block, row_offset, col_offset
    """
    if not block:
        return [], 0, 0

    top = 0
    bottom = len(block) - 1
    left = 0
    right = len(block[0]) - 1

    while top <= bottom and all(not _is_nonblank(x) for x in block[top]):
        top += 1
    while bottom >= top and all(not _is_nonblank(x) for x in block[bottom]):
        bottom -= 1
    while left <= right and all(not _is_nonblank(block[r][left]) for r in range(top, bottom + 1)):
        left += 1
    while right >= left and all(not _is_nonblank(block[r][right]) for r in range(top, bottom + 1)):
        right -= 1

    if top > bottom or left > right:
        return [], 0, 0

    trimmed = [list(block[r][left:right + 1]) for r in range(top, bottom + 1)]
    return trimmed, top, left


def _window_score(window: Sequence[Sequence[str]]) -> Tuple[int, List[List[int]]]:
    """
    分數越高越像合法 1..80 視窗
    score:
      +1 每個可解析整數
      +2 在 1..80 範圍
      +2 不重複
    """
    ints: List[List[int]] = []
    seen = set()
    score = 0

    for row in window:
        out_row: List[int] = []
        for cell in row:
            ok, value = _cell_to_int_or_blank(cell)
            if not ok or value is None:
                out_row.append(-1)
                continue

            out_row.append(value)
            score += 1

            if 1 <= value <= 80:
                score += 2
            if value not in seen:
                score += 2
                seen.add(value)

        ints.append(out_row)

    return score, ints


def _window_to_full_permutation(window: Sequence[Sequence[str]]) -> Tuple[bool, Optional[List[List[int]]], str]:
    grid: List[List[int]] = []
    flat: List[int] = []

    for row in window:
        out_row: List[int] = []
        for cell in row:
            ok, value = _cell_to_int_or_blank(cell)
            if not ok:
                return False, None, "invalid token"
            if value is None:
                return False, None, "blank cell"
            out_row.append(value)
            flat.append(value)
        grid.append(out_row)

    n = len(grid) * len(grid[0])
    if sorted(flat) != list(range(1, n + 1)):
        return False, None, f"not permutation 1..{n}"

    return True, grid, "ok"


def _scan_candidate_block(
    block: Sequence[Sequence[str]],
    file_name: str,
    sheet_name: str,
    base_r: int,
    base_c: int,
) -> List[Dict[str, Any]]:
    rows = len(block)
    cols = len(block[0]) if rows else 0
    found: List[Dict[str, Any]] = []
    dedup: set[Tuple[int, int, Tuple[int, ...]]] = set()

    for shape_rows, shape_cols in ALLOWED_SHAPES:
        if rows < shape_rows or cols < shape_cols:
            continue

        best_near = None
        best_near_score = -1

        for r0 in range(rows - shape_rows + 1):
            for c0 in range(cols - shape_cols + 1):
                window = [list(block[r][c0:c0 + shape_cols]) for r in range(r0, r0 + shape_rows)]

                ok, grid, _ = _window_to_full_permutation(window)
                if ok and grid is not None:
                    flat = tuple(v for row in grid for v in row)
                    key = (shape_rows, shape_cols, flat)
                    if key in dedup:
                        continue
                    dedup.add(key)

                    found.append(
                        {
                            "board_id": f"{Path(file_name).stem}:{sheet_name}:{base_r+r0}:{base_c+c0}",
                            "rows": shape_rows,
                            "cols": shape_cols,
                            "size_class": f"{shape_rows}x{shape_cols}",
                            "grid": grid,
                            "source": "root_xlsx_window_scan_relaxed",
                            "source_file": file_name,
                            "issue_id": Path(file_name).stem,
                            "group_id": f"{Path(file_name).stem}:{sheet_name}:{base_r+r0}:{base_c+c0}",
                            "is_real": True,
                            "source_type": "real",
                            "window_top_left_1_based": [base_r + r0 + 1, base_c + c0 + 1],
                            "match_type": "exact",
                        }
                    )
                else:
                    score, ints = _window_score(window)
                    if score > best_near_score:
                        best_near_score = score
                        best_near = {
                            "rows": shape_rows,
                            "cols": shape_cols,
                            "window_top_left_1_based": [base_r + r0 + 1, base_c + c0 + 1],
                            "score": score,
                            "grid_preview": ints,
                            "match_type": "near_miss",
                        }

        if best_near is not None:
            found.append(best_near)

    return found


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
            try:
                _fill_merged_ranges(ws)
            except Exception:
                pass

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

            components = _find_nonblank_components(table)
            all_found: List[Dict[str, Any]] = []

            # 沒找到 component 時，退回整張表掃描
            if not components:
                components = [(0, 0, rows - 1, cols - 1)] if rows and cols else []

            for r1, c1, r2, c2 in components:
                block = _crop(table, r1, c1, r2, c2)
                block, off_r, off_c = _trim_empty_borders(block)
                if not block:
                    continue

                br = r1 + off_r
                bc = c1 + off_c

                found_windows = _scan_candidate_block(
                    block=block,
                    file_name=xlsx_path.name,
                    sheet_name=ws.title,
                    base_r=br,
                    base_c=bc,
                )
                all_found.extend(found_windows)

            exact_windows = [x for x in all_found if x.get("match_type") == "exact"]
            near_windows = [x for x in all_found if x.get("match_type") == "near_miss"]

            for rec in exact_windows:
                board_id = str(rec["board_id"])
                if board_id in seen_board_ids:
                    continue
                seen_board_ids.add(board_id)
                corpus.append(rec)

            if exact_windows:
                status = "ok"
                message = "found exact 8x10/10x8 permutation 1..80 window(s)"
            elif near_windows:
                status = "near_miss"
                message = "no exact board found, but found candidate window(s) close to valid 1..80 board"
            else:
                status = "no_80_board_found"
                message = "sheet restored successfully, but no 8x10/10x8 candidate found"

            audits.append(
                SheetAudit(
                    file=xlsx_path.name,
                    sheet=ws.title,
                    rows=rows,
                    cols=cols,
                    status=status,
                    format_loss=False,
                    message=message,
                    found_windows=[
                        {
                            "board_id": rec.get("board_id", ""),
                            "size_class": rec.get("size_class", f'{rec["rows"]}x{rec["cols"]}'),
                            "window_top_left_1_based": rec["window_top_left_1_based"],
                            "match_type": rec.get("match_type", "unknown"),
                            "score": rec.get("score"),
                        }
                        for rec in (exact_windows + near_windows)
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

    print(json.dumps({
        "board_count": len(corpus),
        "size_counts": size_counts,
        "output": str(output_path),
        "audit": str(audit_path),
        "preview_dir": str(preview_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()