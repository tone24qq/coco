from pathlib import Path

import pytest

from src.fetch_winwin import _write_debug_snapshot, parse_winwin_html


def test_parse_winwin_html_success_json_object() -> None:
    html = '{"issue":"20260101001","draw_date":"2026-01-01","numbers":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}'
    rows = parse_winwin_html(html)
    assert len(rows) == 1
    assert rows[0].day_issue_index == 1


def test_parse_winwin_html_success_regex_fallback() -> None:
    html = '"issue":"20260101002" "draw_date":"2026/01/01" "numbers":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]'
    rows = parse_winwin_html(html)
    assert rows[0].issue == "20260101002"


def test_parse_winwin_html_fail_fast() -> None:
    with pytest.raises(ValueError):
        parse_winwin_html("<html>invalid</html>")


def test_debug_snapshot_writer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_debug_snapshot("<x>", "https://x", "err")
    files = list((tmp_path / "reports" / "fetch_debug").glob("*.json"))
    assert files


def test_parse_winwin_html_success_table() -> None:
    html = "<table><tr><th>issue</th></tr><tr><td>20260101003</td><td>2026/01/01</td>" + "".join([f"<td>{i}</td>" for i in range(1, 21)]) + "</tr></table>"
    rows = parse_winwin_html(html)
    assert rows[0].issue == "20260101003"
