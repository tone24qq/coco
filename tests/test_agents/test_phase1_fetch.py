from pathlib import Path

import pytest

from src.fetch_winwin import (
    _check_source_health,
    _parse_auzo_html,
    _parse_winwin_dynamic_payload,
    _write_debug_snapshot,
    parse_winwin_html,
)
from src.utils import DataContractError


def test_parse_winwin_html_success_table() -> None:
    html = "<table><tr><th>issue</th></tr><tr><td>20260101003</td><td>2026/01/01</td>" + "".join([f"<td>{i}</td>" for i in range(1, 21)]) + "</tr></table>"
    rows = parse_winwin_html(html)
    assert rows[0].issue == "20260101003"


def test_parse_winwin_html_fail_fast() -> None:
    with pytest.raises(ValueError):
        parse_winwin_html("<html>invalid</html>")


def test_parse_winwin_dynamic_payload_supports_no_bigshoworder_and_opendate() -> None:
    payload = {
        "Data": [
            {
                "No": "20260320001",
                "OpenDate": "2026-03-20 09:05:00",
                "BigShowOrder": "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
                "HighLowTop": "10:10",
                "OddEvenTop": "10:10",
            },
            {
                "No": "20260320002",
                "OpenDate": "2026-03-20 09:10:00",
                "BigShowOrder": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                "HighLowTop": "10:10",
                "OddEvenTop": "10:10",
            },
        ]
    }
    rows = _parse_winwin_dynamic_payload(payload)
    assert len(rows) == 2
    assert rows[0].issue == "20260320001"
    assert rows[1].day_issue_index == 2


def test_parse_auzo_rk_html_success() -> None:
    html = "<table><tr><td>20260320011</td><td>2026/03/20 10:10:00</td>" + "".join([f"<td>{i}</td>" for i in range(1, 21)]) + "</tr></table>"
    rows = _parse_auzo_html(html)
    assert len(rows) == 1
    assert rows[0].issue == "20260320011"


def test_source_health_check_db_error_fail_fast() -> None:
    with pytest.raises(DataContractError):
        _check_source_health("<html>Database Error: temporary unavailable</html>")


def test_debug_snapshot_writer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_debug_snapshot("<x>", "https://x", "err")
    files = list((tmp_path / "reports" / "fetch_debug").glob("*.json"))
    assert files
