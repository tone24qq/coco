import json
import subprocess
from pathlib import Path


def test_board_parser_manual_override_schema(tmp_path: Path) -> None:
    manual_grid = tmp_path / "manual.json"
    manual_grid.write_text(
        json.dumps(
            {
                "grid": [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                    [17, 18, 19, 20],
                ]
            }
        ),
        encoding="utf-8",
    )
    cmd = [
        "python",
        "scripts/parse_board_image.py",
        "--image",
        "gogo/20/IS23120130.jpg",
        "--size-class",
        "20",
        "--strict",
        "--manual-grid",
        str(manual_grid),
    ]
    out = subprocess.check_output(cmd, text=True)
    payload = json.loads(out.strip().splitlines()[-1])
    for key in (
        "status",
        "source_mode",
        "shape",
        "grid",
        "numbers_all",
        "value_to_position",
        "parse_diagnostics",
        "contract_passed",
        "needs_manual_review",
    ):
        assert key in payload
    assert payload["contract_passed"] is True


def test_board_parser_auto_fail_without_manual() -> None:
    cmd = [
        "python",
        "scripts/parse_board_image.py",
        "--image",
        "gogo/20/IS23120130.jpg",
        "--size-class",
        "20",
        "--strict",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    lines = [x for x in proc.stdout.splitlines() if x.strip().startswith("{")]
    payload = json.loads(lines[-1])
    assert payload["status"] in (
        "needs_manual_review",
        "shape_mismatch",
        "incomplete_grid",
        "low_confidence_parse",
        "contract_violation",
    )
