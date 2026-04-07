import json
import subprocess
from pathlib import Path


MANUAL_GRID = {
    "grid": [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
    ]
}


def test_single_image_predict_mode_a_target_cell(tmp_path: Path) -> None:
    manual_grid = tmp_path / "manual.json"
    manual_grid.write_text(json.dumps(MANUAL_GRID), encoding="utf-8")
    cmd = [
        "python",
        "scripts/run_single_image_predict.py",
        "--image",
        "gogo/20/IS23120130.jpg",
        "--rows",
        "5",
        "--cols",
        "4",
        "--manual-grid",
        str(manual_grid),
        "--target-row",
        "0",
        "--target-col",
        "0",
    ]
    out = subprocess.check_output(cmd, text=True)
    payload = json.loads(out)
    assert payload["mode"] == "target_cell_digit"
    assert "top5" in payload


def test_single_image_predict_mode_b_query_number(tmp_path: Path) -> None:
    manual_grid = tmp_path / "manual.json"
    manual_grid.write_text(json.dumps(MANUAL_GRID), encoding="utf-8")
    cmd = [
        "python",
        "scripts/run_single_image_predict.py",
        "--image",
        "gogo/20/IS23120130.jpg",
        "--rows",
        "5",
        "--cols",
        "4",
        "--manual-grid",
        str(manual_grid),
        "--query-number",
        "11",
    ]
    out = subprocess.check_output(cmd, text=True)
    payload = json.loads(out)
    assert payload["mode"] == "query_number_position"
    assert "top5_position_candidates" in payload


def test_single_image_predict_mode_conflict_fail_fast() -> None:
    cmd = [
        "python",
        "scripts/run_single_image_predict.py",
        "--image",
        "gogo/20/IS23120130.jpg",
        "--rows",
        "5",
        "--cols",
        "4",
        "--query-number",
        "11",
        "--target-row",
        "0",
        "--target-col",
        "0",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    assert proc.returncode != 0
