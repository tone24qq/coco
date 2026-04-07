import json
import subprocess
from pathlib import Path


def test_single_image_predict_hybrid_manual_mode(tmp_path: Path) -> None:
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
        "scripts/run_single_image_predict.py",
        "--image",
        "gogo/20/IS23120130.jpg",
        "--size-class",
        "20",
        "--manual-grid",
        str(manual_grid),
        "--target-row",
        "0",
        "--target-col",
        "0",
        "--strict",
    ]
    out = subprocess.check_output(cmd, text=True)
    payload = json.loads(out)
    assert payload["contract_passed"] is True
    assert payload["source_mode"] in ("manual", "hybrid")
    assert "top3" in payload


def test_single_image_predict_reject_when_contract_fails() -> None:
    cmd = [
        "python",
        "scripts/run_single_image_predict.py",
        "--image",
        "gogo/20/IS23120130.jpg",
        "--size-class",
        "20",
        "--target-row",
        "0",
        "--target-col",
        "0",
        "--strict",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    assert proc.returncode != 0
    payload = json.loads(proc.stdout)
    assert payload["status"] == "reject_prediction"
