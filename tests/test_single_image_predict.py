import json
import subprocess


def test_single_image_predict_cli_runs() -> None:
    cmd = [
        "python",
        "scripts/run_single_image_predict.py",
        "--image",
        "gogo/20/IS23120130.jpg",
        "--target-row",
        "0",
        "--target-col",
        "0",
    ]
    out = subprocess.check_output(cmd, text=True)
    payload = json.loads(out)
    assert "top3" in payload or payload.get("status") == "parse_confidence_too_low"
