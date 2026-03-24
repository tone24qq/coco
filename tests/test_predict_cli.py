from pathlib import Path

import pytest

from src import predict as predict_cli


def test_predict_cli_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out_json = tmp_path / "out.json"
    monkeypatch.setattr(
        "src.predict.predict",
        lambda runtime_dir: {
            "latest_known_issue": "1001",
            "target_issue": "1002",
            "top20": [{"number": i, "score": float(i)} for i in range(20, 0, -1)],
            "top3": [
                {"number": 20, "score": 20.0},
                {"number": 11, "score": 11.0},
                {"number": 2, "score": 2.0},
            ],
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        ["predict", "--runtime-dir", str(tmp_path), "--output-json", str(out_json)],
    )

    predict_cli.main()
    captured = capsys.readouterr().out
    if "latest_issue=1001 target_issue=1002" not in captured:
        pytest.fail("predict cli output mismatch")
    if not out_json.exists():
        pytest.fail("predict cli output json missing")
