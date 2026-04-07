import json
from argparse import Namespace
from pathlib import Path

import src.image_parse_api as image_parse_api
from fastapi.testclient import TestClient

from src.image_parse_api import app


client = TestClient(app)


def test_board_parse_api() -> None:
    img_path = Path("gogo/20/IS23120130.jpg")
    with img_path.open("rb") as f:
        resp = client.post(
            "/board/parse",
            files={"image": (img_path.name, f, "image/jpeg")},
            data={
                "rows": "5",
                "cols": "4",
                "manual_grid": json.dumps(
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
            },
        )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["contract_passed"] is True


def test_predict_number_position_api() -> None:
    img_path = Path("gogo/20/IS23120130.jpg")
    with img_path.open("rb") as f:
        resp = client.post(
            "/board/predict-number-position",
            files={"image": (img_path.name, f, "image/jpeg")},
            data={
                "query_number": "11",
                "rows": "5",
                "cols": "4",
                "manual_grid": json.dumps(
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
            },
        )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["query_status"] in ("exact_found", "predicted")


def test_board_parse_api_overlay_toggle(monkeypatch, tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.png"
    overlay.write_bytes(b"fakepng")

    def _stub(_args: Namespace) -> dict[str, object]:
        return {
            "contract_passed": True,
            "grid": [[1]],
            "overlay_path": str(overlay),
            "parse_diagnostics": {},
        }

    monkeypatch.setattr(image_parse_api, "parse_image_hybrid", _stub)
    with Path("gogo/20/IS23120130.jpg").open("rb") as f:
        resp = client.post(
            "/board/parse",
            files={"image": ("x.jpg", f, "image/jpeg")},
            data={"no_overlay": "false"},
        )
    assert resp.status_code == 200
    payload = resp.json()
    assert "overlay_image_base64" in payload
