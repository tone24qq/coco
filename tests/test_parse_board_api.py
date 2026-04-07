import json
from pathlib import Path

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
