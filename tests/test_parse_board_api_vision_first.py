from pathlib import Path

from fastapi.testclient import TestClient

from src.image_parse_api import app


client = TestClient(app)


def test_board_parse_api_minimal_vision_first() -> None:
    img_path = Path("gogo/20/IS23120130.jpg")
    with img_path.open("rb") as f:
        resp = client.post(
            "/board/parse",
            files={"image": (img_path.name, f, "image/jpeg")},
            data={"strict": "true"},
        )
    assert resp.status_code == 200
    payload = resp.json()
    for key in (
        "rows",
        "cols",
        "shape",
        "grid",
        "numbers_all",
        "value_to_position",
        "bounding_boxes",
        "cell_boxes",
    ):
        assert key in payload
