from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_fill_success() -> None:
    board = [[-1, 2], [3, -1]]
    r = client.post("/fill", json={"board": board, "target": 1, "row": 0, "col": 0})
    assert r.status_code == 200
    assert r.json() == [[1, 2], [3, -1]]


def test_fill_cell_not_blank() -> None:
    board = [[4, -1], [-1, 2]]
    r = client.post("/fill", json={"board": board, "target": 3, "row": 0, "col": 0})
    assert r.status_code == 422


def test_fill_target_exists() -> None:
    board = [[-1, 1], [-1, -1]]
    r = client.post("/fill", json={"board": board, "target": 1, "row": 0, "col": 0})
    assert r.status_code == 422
