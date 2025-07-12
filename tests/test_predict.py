from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_root_status():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "OK"


def test_root_head():
    res = client.head("/")
    assert res.status_code == 200
    assert res.text == ""


def test_predict_valid_grid():
    grid = []
    val = 1
    for _ in range(8):
        row = []
        for _ in range(8):
            row.append(val)
            val += 1
        grid.append(row)
    grid[2][3] = -1
    payload = {
        "grid": grid,
        "target_num": 6,
        "iterations": 10,
        "global_iter": 5,
        "focus_iter": 2,
        "top_n": 5,
        "epsilon": 0.1,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert "predictions" in body
    assert "top_predictions" in body
    assert "full_probabilities" not in body
    assert isinstance(body["predictions"], list)
    assert isinstance(body.get("final_recommendations"), list)


def test_invalid_duplicate_numbers():
    payload = {"grid": [[1, 2, 2], [4, 5, 6]]}
    res = client.post("/predict", json=payload)
    assert res.status_code == 500
    assert "duplicate" in res.json()["detail"].lower()


def test_invalid_small_grid():
    payload = {"grid": [[1]]}
    res = client.post("/predict", json=payload)
    assert res.status_code == 500
    assert "at least 2x2" in res.json()["detail"].lower()


def test_target_num_out_of_range():
    grid = [[1, -1], [3, 4]]
    payload = {"grid": grid, "target_num": 5}
    res = client.post("/predict", json=payload)
    assert res.status_code == 400
