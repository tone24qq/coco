from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_valid_small_board_4x5() -> None:
    board = [
        [1, 2, -1, 4, 5],
        [6, -1, 8, 9, 10],
        [11, 12, 13, -1, 15],
        [16, 17, 18, 19, 20],
    ]
    response = client.post("/infer_target_position", json={"board": board, "target_number": 14})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["board_shape"] == {"rows": 4, "cols": 5}
    assert payload["best_cell"] is not None
    assert payload["confidence_score"] == payload["best_confidence_score"]
    assert payload["best_ranking_score"] == payload["best_cell"]["score"]
    assert payload["metadata"]["confidence_1_to_100_is_probability"] is False


def test_valid_large_board_8x10() -> None:
    n = 1
    board = []
    for _ in range(8):
        row = []
        for _ in range(10):
            row.append(n)
            n += 1
        board.append(row)
    board[0][0] = -1
    board[3][5] = -1
    response = client.post("/infer_target_position", json={"board": board, "target_number": 36})
    assert response.status_code == 200
    payload = response.json()
    assert payload["board_shape"] == {"rows": 8, "cols": 10}
    assert len(payload["candidate_cells"]) == 2


def test_target_already_opened() -> None:
    board = [[1, 2], [3, -1]]
    response = client.post("/infer_target_position", json={"board": board, "target_number": 3})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "already_opened"
    assert payload["best_cell"] == {"row": 2, "col": 1, "score": 1.0, "confidence_1_to_100": 100.0}


def test_target_not_opened() -> None:
    board = [[1, -1], [3, 4]]
    response = client.post("/infer_target_position", json={"board": board, "target_number": 2})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_duplicate_opened_number() -> None:
    board = [[1, -1], [1, 4]]
    response = client.post("/infer_target_position", json={"board": board, "target_number": 2})
    assert response.status_code == 422


def test_out_of_range_number() -> None:
    board = [[1, -1], [3, 9]]
    response = client.post("/infer_target_position", json={"board": board, "target_number": 2})
    assert response.status_code == 422


def test_non_rectangular_board() -> None:
    board = [[1, -1], [3]]
    response = client.post("/infer_target_position", json={"board": board, "target_number": 2})
    assert response.status_code == 422


def test_no_unopened_cells() -> None:
    board = [[1, 2], [3, 4]]
    response = client.post("/infer_target_position", json={"board": board, "target_number": 2})
    assert response.status_code == 200
    assert response.json()["status"] == "already_opened"


def test_candidate_cells_sorted_descending() -> None:
    board = [[1, -1, 3], [-1, 5, -1]]
    response = client.post("/infer_target_position", json={"board": board, "target_number": 4})
    assert response.status_code == 200
    cells = response.json()["candidate_cells"]
    scores = [cell["score"] for cell in cells]
    assert scores == sorted(scores, reverse=True)


def test_multi_target_api_returns_unique_assignments() -> None:
    board = [[1, -1], [-1, 4]]
    response = client.post("/infer_multi_target_positions", json={"board": board, "target_numbers": [2, 3]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    cells = {(a["row"], a["col"]) for a in payload["assignments"]}
    assert len(cells) == 2
