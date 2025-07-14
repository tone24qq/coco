import numpy as np
from fastapi.testclient import TestClient

from coco_service.main import app


def test_predict_endpoint_basic():
    client = TestClient(app)
    grid = np.array([[1, -1], [3, 4]])
    response = client.post("/predict", json={"board": grid.tolist(), "target": 2})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == grid.size
    for item in data:
        assert {"row", "col", "score"} <= set(item)
