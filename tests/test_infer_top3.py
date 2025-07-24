import numpy as np

from rf_infer.core import predict_top_k


class DummyModel:
    classes_ = [0, 1]
    n_features_in_ = 4

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pos = (X[:, 0] + X[:, 1]) / 10.0
        neg = 1 - pos
        return np.vstack([neg, pos]).T


def test_binary_model_ok() -> None:
    board = np.array([[-1, -1], [-1, -1]])
    res = predict_top_k(DummyModel(), board, target=1, k=3)
    assert isinstance(res["predictions"], list)
    assert len(res["predictions"]) <= 3
