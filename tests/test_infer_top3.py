import numpy as np

from rf_infer import core


class DummyModel:
    classes_ = [0, 1]
    n_features_in_ = 4

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pos = (X[:, 0] + X[:, 1]) / 10.0
        neg = 1 - pos
        return np.vstack([neg, pos]).T

    def predict(self, X: np.ndarray, num_iteration: int | None = None) -> np.ndarray:
        return (X[:, 0] + X[:, 1]) / 10.0


def test_binary_model_ok() -> None:
    board = np.array([[-1, -1], [-1, -1]])
    res = core.predict_top_k(DummyModel(), board, target=1, k=3)
    assert isinstance(res["predictions"], list)
    assert len(res["predictions"]) <= 3


def test_infer_top3_logging(monkeypatch, caplog) -> None:
    board = np.array([[-1, -1], [-1, -1]])

    class _Dummy(DummyModel):
        pass

    monkeypatch.setattr(core, "_select_model", lambda d, r, c: "dummy")
    monkeypatch.setattr(core, "_load_model", lambda p: _Dummy())

    caplog.set_level("INFO")
    res = core.infer_top3_for_target(board, 1, models_dir="m")
    assert isinstance(res, list)
    msgs = " ".join(r.message for r in caplog.records)
    assert "infer_top3_for_target" in msgs
    assert "Selected model path" in msgs
    assert "predict_top_k returned" in msgs


def test_top3_not_empty(monkeypatch) -> None:
    board = np.array([[-1, 2], [3, 4]])

    class _Dummy(DummyModel):
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.tile(np.array([[0.2, 0.8]]), (X.shape[0], 1))

    monkeypatch.setattr(core, "_select_model", lambda d, r, c: "dummy")
    monkeypatch.setattr(core, "_load_model", lambda p: _Dummy())

    res = core.infer_top3_for_target(board, 1, models_dir="m")
    assert len(res) > 0
    for r, c in res:
        assert isinstance(r, int) and isinstance(c, int)
