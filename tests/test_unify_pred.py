import numpy as np

from rf_infer.core import _unify_pred_output


def test_unify_binary() -> None:
    raw = np.array([[0.2, 0.8], [0.7, 0.3]])
    out = _unify_pred_output(raw)
    assert out.shape == (2,)
    assert np.allclose(out, [0.8, 0.3])


def test_unify_multiclass() -> None:
    raw = np.array([[0.1, 0.5, 0.4], [0.2, 0.1, 0.7]])
    out = _unify_pred_output(raw)
    assert out.shape == (2,)
    assert np.allclose(out, [0.5, 0.7])
