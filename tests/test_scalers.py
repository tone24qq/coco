import numpy as np

from coco_common.scalers import Float32StandardScaler


def test_backward_compatibility_setstate_transform() -> None:
    state = {
        "mean_": np.array([0.0, 1.0], dtype=np.float32),
        "scale_": np.array([1.0, 2.0], dtype=np.float32),
    }
    scaler = Float32StandardScaler.__new__(Float32StandardScaler)
    scaler.__setstate__(state)
    X = np.array([[1.0, 3.0]], dtype=np.float32)
    out = scaler.transform(X)
    assert out.shape == (1, 2)
