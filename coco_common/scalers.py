import numpy as np
from sklearn.preprocessing import StandardScaler


class Float32StandardScaler(StandardScaler):
    """保持 float32，減少記憶體占用。"""

    def __setstate__(self, state: dict) -> None:
        """Restore state with backward compatibility."""
        state.setdefault("with_mean", True)
        state.setdefault("with_std", True)
        state.setdefault("copy", True)
        super().__setstate__(state)

    def fit(self, X, y=None):
        return super().fit(X.astype(np.float32, copy=False), y)

    def transform(self, X):
        return super().transform(X.astype(np.float32, copy=False))
