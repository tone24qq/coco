import argparse
import os
from typing import List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_from_features(
    features_dir: str = "features",
    models_dir: str = "models",
    n_estimators: int = 100,
) -> None:
    """Train RandomForest models from feature .npz files.

    Each subdirectory in ``features_dir`` should be named ``<rows>x<cols>`` and
    contain ``<rows>x<cols>_features.npz``. The trained models are saved to
    ``models_dir`` with the same name and ``.pkl`` extension.
    """
    os.makedirs(models_dir, exist_ok=True)
    for size in os.listdir(features_dir):
        npz_path = os.path.join(features_dir, size, f"{size}_features.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path)
        X, y = data["X"], data["y"]
        clf = RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=-1, random_state=0
        )
        clf.fit(X, y)
        out_path = os.path.join(models_dir, f"{size}.pkl")
        joblib.dump(clf, out_path)
        print(f"Trained and saved model for {size} -> {out_path}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train RF models from features")
    parser.add_argument("--features-dir", default="features")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args(argv)
    train_from_features(args.features_dir, args.models_dir, args.n_estimators)


if __name__ == "__main__":  # pragma: no cover
    main()
