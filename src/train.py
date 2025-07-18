"""Offline training pipeline using LightGBM and Hydra."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from hydra import main
from lightgbm import LGBMClassifier
from omegaconf import DictConfig
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from .data import load_board_dataset
from .evaluate import make_hit_rate_metric
from .features import _board_features


@main(version_base=None, config_path="../config", config_name="config")
def train(cfg: DictConfig) -> None:
    """Train model based on provided configuration."""
    boards, labels = load_board_dataset(cfg.data.path)
    X = np.array([_board_features(b) for b in boards])
    y = labels
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=cfg.data.random_seed
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)

    lgbm = LGBMClassifier(
        objective=cfg.model.objective,
        n_estimators=cfg.model.n_estimators,
        learning_rate=cfg.model.learning_rate,
        num_leaves=cfg.model.num_leaves,
    )

    param_dist = {
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.01, 0.05, 0.1],
    }

    search = RandomizedSearchCV(
        lgbm,
        param_distributions=param_dist,
        n_iter=cfg.search.n_iter,
        scoring=None,
        cv=cfg.search.cv,
        random_state=cfg.data.random_seed,
    )

    hit_rate_metric = make_hit_rate_metric(k=3)
    search.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=hit_rate_metric,
        early_stopping_rounds=cfg.model.early_stopping_rounds,
    )

    best_model = search.best_estimator_
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump((scaler, best_model), "artifacts/model.pkl")

    print(f"Best params: {search.best_params_}")
