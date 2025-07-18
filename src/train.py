"""Training pipeline using LightGBM."""

from __future__ import annotations

import logging
import os
from typing import Any

import hydra
import lightgbm
from lightgbm import LGBMClassifier, log_evaluation
from omegaconf import OmegaConf
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from .data import load_data, split_data
from .evaluate import evaluate_model, hit_rate_score, make_hit_rate_metric
from .features import build_features
from .utils import set_seed

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def train_model(cfg: Any) -> dict[str, float]:
    """Train LightGBM model and return evaluation metrics.

    Parameters
    ----------
    cfg : Any
        Hydra configuration object. Optional keys include ``n_jobs`` for
        feature extraction workers, ``use_gpu`` to enable GPU training and
        ``num_threads`` to limit CPU threads.
    """
    LOGGER.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    X, y = load_data(cfg.train_data, cfg.target_col)
    feature_df = build_features(
        X, tuple(cfg.board_shape), n_jobs=getattr(cfg, "n_jobs", 1)
    )
    X_train, X_valid, y_train, y_valid = split_data(
        feature_df, y, test_size=0.2, seed=cfg.seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    model_params = dict(cfg.model)
    if getattr(cfg, "use_gpu", False):
        model_params.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
    num_threads = getattr(cfg, "num_threads", None)
    if num_threads:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        model_params["n_jobs"] = num_threads

    estimator = LGBMClassifier(random_state=cfg.seed, **model_params)

    param_dist = {
        "num_leaves": sp_randint(15, 63),
        "learning_rate": sp_uniform(0.01, 0.19),
        "n_estimators": sp_randint(50, 200),
    }

    search = RandomizedSearchCV(
        estimator,
        param_dist,
        n_iter=cfg.optuna.n_trials,
        scoring=make_scorer(hit_rate_score, needs_proba=True, k=3),
        random_state=cfg.seed,
        cv=3,
        n_jobs=-1,
    )

    search.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=make_hit_rate_metric(k=3),
        callbacks=[
            lightgbm.early_stopping(stopping_rounds=20),
            log_evaluation(period=0),
        ],
    )

    best_model = search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_valid)
    metrics = evaluate_model(y_valid.to_numpy(), y_pred_proba, k=3)
    LOGGER.info("Validation metrics: %s", metrics)
    return metrics


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: Any) -> None:
    """Main training entry point."""
    train_model(cfg)


if __name__ == "__main__":
    main()
