"""Small encoder-only transformer for deterministic ranking inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class TransformerConfig:
    layers: int = 3
    d_model: int = 128
    feature_dim: int = 7
    seed: int = 42


class SmallTransformerRanker:
    def __init__(
        self, config: TransformerConfig, params: Dict[str, np.ndarray]
    ) -> None:
        self.config = config
        self.params = params

    @staticmethod
    def init_params(config: TransformerConfig) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(config.seed)
        params: Dict[str, np.ndarray] = {
            "input_w": rng.normal(0.0, 0.05, size=(config.feature_dim, config.d_model)),
            "input_b": np.zeros(config.d_model, dtype=np.float64),
            "head_w": np.zeros(config.d_model, dtype=np.float64),
            "head_b": np.zeros(1, dtype=np.float64),
        }
        for layer in range(config.layers):
            params[f"q_{layer}"] = rng.normal(
                0.0, 0.05, size=(config.d_model, config.d_model)
            )
            params[f"k_{layer}"] = rng.normal(
                0.0, 0.05, size=(config.d_model, config.d_model)
            )
            params[f"v_{layer}"] = rng.normal(
                0.0, 0.05, size=(config.d_model, config.d_model)
            )
            params[f"ff1_{layer}"] = rng.normal(
                0.0, 0.05, size=(config.d_model, config.d_model)
            )
            params[f"ff2_{layer}"] = rng.normal(
                0.0, 0.05, size=(config.d_model, config.d_model)
            )
        return params

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def encode(self, features: np.ndarray) -> np.ndarray:
        hidden = features @ self.params["input_w"] + self.params["input_b"]
        hidden = np.tanh(hidden)

        for layer in range(self.config.layers):
            q = hidden @ self.params[f"q_{layer}"]
            k = hidden @ self.params[f"k_{layer}"]
            v = hidden @ self.params[f"v_{layer}"]

            attn = (q @ k.T) / np.sqrt(float(self.config.d_model))
            attn = self._softmax(attn)
            hidden = hidden + (attn @ v)

            ff = np.tanh(hidden @ self.params[f"ff1_{layer}"])
            ff = ff @ self.params[f"ff2_{layer}"]
            hidden = hidden + ff

        return hidden

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        encoded = self.encode(features)
        scores = encoded @ self.params["head_w"] + self.params["head_b"]
        return scores.astype(np.float64)

    def fit_head(
        self, training_features: np.ndarray, training_labels: np.ndarray
    ) -> None:
        encoded_batches: List[np.ndarray] = []
        label_batches: List[np.ndarray] = []

        for x_item, y_item in zip(training_features, training_labels):
            encoded_batches.append(self.encode(x_item))
            label_batches.append(y_item)

        x_flat = np.vstack(encoded_batches)
        y_flat = np.concatenate(label_batches)

        reg = 1e-3
        xtx = x_flat.T @ x_flat + reg * np.eye(self.config.d_model)
        xty = x_flat.T @ y_flat
        head_w = np.linalg.solve(xtx, xty)

        self.params["head_w"] = head_w.astype(np.float64)
        self.params["head_b"] = np.array([float(np.mean(y_flat))], dtype=np.float64)

    def save(self, path: str) -> None:
        params = {key: value for key, value in self.params.items()}
        np.savez_compressed(path, **params)  # type: ignore[arg-type]

    @classmethod
    def load(cls, config: TransformerConfig, path: str) -> "SmallTransformerRanker":
        loaded = np.load(path)
        params = {key: loaded[key] for key in loaded.files}
        return cls(config=config, params=params)
