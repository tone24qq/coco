"""Data loading utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset from CSV."""
    df = pd.read_csv(path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series, *, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and validation sets."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
