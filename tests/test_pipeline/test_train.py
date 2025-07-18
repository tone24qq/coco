import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.train import train_model


def create_dataset(path: str, rows: int = 50) -> None:
    rng = np.random.default_rng(0)
    boards = rng.integers(-1, 9, size=(rows, 16))
    targets = rng.integers(0, 2, size=rows)
    df = pd.DataFrame(boards, columns=[f"cell_{i}" for i in range(16)])
    df["target"] = targets
    df.to_csv(path, index=False)


def test_train_model(tmp_path):
    csv_path = tmp_path / "train.csv"
    create_dataset(csv_path, rows=30)
    cfg = OmegaConf.load("config/config.yaml")
    cfg.train_data = str(csv_path)
    metrics = train_model(cfg)
    assert "hit_rate@3" in metrics
