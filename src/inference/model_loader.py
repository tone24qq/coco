"""Load model checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from ..config import load_config
from ..models.maskgit import MaskGit


def load_model(path: str | Path) -> MaskGit:
    cfg = load_config("configs/small.yaml")
    model = MaskGit(
        cfg["vocab_size"],
        cfg["model"]["d_model"],
        cfg["model"]["n_head"],
        cfg["model"]["num_layers"],
    )
    if Path(path).exists():  # pragma: no cover - optional
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
    return model
