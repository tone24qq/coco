"""Load model checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from ..config import load_config
from ..models.maskgit import MaskGit


def load_model(path: str | Path, *, device: str | torch.device = "cpu") -> MaskGit:
    cfg = load_config("configs/small.yaml")
    model = MaskGit(
        cfg["vocab_size"],
        cfg["model"]["d_model"],
        cfg["model"]["n_head"],
        cfg["model"]["num_layers"],
    )
    p = Path(path)
    if p.exists():  # pragma: no cover - optional
        try:
            if p.stat().st_size > 0:
                state = torch.load(p, map_location=device)
                model.load_state_dict(state)
        except Exception:  # pragma: no cover - best effort
            pass
    model.to(device)
    model.eval()
    return model
