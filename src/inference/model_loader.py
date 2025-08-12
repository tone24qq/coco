"""Load model checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch

from ..config import load_config
from ..models.maskgit import MaskGit


def get_weight_path(
    rows: int, cols: int, base: str | Path = "outputs/checkpoints"
) -> Path:
    """Return the checkpoint path for a given grid size.

    If ``{rows}x{cols}/best.pth`` exists under ``base`` that path is returned,
    otherwise the fallback ``best.pth`` is used.  This allows 部署時根據尺寸
    自動選擇對應的模型權重。
    """

    base_path = Path(base)
    specific = base_path / f"{rows}x{cols}" / "best.pth"
    return specific if specific.exists() else base_path / "best.pth"


MODEL_CACHE: Dict[Tuple[int, int, str], MaskGit] = {}


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


def load_model_for_size(
    rows: int,
    cols: int,
    *,
    base: str | Path = "outputs/checkpoints",
    device: str | torch.device = "cpu",
) -> MaskGit:
    """Load (and cache) a model for a specific grid size.

    依照 ``rows`` 與 ``cols`` 自動尋找對應的模型檔並載入，如該尺寸的
    ``{rows}x{cols}/best.pth`` 不存在則回退到 ``best.pth``。載入後會緩存，
    再次呼叫相同尺寸時會直接回傳快取的模型。所有關鍵訊息皆以中文輸出
    方便追蹤。
    """

    key = (rows, cols, str(device))
    if key not in MODEL_CACHE:
        path = get_weight_path(rows, cols, base)
        print(f"[載入模型] 尺寸 {rows}x{cols} 使用權重 {path}")
        MODEL_CACHE[key] = load_model(path, device=device)
    return MODEL_CACHE[key]
