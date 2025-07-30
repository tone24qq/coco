"""Training utilities such as EarlyStopping and accuracy helpers."""

from __future__ import annotations

import math
from typing import Callable, Dict, Sequence

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


class EarlyStopping:
    """Utility to stop training when validation loss plateaus."""

    def __init__(
        self,
        *,
        patience: int = 5,
        min_delta: float = 0.0,
        restore_best_weights: bool = False,
    ) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.restore_best_weights = restore_best_weights
        self.best_loss: float | None = None
        self.bad_epochs = 0
        self.best_state: dict[str, torch.Tensor] | None = None

    def step(self, val_loss: float, model: torch.nn.Module) -> bool:
        """Return ``True`` if training should stop."""
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.bad_epochs = 0
            if self.restore_best_weights:
                self.best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            return False
        self.bad_epochs += 1
        if self.bad_epochs > self.patience:
            if self.restore_best_weights and self.best_state is not None:
                model.load_state_dict(self.best_state)
            return True
        return False


def masked_topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    topk: Sequence[int] = (1, 3, 5),
) -> Dict[str, float]:
    """Return top-k accuracy for masked positions."""
    max_k = max(topk)
    pred = logits.topk(max_k, dim=-1).indices
    eq = pred.eq(targets.unsqueeze(-1))
    metrics: Dict[str, float] = {}
    for k in topk:
        correct = eq[..., :k].any(dim=-1)
        masked = correct[mask]
        metrics[f"top{k}"] = (
            masked.float().mean().item() if masked.numel() > 0 else float("nan")
        )
    return metrics


def cosine_schedule_with_warmup(
    total_steps: int, warmup_steps: int = 500
) -> "Callable[[int], float]":
    """Return a LR lambda implementing warmup then cosine decay."""

    def _lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return _lambda


def is_zero_loss(loss: float, eps: float = 1e-4) -> bool:
    """Return ``True`` if ``loss`` is close to zero."""

    # 中文註釋：允許極小的浮點誤差視為零損失
    return abs(loss) <= eps
