import logging
from pathlib import Path

import torch


def setup_logger() -> logging.Logger:
    """Configure and return a simple logger."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    return logging.getLogger("scratch")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    directory: str = "checkpoints",
) -> None:
    """Save model checkpoint."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    path = Path(directory) / f"epoch_{epoch}.pth"
    torch.save(ckpt, path)
