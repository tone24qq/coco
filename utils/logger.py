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
    prefix: str = "",
) -> None:
    """Save model checkpoint, optionally with a filename prefix."""
    # Ensure the directory exists
    Path(directory).mkdir(parents=True, exist_ok=True)

    # Prepare checkpoint data
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    # Determine filename with optional prefix
    if prefix:
        filename = f"{prefix}_epoch_{epoch}.pth"
    else:
        filename = f"epoch_{epoch}.pth"

    path = Path(directory) / filename
    torch.save(ckpt, path)
