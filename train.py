import argparse
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from dataset import MASK_TOKEN_ID, ScratchCardDataset
from model import DynamicMET
from utils.io_utils import load_boards_from_archives
from utils.logger import save_checkpoint, setup_logger


def train_epoch(
    model: DynamicMET,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train ``model`` for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        inp = batch["input_vals"].to(device)
        orig = batch["orig_vals"].to(device)
        logits = model(inp)
        # logits: (B, F, N+1); labels: (B, F) with 0 as mask
        loss = torch.nn.functional.cross_entropy(
            logits.permute(0, 2, 1), orig, ignore_index=MASK_TOKEN_ID
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main() -> None:
    """Entry point for training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tabular.yaml")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    rows = int(os.environ.get("BOARD_ROWS", 0))
    cols = int(os.environ.get("BOARD_COLS", 0))
    if rows and cols:
        cfg["model"]["num_fields"] = rows * cols
        cfg["model"]["num_values"] = rows * cols

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boards = load_boards_from_archives(cfg["data"]["data_dir"])
    dataset = ScratchCardDataset(boards, cfg["training"]["mask_ratio"])
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
    model = DynamicMET(
        num_fields=cfg["model"]["num_fields"],
        num_values=cfg["model"]["num_values"],
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        depth=cfg["model"]["depth"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    logger = setup_logger()

    for epoch in range(cfg["training"]["epochs"]):
        loss = train_epoch(model, loader, optimizer, device)
        logger.info("Epoch %s: loss=%.4f", epoch, loss)
        save_checkpoint(model, optimizer, epoch)

    out_dir = Path("checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"met_{rows}x{cols}.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": cfg["training"]["epochs"],
            "rows": rows,
            "cols": cols,
        },
        final_path,
    )
    logger.info("Saved final checkpoint to %s", final_path)


if __name__ == "__main__":
    main()
