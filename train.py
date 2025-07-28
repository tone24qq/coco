import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MASK_TOKEN_ID, ScratchCardDataset, validate_board
from model import DynamicMET
from utils.io_utils import load_boards_from_archives
from utils.logger import save_checkpoint, setup_logger


def train_epoch(
    model: DynamicMET,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train `model` for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        inp = batch["input_vals"].to(device)
        orig = batch["orig_vals"].to(device)
        logits = model(inp)
        loss = torch.nn.functional.cross_entropy(
            logits.permute(0, 2, 1), orig, ignore_index=MASK_TOKEN_ID
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main() -> None:
    """Train DynamicMET models and save checkpoints for prediction service."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tabular.yaml")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    # Load configuration
    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    epochs = int(cfg["training"]["epochs"])

    # Load all boards and mask out the target number so the model learns to
    # infer it from surrounding numbers
    boards = load_boards_from_archives(cfg["data"]["data_dir"], mask_target=True)
    if not boards:
        raise ValueError("No boards loaded from data_dir")
    for b, _ in boards:
        validate_board(b)

    # Group boards by shape
    shape_groups = {}
    for board, target in boards:
        shape_groups.setdefault(board.shape, []).append((board, target))

    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract and parse training hyperparams
    if "mask_ratio_range" in cfg["training"]:
        mask_ratio_cfg = cfg["training"]["mask_ratio_range"]
        mask_ratio = (float(mask_ratio_cfg[0]), float(mask_ratio_cfg[1]))
    else:
        mask_ratio = float(cfg["training"].get("mask_ratio", 0.6))
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"]["weight_decay"])

    # Train a model for each board shape
    for shape, group in shape_groups.items():
        rows, cols = shape
        model_name = f"{rows}x{cols}"
        logger.info(
            f"Starting training for shape {model_name} with {len(group)} samples"
        )

        # Prepare model fields
        cfg["model"]["num_fields"] = rows * cols
        cfg["model"]["num_values"] = rows * cols

        # Prepare dataset and loader
        dataset = ScratchCardDataset(group, mask_ratio)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        # Initialize model and optimizer
        model = DynamicMET(
            num_fields=int(cfg["model"]["num_fields"]),
            num_values=int(cfg["model"]["num_values"]),
            d_model=int(cfg["model"]["d_model"]),
            nhead=int(cfg["model"]["nhead"]),
            depth=int(cfg["model"]["depth"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Epoch loop with progress bar
        for epoch in range(1, epochs + 1):
            pbar = tqdm(
                loader,
                desc=f"{model_name} Epoch {epoch}/{epochs}",
                unit="batch",
            )
            total_loss = 0.0
            for batch_idx, batch in enumerate(pbar, start=1):
                inp = batch["input_vals"].to(device)
                orig = batch["orig_vals"].to(device)
                optimizer.zero_grad()
                logits = model(inp)
                loss = torch.nn.functional.cross_entropy(
                    logits.permute(0, 2, 1), orig, ignore_index=MASK_TOKEN_ID
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                avg_loss = total_loss / batch_idx
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Save epoch checkpoint
            save_checkpoint(model, optimizer, epoch, prefix=model_name)

        # Save final checkpoint
        out_dir = Path("checkpoints")
        out_dir.mkdir(parents=True, exist_ok=True)
        final_path = out_dir / f"met_{model_name}.pth"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epochs,
                "rows": rows,
                "cols": cols,
            },
            final_path,
        )
        # 中文註釋：儲存訓練完成的模型，供實戰預測服務載入使用
        logger.info(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
