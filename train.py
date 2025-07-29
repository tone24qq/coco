import argparse
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MASK_TOKEN_ID, ScratchCardDataset, validate_board
from model import DynamicMET
from utils import load_heatmap
from utils.io_utils import load_boards_from_archives
from utils.logger import save_checkpoint, setup_logger

# fmt: off
# isort: off
from utils.training import (EarlyStopping, cosine_schedule_with_warmup, masked_topk_accuracy)
# isort: on
# fmt: on


class FocalLoss(nn.Module):
    """Compute Focal Loss for classification."""

    def __init__(self, gamma: float = 1.5, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = nn.functional.log_softmax(logits, dim=-1)
        gather = logp.gather(1, target.unsqueeze(1))
        pt = gather.exp()
        loss = -((1 - pt) ** self.gamma) * gather
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


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


def evaluate_epoch(
    model: DynamicMET,
    loader: DataLoader,
    device: torch.device,
    *,
    prior: torch.Tensor | None = None,
    alpha: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """Evaluate ``model`` and return loss and top-k accuracy."""
    model.eval()
    total_loss = 0.0
    metrics = {"top1": 0.0, "top3": 0.0, "top5": 0.0}
    batches = 0
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_vals"].to(device)
            orig = batch["orig_vals"].to(device)
            mask = batch["mask"].to(device)
            logits = model(inp)
            if prior is not None:
                logits = logits + alpha * prior
            loss = torch.nn.functional.cross_entropy(
                logits.permute(0, 2, 1), orig, ignore_index=MASK_TOKEN_ID
            )
            total_loss += loss.item()
            m = masked_topk_accuracy(logits, orig, mask)
            for k, v in m.items():
                if not torch.isnan(torch.tensor(v)):
                    metrics[k] += v
            batches += 1
    if batches:
        for k in metrics:
            metrics[k] /= batches
    return total_loss / len(loader), metrics


def main() -> None:
    """Train DynamicMET models and save checkpoints for prediction service."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tabular.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--lambda_rank", type=float, default=0.3)
    parser.add_argument(
        "--mode",
        choices=["target", "reconstruct"],
        default="target",
        help="Masking mode: 'target' only masks the target cell,"
        " 'reconstruct' masks a random portion of the board",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    epochs = int(cfg["training"]["epochs"])

    # Load all boards; masking is handled by the dataset according to mode
    boards = load_boards_from_archives(cfg["data"]["data_dir"], mask_target=False)
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
    if args.mode == "reconstruct":
        if "mask_ratio_range" in cfg["training"]:
            mask_ratio_cfg = cfg["training"]["mask_ratio_range"]
            mask_ratio = (float(mask_ratio_cfg[0]), float(mask_ratio_cfg[1]))
        else:
            mask_ratio = float(cfg["training"].get("mask_ratio", 0.6))
    else:
        mask_ratio = 0.0
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"]["weight_decay"])

    # Train a model for each board shape
    for shape, group in shape_groups.items():
        rows, cols = shape
        target = group[0][1]
        model_name = f"{rows}x{cols}"
        logger.info(
            f"Starting training for shape {model_name} with {len(group)} samples"
        )

        # Prepare model fields
        cfg["model"]["num_fields"] = rows * cols
        cfg["model"]["num_values"] = rows * cols

        # Prepare datasets and loaders with a validation split
        n_items = len(group)
        if n_items < 2:
            train_pairs = val_pairs = list(group)
        else:
            perm = torch.randperm(n_items)
            split = max(1, int(0.8 * n_items))
            train_pairs = [group[i] for i in perm[:split]]
            val_pairs = [group[i] for i in perm[split:]]

        train_ds = ScratchCardDataset(train_pairs, mask_ratio, mode=args.mode)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
        )
        if args.mode == "reconstruct":
            val_ratios = (0.3, 0.7)
        else:
            val_ratios = (0.0,)
        val_loaders = [
            DataLoader(
                ScratchCardDataset(val_pairs, mask_ratio=r, mode=args.mode),
                batch_size=batch_size,
            )
            for r in val_ratios
        ]

        heat = load_heatmap(rows, cols, target, device=device)
        prior = torch.log(heat.flatten() + 1e-6)

        criterion_cls = FocalLoss(gamma=args.gamma)
        criterion_rank = nn.MarginRankingLoss(margin=args.margin)

        # Initialize model and optimizer
        model = DynamicMET(
            num_fields=int(cfg["model"]["num_fields"]),
            num_values=int(cfg["model"]["num_values"]),
            d_model=int(cfg["model"]["d_model"]),
            nhead=int(cfg["model"]["nhead"]),
            depth=int(cfg["model"]["depth"]),
            dropout=float(cfg["model"].get("dropout", 0.0)),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )

        total_steps = epochs * len(train_loader)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=cosine_schedule_with_warmup(total_steps, warmup_steps=500),
        )
        # Epoch loop with progress bar

        early_stop = EarlyStopping(
            patience=5, min_delta=0.001, restore_best_weights=True
        )

        trained_epochs = 0
        for epoch in range(1, epochs + 1):
            pbar = tqdm(
                train_loader,
                desc=f"{model_name} Epoch {epoch}/{epochs}",
                unit="batch",
            )
            total_loss = 0.0
            for batch_idx, batch in enumerate(pbar, start=1):
                inp = batch["input_vals"].to(device)
                orig = batch["orig_vals"].to(device)
                optimizer.zero_grad()
                logits = model(inp)
                logits = logits + args.alpha * prior

                flat_logits = logits.view(-1, logits.size(-1))
                flat_target = orig.view(-1)
                valid = flat_target.ne(MASK_TOKEN_ID)
                clf_loss = criterion_cls(flat_logits[valid], flat_target[valid])

                k = 3
                neg_idx = torch.randint(
                    0, logits.size(-1), (int(valid.sum().item()), k), device=device
                )
                pos_sc = (
                    flat_logits[valid]
                    .gather(1, flat_target[valid].unsqueeze(1))
                    .squeeze(1)
                )
                neg_sc = flat_logits[valid].gather(1, neg_idx).mean(1)
                rank_loss = criterion_rank(pos_sc, neg_sc, torch.ones_like(pos_sc))

                loss = clf_loss + args.lambda_rank * rank_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                avg_loss = total_loss / batch_idx
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            avg_loss = total_loss / len(train_loader)

            # Validation phase over multiple loaders
            val_losses = []
            topk = {"top1": 0.0, "top3": 0.0, "top5": 0.0}
            for v_loader in val_loaders:
                v_loss, v_metrics = evaluate_epoch(
                    model, v_loader, device, prior=prior, alpha=args.alpha
                )
                val_losses.append(v_loss)
                for k in topk:
                    topk[k] += v_metrics.get(k, 0.0)
            val_loss = sum(val_losses) / len(val_losses)
            for k in topk:
                topk[k] /= len(val_loaders)
            logger.info(
                "%s epoch %s: train_loss=%.4f val_loss=%.4f top1=%.3f top3=%.3f top5=%.3f",
                model_name,
                epoch,
                avg_loss,
                val_loss,
                topk["top1"],
                topk["top3"],
                topk["top5"],
            )
            if early_stop.step(val_loss, model):
                logger.info("Early stopping triggered for %s", model_name)
                trained_epochs = epoch
                break

            trained_epochs = epoch

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
                "epoch": trained_epochs or epochs,
                "rows": rows,
                "cols": cols,
            },
            final_path,
        )
        # 中文註釋：儲存訓練完成的模型，供實戰預測服務載入使用
        logger.info(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
