"""Minimal training loop placeholder."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..config import load_config
from ..data.augment import mask_board
from ..data.collate import collate_boards
from ..data.datasets import GridDataset
from ..models.maskgit import MaskGit
from .loss import masked_cross_entropy


def train(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    ds = (
        GridDataset(Path("data/train.jsonl"))
        if Path("data/train.jsonl").exists()
        else None
    )
    model = MaskGit(
        cfg["vocab_size"],
        cfg["model"]["d_model"],
        cfg["model"]["n_head"],
        cfg["model"]["num_layers"],
    )
    if ds is None:
        return
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg["train"]["batch_size"], shuffle=True
    )
    for board in loader:
        tokens, mask = mask_board(board)
        flat, attn = collate_boards([tokens])
        logits = model(flat)
        loss = masked_cross_entropy(logits, flat, mask.flatten())
        loss.backward()
        opt.step()
        opt.zero_grad()
        break


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
