from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.models.maskgit import MaskGit
from src.training.datasets import JsonBoardsDataset, MaskConfig, pad_collate
from src.training.dep_bias import apply_dep_bias
from src.training.loss_vec import compute_loss_vectorized


def set_seed(seed: int = 42) -> None:
    import random

    import numpy as np

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    use_dep_bias: bool = False,
    dep_alpha: float = 0.5,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_tok = 0
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            target = batch["target"].to(device)
            attn = batch["attn_mask"].to(device)
            N = batch["N"].to(device)
            rows = batch["rows"].to(device)
            cols = batch["cols"].to(device)
            logits = model(tokens, attn)
            if use_dep_bias:
                apply_dep_bias(logits, tokens, target, rows, cols, N, dep_alpha)
            loss = compute_loss_vectorized(logits, target, N)
            cnt = (target != 0).sum().item()
            total_loss += loss.item() * cnt
            total_tok += cnt
    return {"loss": total_loss / max(1, total_tok)}


def main() -> None:  # pragma: no cover - CLI
    p = argparse.ArgumentParser()
    p.add_argument("--train_json", required=True)
    p.add_argument("--val_json", required=True)
    p.add_argument("--outdir", default="runs/gridfill")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bsz", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--config",
        default=os.environ.get("GRIDFILL_CFG", "configs/train_small_mix.yaml"),
    )
    p.add_argument(
        "--use_dep_bias",
        action="store_true",
        help="計算同行+同列直方圖先驗並加到被遮蔽位置的 logits",
    )
    p.add_argument(
        "--dep_alpha",
        type=float,
        default=0.5,
        help="依賴偏置權重（加入 logits 前的縮放係數）",
    )
    args = p.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    model = MaskGit(
        cfg["vocab_size"],
        cfg["model"]["d_model"],
        cfg["model"]["n_head"],
        cfg["model"]["num_layers"],
    ).to(args.device)

    train_ds = JsonBoardsDataset(args.train_json, mask_cfg=MaskConfig())
    val_ds = JsonBoardsDataset(args.val_json, mask_cfg=MaskConfig())

    train_loader = DataLoader(
        train_ds,
        batch_size=args.bsz,
        shuffle=True,
        num_workers=2,
        collate_fn=pad_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.bsz,
        shuffle=False,
        num_workers=2,
        collate_fn=pad_collate,
        pin_memory=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val = float("inf")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    best_ckpt = outdir / "best.ckpt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for step, batch in enumerate(train_loader, 1):
            tokens = batch["tokens"].to(args.device)
            target = batch["target"].to(args.device)
            attn = batch["attn_mask"].to(args.device)
            N = batch["N"].to(args.device)
            rows = batch["rows"].to(args.device)
            cols = batch["cols"].to(args.device)

            logits = model(tokens, attn)
            if args.use_dep_bias:
                apply_dep_bias(logits, tokens, target, rows, cols, N, args.dep_alpha)
            loss = compute_loss_vectorized(logits, target, N)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 50 == 0:
                print(f"[訓練] 第 {epoch} 代，第 {step} 步，損失={loss.item():.4f}")

        val = evaluate(
            model,
            val_loader,
            args.device,
            use_dep_bias=args.use_dep_bias,
            dep_alpha=args.dep_alpha,
        )
        print(
            f"[驗證] 第 {epoch} 代，驗證每 token 損失={val['loss']:.6f}（耗時 {time.time()-t0:.1f} 秒）"
        )

        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(model.state_dict(), best_ckpt)
            print(f"[保存] 目前最佳，已儲存權重：{best_ckpt}")

    torch.save(model.state_dict(), outdir / "final.ckpt")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
