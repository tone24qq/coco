from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.config import load_config
from src.models.maskgit import MaskGit
from src.training.datasets import JsonBoardsDataset, pad_collate
from src.training.dep_bias import apply_dep_bias
from src.training.loss_vec import compute_loss_vectorized
from src.training.train import set_seed


# ----------------------------
# 位置大小分布先驗（EMA 緩存）
class SizePriorCache:
    def __init__(self, vocab_max: int, alpha: float) -> None:
        self.alpha = alpha
        self.vocab_max = vocab_max
        self.maps: dict[tuple[int, int], torch.Tensor] = {}

    def get(self, R: int, C: int, device: torch.device) -> torch.Tensor:
        key = (int(R), int(C))
        if key not in self.maps:
            # [V+1, R, C]；用極小值避免 log(0)
            self.maps[key] = torch.full((self.vocab_max + 1, R, C), 1e-6, device=device)
        return self.maps[key]

    def update_batch(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        targets: torch.Tensor,
        N: torch.Tensor,
    ) -> None:
        # 簡潔實作：全圖衰減，再對命中位置加 α
        B, _ = targets.shape
        device = targets.device
        for b in range(B):
            R = int(rows[b].item())
            C = int(cols[b].item())
            Nb = int(N[b].item())
            H = self.get(R, C, device)
            H.mul_(1.0 - self.alpha)  # EMA 衰減
            t = targets[b, : R * C].view(R, C)
            # 逐格加權（可後續向量化優化）
            for r in range(R):
                for c in range(C):
                    v = int(t[r, c].item())
                    if 1 <= v <= Nb:
                        H[v, r, c] += self.alpha


def apply_size_bias(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    target: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    N: torch.Tensor,
    cache: SizePriorCache,
    beta: float,
) -> None:
    """對被遮蔽位置加入『大小×座標』先驗 bias（log(heatmap)×beta）。"""
    device = logits.device
    V = logits.size(-1)
    mask_pos = ((tokens == 0) & (target != 0)).nonzero(as_tuple=False)
    eps = 1e-6
    for b, lidx in mask_pos.tolist():
        R = int(rows[b].item())
        C = int(cols[b].item())
        r = lidx // C
        c = lidx % C
        Nb = int(N[b].item())
        H = cache.get(R, C, device)
        prior = H[: Nb + 1, r, c]  # [Nb+1]
        log_prior = torch.log(prior + eps) * beta
        bias = torch.full((V,), -1e9, device=device)
        bias[: log_prior.numel()] = log_prior
        logits[b, lidx, :] += bias


def apply_nbr3x3_bias(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    target: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    N: torch.Tensor,
    beta: float,
) -> None:
    """對被遮蔽位置加入 3×3 鄰域直方圖先驗（log(鄰域分布)×beta）。"""
    device = logits.device
    V = logits.size(-1)
    eps = 1e-6
    mask_pos = ((tokens == 0) & (target != 0)).nonzero(as_tuple=False)
    for b, lidx in mask_pos.tolist():
        R = int(rows[b].item())
        C = int(cols[b].item())
        r = int(lidx // C)
        c = int(lidx % C)
        Nb = int(N[b].item())
        Vmax = min(V - 1, Nb)
        grid = tokens[b, : R * C].view(R, C)
        r0 = max(0, r - 1)
        r1 = min(R - 1, r + 1)
        c0 = max(0, c - 1)
        c1 = min(C - 1, c + 1)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].reshape(-1)
        hist = torch.bincount(patch, minlength=Vmax + 1).to(device)
        hist[0] = 0  # 忽略 MASK=0
        prior = hist.float()
        if prior.sum() > 0:
            prior = prior / prior.sum()
        log_prior = torch.log(prior + eps) * beta
        bias = torch.full((V,), -1e9, device=device)
        bias[: log_prior.numel()] = log_prior
        logits[b, lidx, :] += bias


# ----------------------------


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    *,
    use_dep_bias: bool = False,
    dep_alpha: float = 0.5,
    use_size_bias: bool = False,
    size_cache: SizePriorCache | None = None,
    size_beta: float = 0.5,
    use_nbr3x3: bool = False,
    nbr_beta: float = 0.3,
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
            if use_nbr3x3:
                apply_nbr3x3_bias(logits, tokens, target, rows, cols, N, nbr_beta)
            if use_size_bias and size_cache is not None:
                size_cache.update_batch(rows, cols, target, N)
                apply_size_bias(
                    logits, tokens, target, rows, cols, N, size_cache, size_beta
                )

            loss = compute_loss_vectorized(logits, target, N)
            cnt = (target != 0).sum().item()
            total_loss += loss.item() * cnt
            total_tok += cnt
    return {"loss": total_loss / max(1, total_tok)}


def find_datasets(
    data_dir: Path = Path(__file__).parent / "src" / "data",
) -> List[Tuple[int, int, Path]]:
    """Find dataset JSON files like ``4x5.json`` under ``data_dir``."""

    pattern = re.compile(r"^(\d+)x(\d+)\.json$")
    out: List[Tuple[int, int, Path]] = []
    for p in data_dir.glob("*.json"):
        m = pattern.match(p.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), p))
    return out


def train_one(
    path: Path,
    rows: int,
    cols: int,
    *,
    epochs: int = 10,
    bsz: int = 32,
    lr: float = 3e-4,
    seed: int = 42,
    device: str | None = None,
    target_loss: float | None = None,  # 門檻：達標後「等到 epoch 結束」才早停
    use_dep_bias: bool = False,
    dep_alpha: float = 0.5,
    use_size_bias: bool = False,
    size_alpha: float = 0.02,
    size_beta: float = 0.5,
    use_nbr3x3: bool = False,
    nbr_beta: float = 0.3,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[資料] 使用 {path} 訓練 {rows}x{cols} 模型")
    set_seed(seed)
    cfg = load_config("configs/train_small_mix.yaml")
    model = MaskGit(
        cfg["vocab_size"],
        cfg["model"]["d_model"],
        cfg["model"]["n_head"],
        cfg["model"]["num_layers"],
    ).to(device)

    # 若你的資料包含 target 欄位，用 mask_target=True 會只遮住目標；
    # 你也可以改 JsonBoardsDataset 的參數以符合當前資料版型。
    dataset = JsonBoardsDataset(path, mask_target=True, seed=seed)
    if len(dataset) < 2:
        raise ValueError("dataset 太小，至少需 2 筆資料")
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=bsz,
        shuffle=True,
        num_workers=2,
        collate_fn=pad_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bsz,
        shuffle=False,
        num_workers=2,
        collate_fn=pad_collate,
        pin_memory=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf")
    outdir = Path("weights")
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir / f"{rows}x{cols}.ckpt"

    size_cache = (
        SizePriorCache(cfg["vocab_size"] - 1, size_alpha) if use_size_bias else None
    )
    for epoch in range(1, epochs + 1):
        model.train()
        hit_threshold_this_epoch = False  # 做法B：本 epoch 內是否曾達標

        for step, batch in enumerate(train_loader, 1):
            tokens = batch["tokens"].to(device)
            target = batch["target"].to(device)
            attn = batch["attn_mask"].to(device)
            N = batch["N"].to(device)
            rows_t = batch["rows"].to(device)
            cols_t = batch["cols"].to(device)

            logits = model(tokens, attn)
            # 先驗（可並用）
            if use_dep_bias:
                apply_dep_bias(logits, tokens, target, rows_t, cols_t, N, dep_alpha)
            if use_nbr3x3:
                apply_nbr3x3_bias(logits, tokens, target, rows_t, cols_t, N, nbr_beta)
            if use_size_bias and size_cache is not None:
                size_cache.update_batch(rows_t, cols_t, target, N)
                apply_size_bias(
                    logits, tokens, target, rows_t, cols_t, N, size_cache, size_beta
                )

            loss = compute_loss_vectorized(logits, target, N)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 50 == 0:
                print(
                    f"[訓練] {rows}x{cols} 第 {epoch} 代 第 {step} 步 損失={loss.item():.4f}"
                )

            # 標記達標，但不 break —— 等 epoch 結束後再處理
            if target_loss is not None and loss.item() <= target_loss:
                hit_threshold_this_epoch = True

        # ---- 一個 epoch 結束後才評估與可能早停 ----
        val = evaluate(
            model,
            val_loader,
            device,
            use_dep_bias=use_dep_bias,
            dep_alpha=dep_alpha,
            use_size_bias=use_size_bias,
            size_cache=size_cache,
            size_beta=size_beta,
            use_nbr3x3=use_nbr3x3,
            nbr_beta=nbr_beta,
        )
        print(f"[驗證] {rows}x{cols} 第 {epoch} 代 損失={val['loss']:.6f}")
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(model.state_dict(), ckpt_path)
            print(f"[保存] {rows}x{cols} 新最佳模型 -> {ckpt_path}")

        if hit_threshold_this_epoch:
            print(
                f"[早停] 第 {epoch} 代內曾達到門檻 {target_loss:.6g}，於 epoch 結束觸發停止。"
            )
            # 再存一次，確保有最後權重（若上面已是最佳也OK）
            torch.save(model.state_dict(), ckpt_path)
            print(f"[保存] {rows}x{cols} 收斂模型 -> {ckpt_path}")
            break


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="src/data")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bsz", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="cuda / cpu；預設自動偵測")
    p.add_argument(
        "--target_loss",
        type=float,
        default=None,
        help="達到此訓練 loss 門檻時，不中斷步數；等該 epoch 結束後早停",
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
    # 新增：大小×座標先驗
    p.add_argument("--use_size_bias", action="store_true")
    p.add_argument(
        "--size_alpha",
        type=float,
        default=0.02,
        help="大小分布 EMA 係數",
    )
    p.add_argument(
        "--size_beta",
        type=float,
        default=0.5,
        help="加入 logits 的權重",
    )
    # 新增：3×3 鄰域先驗
    p.add_argument("--use_nbr3x3", action="store_true")
    p.add_argument(
        "--nbr_beta",
        type=float,
        default=0.3,
        help="加入 logits 的權重",
    )
    return p.parse_args()


def main() -> None:  # pragma: no cover - CLI
    args = parse_args()
    datasets = find_datasets(Path(args.data_dir))
    if not datasets:
        print("[警告] 未找到任何資料集，請將 NxY.json 放於 src/data 目錄下")
        return
    for rows, cols, path in datasets:
        print(f"[開始] 訓練 {rows}x{cols} 模型")
        train_one(
            path,
            rows,
            cols,
            epochs=args.epochs,
            bsz=args.bsz,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            target_loss=args.target_loss,
            use_dep_bias=args.use_dep_bias,
            dep_alpha=args.dep_alpha,
            use_size_bias=args.use_size_bias,
            size_alpha=args.size_alpha,
            size_beta=args.size_beta,
            use_nbr3x3=args.use_nbr3x3,
            nbr_beta=args.nbr_beta,
        )


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
