#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
- 讀取 data 目錄下的 JSON/JSONL（完整盤，無 -1）
- 依尺寸分桶訓練：每尺寸一個模型（避免可變長度複雜度）
- 動態遮蔽，只在遮蔽格計算 CE
- 迭代解碼 + 唯一性硬約束
- 失敗樣本回灌（fail_buffer）
- 以 EMR(Exact Match Rate) 作課程與早停、另存最佳
依賴：Python 3.10+、PyTorch、numpy、tqdm、orjson(可選)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import orjson as _json

    def jloads(x: bytes):
        return _json.loads(x)

except Exception:

    def jloads(x: bytes):
        return json.loads(x.decode("utf-8"))


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ------------------------- 公用 -------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_boards_from_path(path: Path) -> List[np.ndarray]:
    boards: List[np.ndarray] = []
    if path.suffix.lower() in (".jsonl", ".jsonl.gz"):
        import gzip

        opener = gzip.open if path.suffix.lower().endswith(".gz") else open
        with opener(path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = jloads(line)
                g = obj.get("grid") or obj.get("board")
                if g is None:
                    continue
                arr = np.array(g, dtype=np.int64)
                boards.append(arr)
    elif path.suffix.lower() in (".json", ".json.gz"):
        import gzip

        opener = gzip.open if path.suffix.lower().endswith(".gz") else open
        with opener(path, "rb") as f:
            obj = jloads(f.read())
        if isinstance(obj, dict) and "boards" in obj:
            seq = obj["boards"]
        elif isinstance(obj, list):
            seq = obj
        else:
            seq = []
        for item in seq:
            g = item.get("grid") if isinstance(item, dict) else item
            if g is None:
                continue
            arr = np.array(g, dtype=np.int64)
            boards.append(arr)
    return boards


def load_dataset_dir(data_dir: str) -> Dict[Tuple[int, int], List[np.ndarray]]:
    buckets: Dict[Tuple[int, int], List[np.ndarray]] = {}
    for p in Path(data_dir).rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".json", ".jsonl", ".gz"):
            continue
        try:
            bs = load_boards_from_path(p)
        except Exception:
            continue
        for b in bs:
            r, c = b.shape
            buckets.setdefault((r, c), []).append(b)
    return buckets


# ------------------------- 資料集 -------------------------


class DynamicMaskDataset(Dataset):
    def __init__(
        self, boards: List[np.ndarray], mask_lo=0.2, mask_hi=0.8, mask_token=-1
    ):
        self.boards = [torch.from_numpy(b.copy()).long().view(-1) for b in boards]
        self.mask_lo, self.mask_hi = mask_lo, mask_hi
        self.mask_token = mask_token

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, i):
        gt = self.boards[i]  # [L], 值域 1..V
        L = gt.numel()
        ratio = random.uniform(self.mask_lo, self.mask_hi)
        k = max(1, int(L * ratio))
        idx = torch.randperm(L)[:k]
        inp = gt.clone()
        inp[idx] = self.mask_token
        mask = torch.zeros(L, dtype=torch.bool)
        mask[idx] = True
        return inp, gt, mask


# ------------------------- 模型 -------------------------


class PermAwareTransformer(nn.Module):
    """
    簡潔版：數字嵌入 + 位置嵌入 + TransformerEncoder
    輸入 token 範圍：-1 表示洞，其餘 1..V
    """

    def __init__(
        self, L: int, V: int, d_model=256, nhead=8, depth=6, dim_ff=512, dropout=0.1
    ):
        super().__init__()
        self.V = V
        self.token_emb = nn.Embedding(V + 1, d_model)  # 0 給 -1 映射
        self.pos_emb = nn.Embedding(L, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, depth)
        self.out = nn.Linear(d_model, V)

    def forward(self, tokens):  # [B,L]
        x = tokens.clone()
        x = torch.where(x == -1, torch.zeros_like(x), x)  # -1 -> 0
        B, L = x.size()
        h = self.token_emb(x) + self.pos_emb(
            torch.arange(L, device=x.device)
        ).unsqueeze(0)
        h = self.enc(h)
        logits = self.out(h)  # [B,L,V]
        return logits


# ------------------------- 迭代解碼與約束 -------------------------


@torch.no_grad()
def iterative_fill(
    model, tokens, V, steps=8, tau_start=0.85, tau_end=0.55, per_step_k=5
):
    """
    唯一性硬約束 + 只填高置信前k，逐輪逼近；卡住則自然終止
    """
    B, L = tokens.size()
    mask = tokens.eq(-1)
    for t in range(steps):
        logits = model(tokens)  # [B,L,V]
        probs = F.softmax(logits, dim=-1)
        for b in range(B):
            # 禁止已用數字
            used = [int(x) for x in tokens[b].tolist() if x != -1]
            if used:
                uidx = torch.tensor(
                    [u - 1 for u in used], device=probs.device, dtype=torch.long
                )
                probs[b, :, uidx] = 0.0
            # 非洞位置不填
            probs[b, ~mask[b]] = 0.0

        conf, pred = probs.max(-1)  # [B,L]
        tau = float(tau_start + (tau_end - tau_start) * (t / max(1, steps - 1)))
        updated = 0
        for b in range(B):
            cand = torch.nonzero(mask[b] & (conf[b] >= tau), as_tuple=False).squeeze(1)
            if cand.numel() == 0:
                continue
            k = min(per_step_k, cand.numel())
            topk_idx = torch.topk(conf[b, cand], k).indices
            fill_pos = cand[topk_idx]
            tokens[b, fill_pos] = (pred[b, fill_pos] + 1).to(tokens.dtype)
            mask[b, fill_pos] = False
            updated += fill_pos.numel()
        if updated == 0:
            break
    return tokens


def uniqueness_soft_penalty(logits, tokens, lam=0.5):
    """
    把「已用過的數字」的平均機率壓低，作為軟約束。
    logits: [B,L,V]
    tokens: [B,L]，-1 表示洞，其他 1..V
    """
    B, L, V = logits.shape
    probs = F.softmax(logits, dim=-1)
    loss = 0.0
    cnt = 0
    for b in range(B):
        used = [int(x) for x in tokens[b].tolist() if x != -1]
        if not used:
            continue
        uidx = torch.tensor(
            [u - 1 for u in used], device=logits.device, dtype=torch.long
        )
        loss = loss + probs[b, :, uidx].mean()
        cnt += 1
    if cnt == 0:
        return torch.tensor(0.0, device=logits.device)
    return lam * (loss / cnt)


# ------------------------- 訓練與評估 -------------------------


@torch.no_grad()
def exact_match_rate(model, boards: List[np.ndarray], mask_rate=0.5):
    if not boards:
        return 0.0
    device = next(model.parameters()).device
    ok = 0
    for g in boards:
        gt = torch.from_numpy(g.reshape(1, -1)).long().to(device)
        inp = gt.clone()
        L = gt.numel()
        k = max(1, int(L * mask_rate))
        idx = torch.randperm(L, device=device)[:k]
        inp[0, idx] = -1
        pred = iterative_fill(model, inp, V=gt.max().item(), steps=8).view(-1)
        ok += int(torch.equal(pred.cpu(), gt.view(-1).cpu()))
    return ok / len(boards)


def train_one_bucket(size_rc: Tuple[int, int], boards: List[np.ndarray], args):
    rows, cols = size_rc
    L = rows * cols
    V = L  # 數字 1..V 一次各用一次的情境
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    # 切分 train/val
    rng = np.random.default_rng(42)
    idx = np.arange(len(boards))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_ratio))
    val_idx = idx[:n_val].tolist()
    tr_idx = idx[n_val:].tolist()
    val_boards = [boards[i] for i in val_idx]
    tr_boards = [boards[i] for i in tr_idx]

    ds = DynamicMaskDataset(tr_boards, args.mask_lo, args.mask_hi, -1)
    dl = DataLoader(
        ds, batch_size=args.bsz, shuffle=True, num_workers=0, drop_last=False
    )

    model = PermAwareTransformer(
        L=L,
        V=V,
        d_model=args.d_model,
        nhead=args.nhead,
        depth=args.depth,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_emr = -1.0
    fail_buffer: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    ckpt_dir = Path(args.out_dir) / f"checkpoints/{rows}x{cols}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dl, desc=f"[{rows}x{cols}] epoch {ep}")
        for inp, gt, mask in pbar:
            inp, gt, mask = inp.to(device), gt.to(device), mask.to(device)

            # 混入 fail_buffer
            if fail_buffer and random.random() < 0.5:
                k = min(len(fail_buffer), max(1, args.bsz // 4))
                sel = random.sample(fail_buffer, k)
                finp = torch.stack([t[0] for t in sel]).to(device)
                fgt = torch.stack([t[1] for t in sel]).to(device)
                fmsk = torch.stack([t[2] for t in sel]).to(device)
                inp = torch.cat([inp, finp], dim=0)
                gt = torch.cat([gt, fgt], dim=0)
                mask = torch.cat([mask, fmsk], dim=0)

            logits = model(inp)  # [B,L,V]
            # 只對遮蔽格算 CE；類別 1..V -> 索引 0..V-1
            ce = F.cross_entropy(logits[mask], (gt[mask] - 1).clamp(min=0, max=V - 1))
            up = uniqueness_soft_penalty(
                logits, inp, lam=args.uniq_lambda if args.uniq_soft_penalty else 0.0
            )
            loss = ce + up

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            pbar.set_postfix(
                loss=float(loss.item()),
                ce=float(ce.item()),
                up=float(up.item() if torch.is_tensor(up) else 0.0),
            )

        # 課程：EMR 過門檻則提高遮蔽上限
        model.eval()
        emr = exact_match_rate(
            model, val_boards[: min(64, len(val_boards))], mask_rate=0.5
        )
        if emr > 0.7 and ds.mask_hi < 0.95:
            ds.mask_hi = min(0.95, ds.mask_hi + 0.05)

        # 生成一次遮蔽並測試 fail，塞進 fail_buffer
        with torch.no_grad():
            for _ in range(min(64, len(tr_boards))):
                g = random.choice(tr_boards)
                gt = torch.from_numpy(g.reshape(1, -1)).long().to(device)
                inp = gt.clone()
                Lc = gt.numel()
                k = max(1, int(Lc * args.eval_mask_rate))
                idx = torch.randperm(Lc, device=device)[:k]
                inp[0, idx] = -1
                pred = iterative_fill(
                    model, inp.clone(), V=V, steps=args.decode_steps
                ).view(-1)
                if not torch.equal(pred, gt.view(-1)):
                    fail_buffer.append(
                        (
                            inp.squeeze(0).cpu(),
                            gt.squeeze(0).cpu(),
                            (inp.eq(-1)).squeeze(0).cpu(),
                        )
                    )
            # 限制 fail_buffer 大小
            if len(fail_buffer) > args.fail_buffer_max:
                fail_buffer = fail_buffer[-args.fail_buffer_max :]

        # 保存最佳，若未提升則提前終止
        if emr > best_emr:
            best_emr = emr
            out = ckpt_dir / "best.pth"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "rows": rows,
                    "cols": cols,
                    "L": L,
                    "V": V,
                    "hparams": vars(args),
                    "emr": best_emr,
                },
                out,
            )
            print(
                f"[{rows}x{cols}] epoch {ep} | EMR={emr:.3f} | best={best_emr:.3f} | mask_hi={ds.mask_hi:.2f} | fail_buf={len(fail_buffer)}"
            )
        else:
            print(
                f"[{rows}x{cols}] epoch {ep} | EMR={emr:.3f} | best={best_emr:.3f} | mask_hi={ds.mask_hi:.2f} | fail_buf={len(fail_buffer)} | early stop"
            )
            break

    return best_emr


# ------------------------- 主程式 -------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=str,
        required=True,
        help="資料資料夾，含 *.json / *.jsonl(完整盤)",
    )
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bsz", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--dim_ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mask_lo", type=float, default=0.2)
    ap.add_argument("--mask_hi", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--eval_mask_rate", type=float, default=0.5)
    ap.add_argument("--decode_steps", type=int, default=8)
    ap.add_argument("--uniq_soft_penalty", action="store_true")
    ap.add_argument("--uniq_lambda", type=float, default=0.5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fail_buffer_max", type=int, default=5000)
    args = ap.parse_args()

    set_seed(args.seed)
    buckets = load_dataset_dir(args.data)
    if not buckets:
        print("找不到任何 JSON/JSONL 完整盤資料", file=sys.stderr)
        sys.exit(1)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    logf = Path(args.out_dir) / "metrics.txt"
    with open(logf, "w", encoding="utf-8") as f:
        for (r, c), boards in sorted(buckets.items()):
            print(f"==> 訓練尺寸 {r}x{c}，樣本 {len(boards)}")
            best = train_one_bucket((r, c), boards, args)
            f.write(f"{r}x{c}\tEMR_best={best:.4f}\n")
            f.flush()


if __name__ == "__main__":
    main()
