#!/usr/bin/env python
"""
train_lgbm_resume.py
--------------------
• 只為「缺少的 shard」產生 .npz，再接續訓練
• 完全沿用 v7 的特徵與 NEG_RATIO=3 → 效能不變
"""

import argparse, json, math, os, zipfile, warnings, random
from pathlib import Path
from collections import defaultdict
from typing import Iterator, Tuple, List

import numpy as np, lightgbm as lgb, joblib
from tqdm.auto import tqdm

NEG_RATIO  = 3
SHARD_SIZE = 100_000
TREES_PER  = 200

# ─── JSON helpers ─────────────────────────────────────────
def _yield_json(fp) -> Iterator[object]:
    raw = fp.read()
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            yield from arr;  return
    except Exception: pass
    fp.seek(0)
    for ln in fp:
        ln = ln.strip()
        if ln:
            try: yield json.loads(ln)
            except Exception: continue

def iter_objects(root: Path) -> Iterator[object]:
    for p in root.rglob("*.zip"):
        with zipfile.ZipFile(p) as zf:
            for nm in zf.namelist():
                with zf.open(nm) as fp:
                    yield from _yield_json(fp)
    for p in root.rglob("*.json"):
        with p.open("rb") as fp:
            yield from _yield_json(fp)

# ─── Feature helpers ──────────────────────────────────────
def _local_stats(mat, r, c, k):
    sub = mat[max(0,r-k):r+k+1, max(0,c-k):c+k+1].ravel()
    sub = sub[sub!=-1]
    return (sub.mean(), sub.var()) if sub.size else (0.,0.)

def feats(masked, tgt, pos):
    r,c = pos
    row_sum = np.where(masked!=-1,masked,0).sum(axis=1)
    col_sum = np.where(masked!=-1,masked,0).sum(axis=0)
    m3,v3 = _local_stats(masked,r,c,1)
    m5,v5 = _local_stats(masked,r,c,2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u,s,vt = np.linalg.svd(np.where(masked==-1,masked.mean(),masked), full_matrices=False)
    svd = (u[:,:4] @ np.diag(s[:4]) @ vt[:4,:]).flatten()[:6]
    return [r,c,r*c,r**2,c**2,m3,v3,m5,v5,row_sum[r],col_sum[c],tgt,tgt%10,tgt//10,*svd]

def reshape_flat(arr):
    n=arr.size; r=int(math.isqrt(n))
    while r>1 and n%r: r-=1
    return arr.reshape(r,n//r)

# ─── Flush helper ─────────────────────────────────────────
def flush(sz, buf, out_root, cnt):
    if not buf[sz]: return
    X = np.stack([x for x,_ in buf[sz]])
    y = np.asarray([y for _,y in buf[sz]])
    out = out_root/sz; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/f"part_{cnt[sz]:04d}.npz", X=X, y=y)
    cnt[sz]+=1;  buf[sz].clear()

# ─── Main ────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--root", default=".")
    pa.add_argument("--out-feat", default="features")
    pa.add_argument("--out-model", default="models")
    args = pa.parse_args()

    Path(args.out_feat).mkdir(exist_ok=True)
    Path(args.out_model).mkdir(exist_ok=True)

    buf,cnt = defaultdict(list), defaultdict(int)

    # 1) Streaming only for missing shards
    print("🔄  Resuming feature extraction …")
    for obj in tqdm(iter_objects(Path(args.root)), unit="obj"):
        # decode board(s)
        if isinstance(obj,dict) and "board" in obj:
            boards=[obj["board"]]; tgts=[obj.get("target")]
        elif isinstance(obj,list):
            if obj and isinstance(obj[0],list) and isinstance(obj[0][0],list):
                boards=obj; tgts=[None]*len(boards)
            elif obj and isinstance(obj[0],list):
                boards=[obj]; tgts=[None]
            elif obj and isinstance(obj[0],int):
                boards=[obj]; tgts=[None]
            else: continue
        else: continue

        for bd,tgt in zip(boards,tgts):
            bd=np.asarray(bd,int)
            if bd.ndim==1: bd=reshape_flat(bd)
            if bd.ndim!=2: continue
            R,C=bd.shape; key=f"{R}x{C}"

            # ensure counter aligned with existing files
            while Path(args.out_feat)/key/f"part_{cnt[key]:04d}.npz" in Path(args.out_feat).rglob("part_*.npz"):
                cnt[key]+=1

            if tgt is not None:     # masked
                pos = tuple(zip(*np.where(bd==-1)))[0]
                for r in range(R):
                    for c in range(C):
                        lbl = 1 if (r,c)==pos else 0
                        buf[key].append((feats(bd,tgt,(r,c)), lbl))
            else:                   # full board
                for r in range(R):
                    for c in range(C):
                        tv=int(bd[r,c]); masked=bd.copy(); masked[r,c]=-1
                        buf[key].append((feats(masked,tv,(r,c)),1))
                        negs=[(rr,cc) for rr in range(R) for cc in range(C) if (rr,cc)!=(r,c)]
                        random.shuffle(negs)
                        for rr,cc in negs[:NEG_RATIO]:
                            buf[key].append((feats(masked,tv,(rr,cc)),0))

            if len(buf[key]) >= SHARD_SIZE:
                flush(key,buf,Path(args.out_feat),cnt)
    for k in list(buf): flush(k,buf,Path(args.out_feat),cnt)

    # 2) Training
    params=dict(objective="binary",learning_rate=0.05,num_leaves=48,max_depth=7,
                feature_fraction=0.8,bagging_fraction=0.8,min_data_in_leaf=20,
                metric=["binary_logloss"],verbosity=-1,seed=42)
    print("🏋️  Training models …")
    for sd in sorted(Path(args.out_feat).iterdir()):
        if not sd.is_dir(): continue
        npzs=sorted(sd.glob("*.npz"))
        if not npzs: continue
        booster=None
        for z in npzs:
            d=np.load(z); ds=lgb.Dataset(d['X'], d['y'])
            booster=lgb.train(params,ds,num_boost_round=TREES_PER,
                              init_model=booster,keep_training_booster=True)
        joblib.dump(booster, Path(args.out_model)/f"{sd.name}.pkl")
        print(f"✔ {sd.name} | trees={booster.num_trees()}")
    print("✅  Resume complete. Models in", args.out_model)

if __name__ == "__main__":
    main()
