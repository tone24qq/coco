"""
分析器（priors‑only 版）
====================================
*   只依據 `priors.npz` 熱力圖推論格子機率
*   如有 `data/meta_model.pkl`，自動載入並使用 LogisticRegression 分數
*   行、列皆回傳 **1‑base** 座標，方便與 Excel/JSON 對應
*   熱重載：`priors.npz` 被覆寫後無須重啟服務

用法 (CLI 範例)：
```bash
python -m analyzer_priors sample.json   # 顯示 Top‑3 推薦
```
或在 FastAPI：
```python
from analyzer_priors import predict
results = predict(sample_dict, topk=5)
```
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Dict

import numpy as np

try:
    import joblib
except ModuleNotFoundError:
    joblib = None  # 在無 meta_model 時可省略安裝

# --------------------------------------------------
# 路徑與全域資源
# --------------------------------------------------
BASE = Path(__file__).resolve().parent
_PRIOR_PATH = BASE / "data" / "priors" / "priors.npz"
_MODEL_PATH = BASE / "data" / "meta_model.pkl"

# 內部快取（熱重載 priors）
_PRIORS: np.lib.npyio.NpzFile | None = None
_PRIOR_MTIME: float = 0.0
_MODEL: object | None = None

# --------------------------------------------------
# 輔助：載入 / 熱重載 priors
# --------------------------------------------------

def _ensure_priors() -> None:
    global _PRIORS, _PRIOR_MTIME
    if not _PRIOR_PATH.exists():
        raise FileNotFoundError(f"找不到先驗檔 {_PRIOR_PATH}")
    mtime = _PRIOR_PATH.stat().st_mtime
    if _PRIORS is None or mtime > _PRIOR_MTIME:
        _PRIORS = np.load(_PRIOR_PATH)
        _PRIOR_MTIME = mtime

# --------------------------------------------------
# 輔助：載入 meta‑model（若存在）
# --------------------------------------------------

def _ensure_model() -> None:
    global _MODEL
    if _MODEL is not None:
        return
    if _MODEL_PATH.exists() and joblib is not None:
        _MODEL = joblib.load(_MODEL_PATH)
    else:
        _MODEL = None

# --------------------------------------------------
# 主函式：predict
# --------------------------------------------------

def predict(sample: Dict, topk: int = 3) -> List[Dict]:
    """給定單張樣本 (dict)，回傳 Top‑k 推荐格

    Parameters
    ----------
    sample : Dict
        需含 `grid` (2D list) 和選填 `target` 欄位。
    topk : int
        回傳前 k 名格子 (預設 3)。
    """
    _ensure_priors()
    _ensure_model()

    grid = np.asarray(sample["grid"], dtype=object)
    R, C = grid.shape
    size = f"{R}x{C}"

    # 位置熱力圖
    pos_prior = (_PRIORS.get(f"{size}_pos") if _PRIORS is not None else None)
    if pos_prior is None:
        pos_prior = np.full((R, C), 1.0 / (R * C), dtype=np.float32)

    # 號碼熱力圖（若有 target 且 priors 統計過）
    target_num = sample.get("target", -1)
    num_prior = (_PRIORS.get(f"{size}_num{target_num}") if target_num != -1 else None)
    if num_prior is None:
        num_prior = pos_prior

    # 特徵構建（pos_prior, num_prior, center）
    center = 1 - (np.abs(np.arange(R) - (R + 1) / 2)[:, None] +
                  np.abs(np.arange(C) - (C + 1) / 2)[None, :]) / (R + C)

    X = np.stack([pos_prior, num_prior, center], axis=-1).reshape(-1, 3)

    if _MODEL is None:
        # 無分類器 → fallback 為位置熱力圖
        scores = pos_prior.ravel()
    else:
        scores = _MODEL.predict_proba(X)[:, 1]

    # 取 Top‑k (若 -1 表示已填數字就跳過)
    blank_mask = (grid == -1)
    if not blank_mask.any():
        # 若無空格，任意回空
        return []

    masked_scores = np.where(blank_mask.ravel(), scores, -np.inf)
    k = min(topk, int(blank_mask.sum()))
    idx_topk = np.argpartition(masked_scores, -k)[-k:]
    idx_topk = idx_topk[np.argsort(masked_scores[idx_topk])[::-1]]

    total = masked_scores[idx_topk].sum()
    results = []
    for idx in idx_topk:
        r, c = divmod(int(idx), C)
        score = masked_scores[idx]
        results.append({
            "row": r + 1,
            "col": c + 1,
            "score": float(score if total == 0 else score / total)
        })
    return results

# --------------------------------------------------
# CLI 入口（可直接 python analyzer_priors.py file.json）
# --------------------------------------------------
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Scratchcard predictor (priors only)")
    parser.add_argument("json_file", help="path to sample json")
    parser.add_argument("--topk", type=int, default=3, help="how many positions to return")
    args = parser.parse_args()

    try:
        sample = json.load(open(args.json_file, encoding="utf-8"))
    except Exception as e:
        sys.exit(f"讀取樣本失敗: {e}")

    preds = predict(sample, topk=args.topk)
    print(json.dumps(preds, ensure_ascii=False, indent=2))
