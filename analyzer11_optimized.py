"""
analyzer11_optimized.py - 優化版分析器，支援動態權重調整
"""

import os
import json
import logging
import time
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path

# 從向量化模組導入

from vectorized_modules import SCORING_MODULES

logger = logging.getLogger(**name**)

# 全域變數

MEMORY_SAMPLES: List[Dict[str, Any]] = []
GLOBAL_WEIGHTS: Dict[str, float] = {}
SHAPE_WEIGHTS: Dict[Tuple[int, int], Dict[str, float]] = {}
_last_memory_load_time: float = 0.0

# 快取機制

_score_cache = {}
_cache_hits = 0
_cache_misses = 0

def _load_memory_folder(folder_path: str = "memory_data"):
"""載入歷史樣本"""
global MEMORY_SAMPLES, _last_memory_load_time
MEMORY_SAMPLES.clear()

```
folder = Path(folder_path)
if not folder.exists():
    logger.warning(f"記憶資料夾 {folder_path} 不存在")
    return

max_mtime = 0.0
for json_file in folder.glob("*.json"):
    try:
        mtime = json_file.stat().st_mtime
        max_mtime = max(max_mtime, mtime)
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for sample in data:
                    if all(k in sample for k in ['grid', 'target', 'true_pos']):
                        grid = np.array(sample['grid'])
                        rows, cols = grid.shape
                        sample['card_shape'] = (rows, cols)
                        sample['grid_array'] = grid
                        MEMORY_SAMPLES.append(sample)
    except Exception as e:
        logger.error(f"載入 {json_file} 失敗: {e}")

_last_memory_load_time = max_mtime
logger.info(f"載入了 {len(MEMORY_SAMPLES)} 個歷史樣本")
```

def compute_weights_from_memory() -> Tuple[Dict[str, float], Dict[Tuple[int, int], Dict[str, float]]]:
"""從歷史樣本計算權重"""
if not MEMORY_SAMPLES:
return {}, {}

```
# 全域權重
global_scores = {name: [] for name in SCORING_MODULES}

# 按形狀分組的權重
shape_grouped = {}

for sample in MEMORY_SAMPLES:
    grid = sample['grid_array']
    true_r, true_c = sample['true_pos']
    shape = sample['card_shape']
    
    # 計算每個評分模組的分數
    for name, func in SCORING_MODULES.items():
        try:
            scores = func(grid)
            score_at_true = scores[true_r, true_c]
            global_scores[name].append(score_at_true)
            
            # 按形狀分組
            if shape not in shape_grouped:
                shape_grouped[shape] = {n: [] for n in SCORING_MODULES}
            shape_grouped[shape][name].append(score_at_true)
        except Exception as e:
            logger.error(f"評分模組 {name} 失敗: {e}")

# 計算全域權重
global_weights = {}
total = 0
for name, scores in global_scores.items():
    if scores:
        avg = np.mean(scores)
        global_weights[name] = avg
        total += avg

if total > 0:
    for name in global_weights:
        global_weights[name] /= total

# 計算形狀權重
shape_weights = {}
for shape, module_scores in shape_grouped.items():
    weights = {}
    total = 0
    for name, scores in module_scores.items():
        if scores:
            avg = np.mean(scores)
            weights[name] = avg
            total += avg
    
    if total > 0:
        for name in weights:
            weights[name] /= total
        shape_weights[shape] = weights

return global_weights, shape_weights
```

def get_adaptive_weights(grid: np.ndarray, target: int) -> np.ndarray:
"""根據當前網格和目標數字獲取自適應權重"""
rows, cols = grid.shape
shape = (rows, cols)

```
# 優先使用形狀特定權重
if shape in SHAPE_WEIGHTS and len(SHAPE_WEIGHTS[shape]) > 0:
    weights_dict = SHAPE_WEIGHTS[shape]
else:
    weights_dict = GLOBAL_WEIGHTS

# 如果沒有歷史權重，使用均等權重
if not weights_dict:
    num_modules = len(SCORING_MODULES)
    return np.ones(num_modules) / num_modules

# 將權重字典轉換為數組
weights = []
for name in SCORING_MODULES:
    weights.append(weights_dict.get(name, 1.0 / len(SCORING_MODULES)))

return np.array(weights, dtype=np.float32)
```

def collect_all_scores(grid: np.ndarray, request_id: str = "API") -> np.ndarray:
"""收集所有評分模組的分數（向量化版本）"""
rows, cols = grid.shape
num_modules = len(SCORING_MODULES)

```
# 檢查快取
grid_hash = hash(grid.tobytes())
if grid_hash in _score_cache:
    global _cache_hits
    _cache_hits += 1
    logger.debug(f"快取命中 (命中率: {_cache_hits/(_cache_hits+_cache_misses):.2%})")
    return _score_cache[grid_hash]

global _cache_misses
_cache_misses += 1

# 使用向量化並行計算
tensor = np.zeros((num_modules, rows, cols), dtype=np.float32)

for i, (name, func) in enumerate(SCORING_MODULES.items()):
    try:
        start_time = time.time()
        scores = func(grid)
        tensor[i] = scores
        elapsed = time.time() - start_time
        logger.debug(f"模組 {name} 耗時: {elapsed:.3f}秒")
    except Exception as e:
        logger.error(f"模組 {name} 執行失敗: {e}")
        tensor[i] = 0.0

# 快取結果
_score_cache[grid_hash] = tensor

# 限制快取大小
if len(_score_cache) > 1000:
    # 移除最舊的一半
    keys = list(_score_cache.keys())
    for k in keys[:500]:
        del _score_cache[k]

return tensor
```

def normalize_tensor(tensor: np.ndarray, method: str = "minmax") -> np.ndarray:
"""向量化張量正規化"""
num_modules = tensor.shape[0]

```
if method == "minmax":
    # 向量化 min-max 正規化
    mins = tensor.reshape(num_modules, -1).min(axis=1, keepdims=True)
    maxs = tensor.reshape(num_modules, -1).max(axis=1, keepdims=True)
    
    # 避免除以零
    ranges = maxs - mins
    ranges[ranges < 1e-8] = 1.0
    
    normalized = (tensor.reshape(num_modules, -1) - mins) / ranges
    return normalized.reshape(tensor.shape)

elif method == "zscore":
    # 向量化 z-score 正規化
    means = tensor.reshape(num_modules, -1).mean(axis=1, keepdims=True)
    stds = tensor.reshape(num_modules, -1).std(axis=1, keepdims=True)
    
    # 避免除以零
    stds[stds < 1e-8] = 1.0
    
    normalized = (tensor.reshape(num_modules, -1) - means) / stds
    return normalized.reshape(tensor.shape)

else:
    raise ValueError(f"未知的正規化方法: {method}")
```

def fuse_scores(normed: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
"""向量化分數融合"""
if weights is None:
# 等權平均
return np.mean(normed, axis=0)
else:
# 加權平均（向量化）
weights = weights.reshape(-1, 1, 1)
return np.sum(normed * weights, axis=0)

def get_topk_positions(fused: np.ndarray, grid: np.ndarray, k: int = 3) -> List[Tuple[int, int, float]]:
"""獲取前k個最高分位置（優化版）"""
# 創建空格遮罩
blank_mask = (grid == -1)

```
# 將非空格位置設為負無窮
masked_scores = np.where(blank_mask, fused, -np.inf)

# 使用 argpartition 找出前k大（更快）
flat_scores = masked_scores.flatten()
num_blanks = np.sum(blank_mask)

if num_blanks == 0:
    return []

# 找出前k個最大值的索引
k = min(k, num_blanks)
top_k_indices = np.argpartition(flat_scores, -k)[-k:]
top_k_indices = top_k_indices[np.argsort(flat_scores[top_k_indices])[::-1]]

# 轉換回2D座標
results = []
total_score = np.sum(masked_scores[blank_mask])

for idx in top_k_indices:
    r = idx // fused.shape[1]
    c = idx % fused.shape[1]
    score = fused[r, c]
    confidence = score / total_score if total_score > 0 else 0
    results.append((r, c, confidence))

return results
```

def analyze_with_prior(grid: np.ndarray, target: int, request_id: str = "API") -> List[Tuple[int, int, float]]:
"""主分析函數，整合歷史先驗"""
logger.info(f"[{request_id}] 開始分析 target={target}, grid={grid.shape}")

```
# 1. 收集所有評分
start_time = time.time()
tensor = collect_all_scores(grid, request_id)
collect_time = time.time() - start_time

# 2. 正規化
start_time = time.time()
normed = normalize_tensor(tensor, method="minmax")
norm_time = time.time() - start_time

# 3. 獲取自適應權重
weights = get_adaptive_weights(grid, target)

# 4. 融合分數
start_time = time.time()
fused = fuse_scores(normed, weights)
fuse_time = time.time() - start_time

# 5. 加入歷史先驗（如果有）
shape = grid.shape
prior_samples = [s for s in MEMORY_SAMPLES 
                 if s['card_shape'] == shape and s['target'] == target]

if prior_samples:
    # 計算歷史位置的先驗概率
    prior_map = np.zeros_like(grid, dtype=np.float32)
    for sample in prior_samples:
        r, c = sample['true_pos']
        prior_map[r, c] += 1.0
    
    # 正規化先驗
    if np.sum(prior_map) > 0:
        prior_map /= np.sum(prior_map)
        
        # 與當前分數結合（70%當前分析，30%歷史先驗）
        fused = 0.7 * fused + 0.3 * prior_map

# 6. 獲取Top-K
start_time = time.time()
results = get_topk_positions(fused, grid, k=3)
topk_time = time.time() - start_time

logger.info(f"[{request_id}] 分析完成 - 收集:{collect_time:.3f}s, "
            f"正規化:{norm_time:.3f}s, 融合:{fuse_time:.3f}s, "
            f"Top-K:{topk_time:.3f}s")

return results
```

# 初始化時載入記憶體

_load_memory_folder()
GLOBAL_WEIGHTS, SHAPE_WEIGHTS = compute_weights_from_memory()

logger.info(f"已註冊 {len(SCORING_MODULES)} 個評分模組")
logger.info(f"模組列表: {list(SCORING_MODULES.keys())}")