# analyzer11.py
"""
analyzer11.py：負責載入記憶樣本、計算權重，並提供 collect/normalize/fuse/topk 函式。
"""

import os
import json
import logging
import time
from typing import List, Dict, Tuple, Any

import numpy as np
import new_module3  # 確保 REGISTERED_MODULES_BRAIN 在此可取

logger = logging.getLogger(__name__)

# --- 全域變數 ---
MEMORY_SAMPLES: List[Dict[str, Any]] = []            # 所有從 memory_data/*.json 載入的樣本
GLOBAL_WEIGHTS: Dict[str, float] = {}                # 全域平均權重
SHAPE_WEIGHTS: Dict[Tuple[int,int], Dict[str,float]] = {}  # 依尺寸分群的權重

# 記錄上次載入記憶檔時間戳（秒），用於檔案變動檢查
_last_memory_load_time: float = 0.0


def _load_memory_folder(folder_path: str):
    """
    掃描 folder_path 下所有 .json，將有效樣本 append 到 MEMORY_SAMPLES，
    並為每筆樣本加入 "card_shape": (rows, cols)。
    """
    global MEMORY_SAMPLES, _last_memory_load_time
    MEMORY_SAMPLES.clear()

    if not os.path.isdir(folder_path):
        logger.warning(f"記憶資料夾 {folder_path!r} 不存在，略過載入。")
        return

    max_mtime = 0.0
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".json"):
            continue
        fullpath = os.path.join(folder_path, fname)
        try:
            mtime = os.path.getmtime(fullpath)
            if mtime > max_mtime:
                max_mtime = mtime
            with open(fullpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.warning(f"{fname!r} 內容不是 list，已略過。")
                    continue
                for sample in data:
                    grid = sample.get("grid", None)
                    scores = sample.get("scores", None)
                    true_pos = sample.get("true_pos", None)
                    if grid is None or scores is None or true_pos is None:
                        logger.warning(f"{fname} 中有樣本欄位不齊，已略過該筆。")
                        continue
                    rows = len(grid)
                    cols = len(grid[0]) if rows > 0 else 0
                    if any(len(row) != cols for row in grid):
                        logger.warning(f"{fname} 中有一筆 grid 不是矩形，已略過該筆。")
                        continue
                    sample["card_shape"] = (rows, cols)
                    MEMORY_SAMPLES.append(sample)
        except Exception as ex:
            logger.error(f"載入 {fullpath!r} 發生錯誤：{ex}")

    _last_memory_load_time = max_mtime
    logger.info(f"總共載入 {len(MEMORY_SAMPLES)} 筆記憶樣本。")


def compute_global_weights_from_memory() -> Dict[str, float]:
    """
    計算全域權重：對所有 MEMORY_SAMPLES，
    將每支模組在 true_pos 上的 raw score 平均後 normalize。
    """
    if not MEMORY_SAMPLES:
        logger.warning("記憶樣本為空，無法計算 GLOBAL_WEIGHTS。")
        return {}

    sum_scores: Dict[str, float] = {}
    count_scores: Dict[str, int] = {}

    for sample in MEMORY_SAMPLES:
        raw_scores: Dict[str, float] = sample["scores"]
        for mod_name, sc in raw_scores.items():
            sum_scores[mod_name] = sum_scores.get(mod_name, 0.0) + float(sc)
            count_scores[mod_name] = count_scores.get(mod_name, 0) + 1

    avg_scores: Dict[str, float] = {}
    for mod_name, total in sum_scores.items():
        cnt = count_scores.get(mod_name, 1)
        avg_scores[mod_name] = total / cnt

    total_sum = sum(avg_scores.values())
    if total_sum < 1e-8:
        return {}
    for mod_name in avg_scores:
        avg_scores[mod_name] /= total_sum

    return avg_scores


def compute_shape_weights_from_memory() -> Dict[Tuple[int,int], Dict[str, float]]:
    """
    將 MEMORY_SAMPLES 按照 card_shape 分群，計算各群組權重。
    回傳 { (rows,cols): {mod_name: weight} }。
    """
    if not MEMORY_SAMPLES:
        logger.warning("記憶樣本為空，無法計算 SHAPE_WEIGHTS。")
        return {}

    sum_by_shape: Dict[Tuple[int,int], Dict[str, float]] = {}
    count_by_shape: Dict[Tuple[int,int], Dict[str, int]] = {}

    for sample in MEMORY_SAMPLES:
        shape = sample["card_shape"]
        raw_scores: Dict[str, float] = sample["scores"]
        if shape not in sum_by_shape:
            sum_by_shape[shape] = {}
            count_by_shape[shape] = {}
        for mod_name, sc in raw_scores.items():
            sum_by_shape[shape][mod_name] = sum_by_shape[shape].get(mod_name, 0.0) + float(sc)
            count_by_shape[shape][mod_name] = count_by_shape[shape].get(mod_name, 0) + 1

    shape_weights: Dict[Tuple[int,int], Dict[str,float]] = {}
    for shape, sums in sum_by_shape.items():
        counts = count_by_shape[shape]
        avg_scores: Dict[str, float] = {}
        for mod_name, total in sums.items():
            cnt = counts.get(mod_name, 1)
            avg_scores[mod_name] = total / cnt
        S = sum(avg_scores.values())
        if S < 1e-8:
            shape_weights[shape] = {}
        else:
            for mod_name in avg_scores:
                avg_scores[mod_name] /= S
            shape_weights[shape] = avg_scores

    return shape_weights


def maybe_reload_memory():
    """
    每隔一段時間(由 main14.py 背景任務呼叫)檢查 memory_data/ 下是否有檔案更新。
    若發現 modification time > _last_memory_load_time，則重載樣本並更新權重。
    """
    global GLOBAL_WEIGHTS, SHAPE_WEIGHTS, _last_memory_load_time

    folder = "memory_data"
    if not os.path.isdir(folder):
        return

    latest_mtime = _last_memory_load_time
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".json"):
            continue
        fullpath = os.path.join(folder, fname)
        try:
            mtime = os.path.getmtime(fullpath)
            if mtime > latest_mtime:
                latest_mtime = mtime
        except:
            continue

    if latest_mtime > _last_memory_load_time:
        logger.info("偵測到 memory_data 資料夾有更新，重新載入樣本並更新權重。")
        _load_memory_folder(folder)
        GLOBAL_WEIGHTS = compute_global_weights_from_memory()
        SHAPE_WEIGHTS = compute_shape_weights_from_memory()


def get_weights_for_shape(shape: Tuple[int,int]) -> Dict[str, float]:
    """
    根據傳入的 (rows,cols)，
    若 SHAPE_WEIGHTS 中有對應且該群組樣本足夠，回該組權重；
    否則回 GLOBAL_WEIGHTS；若兩者皆空則回空 dict。
    """
    wdict = SHAPE_WEIGHTS.get(shape, {})
    # 若該 shape 欄位沒有樣本或權重為空，則 fallback 到全域權重
    if not wdict:
        return GLOBAL_WEIGHTS
    # 可加額外判斷：若該 shape 樣本數過少，也 fallback
    sample_count = sum(1 for s in MEMORY_SAMPLES if s["card_shape"] == shape)
    if sample_count < 10:  # threshold 可自行調整
        return GLOBAL_WEIGHTS
    return wdict


# ===== 以下為原先負責向量化、正規化、融合、Top‐K 的函式範例，需依實際邏輯填充 =====

def collect_all_scores(grid: np.ndarray, request_id: str = "API") -> np.ndarray:
    """
    自動從 new_module3.REGISTERED_MODULES_BRAIN 取出所有 mod_name, func，
    依序呼叫 func(grid, request_id)，
    並將 (num_mod, rows, cols) 的分數 tensor 回傳。
    """
    modules = new_module3.REGISTERED_MODULES_BRAIN
    rows, cols = grid.shape
    num_mod = len(modules)
    tensor = np.zeros((num_mod, rows, cols), dtype=np.float32)

    for i, (name, func) in enumerate(modules.items()):
        try:
            scores = func(grid, request_id=request_id)
            tensor[i, :, :] = scores.astype(np.float32)
        except Exception as ex:
            logger.error(f"Module {name} 計算失敗: {ex}")
            # 若失敗則填 0
            tensor[i, :, :] = 0.0

    return tensor


def normalize_tensor(tensor: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    method: "minmax" or "zscore"
    回傳同 shape 的 normalized tensor (float32)。
    """
    tensor = tensor.astype(np.float32)
    num_mod, rows, cols = tensor.shape
    normed = np.zeros_like(tensor, dtype=np.float32)

    if method == "minmax":
        for i in range(num_mod):
            arr = tensor[i]
            minv = np.nanmin(arr)
            maxv = np.nanmax(arr)
            if np.isclose(maxv, minv):
                normed[i] = 0.0
            else:
                normed[i] = (arr - minv) / (maxv - minv)
    elif method == "zscore":
        for i in range(num_mod):
            arr = tensor[i]
            mean = np.nanmean(arr)
            std = np.nanstd(arr)
            if np.isclose(std, 0.0):
                normed[i] = 0.0
            else:
                normed[i] = (arr - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normed


def fuse_scores(tensor: np.ndarray, weights: List[float] = None) -> np.ndarray:
    """
    若 weights 為 None，則做等權平均；否則 weights shape=(num_mod,)，
    回傳 shape=(rows,cols) 的加權融合結果 (float32)。
    """
    if weights is None:
        return np.nanmean(tensor, axis=0).astype(np.float32)
    w = np.array(weights, dtype=np.float32)
    # tensordot: (num_mod,) 與 (num_mod,rows,cols) → (rows,cols)
    fused = np.tensordot(w, tensor, axes=([0], [0]))
    return fused.astype(np.float32)


def get_topk_positions(fused: np.ndarray, grid: np.ndarray, k: int = 3) -> List[Tuple[Tuple[int,int], float]]:
    """
    只從 grid == -1 的空格挑 top-k 分數位置。
    回傳 List of ((row, col), score_normalized)，其中 row, col 皆 0-based。
    """
    mask = (grid == -1)
    if not np.any(mask):
        return []

    # 取哪 k 個最大值
    flat_scores = fused.copy()
    flat_scores[~mask] = -np.inf
    flat_idx = np.argsort(flat_scores.flatten())[::-1]
    results: List[Tuple[Tuple[int,int], float]] = []
    added = 0
    for idx in flat_idx:
        if added >= k:
            break
        r = idx // fused.shape[1]
        c = idx % fused.shape[1]
        if mask[r, c]:
            results.append(((r, c), float(fused[r, c])))
            added += 1
    return results


# --- 初次模組載入時，先 load memory_data 並計算權重 ---
_load_memory_folder("memory_data")
GLOBAL_WEIGHTS = compute_global_weights_from_memory()
SHAPE_WEIGHTS = compute_shape_weights_from_memory()