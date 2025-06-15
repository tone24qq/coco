"""
analyzer.py
===========

核心職責
--------
1. Monte-Carlo / 公式混合方式，推測 -1 位置的可能數字分佈。
2. 回傳 Top-3 推測結果與完整機率分佈。
3. 提供 check_sequences 等工具函式，供其他模組交互使用。

外部依賴
--------
* new_module.score_full_board(board) : 回傳浮點分數（越高越好）
* modules.FORMULA_REGISTRY          : Dict[str, Callable[..., float]]，自定義評分函式
* BrainCore (brain.py)              : 提供 rank_candidates 等進階比對
"""

from __future__ import annotations

import itertools
import logging
import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

# ────────────────────────────────────────────────────────────
# 動態匯入外部依賴（若不存在則降級處理）
# ────────────────────────────────────────────────────────────
try:
    from new_module import score_full_board  # type: ignore
except ImportError:  # Fallback
    def score_full_board(board: np.ndarray) -> float:  # noqa: D401
        """Fallback: 無權重時每盤分數 = 1.0"""
        return 1.0


try:
    from modules import FORMULA_REGISTRY  # type: ignore
except ImportError:
    FORMULA_REGISTRY: Dict[str, Any] = {}

try:
    # 與 brain.py 同層目錄
    from app.brain import create_brain_from_env  # type: ignore
except ImportError:
    # 若專案結構不同，改從根目錄嘗試
    try:
        from brain import create_brain_from_env  # type: ignore
    except ImportError:
        create_brain_from_env = None  # type: ignore

# ────────────────────────────────────────────────────────────
# 環境變數
# ────────────────────────────────────────────────────────────
USE_FORMULA_ONLY = os.getenv("USE_FORMULA_ONLY", "0") == "1"
DEFAULT_ITER = int(os.getenv("ITER", "500000"))
TOP_K_CAND_NUM = 3        # 每格回傳多少候選數字
GLOBAL_TOP_POS = 3        # 挑出信心最高的多少格回傳

# ────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────
logger = logging.getLogger("analyzer")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s:%(name)s] %(message)s",
    )


# ────────────────────────────────────────────────────────────
# 公開函式
# ────────────────────────────────────────────────────────────
def predict_scratch_card(
    grid: List[List[int]],
    *,
    n_iter: int | None = None,
    top_pos: int = GLOBAL_TOP_POS,
    top_num: int = TOP_K_CAND_NUM,
) -> Dict[str, Any]:
    """
    高階 API：直接輸入 2D list[int]，輸出 Top-k 推測。

    Returns
    -------
    {
        "iterations": 100000,
        "predictions": [
            {
                "row": 1, "col": 2,
                "candidates": [13, 9, 7],
                "confidences": [0.62, 0.21, 0.17]
            },
            ...
        ],
        "distribution": {
            "(0,1)": { "5": 0.33, "6": 0.67, ... },
            ...
        }
    }
    """
    _validate_grid(grid)
    n_iter = n_iter or DEFAULT_ITER
    h, w = len(grid), len(grid[0])

    # 先計算共用資源
    empty_pos = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == -1]
    if not empty_pos:
        raise ValueError("Grid 內無 -1，無需預測。")

    all_nums = set(range(1, h * w + 1))
    used_nums = {grid[r][c] for r in range(h) for c in range(w) if grid[r][c] != -1}
    remaining_nums = list(all_nums - used_nums)
    if len(remaining_nums) < len(empty_pos):
        raise ValueError("已用數字 + 空格數量 > 最大可用數字範圍，資料不合法。")

    logger.info(
        "Analyze %dx%d board | blank=%d | iter=%s | USE_FORMULA_ONLY=%s",
        h,
        w,
        len(empty_pos),
        f"{n_iter:,}",
        USE_FORMULA_ONLY,
    )

    # 蒙地卡羅模擬
    counts: Dict[Tuple[int, int], Dict[int, float]] = defaultdict(lambda: defaultdict(float))

    rng = random.Random(0xC0FFEE)  # 固定種子 ⇒ 可重現
    grid_arr = np.array(grid, dtype=np.int64)

    # 盡量向量化：一次產生 batch permutations
    batch_size = 256
    iter_left = n_iter
    while iter_left > 0:
        cur_batch = min(batch_size, iter_left)
        iter_left -= cur_batch

        # 每次隨機抽樣 cur_batch 個排列
        perms = _sample_permutations(remaining_nums, len(empty_pos), cur_batch, rng)

        for perm in perms:
            # 填入 -1 位置
            board = grid_arr.copy()
            for (r, c), v in zip(empty_pos, perm):
                board[r, c] = v

            # 計算分數
            score = _board_score(board)
            if math.isinf(score) or math.isnan(score):
                score = 0.0

            # 更新 counts
            for (r, c), v in zip(empty_pos, perm):
                counts[(r, c)][v] += score

    # 機率化
    distribution: Dict[str, Dict[int, float]] = {}
    for pos, num_dict in counts.items():
        total = sum(num_dict.values())
        if total == 0:  # guard
            total = 1e-9
        distribution[str(pos)] = {n: v / total for n, v in num_dict.items()}

    # 產生 predictions list
    preds: List[Dict[str, Any]] = []
    for (r, c), num_dict in counts.items():
        sorted_items = sorted(num_dict.items(), key=lambda kv: kv[1], reverse=True)
        top_items = sorted_items[:top_num]
        cand_nums, cand_scores = zip(*top_items)
        total = sum(num_dict.values()) or 1e-9
        probs = [s / total for s in cand_scores]
        preds.append(
            {
                "row": r,
                "col": c,
                "candidates": list(cand_nums),
                "confidences": probs,
            }
        )

    # 依第一候選之置信度排序，取前 top_pos
    preds.sort(key=lambda x: x["confidences"][0], reverse=True)
    output = {
        "iterations": n_iter,
        "predictions": preds[:top_pos],
        "distribution": distribution,
    }
    return output


def analyze_board(grid: List[List[int]], **kwargs: Any) -> Dict[str, Any]:
    """向後相容 alias。"""
    return predict_scratch_card(grid, **kwargs)


# ────────────────────────────────────────────────────────────
# check_sequences —— 提供外部模組呼叫的序列偵測工具
# ────────────────────────────────────────────────────────────
def check_sequences(
    board: np.ndarray,
    ref_board: np.ndarray,
    *,
    min_len: int = 3,
    allow_gaps: int = 1,
) -> bool:
    """
    判斷 `board` 與 `ref_board` 是否在任一 row/col/diag 有
    「相同數字且連續（可容許 gaps）」長度 >= min_len。

    Parameters
    ----------
    board :
        已完整填入的盤。
    ref_board :
        原始盤，通常含 -1。
    min_len :
        判定序列長度下限。
    allow_gaps :
        允許多少個非相等值當作「空隙」仍視為連續。
    """
    if board.shape != ref_board.shape:
        raise ValueError("board / ref_board shape 不一致")

    h, w = board.shape

    def _iter_lines() -> List[np.ndarray]:
        # rows
        for r in range(h):
            yield board[r, :]
        # cols
        for c in range(w):
            yield board[:, c]
        # 主對角線
        yield board.diagonal()
        # 副對角線
        yield np.fliplr(board).diagonal()

    for line in _iter_lines():
        streak = gap = 0
        for idx, val in enumerate(line):
            ref_val = ref_board.flatten()[idx]
            if ref_val == -1 or val == ref_val:
                streak += 1
                gap = 0
            else:
                gap += 1
                if gap > allow_gaps:
                    streak = 0
                    gap = 0
            if streak >= min_len:
                return True
    return False


# ────────────────────────────────────────────────────────────
# 內部工具
# ────────────────────────────────────────────────────────────
def _validate_grid(grid: List[List[int]]) -> None:
    if not grid or not grid[0]:
        raise ValueError("Grid 為空")
    widths = {len(r) for r in grid}
    if len(widths) != 1:
        raise ValueError("Grid 各列長度不一致 (非矩形)")


def _sample_permutations(
    pool: List[int],
    k: int,
    count: int,
    rng: random.Random,
) -> List[Tuple[int, ...]]:
    """
    從 pool 中不重複隨機抽 k 個，重複 count 次，回傳 permutations list。
    隨機性夠即可，無需涵蓋全部排列（計算複雜度太高）。
    """
    perms: List[Tuple[int, ...]] = []
    for _ in range(count):
        if k == 0:
            perms.append(tuple())
            continue
        sample = rng.sample(pool, k)
        rng.shuffle(sample)
        perms.append(tuple(sample))
    return perms


def _board_score(board: np.ndarray) -> float:
    """
    綜合分數：
        base = score_full_board()
        + 其他 FORMULA_REGISTRY
    若 `USE_FORMULA_ONLY`=1 則僅使用數學公式；否則可加權 LLM / BrainCore 之評估。
    """
    base = score_full_board(board)

    # 套用自定義公式
    formula_score = 0.0
    for name, fn in FORMULA_REGISTRY.items():
        try:
            formula_score += fn(board)
        except Exception as exc:  # noqa: BLE001
            logger.debug("公式 %s 失敗：%s", name, exc)

    total_score = base + formula_score

    if not USE_FORMULA_ONLY and create_brain_from_env is not None:
        # 以 board.ravel() 作為向量交給 BrainCore 做一次相似度評分
        brain = create_brain_from_env()
        # 假設 unified baseline 向量 = 升序 1..N
        h, w = board.shape
        baseline = np.arange(1, h * w + 1, dtype=np.float64)
        total_score_vec = brain.rank_candidates(baseline, [board.ravel()], top_k=1)[0][1]
        total_score += total_score_vec

    return total_score


# ────────────────────────────────────────────────────────────
# 模組自測
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 小型單元測試示範（不影響外部 import）
    demo_grid = [
        [1, 2, -1],
        [4, -1, 6],
        [-1, 8, 9],
    ]
    result = predict_scratch_card(demo_grid, n_iter=2000, top_pos=3, top_num=2)
    from pprint import pprint

    pprint(result["predictions"])