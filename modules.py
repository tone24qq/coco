"""
modules.py
==========

* 集中放置所有可重用的數學工具、特徵計算與公式評分模組。
* FORMULA_REGISTRY 透過 `@register_formula` 自動收錄，供 analyzer.py
  以向量化方式加權評分。

⚠️ 注意：此檔會被多處 import，請勿在最上層做重計算以免拖慢啟動。
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import statistics
from functools import lru_cache
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

# 嘗試載入 scipy；若環境未安裝則以 NumPy 替代部份簡易運算
try:
    from scipy.signal import convolve2d  # type: ignore
except ImportError:  # pragma: no cover
    def convolve2d(a: np.ndarray, b: np.ndarray, mode: str = "valid", **_) -> np.ndarray:  # noqa: D401
        """簡易替代：僅支援 mode='valid' 且無 padding。"""
        if mode != "valid":
            raise NotImplementedError("convolve2d fallback 只支援 mode='valid'")
        kh, kw = b.shape
        h, w = a.shape
        out_h, out_w = h - kh + 1, w - kw + 1
        res = np.empty((out_h, out_w), dtype=float)
        for i in range(out_h):
            for j in range(out_w):
                res[i, j] = (a[i : i + kh, j : j + kw] * b).sum()
        return res


# ────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s:%(name)s] %(message)s",
    handlers=[logging.FileHandler("logs/modules.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# FORMULA_REGISTRY 相關
# ────────────────────────────────────────────────────────────
FORMULA_REGISTRY: Dict[str, Callable[[np.ndarray], float]] = {}


def register_formula(name: str) -> Callable[[Callable[[np.ndarray], float]], Callable[[np.ndarray], float]]:
    """裝飾器：註冊評分公式到全域 FORMULA_REGISTRY。"""

    def _decorator(func: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
        if name in FORMULA_REGISTRY:
            logger.warning("公式 %s 已存在，將被覆寫。", name)
        FORMULA_REGISTRY[name] = func
        return func

    return _decorator


# ────────────────────────────────────────────────────────────
# 基礎工具
# ────────────────────────────────────────────────────────────
class MathUtils:
    """常用數學與距離計算輔助函式。"""

    @staticmethod
    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @staticmethod
    def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def variance(arr: Sequence[float]) -> float:
        return statistics.pvariance(arr) if arr else 0.0

    @staticmethod
    def normalize(arr: Sequence[float]) -> List[float]:
        total = sum(arr) or 1e-9
        return [x / total for x in arr]


class BoardAnalyzerUtils:
    """
    與盤面（NumPy 陣列）相關的靜態工具函式。
    """

    @staticmethod
    def get_neighbors(board: np.ndarray, r: int, c: int) -> List[int]:
        h, w = board.shape
        nbrs: List[int] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    nbrs.append(int(board[nr, nc]))
        return nbrs

    @staticmethod
    def check_sequences(
        board: np.ndarray,
        ref_board: np.ndarray,
        *,
        min_len: int = 3,
        allow_gaps: int = 1,
    ) -> bool:
        """
        與 analyzer.check_sequences 相同邏輯，提供模組端呼叫。
        """
        from analyzer import check_sequences as _chk

        return _chk(board, ref_board, min_len=min_len, allow_gaps=allow_gaps)


# ────────────────────────────────────────────────────────────
# 全域特徵計算
# ────────────────────────────────────────────────────────────
@lru_cache(maxsize=4096)
def compute_global_features(board_hash: str, board: np.ndarray) -> Dict[str, float]:
    """
    快取全盤特徵；board_hash 建議用 board.tobytes()[:64] 或自訂 key。

    特徵包含：
        mean, std, missing_ratio, seq_score 等
    """
    h, w = board.shape
    values = board[board != -1]
    missing_ratio = (h * w - values.size) / (h * w)

    # 序列連續性：使用 3x1 / 1x3 kernel 判斷連續數字權重
    # 為避免過度計算，僅在 board 內部做簡易卷積評分
    horiz_kernel = np.array([[1, 1, 1]])
    vert_kernel = horiz_kernel.T

    def _seq_score(kernel: np.ndarray) -> float:
        if min(board.shape) < kernel.shape[0]:
            return 0.0
        conv = convolve2d(board == np.roll(board, -1, axis=1), kernel, mode="valid")
        return float(conv.sum())

    seq_score = _seq_score(horiz_kernel) + _seq_score(vert_kernel)

    features = {
        "mean": float(values.mean()) if values.size else 0.0,
        "std": float(values.std()) if values.size else 0.0,
        "missing_ratio": missing_ratio,
        "seq_score": seq_score,
    }
    return features


# ────────────────────────────────────────────────────────────
# AdaptiveWeights —— 讓多公式可學習性加權
# ────────────────────────────────────────────────────────────
class AdaptiveWeights:
    """
    透過增量學習自動微調公式權重；本版提供簡化乘積更新規則。
    """

    def __init__(self, names: Sequence[str] | None = None) -> None:
        self.weights: Dict[str, float] = {n: 1.0 for n in names or []}
        self.lr = 0.05

    def update(self, scores: Dict[str, float], correct: bool) -> None:
        """
        若預測正確，強化高分公式；否則弱化高分公式。
        """
        for name, sc in scores.items():
            delta = self.lr * sc if correct else -self.lr * sc
            self.weights[name] = max(0.01, self.weights.get(name, 1.0) + delta)

    def weighted_sum(self, scores: Dict[str, float]) -> float:
        return sum(scores[n] * self.weights.get(n, 1.0) for n in scores)


# ────────────────────────────────────────────────────────────
# 實際「評分公式」實作
# ────────────────────────────────────────────────────────────
def _feature(board: np.ndarray, key: str) -> float:
    """
    共用：根據 feature key 取 compute_global_features 結果。
    """
    bhash = hash(board.tobytes())
    feats = compute_global_features(str(bhash), board)
    return feats[key]


# ----- A 系列 ─────────────────────────── #
@register_formula("A2")
def formula_a2(board: np.ndarray) -> float:
    """偏好低標準差（數字分布均勻）。"""
    std = _feature(board, "std")
    return 1.0 / (1.0 + std)


@register_formula("A6")
def formula_a6(board: np.ndarray) -> float:
    """偏好高序列分數。"""
    return _feature(board, "seq_score") / 10.0


# ----- M 系列 ─────────────────────────── #
@register_formula("M1")
def formula_m1(board: np.ndarray) -> float:
    """缺值比例越低越好。"""
    return 1.0 - _feature(board, "missing_ratio")


@register_formula("M3")
def formula_m3(board: np.ndarray) -> float:
    """缺值 + 標準差聯合懲罰。"""
    miss = _feature(board, "missing_ratio")
    std = _feature(board, "std")
    return math.exp(-(miss + std))


@register_formula("M4")
def formula_m4(board: np.ndarray) -> float:
    return random.random() * 0.01 + formula_m1(board)


@register_formula("M5")
def formula_m5(board: np.ndarray) -> float:
    return math.sqrt(_feature(board, "seq_score") + 1.0)


@register_formula("M6")
def formula_m6(board: np.ndarray) -> float:
    return 1.0 / (1.0 + _feature(board, "mean"))


@register_formula("M7")
def formula_m7(board: np.ndarray) -> float:
    return formula_m6(board) * formula_m1(board)


@register_formula("M9")
def formula_m9(board: np.ndarray) -> float:
    return math.log1p(_feature(board, "seq_score"))


@register_formula("M10")
def formula_m10(board: np.ndarray) -> float:
    return formula_m3(board) * 1.2


@register_formula("M11")
def formula_m11(board: np.ndarray) -> float:
    return 0.5 * formula_m1(board) + 0.5 * formula_a2(board)


# ----- F 系列 ─────────────────────────── #
@register_formula("F2")
def formula_f2(board: np.ndarray) -> float:
    return float(np.count_nonzero(board == -1))


@register_formula("F3")
def formula_f3(board: np.ndarray) -> float:
    return -formula_f2(board)  # 少缺值更好


@register_formula("F7")
def formula_f7(board: np.ndarray) -> float:
    return formula_a6(board) * 2.0


@register_formula("F8")
def formula_f8(board: np.ndarray) -> float:
    return random.uniform(0, 0.5)


@register_formula("F10")  # 公平排序一致性（簡化版）
def formula_f10(board: np.ndarray) -> float:
    """衡量行列平均差異是否一致。"""
    row_means = board.mean(axis=1)
    col_means = board.mean(axis=0)
    return -abs(row_means.std() - col_means.std())


# ----- R 系列 ─────────────────────────── #
@register_formula("R2")
def formula_r2(board: np.ndarray) -> float:
    return formula_a2(board) * 1.1


@register_formula("R7")
def formula_r7(board: np.ndarray) -> float:
    return formula_m5(board) * 0.8


@register_formula("R5")
def formula_r5(board: np.ndarray) -> float:
    return formula_r2(board) + formula_r7(board)


# ----- P 系列 ─────────────────────────── #
@register_formula("P1")
def formula_p1(board: np.ndarray) -> float:
    return -_feature(board, "std")


@register_formula("P2")
def formula_p2(board: np.ndarray) -> float:
    return math.log1p(_feature(board, "seq_score"))


@register_formula("P4")
def formula_p4(board: np.ndarray) -> float:
    return formula_p1(board) * 0.5 + formula_p2(board) * 0.5


@register_formula("P7")
def formula_p7(board: np.ndarray) -> float:
    return random.random()


# ----- L 系列 ─────────────────────────── #
@register_formula("L1")
def formula_l1(board: np.ndarray) -> float:
    return float(np.max(board[board != -1])) / (board.size + 1)


@register_formula("L3")
def formula_l3(board: np.ndarray) -> float:
    return formula_l1(board) / (_feature(board, "mean") + 1e-3)


# ----- D 系列 ─────────────────────────── #
@register_formula("D3")
def formula_d3(board: np.ndarray) -> float:
    """低缺值 + 低標準差偏好。"""
    return formula_m1(board) + formula_a2(board)


# ----- GM 系列 (GM1 ~ GM26) ─────────────── #
def _auto_gm_formula(idx: int) -> Callable[[np.ndarray], float]:
    """動態生成 GM 系列公式：各自使用不同權重組合。"""

    @register_formula(f"GM{idx}")
    def _gm(board: np.ndarray) -> float:  # noqa: D401
        w1, w2, w3 = (idx % 5 + 1) / 10, (idx % 7 + 1) / 12, (idx % 3 + 1) / 8
        return (
            w1 * formula_a2(board)
            + w2 * formula_m3(board)
            + w3 * formula_f10(board)
            - (idx % 4) * 0.01
        )

    return _gm


for _idx in range(1, 27):  # GM1~GM26
    _auto_gm_formula(_idx)

# ────────────────────────────────────────────────────────────
# DEBUG: 顯示已註冊公式數量
# ────────────────────────────────────────────────────────────
logger.info("已註冊公式 %d 個：%s", len(FORMULA_REGISTRY), list(FORMULA_REGISTRY.keys()))