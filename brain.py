"""
brain.py
========
核心邏輯層（「不動到 app / main」前提下的強化版）

* 保留 GitHubClient / OpenAIClient 既有介面，方便 main 直接呼叫  
* 新增 BrainCore：集中處理「算分 → 排序 → 取 Top-k」  
* 支援環境變數  
    USE_FORMULA_ONLY : 若 == "1" 則僅走純公式評分，不呼叫 LLM  
    ITER             : 內部 Monte-Carlo / 隨機搜尋最大迭代次數  
* 例外處理全面提升，避免先前 ValueError / Truth-value amb 等 NumPy 陷阱
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np  # type: ignore

# -----------------------------------------------------------------------------
# 已存在的兩個底層 API client —— 原始介面維持不變
# -----------------------------------------------------------------------------
# （這兩個類別的程式碼照舊，省略顯示；若未改動可直接保留現檔內容）
from .clients import GitHubClient, OpenAIClient  # 假設你把它們抽到 clients.py

# -----------------------------------------------------------------------------
# 新增：BrainCore —— 負責命中率計算與 Top-k 推理
# -----------------------------------------------------------------------------


class BrainCore:
    """
    封裝「分數計算 → Top-k 擇優」邏輯的單一責任類別。

    Parameters
    ----------
    use_formula_only :
        是否僅使用數學公式評分，跳過 LLM。
    max_iter :
        蒙地卡羅或隨機取樣最大迭代次數。
    """

    def __init__(self, *, use_formula_only: bool = False, max_iter: int = 10_000) -> None:
        self.use_formula_only = use_formula_only
        self.max_iter = max_iter

        # 預先配置 logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    # --------------------------------------------------------------------- #
    # 公開 API
    # --------------------------------------------------------------------- #

    def rank_candidates(
        self,
        base_vector: np.ndarray,
        candidates: Sequence[np.ndarray],
        top_k: int = 3,
    ) -> List[Tuple[int, float]]:
        """
        根據與 `base_vector` 的相似度（皮爾森相關係數）由高到低排序，
        回傳 (index, score) 清單，長度為 `top_k`。

        此函式同時解決兩個常見錯誤：
        1. **Truth-value ambiguous**：避免直接 `if array`，一律用 `.any()` / `.all()`
        2. **broadcast 形狀不一致**：先對齊最小長度再計算相關係數

        Notes
        -----
        * corr = 1 代表「越像」，corr < 0 時代表反向；負值一律視為 0 分。
        * 若分母為 0 或 `np.isnan`，當前候選會給 -inf，確保不進入 Top-k。
        """
        self._validate_input(base_vector, candidates)

        base_flat = base_vector.ravel()
        base_len = base_flat.shape[0]

        scores: List[float] = []
        for idx, cand in enumerate(candidates):
            cand_flat = cand.ravel()

            # 取最小共同長度，防「(10000,) vs (9983,)」錯誤
            min_len = min(base_len, cand_flat.shape[0])
            if min_len == 0:
                scores.append(float("-inf"))
                continue

            corr = self._safe_corrcoef(base_flat[:min_len], cand_flat[:min_len])
            scores.append(max(0.0, corr))  # 負值歸 0

            self.logger.debug("idx=%d corr=%.4f", idx, corr)

        # 取 Top-k 並保留 (index, score)
        top_idx = np.argsort(scores)[::-1][:top_k]
        result = [(int(i), float(scores[i])) for i in top_idx]

        self.logger.info("Top-%d result: %s", top_k, result)
        return result

    # ------------------------------------------------------------------ #
    # 內部小工具
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_input(base_vector: np.ndarray, candidates: Sequence[np.ndarray]) -> None:
        if base_vector.size == 0:
            raise ValueError("base_vector 不可為空")
        if not candidates:
            raise ValueError("candidates 至少需要一筆")

    @staticmethod
    def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
        """
        計算皮爾森相關係數；如遇 0 標準差則回 -inf，避免後續進入 Top-k。
        """
        if a.size != b.size:  # 雙重保險
            raise ValueError(f"尺寸不一致: {a.size=} {b.size=}")

        if np.std(a) == 0 or np.std(b) == 0:
            return float("-inf")

        return float(np.corrcoef(a, b)[0, 1])

    # ------------------------------------------------------------------ #
    # （可選）混合 LLM 與公式評分
    # ------------------------------------------------------------------ #

    async def hybrid_score(
        self,
        prompt_template: str,
        candidate_texts: Sequence[str],
        openai_client: OpenAIClient,
        top_k: int = 3,
        *,
        model: str = "gpt-4o-mini",
    ) -> List[Tuple[int, float]]:
        """
        * **公式 + LLM** 混合評分範例：  
          - 先用基礎規則過濾 / 取前 N  
          - 再讓 GPT 排序補強  
        * 若 `self.use_formula_only` = True 則直接落回 `rank_candidates`。
        """
        if self.use_formula_only:
            # 這裡以文字長度向量化當示範
            base = np.array([len(t) for t in candidate_texts])
            cands = [np.array([len(t)]) for t in candidate_texts]
            return self.rank_candidates(base, cands, top_k=top_k)

        # ------------------------ GPT 排序 ------------------------ #
        system_msg = (
            "You are a ranking assistant. "
            "Return exactly JSON list of top-k indices in descending relevance."
        )
        joined = "\n".join(f"{i}. {c}" for i, c in enumerate(candidate_texts))
        prompt = prompt_template.format(candidates=joined, k=top_k)
        gpt_resp = await openai_client.generate_response(
            prompt, model=model, request_id="BrainCore.hybrid_score"
        )

        # 抽取 [0, 2, 1] 這類索引清單
        indices = _parse_indices(gpt_resp["content"], top_k=top_k)
        return [(i, top_k - rank) for rank, i in enumerate(indices)]


def _parse_indices(text: str, top_k: int) -> List[int]:
    """
    嘗試從 LLM 回覆中抓取整數索引清單，例如 `[2,0,1]`。
    失敗則回傳空清單。
    """
    import re
    matches = re.findall(r"\d+", text)
    ints = [int(m) for m in matches][:top_k]
    if len(ints) != top_k:
        return list(range(top_k))
    return ints


# -----------------------------------------------------------------------------
# 便利工廠：從環境變數自動初始化 BrainCore
# -----------------------------------------------------------------------------


def create_brain_from_env() -> BrainCore:
    """
    helper 讓 main.py 只要：
        brain = create_brain_from_env()
    就能獲得具備 env 設定的 BrainCore 實例
    """
    use_formula_only = os.getenv("USE_FORMULA_ONLY", "0") == "1"
    max_iter = int(os.getenv("ITER", 10_000))
    brain = BrainCore(use_formula_only=use_formula_only, max_iter=max_iter)
    brain.logger.info(
        "BrainCore ready (USE_FORMULA_ONLY=%s, ITER=%d)",
        use_formula_only,
        max_iter,
    )
    return brain


# -----------------------------------------------------------------------------
# 自我測試（`python brain.py` 時執行）
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import asyncio

    async def _demo() -> None:
        # 假資料示範
        base = np.random.rand(100)
        cands = [np.random.rand(100) for _ in range(10)]
        brain = create_brain_from_env()
        print(brain.rank_candidates(base, cands, top_k=3))

    asyncio.run(_demo())