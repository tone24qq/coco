# main.py
"""
入口程式：
    1. 解析網格字串或檔案
    2. 依環境變數 / CLI 參數決定迭代次數
    3. 呼叫 app.analyzer.predict_scratch_card 取得 Top-3 預測
    4. 以人類可讀或純 JSON 方式輸出
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np  # noqa: F401  # 僅確保依賴存在

# 確保工作目錄在 sys.path（因 Render / 手機環境常缺這一步）
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from app.analyzer import predict_scratch_card  # noqa: E402

# ────────────────────────────────────────────────────────────
# 環境變數
# ────────────────────────────────────────────────────────────
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    return default if v is None else v.lower() in {"1", "true", "yes", "y", "on"}


USE_FORMULA_ONLY: bool = _env_bool("USE_FORMULA_ONLY", False)
DEFAULT_ITER: int = int(os.getenv("ITER", "500000"))

# ────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s:%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


# ────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────
def parse_grid(grid_str: str) -> List[List[int]]:
    """將 '1,2,-1;3,-1,5' 轉為 2D int list。"""
    try:
        rows = [
            [int(cell.strip()) for cell in row.split(",") if cell.strip()]
            for row in grid_str.strip().split(";")
        ]
    except ValueError as exc:
        raise ValueError(f"Grid 內含非整數：{exc}") from None

    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError("Grid 不是矩形：各列長度不一致")
    return rows


def read_grid_from_file(path: str) -> List[List[int]]:
    """支援 .json（list of list）或 .txt（與 CLI 相同語法）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, list):
            raise ValueError("JSON 格式須為 list[list[int]]")
        return [[int(c) for c in row] for row in data]

    with open(path, "r", encoding="utf-8") as fp:
        return parse_grid(fp.read())


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scratch-card hidden-number predictor")
    p.add_argument(
        "--grid",
        required=True,
        help="網格字串 '1,2,-1;3,4,-1' 或 .json/.txt 檔案路徑",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Monte-Carlo 迭代次數（覆寫 $ITER）",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="只輸出 JSON（不印美化結果）",
    )
    return p


# ────────────────────────────────────────────────────────────
# Main routine
# ────────────────────────────────────────────────────────────
def main() -> Dict[str, Any]:
    args = build_arg_parser().parse_args()

    # 解析 Grid
    if os.path.isfile(args.grid):
        grid = read_grid_from_file(args.grid)
        logger.info("載入網格檔 %s → shape %dx%d", args.grid, len(grid), len(grid[0]))
    else:
        grid = parse_grid(args.grid)
        logger.info("解析網格字串 → shape %dx%d", len(grid), len(grid[0]))

    if not any(-1 in row for row in grid):
        raise ValueError("Grid 內沒有 -1（空格），無需預測。")

    n_iter = args.iterations if args.iterations else DEFAULT_ITER
    if n_iter <= 0:
        raise ValueError("iterations 必須為正整數")

    logger.info("n_iter=%s | USE_FORMULA_ONLY=%s", f"{n_iter:,}", USE_FORMULA_ONLY)

    try:
        result = predict_scratch_card(grid, n_iter=n_iter)
    except Exception:
        logger.exception("預測失敗")
        sys.exit(1)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        logger.info("── Top-3 預測 ──")
        for pred in result["predictions"][:3]:
            row, col = pred["row"], pred["col"]
            nums = ", ".join(map(str, pred["candidates"]))
            confs = ", ".join(f"{c:.3f}" for c in pred["confidences"])
            logger.info("(%d, %d) → %s   conf=%s", row, col, nums, confs)
        logger.info("完整機率分佈可加 --json 參數輸出")

    return result


if __name__ == "__main__":
    main()