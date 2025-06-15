# main.py
"""
Scratch-Card Hidden-Number Predictor ── CLI 入口

1. 解析網格（字串或檔案）
2. 依環境變數 / CLI 決定迭代次數與引擎
3. 呼叫 analyzer.predict_scratch_card() 取得 Top-3
4. 輸出人類可讀或純 JSON

環境變數
--------
ITER              : Monte-Carlo 迭代次數（預設 500_000）
USE_FORMULA_ONLY  : "1"→僅跑公式，不用 BrainCore
USE_LEGACY        : "1"→舊版 2 公式極速引擎
LOG_LEVEL         : 預設 INFO
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# 確保專案根目錄在 sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyzer import predict_scratch_card

# ────────────────────────────────────────────────────────────
# 環境參數
# ────────────────────────────────────────────────────────────
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    return default if v is None else v.lower() in {"1", "true", "yes", "y", "on"}


USE_FORMULA_ONLY: bool = _env_bool("USE_FORMULA_ONLY", False)
USE_LEGACY: bool = _env_bool("USE_LEGACY", False)
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
def _parse_grid_str(grid_str: str) -> List[List[int]]:
    """
    將 '1,2,-1;4,-1,6' 轉二維 int list
    """
    try:
        rows = [
            [int(cell.strip()) for cell in row.split(",") if cell.strip()]
            for row in grid_str.strip().split(";")
        ]
    except ValueError as exc:
        raise ValueError(f"Grid 內含非整數：{exc}") from None

    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError("Grid 不是矩形，各列長度不一致")
    return rows


def _read_grid(path: Path) -> List[List[int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON 格式需為 list[list[int]]")
        return [[int(c) for c in row] for row in data]
    return _parse_grid_str(path.read_text(encoding="utf-8"))


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scratch-card hidden-number predictor")
    p.add_argument(
        "--grid",
        required=True,
        help="網格字串 '1,2,-1;...' 或 .json/.txt 檔路徑",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Monte-Carlo 迭代數（覆寫 $ITER）",
    )
    p.add_argument("--json", action="store_true", help="只輸出 JSON")
    p.add_argument("--legacy", action="store_true", help="強制使用舊版 2 公式引擎")
    return p


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main() -> Dict[str, Any]:
    args = _build_argparser().parse_args()

    # 讀取 / 解析網格
    if Path(args.grid).is_file():
        grid = _read_grid(Path(args.grid))
        logger.info("載入網格檔 %s → shape %dx%d", args.grid, len(grid), len(grid[0]))
    else:
        grid = _parse_grid_str(args.grid)
        logger.info("解析字串 → shape %dx%d", len(grid), len(grid[0]))

    n_iter = args.iterations or DEFAULT_ITER
    engine_legacy = args.legacy or USE_LEGACY
    logger.info(
        "iterations=%s | USE_FORMULA_ONLY=%s | USE_LEGACY=%s",
        f"{n_iter:,}",
        USE_FORMULA_ONLY,
        engine_legacy,
    )

    result = predict_scratch_card(grid, n_iter=n_iter, use_legacy=engine_legacy)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        logger.info("── Top-3 推測 ──")
        for pred in result["predictions"]:
            row, col = pred["row"], pred["col"]
            nums = ", ".join(map(str, pred["candidates"]))
            confs = ", ".join(f"{c:.3f}" for c in pred["confidences"])
            logger.info("(%d, %d) → %s  conf=%s", row, col, nums, confs)
        logger.info("完整機率分佈可加 --json 輸出")
    return result


if __name__ == "__main__":
    main()