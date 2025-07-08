import json
import logging
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SAMPLES_DIR = Path("samples")
OUTPUT_DIR = Path("priors")


def iter_all_json_from_zip(zip_path: Path):
    """
    逐筆從 zip 中讀取 JSON，支援三種格式：
      1. {"grid": [...], "rows": X, "cols": Y}
      2. 裸 list：單一盤面或多盤面 [[...], [...], ...]
      3. {"8x10": [盤面, ...], "4x5": [盤面, ...], ...}
    遇到尺寸不一致或讀取失敗都會輸出 warning。
    """
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".json"):
                continue

            try:
                raw = zf.read(name)
                data = json.loads(raw)
            except Exception as exc:
                logger.warning(f"❌ 讀取失敗：{name} in {zip_path.name} - {exc}")
                continue

            # ---- 格式 1：grid + rows/cols ----
            if (
                isinstance(data, dict)
                and "grid" in data
                and isinstance(data["grid"], list)
            ):
                grid = data["grid"]
                rows = int(data.get("rows", len(grid)))
                cols = int(data.get("cols", len(grid[0]) if grid else 0))
                if (
                    rows > 0
                    and cols > 0
                    and all(isinstance(r, list) and len(r) == cols for r in grid)
                ):
                    yield rows, cols, grid
                    count += 1
                else:
                    logger.warning(
                        f"❌ 跳過 format1：{name} rows={rows}, cols={cols} 與 grid 不符"
                    )
                continue

            # ---- 格式 2：裸 list of boards 或單一 board ----
            if isinstance(data, list) and data and isinstance(data[0], list):
                # 判斷「多盤面」 vs 「單一盤面」
                # 若第一層元素裡還是 list of lists，當作多盤面
                if all(isinstance(row, list) for row in data[0]):
                    # 多盤面
                    for i, board in enumerate(data):
                        rows0 = len(board)
                        cols0 = len(board[0]) if rows0 else 0
                        if (
                            rows0
                            and cols0
                            and all(
                                isinstance(r, list) and len(r) == cols0 for r in board
                            )
                        ):
                            yield rows0, cols0, board
                            count += 1
                        else:
                            logger.warning(
                                f"❌ 跳過 format2 board#{i}：行列 {rows0}x{cols0} 不一致 ({name})"
                            )
                else:
                    # 單一盤面
                    rows0 = len(data)
                    cols0 = len(data[0])
                    if (
                        rows0
                        and cols0
                        and all(isinstance(r, list) and len(r) == cols0 for r in data)
                    ):
                        yield rows0, cols0, data
                        count += 1
                    else:
                        logger.warning(
                            f"❌ 跳過單一盤面：{name} 行列 {rows0}x{cols0} 不一致"
                        )
                continue

            # ---- 格式 3：{"8x10": [盤面, ...], ...} ----
            if isinstance(data, dict):
                for key, boards in data.items():
                    if not isinstance(boards, list):
                        continue
                    key_norm = key.lower().replace(" ", "")
                    m = re.match(r"^(\d+)x(\d+)$", key_norm)
                    if not m:
                        continue
                    rows0, cols0 = map(int, m.groups())
                    for i, board in enumerate(boards):
                        if (
                            isinstance(board, list)
                            and board
                            and all(
                                isinstance(r, list) and len(r) == cols0 for r in board
                            )
                        ):
                            yield rows0, cols0, board
                            count += 1
                        else:
                            logger.warning(
                                f"❌ 跳過 format3 board#{i} for key '{key}': size mismatch"
                            )
                continue

    logger.info(f"✅ {zip_path.name} 讀取 {count} 筆資料")


def build_and_save_all_pos_freq(samples_dir: Path, output_dir: Path):
    shape_counts = defaultdict(lambda: None)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 掃描並累加
    for zip_file in samples_dir.glob("*.zip"):
        logger.info(f"📦 掃描 {zip_file.name}")
        for rows, cols, grid in iter_all_json_from_zip(zip_file):
            grid_np = np.asarray(grid, dtype=int)
            shape = (rows, cols)
            if shape_counts[shape] is None:
                shape_counts[shape] = np.zeros((rows, cols), dtype=int)
            shape_counts[shape] += (grid_np != -1).astype(int)

    if not shape_counts:
        logger.warning("⚠️ 沒有有效樣本可供輸出")
        return

    # 列出所有實際讀到的尺寸，方便檢查
    all_shapes = sorted(shape_counts.keys())
    logger.info(f"🔍 一共找到這些盤面尺寸：{all_shapes}")

    # 計算頻率並輸出
    for (rows, cols), count_mat in shape_counts.items():
        total = int(count_mat.sum()) or 1
        freq = count_mat.astype(float) / float(total)
        out_path = output_dir / f"pos_freq_{rows}x{cols}.npz"
        np.savez_compressed(out_path, freq=freq)
        logger.info(f"✅ 儲存 {out_path.name} ({rows}x{cols})")


if __name__ == "__main__":
    build_and_save_all_pos_freq(SAMPLES_DIR, OUTPUT_DIR)
