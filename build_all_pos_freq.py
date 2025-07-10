import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SAMPLES_DIR = Path("samples")
OUTPUT_DIR = Path("priors")


def iter_all_json_from_zip(npz_path: Path):
    """Yield boards from an ``.npz`` file."""
    with np.load(npz_path) as data:
        boards = data.get("boards")
        if boards is None:
            return
    if boards.ndim == 2:
        boards = boards[None, ...]
    rows, cols = boards.shape[1:3]
    for board in boards:
        yield rows, cols, board.tolist()
    logger.info(f"✅ {npz_path.name} 讀取 {boards.shape[0]} 筆資料")


def build_and_save_all_pos_freq(samples_dir: Path, output_dir: Path):
    shape_counts = defaultdict(lambda: None)
    output_dir.mkdir(parents=True, exist_ok=True)

    for npz_file in samples_dir.glob("*.npz"):
        logger.info(f"📦 掃描 {npz_file.name}")
        for rows, cols, grid in iter_all_json_from_zip(npz_file):
            grid_np = np.asarray(grid, dtype=int)
            shape = (rows, cols)
            if shape_counts[shape] is None:
                shape_counts[shape] = np.zeros((rows, cols), dtype=int)
            shape_counts[shape] += (grid_np != -1).astype(int)

    if not shape_counts:
        logger.warning("⚠️ 沒有有效樣本可供輸出")
        return

    all_shapes = sorted(shape_counts.keys())
    logger.info(f"🔍 一共找到這些盤面尺寸：{all_shapes}")

    for (rows, cols), count_mat in shape_counts.items():
        total = int(count_mat.sum()) or 1
        freq = count_mat.astype(float) / float(total)
        out_path = output_dir / f"pos_freq_{rows}x{cols}.npz"
        np.savez_compressed(out_path, freq=freq)
        logger.info(f"✅ 儲存 {out_path.name} ({rows}x{cols})")


if __name__ == "__main__":
    build_and_save_all_pos_freq(SAMPLES_DIR, OUTPUT_DIR)
