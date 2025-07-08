import json
import zipfile
from pathlib import Path

import numpy as np

from analyzer import compute_global_distribution, load_global_pos_freq_npz

SAMPLES_DIR = Path("samples")


def main() -> None:
    """Generate heatmap NPZ files and pre-load one to warm caches."""

    # 強制讀一次真實熱力圖，避免 build 時出現 "no heatmap .npz files loaded"
    try:
        _ = load_global_pos_freq_npz((4, 5))  # 可換其他尺寸
    except Exception:  # noqa: BLE001 - best effort warm-up
        pass

    shapes = set()
    for zp in SAMPLES_DIR.glob("*.zip"):
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                try:
                    data = json.loads(zf.read(name))

                    # ✅ Case 1: dict 格式，標準結構
                    if isinstance(data, dict) and "rows" in data and "cols" in data:
                        rows = int(data["rows"])
                        cols = int(data["cols"])
                        shapes.add((rows, cols))

                    # ✅ Case 2: 純 grid 陣列 → [[1,2], [3,4]]
                    elif isinstance(data, list) and data and isinstance(data[0], list):
                        rows = len(data)
                        cols = len(data[0])
                        shapes.add((rows, cols))

                    # （可選擴充）Case 3: list of dicts
                    elif isinstance(data, list) and data and isinstance(data[0], dict):
                        for item in data:
                            if "rows" in item and "cols" in item:
                                rows = int(item["rows"])
                                cols = int(item["cols"])
                                shapes.add((rows, cols))

                    else:
                        raise ValueError("Unrecognized JSON structure")

                except Exception as e:
                    print(f"⚠️ Skip malformed: {name} in {zp.name}: {e}")
                    continue

    for rows, cols in sorted(shapes):
        out_path = SAMPLES_DIR / f"pos_freq_{rows}x{cols}.npz"
        if out_path.exists():
            print(f"Skip {out_path}, already exists")
            continue
        freq = compute_global_distribution(str(SAMPLES_DIR), rows, cols)
        np.savez(out_path, freq=freq)
        print(f"✅ Generated {out_path}")

    # fallback alias: only if there's exactly one size
    if len(shapes) == 1:
        rows, cols = next(iter(shapes))
        src = SAMPLES_DIR / f"pos_freq_{rows}x{cols}.npz"
        dst = SAMPLES_DIR / "pos_freq.npz"
        if not dst.exists() and src.exists():
            src.replace(dst)
            print(f"✅ Created fallback alias: {dst}")


if __name__ == "__main__":
    main()
