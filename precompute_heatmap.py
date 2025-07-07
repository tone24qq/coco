import json
import zipfile
from pathlib import Path

import numpy as np

from analyzer import compute_global_distribution

SAMPLES_DIR = Path("samples")


def main() -> None:
    shapes: set[tuple[int, int]] = set()
    for zp in SAMPLES_DIR.glob("*.zip"):
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                data = json.loads(zf.read(name))
                if isinstance(data, dict) and "rows" in data and "cols" in data:
                    shapes.add((int(data["rows"]), int(data["cols"])))
                elif isinstance(data, list) and data:
                    rows = len(data[0])
                    cols = len(data[0][0]) if data[0] else 0
                    shapes.add((rows, cols))
                else:
                    for key in list(data.keys()):
                        if "x" in key:
                            try:
                                r, c = [int(x) for x in key.split("x")]
                            except Exception:
                                continue
                            shapes.add((r, c))
    for rows, cols in sorted(shapes):
        out_path = SAMPLES_DIR / f"pos_freq_{rows}x{cols}.npz"
        if out_path.exists():
            print(f"Skip {out_path}, already exists")
            continue
        freq = compute_global_distribution(str(SAMPLES_DIR), rows, cols)
        np.savez(out_path, freq=freq)
        print(f"Generated {out_path}")


if __name__ == "__main__":
    main()
