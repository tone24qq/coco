import json
import zipfile
from pathlib import Path

import numpy as np

from analyzer import compute_global_distribution

SAMPLES_DIR = Path("samples")


def main() -> None:
    shapes = set()
    for zp in SAMPLES_DIR.glob("*.zip"):
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                try:
                    data = json.loads(zf.read(name))
                    rows = int(data["rows"])
                    cols = int(data["cols"])
                    shapes.add((rows, cols))
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

    if len(shapes) == 1:
        rows, cols = next(iter(shapes))
        src = SAMPLES_DIR / f"pos_freq_{rows}x{cols}.npz"
        dst = SAMPLES_DIR / "pos_freq.npz"
        if not dst.exists() and src.exists():
            src.replace(dst)
            print(f"✅ Created fallback alias: {dst}")


if __name__ == "__main__":
    main()