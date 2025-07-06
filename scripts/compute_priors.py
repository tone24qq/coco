import json
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

SAMPLE_DIR = Path("samples")
OUT_DIR = Path("priors")
OUT_DIR.mkdir(exist_ok=True)


def iter_json(zippath: Path):
    with zipfile.ZipFile(zippath) as zf:
        for name in zf.namelist():
            if name.endswith(".json"):
                with zf.open(name) as f:
                    yield json.loads(f.read())


def main() -> None:
    sizes: dict[tuple[int, int], np.ndarray] = {}
    for zp in tqdm(list(SAMPLE_DIR.glob("*.zip"))):
        for data in iter_json(zp):
            grid = data.get("grid")
            if not grid:
                continue
            rows, cols = len(grid), len(grid[0])
            key = (rows, cols)
            if key not in sizes:
                sizes[key] = np.zeros((rows, cols), dtype=np.int64)
            n = data.get("target_num")
            if n is None:
                continue
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == n:
                        sizes[key][r, c] += 1

    for (r, c), freq in sizes.items():
        total = freq.sum() or 1
        prob = freq.astype(float) / float(total)
        np.save(OUT_DIR / f"{r}x{c}.npy", prob)
        print(f"saved {r}x{c} prior to {OUT_DIR}")


if __name__ == "__main__":
    main()
