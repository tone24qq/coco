import sys
from pathlib import Path

import numpy as np

from analyzer import dtype_for_shape


def check_file(path: Path) -> bool:
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:
        print(f"failed to load {path}: {exc}", file=sys.stderr)
        return False
    if "freq" not in data:
        print(f"freq missing in {path}", file=sys.stderr)
        return False
    freq = data["freq"]
    rows, cols = freq.shape[:2]
    if freq.ndim == 3 and freq.shape[2] != rows * cols + 1:
        print(f"shape mismatch in {path}", file=sys.stderr)
        return False
    if np.issubdtype(freq.dtype, np.integer):
        expected = dtype_for_shape(rows, cols)
        if freq.dtype != expected:
            print(f"dtype {freq.dtype} != {expected} in {path}", file=sys.stderr)
            return False
    meta = data.get("meta")
    if meta is not None:
        if isinstance(meta, np.ndarray) and meta.shape == ():
            meta = meta.item()
        if not isinstance(meta, dict):
            print(f"meta type invalid in {path}", file=sys.stderr)
            return False
        if meta.get("schema_version") != 1 or "generated_at" not in meta:
            print(f"meta schema invalid in {path}", file=sys.stderr)
            return False
    return True


def main(argv: list[str]) -> int:
    ok = True
    targets = argv or ["samples"]
    for a in targets:
        p = Path(a)
        if p.is_dir():
            files = list(p.glob("*.npz"))
        else:
            files = [p]
        for f in files:
            if not check_file(f):
                ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
