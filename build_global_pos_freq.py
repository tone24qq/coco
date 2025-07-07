#!/usr/bin/env python3
"""
Auto-generate global_pos_freq .npz files for specified or detected board shapes in a samples directory.
"""
# coding: utf-8
import json
import zipfile
import numpy as np
from pathlib import Path
from analyzer import compute_global_distribution

def detect_shapes(samples_dir: Path) -> set[tuple[int, int]]:
    """
    Scan all .zip files in samples_dir, inspect every JSON inside each,
    and collect unique (rows, cols) shapes.
    """
    shapes: set[tuple[int, int]] = set()
    for zp in samples_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(zp) as zf:
                for name in zf.namelist():
                    if name.lower().endswith(".json"):
                        try:
                            data = json.loads(zf.read(name))
                            r = int(data.get("rows", 0))
                            c = int(data.get("cols", 0))
                            shapes.add((r, c))
                        except Exception:
                            continue  # skip malformed
        except zipfile.BadZipFile:
            continue
    return shapes

def parse_shapes(shape_list: list[str]) -> list[tuple[int, int]]:
    """
    Parse list of strings like ['4x5','8x10'] into list of (4,5),(8,10)
    """
    result = []
    for s in shape_list:
        try:
            r, c = map(int, s.lower().split('x'))
            result.append((r, c))
        except Exception:
            raise ValueError(f"Invalid shape '{s}'. Must be in RxC format, e.g. 4x5.")
    return result

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate .npz frequency maps for shapes in samples folder"
    )
    parser.add_argument(
        "-s", "--samples",
        type=Path,
        required=True,
        help="Directory containing your .zip sample files"
    )
    parser.add_argument(
        "-o", "--outdir",
        type=Path,
        default=Path("out_npz"),
        help="Output directory for .npz files"
    )
    parser.add_argument(
        "-S", "--shapes",
        nargs='+',
        help="Optional list of shapes to process, e.g. 4x5 8x10. If omitted, will auto-detect."
    )
    args = parser.parse_args()

    samples_dir = args.samples
    if not samples_dir.exists() or not samples_dir.is_dir():
        print(f"ERROR: Samples directory {samples_dir} invalid.")
        return

    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.shapes:
        shapes = parse_shapes(args.shapes)
    else:
        shapes = detect_shapes(samples_dir)

    if not shapes:
        print("ERROR: No shapes to process. Provide --shapes or ensure ZIPs contain JSON with rows/cols.")
        return

    print(f"Shapes to process: {sorted(shapes)}")
    for rows, cols in sorted(shapes):
        print(f"Processing shape {rows}x{cols}...")
        freq_maps = compute_global_distribution(str(samples_dir), rows, cols)
        out_file = out_dir / f"global_pos_freq_{rows}x{cols}.npz"
        np.savez_compressed(str(out_file), freq=freq_maps)
        mb = out_file.stat().st_size / 1024**2
        print(f"Saved {out_file.name} ({mb:.1f} MB)")

if __name__ == '__main__':
    main()
