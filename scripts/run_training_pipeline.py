from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict

from src.main_ranker import write_model_registry
from src.safe_io import read_dataset_auto


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _train_one(train: Path, valid: Path, holdout: Path, out_dir: Path, size_class: str, max_workers: int) -> Dict[str, str]:
    _run(
        [
            "python",
            "scripts/train_local_ranker.py",
            "--train-path",
            str(train),
            "--valid-path",
            str(valid),
            "--holdout-path",
            str(holdout),
            "--size-class",
            size_class,
            "--artifacts-dir",
            str(out_dir),
            "--max-workers",
            str(max_workers),
        ]
    )
    meta = json.loads((out_dir / "main_ranker_meta.json").read_text(encoding="utf-8"))
    return {
        "artifact_path": str(out_dir / "main_ranker.pkl"),
        "meta_path": str(out_dir / "main_ranker_meta.json"),
        "backend": meta["backend"],
        "feature_columns": meta["feature_columns"],
        "train_rows": meta["train_rows"],
        "holdout_rows": meta["holdout_rows"],
        "holdout_metrics": meta["holdout_metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=".")
    parser.add_argument("--glob", default="*.xlsx")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--generate-synthetic", action="store_true")
    parser.add_argument("--per-real", type=int, default=12)
    parser.add_argument("--mask-ratios", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["by_board", "by_lineage"], default="by_lineage")
    parser.add_argument("--model-strategy", choices=["auto", "per_size", "global_only"], default="auto")
    parser.add_argument("--min-real-boards-per-size", type=int, default=5)
    parser.add_argument("--max-file-mb", type=int, default=100)
    parser.add_argument("--enable-inference", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.output_root)
    full = root / "data/full_boards/full_board_corpus.jsonl"
    synth = root / "data/full_boards/synthetic_board_corpus.jsonl"
    rank = root / "data/ranking/ranking_dataset.parquet"
    split_root = root / "data/ranking/splits"
    artifacts = root / "artifacts"

    _run(
        [
            "python",
            "scripts/build_real_board_corpus.py",
            "--input-dir",
            args.input_dir,
            "--glob",
            args.glob,
            "--output",
            str(full),
            "--max-file-mb",
            str(args.max_file_mb),
            "--valid-real-only",
            "--holdout-real-only",
            "--exclude-synth-from-valid",
        ]
    )
    if args.generate_synthetic:
        _run(
            [
                "python",
                "scripts/generate_synthetic_boards.py",
                "--real-corpus",
                str(full),
                "--output",
                str(synth),
                "--per-real",
                str(args.per_real),
                "--max-file-mb",
                str(args.max_file_mb),
            ]
        )
    _run(
        [
            "python",
            "scripts/build_masked_ranking_dataset.py",
            "--real-corpus",
            str(full),
            "--synthetic-corpus",
            str(synth),
            "--output",
            str(rank),
            "--mask-ratios",
            args.mask_ratios,
            "--max-file-mb",
            str(args.max_file_mb),
            "--valid-real-only",
            "--holdout-real-only",
            "--exclude-synth-from-valid",
        ]
    )
    _run(
        [
            "python",
            "scripts/split_ranking_dataset.py",
            "--dataset-path",
            str(rank),
            "--output-root",
            str(split_root),
            "--holdout-ratio",
            str(args.holdout_ratio),
            "--split-mode",
            args.split_mode,
            "--max-file-mb",
            str(args.max_file_mb),
            "--valid-real-only",
            "--holdout-real-only",
            "--exclude-synth-from-valid",
        ]
    )

    train = split_root / "train.parquet"
    valid = split_root / "valid.parquet"
    holdout = split_root / "holdout.parquet"

    full_df = read_dataset_auto(full)
    real_size_counts = Counter(full_df["size_class"].tolist())

    registry = {
        "model_strategy": "size_specific_with_global_fallback",
        "global": {},
        "per_size": {},
        "feature_schema_path": "artifacts/feature_schema.json",
        "backend": "mixed",
        "train_stats": {"real_size_counts": dict(real_size_counts)},
    }

    global_info = _train_one(train, valid, holdout, artifacts / "global", "", args.max_workers)
    registry["global"] = global_info

    if args.model_strategy != "global_only":
        for size_class, count in sorted(real_size_counts.items()):
            if args.model_strategy == "per_size" or count >= args.min_real_boards_per_size:
                out = artifacts / "sizes" / size_class
                registry["per_size"][size_class] = _train_one(train, valid, holdout, out, size_class, args.max_workers)

    write_model_registry(registry, artifacts / "model_registry.json")


    split_summary_path = split_root / "split_summary.json"
    split_summary = json.loads(split_summary_path.read_text(encoding="utf-8")) if split_summary_path.exists() else {}
    synth_count = 0
    if synth.exists():
        try:
            synth_df = read_dataset_auto(synth)
            synth_count = int(len(synth_df))
        except Exception:
            synth_count = 0

    readiness = {
        "real_full_board_count": int(len(full_df)),
        "per_size_real_board_count": dict(real_size_counts),
        "synthetic_board_count": synth_count,
        "split": split_summary,
        "model_strategy": registry.get("model_strategy"),
        "global_model_present": bool(Path(registry["global"]["artifact_path"]).exists()) if registry.get("global") else False,
        "per_size_model_present": {k: bool(Path(v["artifact_path"]).exists()) for k, v in registry.get("per_size", {}).items()},
        "inference_strict_missing_artifact": True,
        "ready_for_runtime": bool(registry.get("global")) and bool(split_summary.get("valid_real_rows", 0) > 0) and bool(split_summary.get("holdout_real_rows", 0) > 0),
    }
    rep_path = root / "reports/runtime_readiness_report.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.enable_inference:
        cfg_path = Path("configs/inference.yaml")
        text = cfg_path.read_text(encoding="utf-8")
        if "strict_missing_artifact" not in text:
            text += "\ntrained_ranker:\n  enabled: true\n  strict_missing_artifact: true\n"
            cfg_path.write_text(text, encoding="utf-8")

    print(json.dumps({"status": "ok", "registry": str(artifacts / 'model_registry.json')}, ensure_ascii=False))


if __name__ == "__main__":
    main()
