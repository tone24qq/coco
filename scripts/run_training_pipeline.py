from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main_ranker import write_model_registry  # noqa: E402
from src.safe_io import read_dataset_auto  # noqa: E402


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    subprocess.run(cmd, check=True, env=env)


def _train_one(
    train: Path,
    valid: Path,
    holdout: Path,
    out_dir: Path,
    size_class: str,
    max_workers: int,
    feature_schema: Path,
) -> Dict[str, Any]:
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
            "--feature-schema",
            str(feature_schema),
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


def _build_multisize_summary(corpus_df, split_summary: Dict[str, Any], registry: Dict[str, Any]) -> Dict[str, Any]:
    real_by_size = Counter(corpus_df["size_class"].tolist()) if len(corpus_df) else Counter()
    split_per_size = split_summary.get("per_size", {})
    train_ps = split_per_size.get("train", {})
    valid_ps = split_per_size.get("valid", {})
    holdout_ps = split_per_size.get("holdout", {})

    per_size_training: Dict[str, Dict[str, Any]] = {}
    for size in sorted(set(real_by_size) | set(train_ps) | set(valid_ps) | set(holdout_ps)):
        model_item = registry.get("per_size", {}).get(size)
        per_size_training[size] = {
            "real_boards": int(real_by_size.get(size, 0)),
            "train_rows": int(train_ps.get(size, {}).get("rows", 0)),
            "valid_rows": int(valid_ps.get(size, {}).get("rows", 0)),
            "holdout_rows": int(holdout_ps.get(size, {}).get("rows", 0)),
            "model_trained": bool(model_item),
            "artifact_path": model_item.get("artifact_path") if model_item else "",
        }

    return {
        "real_board_count": int(len(corpus_df)),
        "real_board_per_size": dict(real_by_size),
        "split_summary": split_summary,
        "global_model": registry.get("global", {}),
        "per_size_training": per_size_training,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=".")
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
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    print("[主線訓練進度 5%] 參數解析完成，準備啟動資料建置。")

    root = Path(args.output_root)
    full = root / "data/full_boards/full_board_corpus.jsonl"
    synth = root / "data/full_boards/synthetic_board_corpus.jsonl"
    rank = root / "data/ranking/ranking_dataset.parquet"
    split_root = root / "data/ranking/splits"
    artifacts = root / "artifacts"
    reports = root / "reports"
    feature_schema = artifacts / "feature_schema_residue.json"

    _run(
        [
            "python",
            "scripts/build_root_xlsx_corpus_80.py",
            "--input-dir",
            args.input_dir,
            "--output",
            str(full),
            "--audit",
            str(reports / "full_board_corpus_audit.json"),
            "--preview-dir",
            str(reports / "root_xlsx_previews"),
        ]
    )
    print("[主線訓練進度 20%] 多尺寸真實盤面語料建置完成。")

    if args.generate_synthetic:
        profile = artifacts / "synthetic_generator_profile.json"
        _run(
            [
                "python",
                "scripts/fit_real_board_generator.py",
                "--real-corpus",
                str(full),
                "--output",
                str(profile),
            ]
        )
        _run(
            [
                "python",
                "scripts/generate_synthetic_boards.py",
                "--real-corpus",
                str(full),
                "--profile",
                str(profile),
                "--output",
                str(synth),
                "--per-real",
                str(args.per_real),
                "--max-file-mb",
                str(args.max_file_mb),
            ]
        )
        print("[主線訓練進度 30%] 合成盤面語料建置完成。")

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
            "--feature-schema",
            str(feature_schema),
            "--max-file-mb",
            str(args.max_file_mb),
        ]
    )
    print("[主線訓練進度 45%] ranking dataset 建置完成。")

    split_cmd = [
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
    _run(split_cmd)
    print("[主線訓練進度 55%] train/valid/holdout 切分完成。")

    train = split_root / "train.parquet"
    valid = split_root / "valid.parquet"
    holdout = split_root / "holdout.parquet"

    full_df = read_dataset_auto(full)
    real_size_counts = Counter(full_df["size_class"].tolist()) if len(full_df) else Counter()
    split_summary_path = split_root / "split_summary.json"
    split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
    if split_summary.get("valid_real_rows", 0) == 0 or split_summary.get("holdout_real_rows", 0) == 0:
        relaxed_cmd = [
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
            "--include-synth-in-holdout",
        ]
        _run(relaxed_cmd)
        split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))

    registry = {
        "model_strategy": "size_specific_with_global_fallback",
        "global": {},
        "per_size": {},
        "feature_schema_path": str(feature_schema),
        "backend": "mixed",
        "train_stats": {"real_size_counts": dict(real_size_counts)},
    }

    global_info = _train_one(train, valid, holdout, artifacts / "global", "", args.max_workers, feature_schema)
    registry["global"] = global_info
    print("[主線訓練進度 75%] global 模型訓練完成。")
    per_size_splits = split_summary.get("per_size", {})
    train_ps = per_size_splits.get("train", {})
    valid_ps = per_size_splits.get("valid", {})
    holdout_ps = per_size_splits.get("holdout", {})

    for size_class, count in sorted(real_size_counts.items()):
        if args.model_strategy == "global_only":
            continue
        should_train = args.model_strategy == "per_size" or count >= args.min_real_boards_per_size
        if not should_train:
            continue
        if train_ps.get(size_class, {}).get("rows", 0) <= 0:
            continue
        if valid_ps.get(size_class, {}).get("rows", 0) <= 0:
            continue
        if holdout_ps.get(size_class, {}).get("rows", 0) <= 0:
            continue

        out = artifacts / "sizes" / size_class
        registry["per_size"][size_class] = _train_one(
            train,
            valid,
            holdout,
            out,
            size_class,
            args.max_workers,
            feature_schema,
        )
        print(f"[主線訓練進度 85%] size={size_class} 模型訓練完成。")

    write_model_registry(registry, artifacts / "model_registry.json")

    reports.mkdir(parents=True, exist_ok=True)

    corpus_audit = json.loads((reports / "full_board_corpus_audit.json").read_text(encoding="utf-8"))
    multisize_corpus_summary = {
        "scanned_xlsx": corpus_audit.get("files_checked", []),
        "files": corpus_audit.get("files", []),
        "size_counts": corpus_audit.get("size_counts", {}),
        "board_count": corpus_audit.get("board_count", 0),
    }
    (reports / "multisize_corpus_summary.json").write_text(
        json.dumps(multisize_corpus_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    training_summary = _build_multisize_summary(full_df, split_summary, registry)
    (reports / "multisize_training_summary.json").write_text(
        json.dumps(training_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readiness = {
        "real_full_board_count": int(len(full_df)),
        "per_size_real_board_count": dict(real_size_counts),
        "split": split_summary,
        "model_strategy": registry.get("model_strategy"),
        "global_model_present": bool(Path(registry["global"]["artifact_path"]).exists()) if registry.get("global") else False,
        "per_size_model_present": {k: bool(Path(v["artifact_path"]).exists()) for k, v in registry.get("per_size", {}).items()},
        "inference_strict_missing_artifact": True,
        "ready_for_runtime": bool(registry.get("global"))
        and bool(split_summary.get("valid_real_rows", 0) > 0)
        and bool(split_summary.get("holdout_real_rows", 0) > 0),
        "global_artifact_path": registry.get("global", {}).get("artifact_path", ""),
        "per_size_artifact_paths": {k: v.get("artifact_path", "") for k, v in registry.get("per_size", {}).items()},
    }
    (reports / "runtime_readiness_report.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[主線訓練進度 95%] registry 與 readiness report 輸出完成。")

    if args.enable_inference:
        cfg_path = Path("configs/inference.yaml")
        if cfg_path.exists():
            text = cfg_path.read_text(encoding="utf-8")
            if "strict_missing_artifact" not in text:
                text += "\ntrained_ranker:\n  enabled: true\n  strict_missing_artifact: true\n"
                cfg_path.write_text(text, encoding="utf-8")

    print(json.dumps({"status": "ok", "registry": str(artifacts / 'model_registry.json')}, ensure_ascii=False))
    print("[主線訓練進度 100%] 主線訓練流程完成。")


if __name__ == "__main__":
    main()
