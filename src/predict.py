"""CLI prediction entrypoint sharing inference core."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.inference import predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict with runtime transformer artifacts"
    )
    parser.add_argument(
        "--runtime-dir", default=Path("data/runtime_history"), type=Path
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _extract_numbers(items: Any) -> List[int]:
    if not isinstance(items, list):
        raise ValueError("Prediction output schema mismatch: expected list")
    return [int(item["number"]) for item in items]


def main() -> None:
    args = parse_args()
    result: Dict[str, Any] = predict(runtime_dir=args.runtime_dir)

    if args.output_json:
        args.output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    latest = result["latest_known_issue"]
    target = result["target_issue"]
    print(f"latest_issue={latest} target_issue={target}")
    print("top20=", _extract_numbers(result["top20"]))
    print("top3=", _extract_numbers(result["top3"]))


if __name__ == "__main__":
    main()
