import argparse
import json
import logging
import os
from typing import List

from .core import batch_predict


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RandomForest board inference")
    p.add_argument("--input", required=True, help="input JSON file or glob pattern")
    p.add_argument("--output", required=True, help="output JSON file or directory")
    p.add_argument("--model", help="model path; auto select if omitted")
    p.add_argument("--k", type=int, default=3, help="top-k candidates")
    p.add_argument(
        "--models-dir", default="models", help="models directory for auto selection"
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    results = batch_predict(args.model, args.input, args.k, models_dir=args.models_dir)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info("Wrote results to %s", args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
