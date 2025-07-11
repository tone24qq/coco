import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import ray
except ModuleNotFoundError:  # FIXME optional dependency
    ray = None

from strategy_types import Strategy

# fmt: off
# isort: off
import analyzer
from analyzer import (
    probability_heatmap,
    predict_scratch_card,
    render_heatmap,
)
# isort: on
# fmt: on

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

priors: Dict[int, float] = {}


@dataclass
class CLIConfig:
    """Command-line configuration."""

    grid: str
    iterations: int
    global_iter: Optional[int]
    focus_iter: Optional[int]
    top_n: int
    top_k: Optional[int]
    epsilon: float
    target: Optional[int]
    heatmap_k: Optional[int]
    heatmap_iter: int
    heatmap_format: str
    sample_gamma: float
    use_neighbor_lock: bool
    strategy: Strategy


def parse_args(argv: Optional[List[str]] = None) -> CLIConfig:
    """Parse command-line arguments for grid input and iterations."""
    parser = argparse.ArgumentParser(
        description="Predict hidden numbers in a scratch card grid."
    )
    parser.add_argument(
        "--grid",
        type=str,
        required=True,
        help="2D grid as a comma-separated string, e.g., '1,2,-1;3,-1,5;-1,4,6'",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.getenv("ITER", "5000")),
        help="Number of Monte Carlo iterations",
    )
    parser.add_argument(
        "--global-iter",
        type=int,
        default=None,
        help="Phase-1 global iteration count",
    )
    parser.add_argument(
        "--focus-iter",
        type=int,
        default=None,
        help="Phase-2 focused iteration count",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Top cells to refine")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of top candidates to output",
    )
    parser.add_argument("--epsilon", type=float, default=0.05, help="Exploration rate")
    parser.add_argument(
        "--target", type=int, default=None, help="Target number to predict"
    )
    parser.add_argument(
        "--heatmap-k",
        type=int,
        default=None,
        help="Generate probability heatmap for this number (None to skip)",
    )
    parser.add_argument(
        "--heatmap-iter",
        type=int,
        default=1000,
        help="Iterations for heatmap simulation",
    )
    parser.add_argument(
        "--heatmap-format",
        type=str,
        choices=["raw", "base64", "png_bytes"],
        default="png_bytes",
        help="Format for heatmap output",
    )
    parser.add_argument(
        "--sample-gamma",
        type=float,
        default=0.9,
        help="Weight for sample-based frequency prior",
    )
    parser.add_argument(
        "--use-neighbor-lock",
        action="store_true",
        default=True,
        help="Enable neighbor lock strategy",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=[s.value for s in Strategy],
        default=Strategy.LEGACY.value,
        help="Prediction ranking strategy",
    )
    ns = parser.parse_args(argv)
    return CLIConfig(
        grid=ns.grid,
        iterations=ns.iterations,
        global_iter=ns.global_iter,
        focus_iter=ns.focus_iter,
        top_n=ns.top_n,
        top_k=ns.top_k,
        epsilon=ns.epsilon,
        target=ns.target,
        heatmap_k=ns.heatmap_k,
        heatmap_iter=ns.heatmap_iter,
        heatmap_format=ns.heatmap_format,
        sample_gamma=ns.sample_gamma,
        use_neighbor_lock=ns.use_neighbor_lock,
        strategy=Strategy(ns.strategy),
    )


def parse_grid(grid_str: str) -> List[List[int]]:
    """Parse string input into 2D grid."""
    try:
        rows = grid_str.strip().split(";")
        grid = [[int(x) for x in row.split(",")] for row in rows]
        grid_np = np.array(grid, dtype=int)
        if grid_np.ndim != 2:
            raise ValueError("Grid must be a 2D matrix")
        r, c = grid_np.shape
        if r < 2 or c < 2:
            raise ValueError("Grid must be at least 2x2 with consistent row length")
        return grid_np.tolist()
    except ValueError as e:
        logging.error(f"Invalid grid format: {e} - 格子字串格式錯誤")
        # 中文說明：解析命令列輸入的格子失敗，提示使用者檢查格式
        raise


def main():
    """Main function to run scratch card prediction."""
    args = parse_args()
    args.strategy = Strategy(args.strategy)
    try:
        p = Path("output/cleaned_data.json")
        global priors
        if p.exists():
            priors = json.loads(p.read_text(encoding="utf-8"))
        else:
            priors = {}
        grid = parse_grid(args.grid)
        iterations = args.iterations
        grid_np = np.array(grid, dtype=np.int64)

        # Validate grid
        known_vals = grid_np[grid_np != -1]
        rows, cols = grid_np.shape
        max_val = rows * cols
        if known_vals.size != np.unique(known_vals).size:
            raise ValueError("Grid contains duplicate numbers")
        if np.any((known_vals < 1) | (known_vals > max_val)):
            raise ValueError(f"Numbers must be between 1 and {max_val}")

        # Disable Ray dashboard to avoid excessive port scanning
        if ray is not None:
            ray.init(num_cpus=4, include_dashboard=False)
        result = predict_scratch_card(
            grid,
            target_num=args.target,
            iterations=iterations,
            global_iter=args.global_iter,
            focus_iter=args.focus_iter,
            top_n=args.top_n,
            epsilon=args.epsilon,
            result_top_k=args.top_k,
            priors=priors,
            sample_gamma=args.sample_gamma,
            use_neighbor_lock=args.use_neighbor_lock,
            fusion_alpha=0.1,
            force_legacy=False,
            strategy=args.strategy,
        )
        if ray is not None:
            ray.shutdown()

        if args.heatmap_k is not None:
            prob = probability_heatmap(
                grid_np,
                args.heatmap_k if args.heatmap_k != -1 else None,
                args.heatmap_iter,
                sample_gamma=args.sample_gamma,
                history_dir="samples",
            )
            if isinstance(prob, dict):
                logging.info(
                    "Full probability maps computed (no image) - 完成機率矩陣計算"
                )
                # 中文說明：僅輸出數值矩陣，不產生圖片
            else:
                rendered = render_heatmap(prob, args.heatmap_format)
                if isinstance(rendered, bytes):
                    with open("heatmap.png", "wb") as f:
                        f.write(rendered)
                    logging.info(
                        "Heatmap saved to heatmap.png - 熱力圖已存成 heatmap.png"
                    )
                    # 中文說明：熱力圖已存成 PNG 檔案
                elif isinstance(rendered, str):
                    with open("heatmap.txt", "w") as f:
                        f.write(rendered)
                    logging.info(
                        "Heatmap base64 saved to heatmap.txt - 熱力圖 base64 已存至 heatmap.txt"
                    )
                    # 中文說明：將熱力圖的 base64 字串寫入檔案
        logging.info(
            "Prediction results (strategy=%s) - 預測結果", result.get("strategy")
        )
        # 中文說明：列出採用的策略名稱
        for pred in result["predictions"]:
            r = int(pred.get("row", 0)) + 1
            c = int(pred.get("col", 0)) + 1
            if "probability" in pred:
                msg = f"Cell ({r}, {c}) prob {pred['probability']:.2f}%"
            elif "score" in pred:
                msg = f"Cell ({r}, {c}) score {pred['score']:.4f}"
            else:
                msg = f"Cell ({r}, {c})"
            logging.info(f"{msg} - 單格預測結果")
            # 中文說明：逐行列印每個格子的預測分數或機率
        logging.info(
            "Full probabilities available in result['full_probabilities'] - 全機率矩陣已包含於結果"
        )
        # 中文說明：提醒使用者結果內包含完整機率矩陣
        logging.info("Complete! - 程序結束")
        # 中文說明：CLI 流程結束
        return result
    except (ValueError, Exception) as e:
        logging.error(f"Error during prediction: {e} - 預測過程發生錯誤")
        # 中文說明：預測過程發生錯誤，程式將以非零狀態結束
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "dump_prior":
        analyzer.dump_prior(sys.argv[2], sys.argv[3])
    else:
        main()
