import argparse
import json
import os
import logging
from typing import Dict, Optional
from brain import process_single_board, process_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the scratch card analysis tool.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="橘子刮樂分析工具")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input-file', type=str, help='單一盤面檔案 (JSON/CSV/Excel)')
    group.add_argument('--input-folder', type=str, help='盤面檔案資料夾 (samples/data/)')
    parser.add_argument('--output', type=str, required=True, help='輸出檔案或資料夾')
    parser.add_argument('--weights', type=str, default=None, help='模組權重 JSON 字串')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['heatmap', 'predict'],
        default='predict',
        help='模式: heatmap 或 predict'
    )
    parser.add_argument('--target-num', type=int, default=None, help='指定數字')
    parser.add_argument(
        '--json-heatmap',
        type=str,
        default='samples/data/json',
        help='JSON 熱力圖資料夾路徑'
    )
    return parser.parse_args()

def main() -> None:
    """
    Main function to execute scratch card analysis.

    Processes either a single file or a batch of files based on command-line arguments.
    """
    args = parse_args()

    weights: Dict[str, float]
    if args.weights:
        try:
            weights = json.loads(args.weights)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid weights JSON: {e}")
            raise
    else:
        weights = {
            "compute_dynamic_hot_cold_vectorized": 0.15,
            "compute_dynamic_hot_cold_advanced": 0.2,
            "compute_block_heatmap_vectorized": 0.1,
            "idw_vectorized": 0.1,
            "compute_global_diff_heatmap": 0.05,
            "compute_focus_score": 0.1,
            "detect_skip_patterns": 0.05,
            "compute_difference_trend": 0.05,
            "detect_mirror_sequences": 0.05,
            "connectivity_heatmap": 0.05,
            "sequence_tail_analyzer": 0.05,
            "analyze_number_patterns": 0.05
        }

    return_predictions = (args.mode == 'predict')

    try:
        os.makedirs(args.json_heatmap, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create heatmap directory {args.json_heatmap}: {e}")
        raise

    if args.input_file:
        input_path = args.input_file
        output_prefix = args.output
        logger.info(f"Analyzing single file: {input_path}")
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file {input_path} does not exist")
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            sheet_heatmap_path = os.path.join(args.json_heatmap, f"{base_name}_sheet1.json")
            process_single_board(
                input_path,
                weights,
                return_predictions,
                output_prefix,
                args.target_num,
                sheet_heatmap_path
            )
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
    else:
        input_folder = args.input_folder
        output_folder = args.output
        try:
            os.makedirs(output_folder, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {output_folder}: {e}")
            raise
        logger.info(f"Batch analyzing folder: {input_folder}, outputting to: {output_folder}")
        try:
            if not os.path.exists(input_folder):
                raise FileNotFoundError(f"Input folder {input_folder} does not exist")
            process_batch(
                input_folder,
                weights,
                return_predictions,
                output_folder,
                args.target_num,
                args.json_heatmap
            )
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise

if __name__ == "__main__":
    main()