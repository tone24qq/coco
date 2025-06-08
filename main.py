import argparse
import json
import os
from brain import process_single_board, process_batch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="橘子刮樂分析工具")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input-file', type=str, help='單一盤面檔案 (JSON/CSV/Excel)')
    group.add_argument('--input-folder', type=str, help='盤面檔案資料夾 (samples/data/)')
    parser.add_argument('--output', type=str, required=True, help='輸出檔案或資料夾')
    parser.add_argument('--weights', type=str, default=None, help='模組權重 JSON 字串')
    parser.add_argument('--mode', type=str, choices=['heatmap', 'predict'], default='predict', help='模式: heatmap 或 predict')
    parser.add_argument('--target-num', type=int, default=None, help='指定數字')
    parser.add_argument('--json-heatmap', type=str, default='samples/data/json', help='JSON 熱力圖資料夾路徑')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.weights:
        weights = json.loads(args.weights)
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

    os.makedirs(args.json_heatmap, exist_ok=True)

    if args.input_file:
        input_path = args.input_file
        output_prefix = args.output
        logger.info(f"分析單一檔案: {input_path}")
        try:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            sheet_heatmap_path = os.path.join(args.json_heatmap, f"{base_name}_sheet1.json")
            process_single_board(input_path, weights, return_predictions, output_prefix, args.target_num, sheet_heatmap_path)
        except Exception as e:
            logger.error(f"處理失敗: {e}")
    else:
        input_folder = args.input_folder
        output_folder = args.output
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"批次分析資料夾: {input_folder}，輸出到: {output_folder}")
        try:
            process_batch(input_folder, weights, return_predictions, output_folder, args.target_num, args.json_heatmap)
        except Exception as e:
            logger.error(f"批次處理失敗: {e}")