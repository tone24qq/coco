import argparse
import json
import os
from brain import process_single_board, process_batch
import logging

# 日誌設置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行參數，新增對新模組和熱力圖路徑的支援"""
    parser = argparse.ArgumentParser(description="橘子刮樂分析工具")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input-file', type=str, help='單一盤面檔案 (JSON/CSV/Excel)')
    group.add_argument('--input-folder', type=str, help='盤面檔案資料夾 (samples/data/)')
    parser.add_argument('--output', type=str, required=True, help='輸出檔案或資料夾')
    parser.add_argument('--weights', type=str, default=None, help='模組權重 JSON 字串')
    parser.add_argument('--mode', type=str, choices=['heatmap', 'predict'], default='predict', help='模式: heatmap 或 predict')
    parser.add_argument('--target-num', type=int, default=None, help='指定數字')
    parser.add_argument('--json-heatmap', type=str, default='samples/data/heatmaps', help='熱力圖資料夾路徑')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # 默認權重，包含新模組
    if args.weights:
        weights = json.loads(args.weights)
    else:
        weights = {
            "compute_dynamic_hot_cold_vectorized": 0.15,
            "compute_dynamic_hot_cold_advanced": 0.2,  # 新模組
            "compute_block_heatmap_vectorized": 0.1,
            "idw_vectorized": 0.1,
            "compute_global_diff_heatmap": 0.05,
            "compute_focus_score": 0.1,
            "detect_skip_patterns": 0.05,
            "compute_difference_trend": 0.05,
            "detect_mirror_sequences": 0.05,
            "connectivity_heatmap": 0.05,
            "sequence_tail_analyzer": 0.05,
            "analyze_number_patterns": 0.05  # 新模組
        }

    return_predictions = (args.mode == 'predict')

    # 確保熱力圖資料夾存在
    if not os.path.exists(args.json_heatmap):
        os.makedirs(args.json_heatmap, exist_ok=True)

    if args.input_file:
        input_path = args.input_file
        output_prefix = args.output
        logger.info(f"分析單一檔案: {input_path}")
        try:
            # 動態生成熱力圖路徑
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

# 測試運行
if __name__ == "__main__":
    # 模擬命令行測試，使用 Sheet1 進行示例
    import sys
    sys.argv = ['main.py', '--input-file', 'samples/data/8x10-10x12.xlsx', '--output', 'samples/output/test_result', '--mode', 'predict', '--target-num', 42, '--json-heatmap', 'samples/data/heatmaps']
    args = parse_args()
    if args.input_file:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        sheet_heatmap_path = os.path.join(args.json_heatmap, f"{base_name}_sheet1.json")
        process_single_board(args.input_file, weights, return_predictions, args.output, args.target_num, sheet_heatmap_path)