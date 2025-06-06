import argparse
import json
import os
from brain import process_single_board, process_batch

def parse_args():
    parser = argparse.ArgumentParser(description="刮刮樂分析工具")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input-file', type=str, help='單一盤面檔案 (JSON/CSV/Excel)')
    group.add_argument('--input-folder', type=str, help='盤面檔案資料夾 (samples/data/)')
    parser.add_argument('--output', type=str, required=True, help='輸出檔案或資料夾。如果是單一檔案，請包含前綴名(不含副檔名)。如果是 folder，則指定輸出資料夾路徑。')
    parser.add_argument('--weights', type=str, default=None, help='各模組權重的 JSON 字串，例如 \'{"focus":0.2,"skip":0.15,"diff":0.15,"mirror":0.2,"conn":0.15,"tail":0.15}\'。若不指定，使用預設權重。')
    parser.add_argument('--mode', type=str, choices=['heatmap', 'predict'], default='heatmap', help='heatmap: 只輸出熱力圖；predict: 同時輸出預測值')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.weights:
        weights = json.loads(args.weights)
    else:
        weights = {
            "focus": 0.2,
            "skip": 0.15,
            "diff": 0.15,
            "mirror": 0.2,
            "conn": 0.15,
            "tail": 0.15
        }

    return_predictions = (args.mode == 'predict')

    if args.input_file:
        input_path = args.input_file
        output_prefix = args.output
        print(f"分析單一檔案: {input_path}")
        process_single_board(input_path, weights, return_predictions, output_prefix)
    else:
        input_folder = args.input_folder
        output_folder = args.output
        print(f"批次分析資料夾: {input_folder}，輸出到: {output_folder}")
        process_batch(input_folder, weights, return_predictions, output_folder)