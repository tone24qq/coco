import argparse
import json
import os
from brain import load_grid_from_file, process_single_board
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行參數，接收輸入檔案和輸出設置。"""
    parser = argparse.ArgumentParser(description="刮刮樂分析工具")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input-file', type=str, help='單一盤面檔案 (JSON/CSV/Excel)')
    group.add_argument('--input-folder', type=str, help='盤面檔案資料夾 (samples/data/)')
    parser.add_argument('--output', type=str, required=True, help='輸出檔案或資料夾。如果是單一檔案，請包含前綴名(不含副檔名)。如果是 folder，則指定輸出資料夾路徑。')
    parser.add_argument('--weights', type=str, default=None, help='各模組權重的 JSON 字串，例如 \'{"focus":0.2,"skip":0.15,"diff":0.15,"mirror":0.2,"conn":0.15,"tail":0.15,"constraint":0.1,"tensor":0.1,"json":0.1}\'。若不指定，使用預設權重。')
    parser.add_argument('--mode', type=str, choices=['heatmap', 'predict'], default='heatmap', help='heatmap: 只輸出熱力圖；predict: 同時輸出預測值')
    parser.add_argument('--target-num', type=int, default=None, help='指定數字，預測其位置')
    parser.add_argument('--json-heatmap', type=str, default=None, help='JSON熱力圖檔案路徑')
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
            "tail": 0.15,
            "constraint": 0.1,
            "tensor": 0.1,
            "json": 0.1
        }

    return_predictions = (args.mode == 'predict')

    if args.input_file:
        input_path = args.input_file
        output_prefix = args.output
        logger.info(f"分析單一檔案: {input_path}")
        try:
            grids = load_grid_from_file(input_path)
            for idx, grid in enumerate(grids):
                sheet_output_prefix = f"{output_prefix}_sheet{idx+1}"
                out_format = os.path.splitext(output_prefix)[1].lower().strip('.')
                if out_format not in ['json', 'csv', 'xls', 'xlsx']:
                    sheet_output_prefix += '.json'
                    out_format = 'json'
                process_single_board(grid, weights, return_predictions, sheet_output_prefix, args.target_num, args.json_heatmap)
                logger.info(f"處理工作表 {idx+1} 完成，輸出到: {sheet_output_prefix}")
        except Exception as e:
            logger.error(f"處理檔案 {input_path} 失敗: {e}")
            raise
    else:
        input_folder = args.input_folder
        output_folder = args.output
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        logger.info(f"批次分析資料夾: {input_folder}，輸出到: {output_folder}")
        # 這裡假設 process_batch 函數已定義（參考原始 main.py）
        # 如果需要完整實現，請告訴我，我可以補充