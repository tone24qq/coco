import os
import json
import numpy as np
import pandas as pd
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_grid_from_file(path: str) -> list[np.ndarray]:
    """讀取 Excel、JSON 或 CSV 檔案，返回所有工作表的 np.ndarray 列表，符合預處理條例。
    
    Args:
        path (str): 輸入檔案路徑，支持 .xls, .xlsx, .json, .csv 格式。
    
    Returns:
        list[np.ndarray]: 包含所有工作表的數值陣列列表。
    
    Raises:
        ValueError: 如果檔案格式不支援或內容無效。
        FileNotFoundError: 如果檔案不存在。
    """
    if not os.path.exists(path):
        logger.error(f"檔案不存在: {path}")
        raise FileNotFoundError(f"檔案不存在: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in ['.xls', '.xlsx']:
        try:
            xls = pd.ExcelFile(path, engine='openpyxl')
            grids = []
            for sheet_name in xls.sheet_names:
                try:
                    df = pd.read_excel(path, sheet_name=sheet_name, header=None, engine='openpyxl', dtype=str)
                    # 填補空值並清理資料
                    df = df.fillna("")
                    cleaned_data = []
                    for row in df.values:
                        cleaned_row = []
                        for cell in row:
                            if pd.isna(cell) or cell.strip() == "":
                                cleaned_row.append(-1)
                            else:
                                cell = cell.replace('O', '0').replace('I', '1')
                                if cell.isdigit():
                                    cleaned_row.append(int(cell))
                                else:
                                    cleaned_row.append(-1)  # 非數字轉為 -1
                        cleaned_data.append(cleaned_row)
                    grid = np.array(cleaned_data)
                    # 檢查形狀是否合理
                    if grid.size == 0 or grid.shape[0] > 20 or grid.shape[1] > 20:
                        logger.warning(f"Sheet {sheet_name} 形狀異常: {grid.shape}, 跳過")
                        continue
                    grids.append(grid)
                except ValueError as e:
                    logger.error(f"Sheet {sheet_name} 解析失敗: {e}")
                    continue
            if not grids:
                logger.warning(f"檔案 {path} 無有效工作表")
                return [np.zeros((1, 1), dtype=int) - 1]  # 回傳預設空陣列
            return grids
        except Exception as e:
            logger.error(f"讀取 Excel 檔案 {path} 失敗: {e}")
            raise
    elif ext == '.json':
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [np.array(data)]
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失敗: {path}, 錯誤: {e}")
            raise
    elif ext == '.csv':
        try:
            df = pd.read_csv(path, header=None, dtype=str)
            df = df.fillna("")
            cleaned_data = []
            for row in df.values:
                cleaned_row = []
                for cell in row:
                    cell = cell.replace('O', '0').replace('I', '1')
                    if cell.isdigit():
                        cleaned_row.append(int(cell))
                    else:
                        cleaned_row.append(-1)
                cleaned_data.append(cleaned_row)
            return [np.array(cleaned_data)]
        except Exception as e:
            logger.error(f"讀取 CSV 檔案 {path} 失敗: {e}")
            raise
    else:
        raise ValueError(f"不支援的檔案格式: {ext}")

def save_results_to_file(score: np.ndarray, pred: np.ndarray, best_pos: tuple, out_prefix: str, out_format: str = 'json'):
    """將結果保存到檔案。"""
    M, N = score.shape
    result = {
        'heatmap': score.tolist(),
        'best_position': best_pos if best_pos else None
    }
    if pred is not None:
        result['prediction'] = pred.tolist()
    if out_format == 'json':
        with open(out_prefix + '.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    elif out_format in ['xls', 'xlsx']:
        writer = pd.ExcelWriter(f"{out_prefix}.xlsx", engine='openpyxl')
        df_score = pd.DataFrame(score)
        df_score.to_excel(writer, sheet_name='heatmap', index=False, header=False)
        if pred is not None:
            df_pred = pd.DataFrame(pred)
            df_pred.to_excel(writer, sheet_name='prediction', index=False, header=False)
        writer.close()
    else:
        raise ValueError(f"不支援的輸出格式: {out_format}")

def print_aligned_grid(grid: np.ndarray):
    """以固定寬度對齊補齊輸出，行列索引從 1 開始。"""
    if grid.size == 0:
        print("空盤面")
        return
    M, N = grid.shape
    max_width = max(len(str(abs(x))) for x in grid.flatten()) + 2
    col_labels = " " * max_width + " ".join(f"{j+1:>{max_width}}" for j in range(N))
    print(col_labels)
    for i in range(M):
        row_str = f"{i+1:>{max_width}}" + " ".join(f"{x:>{max_width}}" for x in grid[i])
        print(row_str)

def process_single_board(grid, weights, return_predictions, output_prefix, target_num=None, json_heatmap=None):
    """處理單一盤面並輸出結果。"""
    from analyzer import analyze_board
    score, pred, best_pos = analyze_board(grid, weights, return_predictions, target_num, json_heatmap)
    out_format = os.path.splitext(output_prefix)[1].lower().strip('.')
    save_results_to_file(score, pred, best_pos, output_prefix, out_format)
    print(f"原始盤面:")
    print_aligned_grid(grid)
    print(f"熱力圖:")
    print_aligned_grid(score)
    if pred is not None:
        print(f"預測值:")
        print_aligned_grid(pred)
    if best_pos:
        print(f"指定數字 {target_num} 最可能位置: {best_pos}")