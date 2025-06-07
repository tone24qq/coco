import os
import json
import numpy as np
import pandas as pd
from analyzer import analyze_board

def load_grid_from_file(path: str) -> list[np.ndarray]:
    """讀取 Excel 檔案，返回所有工作表的 np.ndarray 列表，符合預處理條例。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.xls', '.xlsx']:
        xls = pd.ExcelFile(path)
        grids = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet_name, header=None, dtype=str)
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
            grid = np.array(cleaned_data)
            grids.append(grid)
        return grids
    elif ext == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [np.array(data)]
    elif ext == '.csv':
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
    else:
        raise ValueError(f"不支援的檔案格式: {ext}")

def save_results_to_file(score: np.ndarray, pred: np.ndarray, best_pos: tuple, out_prefix: str, out_format: str = 'json'):
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

def process_single_board(filepath: str, weights: dict, return_predictions: bool, output_prefix: str, target_num: int = None, json_heatmap: str = None):
    grids = load_grid_from_file(filepath)
    for idx, grid in enumerate(grids):
        sheet_output_prefix = f"{output_prefix}_sheet{idx+1}"
        score, pred, best_pos = analyze_board(grid, weights, return_predictions, target_num, json_heatmap)
        out_format = os.path.splitext(output_prefix)[1].lower().strip('.')
        if out_format not in ['json', 'csv', 'xls', 'xlsx']:
            sheet_output_prefix += '.json'
            out_format = 'json'
        save_results_to_file(score, pred, best_pos, sheet_output_prefix, out_format)
        print(f"處理工作表 {idx+1} 完成，輸出到: {sheet_output_prefix}")
        print(f"原始盤面 (Sheet {idx+1}):")
        print_aligned_grid(grid)
        print(f"熱力圖 (Sheet {idx+1}):")
        print_aligned_grid(score)
        if pred is not None:
            print(f"預測值 (Sheet {idx+1}):")
            print_aligned_grid(pred)
        if best_pos:
            print(f"指定數字 {target_num} 最可能位置: {best_pos}")

def process_batch(folder: str, weights: dict, return_predictions: bool, output_dir: str, target_num: int = None, json_heatmap: str = None):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(folder):
        fullpath = os.path.join(folder, fname)
        ext = os.path.splitext(fname)[1].lower()
        if ext in ['.json', '.csv', '.xls', '.xlsx']:
            base = os.path.splitext(fname)[0]
            output_prefix = os.path.join(output_dir, base + '_result')
            print(f"處理檔案: {fullpath} → 輸出: {output_prefix}")
            process_single_board(fullpath, weights, return_predictions, output_prefix, target_num, json_heatmap)