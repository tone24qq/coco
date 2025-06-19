import numpy as np
import argparse
import logging
from analyzer import predict_scratch_card

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description="預測刮刮樂盤面中的隱藏數字")
    parser.add_argument("--grid", type=str, required=True, help="盤面字符串，例如 '1,2,-1;3,-1,5;-1,4,6'")
    parser.add_argument("--iterations", type=int, default=10000, help="蒙特卡洛模擬次數")
    parser.add_argument("--target", type=int, default=None, help="目標數字")
    return parser.parse_args()

def parse_grid(grid_str: str) -> np.ndarray:
    """將字符串解析為盤面"""
    rows = grid_str.strip().split(';')
    grid = [[int(x) if x.strip() != '' else -1 for x in row.split(',')] for row in rows]
    grid_np = np.array(grid, dtype=np.int64)
    if not (4 <= grid_np.shape[0] <= 20 and 4 <= grid_np.shape[1] <= 20):
        raise ValueError("盤面必須在 4x4 到 20x20 之間")
    max_val = grid_np.shape[0] * grid_np.shape[1]
    known = grid_np[grid_np != -1]
    if len(known) != len(np.unique(known)):
        raise ValueError("盤面數字不可重複")
    if any(v < 1 or v > max_val for v in known):
        raise ValueError(f"數字必須在 1 到 {max_val} 之間")
    return grid_np

def main():
    """主函數"""
    args = parse_args()
    grid = parse_grid(args.grid)
    result = predict_scratch_card(grid, target_num=args.target, iterations=args.iterations)
    logging.info("預測結果：")
    for pred in result["predictions"]:
        logging.info(f"格子 ({pred['row']}, {pred['col']})：{pred['candidates']}，概率 {pred['probability']:.2f}%")
    return result

if __name__ == "__main__":
    main()