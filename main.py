# main.py
import argparse
import json
import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from brain import process_single_board, process_batch, load_grid_from_file
from analyzer import generate_masked_samples, train_extended_model
from modules import ScratchSolver
from joblib import Parallel, delayed
import zipfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s:%(name)s] %(message)s",
    handlers=[logging.FileHandler("logs/main.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "compute_dynamic_hot_cold_vectorized": 0.15,
    "compute_dynamic_hot_cold_advanced": 0.5,
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scratch Card Analysis Tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-file", type=str, help="Single input file (JSON/CSV/Excel)")
    group.add_argument("--input-folder", type=str, help="Input folder (samples/data/)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output file or folder")
    parser.add_argument("--weights", type=str, default=None, help="Module weights JSON")
    parser.add_argument(
        "--mode", choices=["heatmap", "predict"], default="predict", help="Analysis mode"
    )
    parser.add_argument("--target-num", type=str, default=None, help="Target number(s)")
    parser.add_argument(
        "--json-heatmap", default="samples/data/json", type=str,
        help="JSON heatmap folder"
    )
    parser.add_argument(
        "--train", action="store_true", help="Enable training mode"
    )
    parser.add_argument(
        "--model-dir", default="stats/models", type=str, help="Model output folder"
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--global-heatmap", action="store_true", help="Generate global heatmap from all files")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size for processing")
    return parser.parse_args()

def generate_random_grid(m: int, n: int, open_ratio: float = 0.5, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    total = m * n
    nums = np.random.permutation(np.arange(1, total + 1))
    grid = np.full((m, n), -1, dtype=np.int64)
    open_cells = int(total * open_ratio)
    idx = np.random.choice(total, open_cells, replace=False)
    grid[np.unravel_index(idx, (m, n))] = nums[:open_cells]
    return grid

def balance_samples(grids: List[np.ndarray], target_nums: List[int]) -> List[Tuple[np.ndarray, int]]:
    freq = {num: 0 for num in target_nums}
    for grid in grids:
        for num in grid[grid != -1].flatten():
            if num in freq:
                freq[num] += 1
    min_freq = min(freq.values()) if freq else 0
    samples = []
    for grid in grids:
        m, n = grid.shape
        remaining = list(set(target_nums).intersection(set(range(1, m * n + 1)) - set(grid[grid != -1].flatten())))
        for num in remaining:
            if freq[num] < min_freq * 1.5:
                for _ in range(int(min_freq * 1.5 - freq[num]) + 1):
                    samples.append((grid.copy(), num))
    return samples

async def process_batch_files(input_path: str, batch_size: int, batch_index: int):
    file_paths = []
    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        if filename.endswith(('.json', '.zip')):
            if filename.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(input_path, os.path.splitext(filename)[0]))
                    for json_file in zip_ref.namelist():
                        if json_file.endswith('.json'):
                            extracted_path = os.path.join(input_path, os.path.splitext(filename)[0], json_file)
                            file_paths.append(extracted_path)
            else:
                file_paths.append(file_path)
    start_idx = batch_index * batch_size
    end_idx = min((batch_index + 1) * batch_size, len(file_paths))
    return file_paths[start_idx:end_idx], len(file_paths) > end_idx

async def main() -> None:
    args = parse_args()
    
    weights: Dict[str, float] = json.loads(args.weights) if args.weights else DEFAULT_WEIGHTS
    return_predictions: bool = args.mode == "predict"
    target_nums: Optional[List[int]] = [int(x) for x in args.target_num.split(",")] if args.target_num else None
    
    os.makedirs(args.json_heatmap, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    global_heatmap_path = os.path.join("samples/data", "global_heatmap.json") if args.global_heatmap else None
    
    batch_index = 0
    while True:
        if args.global_heatmap:
            solver = ScratchSolver()
            input_path = args.input_folder or args.input_file
            batch_files, has_next = await process_batch_files(input_path, args.batch_size, batch_index)
            if not batch_files:
                break
            solver.compute_global_heatmap_from_files(batch_files, batch_size=args.batch_size, output_path=global_heatmap_path)
            if not has_next:
                break
            input()
            batch_index += 1
        
        elif args.train:
            grids: List[np.ndarray] = []
            if not os.path.isdir(args.input_folder):
                raise NotADirectoryError(f"Input folder {args.input_folder} is not a directory")
            for filename in os.listdir(args.input_folder):
                if filename.endswith(('.json', '.csv', '.xls', '.xlsx')):
                    grids.extend(load_grid_from_file(os.path.join(args.input_folder, filename)))
            
            if len(grids) < 100:
                for i in range(100 - len(grids)):
                    grids.append(generate_random_grid(8, 10, 0.5, seed=i))
            
            samples: List[Tuple[np.ndarray, int, Dict[str, Any]]] = []
            def process_grid(grid):
                m, n = grid.shape
                nums = list(set(range(1, m * n + 1)) - set(grid[grid != -1].flatten()))
                return generate_masked_samples(grid, target_nums=nums if not target_nums else target_nums)
            samples_list = Parallel(n_jobs=args.n_jobs)(delayed(process_grid)(grid) for grid in grids[:100])
            for sub_samples in samples_list:
                samples.extend(sub_samples)
            
            balanced_samples = balance_samples(grids, nums if not target_nums else target_nums)
            samples.extend([
                (grid, num, {"features": compute_all_module_scores(grid, (0, 0), grid.shape)})
                for grid, num in balanced_samples
            ])
            
            if not samples:
                raise ValueError("No valid training samples generated")
            
            os.makedirs(os.path.dirname(args.model_dir), exist_ok=True)
            feature_log = os.path.join(args.output_dir, "features_log.json")
            train_extended_model(samples, os.path.join(args.model_dir, "model.pkl"), feature_log)
            break
        
        elif args.input_file:
            output_prefix = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.input_file))[0])
            base_name = os.path.splitext(os.path.basename(args.input_file))[0]
            heatmap_path = os.path.join(args.json_heatmap, f"{base_name}_sheet1.json")
            
            await process_single_board(
                args.input_file, weights, return_predictions, output_prefix,
                target_nums[0] if target_nums else None, heatmap_path, global_heatmap_path
            )
            break
        
        else:
            output_folder = args.output_dir
            os.makedirs(output_folder, exist_ok=True)
            await process_batch(
                args.input_folder, weights, return_predictions, output_folder,
                target_nums[0] if target_nums else None, args.json_heatmap, global_heatmap_path
            )
            break

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())