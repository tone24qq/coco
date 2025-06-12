# analyzer.py
import numpy as np
import logging
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from modules import ScratchSolver
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_all_module_scores(grid: np.ndarray, target_pos: Tuple[int, int], grid_shape: Tuple[int, int]) -> np.ndarray:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
        return np.array([])
    solver = ScratchSolver()
    solver.update_tree(grid)
    features = []
    for mod_name, mod_func in solver.MODULE_REGISTRY.items():
        try:
            result = mod_func(grid)
            scores = result[0] if isinstance(result, tuple) else result
            empty_yx = np.argwhere(grid == -1)
            idx = np.where((empty_yx == target_pos).all(axis=1))[0]
            features.append(scores[idx[0]] if idx.size > 0 else 0.1)
        except Exception:
            features.append(0.1)
    return np.array(features)

def extract_extended_features(grid: np.ndarray) -> Dict[str, float]:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
        return {}
    features = {}
    M, N = grid.shape
    open_nums = grid[grid != -1]
    
    for i in range(M):
        row = grid[i][grid[i] != -1]
        features[f"row_{i}_mean"] = np.mean(row) if row.size else 0
        features[f"row_{i}_std"] = np.std(row) if row.size > 1 else 0
    
    for j in range(N):
        col = grid[:, j][grid[:, j] != -1]
        features[f"col_{j}_mean"] = np.mean(col) if col.size else 0
        features[f"col_{j}_std"] = np.std(col) if col.size > 1 else 0
    
    diag = np.diagonal(grid)
    anti_diag = np.diagonal(np.fliplr(grid))
    features["diag_mean"] = np.mean(diag[diag != -1]) if np.any(diag != -1) else 0
    features["anti_diag_std"] = np.std(anti_diag[anti_diag != -1]) if np.any(anti_diag != -1) else 0
    
    solver = ScratchSolver()
    heatmap = solver.compute_dynamic_hot_cold_vectorized(grid)
    features["heatmap_top5_mean"] = np.mean(np.sort(heatmap)[-min(5, len(heatmap)):]) if heatmap.size else 0
    features["global_variance"] = np.var(open_nums) if open_nums.size else 0
    
    return features

def generate_masked_samples(grid: np.ndarray, target_nums: Optional[List[int]] = None) -> List[Tuple[np.ndarray, int, Dict[str, Any]]]:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
        return []
    samples = []
    M, N = grid.shape
    remaining_nums = list(set(range(1, M * N + 1)) - set(grid[grid != -1].flatten()))
    target_nums = target_nums if target_nums else remaining_nums
    
    extended_features = extract_extended_features(grid)
    
    for i in range(M):
        for j in range(N):
            true_val = grid[i, j]
            if true_val == -1 or true_val not in target_nums:
                continue
            masked = grid.copy()
            masked[i, j] = -1
            module_scores = compute_all_module_scores(masked, (i, j), (M, N))
            sample_features = {
                "module_scores": module_scores.tolist(),
                "extended_features": extended_features,
                "position": (i, j),
                "remaining_nums": remaining_nums
            }
            samples.append((masked, int(true_val), sample_features))
    
    return samples

def train_extended_model(samples: List[Tuple[np.ndarray, int, Dict[str, Any]]], model_path: str, feature_log_path: str) -> None:
    if not samples:
        return
    try:
        X = []
        y = []
        for _, true_val, sample_features in samples:
            features = np.concatenate([
                np.array(sample_features["module_scores"]),
                np.array([v for v in sample_features["extended_features"].values()])
            ])
            X.append(features)
            y.append(true_val)
        
        clf = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=7,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42
        )
        clf.fit(np.array(X), np.array(y))
        joblib.dump(clf, model_path)
        
        feature_log = [s[2] for s in samples]
        os.makedirs(os.path.dirname(feature_log_path), exist_ok=True)
        with open(feature_log_path, "w", encoding="utf-8") as f:
            json.dump(feature_log, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def predict_topk(masked_grid: np.ndarray, model_path: str, target_num: int, k: int = 3) -> List[Tuple[int, int, int, float, Dict[str, Any]]]:
    if not isinstance(masked_grid, np.ndarray) or masked_grid.ndim != 2 or masked_grid.size == 0:
        return []
    try:
        clf = joblib.load(model_path)
    except FileNotFoundError:
        return []
    
    M, N = masked_grid.shape
    extended_features = extract_extended_features(masked_grid)
    candidates = []
    
    for i in range(M):
        for j in range(N):
            if masked_grid[i, j] == -1:
                features = compute_all_module_scores(masked_grid, (i, j), (M, N))
                combined = np.concatenate([
                    features,
                    np.array([v for v in extended_features.values()])
                ])
                probs = clf.predict_proba([combined])[0]
                target_idx = np.where(clf.classes_ == target_num)[0]
                confidence = probs[target_idx[0]] if target_idx.size else 0.0
                reasoning = {
                    "position": (i, j),
                    "module_scores": features.tolist(),
                    "extended_features": extended_features,
                    "confidence_contributors": {
                        name: float(features[idx]) for idx, name in enumerate(ScratchSolver.MODULE_REGISTRY.keys())
                    }
                }
                candidates.append((i, j, target_num, confidence, reasoning))
    
    return sorted(candidates, key=lambda x: x[3], reverse=True)[:k] if candidates else []

def analyze_board(grid: np.ndarray, weights: Dict[str, float], return_predictions: bool = False, target_num: Optional[int] = None, json_heatmap_path: Optional[str] = None, knowledge_base: Optional[List[Dict[str, Any]]] = None, heatmap_data: Optional[Dict[str, Any]] = None, model_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, float], List[str]]:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
        return np.zeros_like(grid, dtype=float), np.array(grid), [], {"accuracy": 0}, ["Invalid grid input"]

    M, N = grid.shape
    heatmaps = {}
    if target_num is not None and target_num not in grid[grid != -1]:
        prior = np.where(grid == -1, 1.0, 0.0) / np.count_nonzero(grid == -1)
        heatmaps['uniform_prior'] = prior

    solver = ScratchSolver()
    solver.update_tree(grid)

    for mod_name, mod_func in solver.MODULE_REGISTRY.items():
        try:
            result = mod_func(grid)
            if isinstance(result, tuple):
                heatmaps[mod_name] = result[0]
            else:
                heatmaps[mod_name] = result
        except Exception:
            heatmaps[mod_name] = np.zeros_like(grid, dtype=float)

    final_score = solver.fuse_scores_vectorized(heatmaps, solver.classify_board_type(heatmaps.get('compute_dynamic_hot_cold_vectorized', np.zeros_like(grid))), weights)
    patterns = solver.analyze_number_patterns(grid)
    predictions, confidence = solver.integrate_predictions(grid, final_score, patterns)

    top3 = []
    reasoning_steps = [
        f"Remaining numbers: {list(set(range(1, grid.size + 1)) - set(grid[grid != -1].flatten()))}",
        f"Target number: {target_num if target_num is not None else 'auto'}"
    ]

    if model_path and os.path.exists(model_path):
        top3_predictions = predict_topk(grid, model_path, target_num or 0, k=3)
        top3 = [
            {
                "row": int(p[0]), "col": int(p[1]), "predicted_digit": int(p[2]),
                "confidence": float(p[3]), "module_scores": p[4]["confidence_contributors"]
            }
            for p in top3_predictions
        ]
        reasoning_steps.extend([f"Candidate at {p[4]['position']} with confidence {p[3]}" for p in top3_predictions])
    else:
        empty_yx = np.argwhere(grid == -1)
        if len(empty_yx) == 0:
            return np.zeros_like(grid, dtype=float), np.array(grid), [], {"accuracy": 0}, ["No hidden cells"]
        top3_pred = solver.predict_top3_vectorized(final_score, empty_yx, target_num)
        top3 = [
            {
                "row": int(pos[0]), "col": int(pos[1]), "predicted_digit": target_num or 0,
                "confidence": float(pos[2]), "module_scores": pos[3]
            }
            for pos in top3_pred
        ]

    true_values = grid.copy()
    remaining_nums = list(set(range(1, grid.size + 1)) - set(grid[grid != -1].flatten()))
    np.random.shuffle(remaining_nums)
    for (i, j), num in zip(np.argwhere(grid == -1), remaining_nums):
        true_values[i, j] = num
    metrics = solver.evaluate_prediction(grid, predictions, true_values)

    return final_score, predictions, top3, metrics, reasoning_steps