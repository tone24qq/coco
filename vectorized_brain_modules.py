"""
vectorized_brain_modules.py - 4模組向量化實現，整合熱力圖邏輯
"""
import numpy as np
from scipy.ndimage import convolve
import numba as nb
import logging
import time
from typing import Tuple

logger = logging.getLogger(__name__)

class VectorizedBrainModules:
    """4模組向量化分析器"""
    
    def edge_proximity_fusion(self, grid: np.ndarray) -> np.ndarray:
        """模組1：邊緣鄰近融合，低值偏邊緣"""
        try:
            rows, cols = grid.shape
            legal_mask = (grid == -1).astype(np.float32)
            filled_mask = (grid > 0).astype(np.float32)
            
            radius = 2
            size = 2 * radius + 1
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            distances = np.abs(x) + np.abs(y)
            distances[radius, radius] = np.inf
            kernel = 1.0 / (distances ** 1.5)
            kernel[radius, radius] = 0
            
            value_weights = np.where(filled_mask, grid * 0.1, 0)
            scores = convolve(value_weights, kernel, mode='constant')
            
            max_val = np.max(scores[legal_mask]) if np.any(legal_mask) else 1.0
            scores = np.where(legal_mask, scores / max_val, 0)
            return scores
        except Exception as e:
            logger.error(f"EdgeProximityFusion 失敗: {e}")
            return np.zeros_like(grid, dtype=np.float32)

    @nb.njit(parallel=True)
    def sequence_tail_analyzer(grid: np.ndarray) -> np.ndarray:
        """模組2：序列尾數分析，等差數列"""
        rows, cols = grid.shape
        scores = np.zeros((rows, cols), dtype=np.float32)
        blank_mask = (grid == -1)
        
        directions = [(0,1), (1,0), (1,1), (1,-1), (0,-1), (-1,0), (-1,-1), (-1,1)]
        
        for r in nb.prange(rows):
            for c in range(cols):
                if not blank_mask[r, c]:
                    continue
                max_score = 0.0
                for dr, dc in directions:
                    r1, c1 = r + dr, c + dc
                    r2, c2 = r + 2*dr, c + 2*dc
                    if (0 <= r1 < rows and 0 <= c1 < cols and 
                        0 <= r2 < rows and 0 <= c2 < cols and
                        grid[r1, c1] > 0 and grid[r2, c2] > 0):
                        v1, v2 = grid[r1, c1], grid[r2, c2]
                        if (v2 - v1) % 2 == 0:
                            expected = (v1 + v2) // 2
                            if 1 <= expected <= rows * cols:
                                max_score = max(max_score, 0.8)
                scores[r, c] = max_score
        return scores

    def connectivity_heatmap(self, grid: np.ndarray) -> np.ndarray:
        """模組3：連通熱圖，基於已知數字分佈（融合熱力圖邏輯）"""
        try:
            rows, cols = grid.shape
            legal_mask = (grid == -1).astype(np.float32)
            filled_mask = (grid > 0).astype(np.float32)
            
            # 卷積計算鄰域密度（模擬熱力圖）
            kernel = np.ones((3, 3), dtype=np.float32) / 8
            kernel[1, 1] = 0
            heatmap = convolve(filled_mask, kernel, mode='constant')
            
            # 加入數值加權（模擬歷史樣本的熱力分佈）
            value_weights = np.where(filled_mask, grid / (rows * cols), 0)
            value_heatmap = convolve(value_weights, kernel, mode='constant')
            
            # 融合密度與數值熱圖
            combined_heatmap = 0.7 * heatmap + 0.3 * value_heatmap
            scores = np.where(legal_mask, combined_heatmap / np.max(combined_heatmap + 1e-8), 0)
            return scores
        except Exception as e:
            logger.error(f"ConnectivityHeatmap 失敗: {e}")
            return np.zeros_like(grid, dtype=np.float32)

    def entropy_risk_fusion(self, grid: np.ndarray) -> np.ndarray:
        """模組4：熵風險融合，局部熵"""
        try:
            rows, cols = grid.shape
            legal_mask = (grid == -1).astype(np.float32)
            
            scores = np.zeros((rows, cols), dtype=np.float32)
            for r in range(rows):
                for c in range(cols):
                    if not legal_mask[r, c]:
                        continue
                    window = grid[max(0, r-1):r+2, max(0, c-1):c+2]
                    valid_vals = window[window > 0]
                    if len(valid_vals) > 1:
                        counts = np.bincount(valid_vals.astype(int))
                        probs = counts[counts > 0] / len(valid_vals)
                        entropy = -np.sum(probs * np.log2(probs + 1e-10))
                        scores[r, c] = entropy / np.log2(len(valid_vals) + 1e-10)
            
            scores = np.where(legal_mask, scores / np.max(scores + 1e-8), 0)
            return scores
        except Exception as e:
            logger.error(f"EntropyRiskFusion 失敗: {e}")
            return np.zeros_like(grid, dtype=np.float32)

    def test_with_masking(
        self, original_grid: np.ndarray, n_mask: int = 40, target: int = 7, n_trials: int = 20
    ) -> Tuple[float, float]:
        """隨機遮蔽測試，計算命中率"""
        try:
            true_positions = np.where(original_grid == target)
            if len(true_positions[0]) == 0:
                raise ValueError(f"目標數字 {target} 不存在於盤面")
            
            accuracies = []
            for trial in range(n_trials):
                grid = original_grid.copy()
                indices = np.random.choice(original_grid.size, n_mask, replace=False)
                grid.flat[indices] = -1
                
                for r, c in zip(true_positions[0], true_positions[1]):
                    grid[r, c] = -1
                
                from analyzer11_optimized import analyze_with_prior
                results = analyze_with_prior(grid, target, request_id=f"test_{trial}")
                
                correct = False
                for r, c, conf in results:
                    if original_grid[r, c] == target:
                        correct = True
                        logger.info(f"試驗 {trial+1}: 預測位置 {(r,c)}, 信心分數 {conf:.3f}, 正確")
                        break
                if not correct:
                    logger.info(f"試驗 {trial+1}: 預測失敗")
                
                accuracies.append(1.0 if correct else 0.0)
            
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            logger.info(f"平均命中率: {mean_acc:.3f} ({mean_acc*100:.1f}%), 標準差: {std_acc:.3f}")
            return mean_acc, std_acc
        except Exception as e:
            logger.error(f"測試失敗: {e}")
            return 0.0, 0.0

def performance_test():
    """性能測試"""
    test_grid = np.random.randint(-1, 60, (6, 10))
    test_grid[test_grid == 0] = -1
    brain = VectorizedBrainModules()
    
    start_time = time.time()
    scores = [
        brain.edge_proximity_fusion(test_grid),
        brain.sequence_tail_analyzer(test_grid),
        brain.connectivity_heatmap(test_grid),
        brain.entropy_risk_fusion(test_grid)
    ]
    elapsed = time.time() - start_time
    
    print(f"4模組計算時間: {elapsed:.4f}秒")
    return scores

def run_masking_test():
    """運行隨機遮蔽測試"""
    sample_grid = np.array([
        [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
        [51, 52, 53, 54, 55, 56,  7, 58, 59, 60]
    ])
    brain = VectorizedBrainModules()
    np.random.seed(42)
    mean_acc, std_acc = brain.test_with_masking(sample_grid, n_mask=40, target=7, n_trials=20)
    print(f"最終結果: 平均命中率 {mean_acc*100:.1f}%, 標準差 {std_acc:.3f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    performance_test()
    run_masking_test()