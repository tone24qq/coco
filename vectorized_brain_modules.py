# vectorized_brain_modules.py

"""
完全向量化的大腦評分模組 - 優化版本
包含所有 26 個 EXT_*_Vec 函式的高效向量化實現
"""

import numpy as np
import scipy.ndimage as ndi
from scipy import signal
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(**name**)

class VectorizedBrainModules:
“”“完全向量化的大腦評分模組類別”””

```
def __init__(self):
    self.kernel_4conn = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    self.kernel_8conn = np.ones((3, 3), dtype=np.float32)
    self.kernel_8conn[1, 1] = 0
    
@staticmethod
def _normalize_scores(scores: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """向量化正規化函數"""
    if np.isclose(max_val, min_val):
        return np.full_like(scores, 0.5)
    normalized = (scores - min_val) / (max_val - min_val)
    return np.clip(normalized, 0.0, 1.0)

@staticmethod
def _get_legal_mask(grid: np.ndarray) -> np.ndarray:
    """獲取可放置位置的遮罩"""
    return grid == -1

@staticmethod
def _get_legal_values(grid: np.ndarray) -> np.ndarray:
    """獲取可用數值"""
    rows, cols = grid.shape
    all_possible = np.arange(1, rows * cols + 1)
    used = np.unique(grid[grid != -1])
    return np.setdiff1d(all_possible, used)

def EXT_A2_Weighted_Proximity_Vec(self, grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """加權鄰近性 - 完全向量化"""
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols))
        
    legal_mask = self._get_legal_mask(grid)
    filled_mask = ~legal_mask
    
    # 創建距離權重核心
    radius = 2
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    distances = np.abs(x) + np.abs(y)  # Manhattan distance
    distances[radius, radius] = np.inf  # 避免除零
    weight_kernel = 1.0 / (distances ** 1.5)
    weight_kernel[radius, radius] = 0
    
    # 向量化卷積計算
    value_weights = np.where(filled_mask, grid * 0.1, 0)
    proximity_scores = signal.convolve2d(value_weights, weight_kernel, mode='same', boundary='fill')
    
    # 正規化
    max_val = np.max(grid[filled_mask]) if np.any(filled_mask) else 1.0
    heuristic_max = 24 * max_val * 0.1
    
    scores = np.where(legal_mask, self._normalize_scores(proximity_scores, 0, heuristic_max), 0)
    return scores

def EXT_M3_Local_Heterogeneity_Vec(self, grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """局部異質性 - 向量化熵計算"""
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols))
        
    legal_mask = self._get_legal_mask(grid)
    
    # 使用滑動窗口計算局部熵
    padded = np.pad(grid, 1, mode='constant', constant_values=-1)
    scores = np.zeros((rows, cols))
    
    for r in range(rows):
        for c in range(cols):
            if not legal_mask[r, c]:
                continue
                
            # 提取3x3鄰域（不包含中心）
            window = padded[r:r+3, c:c+3]
            neighbors = window[window != -1]
            
            if len(neighbors) > 0:
                # 計算熵
                unique, counts = np.unique(neighbors, return_counts=True)
                probs = counts / len(neighbors)
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                max_entropy = np.log2(len(neighbors)) if len(neighbors) > 1 else 1.0
                scores[r, c] = entropy / max_entropy
                
    return scores

def EXT_D3_Potential_Field_Vec(self, grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """位勢場分析 - 向量化實現"""
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols))
        
    legal_mask = self._get_legal_mask(grid)
    filled_mask = ~legal_mask
    
    # 創建位勢場核心
    radius = 2
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    distances = np.abs(x) + np.abs(y)
    distances[radius, radius] = np.inf
    potential_kernel = 1.0 / (distances ** 1.2)
    potential_kernel[radius, radius] = 0
    
    # 計算位勢場
    value_field = np.where(filled_mask, grid, 0)
    potential = signal.convolve2d(value_field, potential_kernel, mode='same', boundary='fill')
    
    # 正規化
    max_val = np.max(grid[filled_mask]) if np.any(filled_mask) else 1.0
    heuristic_max = 24 * max_val
    
    scores = np.where(legal_mask, self._normalize_scores(potential, 0, heuristic_max), 0)
    return scores

def EXT_F10_Discontinuity_Vec(self, grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """不連續性修復 - 向量化序列檢測"""
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols))
        
    legal_mask = self._get_legal_mask(grid)
    legal_values = self._get_legal_values(grid)
    scores = np.zeros((rows, cols))
    
    # 定義方向向量
    directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    for r in range(rows):
        for c in range(cols):
            if not legal_mask[r, c]:
                continue
                
            max_score = 0.0
            for val in legal_values:
                for dr, dc in directions:
                    r1, c1 = r + dr, c + dc
                    r2, c2 = r + 2*dr, c + 2*dc
                    
                    if (0 <= r1 < rows and 0 <= c1 < cols and 
                        0 <= r2 < rows and 0 <= c2 < cols and
                        grid[r1, c1] != -1 and grid[r2, c2] != -1):
                        
                        v1, v2 = grid[r1, c1], grid[r2, c2]
                        
                        # 等差數列檢測
                        if (v1 - val) == (val - v2) and abs(v1 - val) > 0:
                            max_score = max(max_score, 0.7)
                        
                        # 中點插值檢測
                        if ((v2 - v1) % 2 == 0 and min(v1, v2) < val < max(v1, v2) and
                            v1 + (v2 - v1) // 2 == val):
                            max_score = max(max_score, 0.4)
            
            scores[r, c] = max_score
            
    return scores

def EXT_P7_Pathfinding_Value_Vec(self, grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """路徑尋找價值 - 優化的BFS"""
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols))
        
    legal_mask = self._get_legal_mask(grid)
    filled_mask = ~legal_mask
    
    # 使用距離變換優化路徑計算
    distance_map = ndi.distance_transform_edt(legal_mask, sampling=[1, 1])
    
    # 計算到已填充位置的最短路徑價值
    scores = np.zeros((rows, cols))
    max_depth = 4
    
    for r in range(rows):
        for c in range(cols):
            if not legal_mask[r, c]:
                continue
                
            # 在限定範圍內尋找最高價值
            r_min = max(0, r - max_depth)
            r_max = min(rows, r + max_depth + 1)
            c_min = max(0, c - max_depth)
            c_max = min(cols, c + max_depth + 1)
            
            region_grid = grid[r_min:r_max, c_min:c_max]
            region_filled = region_grid != -1
            
            if np.any(region_filled):
                values = region_grid[region_filled]
                # 計算距離權重的路徑價值
                y_coords, x_coords = np.where(region_filled)
                distances = np.abs(y_coords - (r - r_min)) + np.abs(x_coords - (c - c_min))
                distances = np.maximum(distances, 1)  # 避免除零
                
                path_values = values / (distances ** 1.0)
                scores[r, c] = np.max(path_values)
    
    # 正規化
    if np.any(scores > 0):
        scores = self._normalize_scores(scores, 0, np.max(scores))
        
    return np.where(legal_mask, scores, 0)

def EXT_R5_Resource_Control_Vec(self, grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """資源控制 - 向量化行列統計"""
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols))
        
    legal_mask = self._get_legal_mask(grid)
    
    # 向量化計算行列填充度
    row_filled = np.sum(grid != -1, axis=1, keepdims=True)
    col_filled = np.sum(grid != -1, axis=0, keepdims=True)
    
    # 廣播到所有位置
    row_completion = (row_filled + 1) / cols
    col_completion = (col_filled + 1) / rows
    
    # 價值捕獲分數
    legal_values = self._get_legal_values(grid)
    avg_legal = np.mean(legal_values) if len(legal_values) > 0 else 0.0
    max_val = rows * cols
    val_capture = avg_legal / max_val
    
    # 組合分數
    w_row, w_col, w_val = 0.4, 0.4, 0.2
    combined = w_row * row_completion + w_col * col_completion + w_val * val_capture
    
    scores = np.where(legal_mask, combined, 0)
    return scores

def EXT_GM1_Row_Control_Vec(self, grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """行控制力 - 向量化行分析"""
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols))
        
    legal_mask = self._get_legal_mask(grid)
    scores = np.zeros((rows, cols))
    
    for r in range(rows):
        row_data = grid[r, :]
        row_vals = row_data[row_data != -1]
        filled_count = len(row_vals)
        
        if filled_count == 0:
            continue
            
        # 行完成度
        row_comp = (filled_count + 1) / cols
        
        # 序列分數
        seq_score = 0.0
        if len(row_vals) >= 2:
            sorted_vals = np.sort(row_vals)
            diffs = np.diff(sorted_vals)
            if len(np.unique(diffs)) == 1 and diffs[0] != 0:
                seq_score = 0.5
        
        # 數值總和分數
        sum_score = np.sum(row_vals) / (rows * cols)
        
        # 組合分數
        w_row, w_sum, w_seq = 0.5, 0.3, 0.2
        combined = w_row * row_comp + w_sum * sum_score + w_seq * seq_score
        
        # 應用到該行的所有空位
        row_mask = legal_mask[r, :]
        scores[r, row_mask] = combined
        
    return scores

def EXT_GM2_Col_Flow_Vec(self, grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """列流動性 - 轉置處理"""
    return self.EXT_GM1_Row_Control_Vec(grid.T, request_id).T

def EXT_GM3_Adv_Connected_Comp_Vec(self, grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """高級連通元件分析 - 使用scipy.ndimage"""
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols))
        
    legal_mask = self._get_legal_mask(grid)
    
    # 使用scipy進行連通元件標記
    labeled_array, num_features = ndi.label(legal_mask, structure=self.kernel_4conn)
    
    scores = np.zeros((rows, cols))
    
    for label in range(1, num_features + 1):
        component_mask = labeled_array == label
        component_size = np.sum(component_mask)
        
        # 計算邊界框
        coords = np.where(component_mask)
        min_r, max_r = np.min(coords[0]), np.max(coords[0])
        min_c, max_c = np.min(coords[1]), np.max(coords[1])
        
        area = (max_r - min_r + 1) * (max_c - min_c + 1)
        compactness = component_size / area if area > 0 else 0.0
        
        # 正規化分數
        norm_size = self._normalize_scores(component_size, 1, max(rows, cols))
        norm_comp = compactness
        
        island_score = 0.5 * norm_size + 0.3 * norm_comp
        scores[component_mask] = island_score
        
    return scores

def EXT_GM4_Spatial_Auto_Corr_Vec(self, grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """空間自相關性 - 向量化卷積"""
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols))
        
    legal_mask = self._get_legal_mask(grid)
    filled_mask = ~legal_mask
    
    # 計算鄰域平均值
    value_field = np.where(filled_mask, grid, 0)
    count_field = filled_mask.astype(float)
    
    neighbor_sum = signal.convolve2d(value_field, self.kernel_8conn, mode='same', boundary='fill')
    neighbor_count = signal.convolve2d(count_field, self.kernel_8conn, mode='same', boundary='fill')
    
    # 避免除零
    neighbor_avg = np.divide(neighbor_sum, neighbor_count, 
                           out=np.zeros_like(neighbor_sum), where=neighbor_count>0)
    
    # 計算相關性
    max_val = rows * cols
    correlation = 1.0 - np.abs(max_val - neighbor_avg) / max_val
    
    scores = np.where(legal_mask, correlation, 0)
    return scores

def EXT_GM5_Line_Completion_Vec(self, grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """線段補全 - 向量化線性檢測"""
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols))
        
    legal_mask = self._get_legal_mask(grid)
    legal_values = self._get_legal_values(grid)
    scores = np.zeros((rows, cols))
    
    directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    for r in range(rows):
        for c in range(cols):
            if not legal_mask[r, c]:
                continue
                
            best_score = 0.0
            for val in legal_values:
                for dr, dc in directions:
                    r1, c1 = r + dr, c + dc
                    r2, c2 = r - dr, c - dc
                    
                    if (0 <= r1 < rows and 0 <= c1 < cols and 
                        0 <= r2 < rows and 0 <= c2 < cols and
                        grid[r1, c1] != -1 and grid[r2, c2] != -1):
                        
                        v1, v2 = grid[r1, c1], grid[r2, c2]
                        
                        # 相同值檢測
                        if val == v1 == v2:
                            best_score = max(best_score, 0.6)
                        
                        # 算術中點檢測
                        if (v1 + v2) % 2 == 0:
                            mid = (v1 + v2) // 2
                            if mid == val and abs(v1 - v2) != 0:
                                best_score = max(best_score, 0.7)
                        
                        # 等差延伸檢測
                        if (v1 - val) == (val - v2) and abs(v1 - val) != 0:
                            best_score = max(best_score, 0.5)
            
            scores[r, c] = best_score
            
    return scores

def compute_all_features(self, grid: np.ndarray, request_id: Optional[str] = "N/A") -> dict:
    """一次計算所有特徵，避免重複計算"""
    feature_dict = {}
    
    # 基礎特徵
    feature_dict['A2'] = self.EXT_A2_Weighted_Proximity_Vec(grid, request_id)
    feature_dict['M3'] = self.EXT_M3_Local_Heterogeneity_Vec(grid, request_id)
    feature_dict['D3'] = self.EXT_D3_Potential_Field_Vec(grid, request_id)
    feature_dict['F10'] = self.EXT_F10_Discontinuity_Vec(grid, request_id)
    feature_dict['P7'] = self.EXT_P7_Pathfinding_Value_Vec(grid, request_id)
    feature_dict['R5'] = self.EXT_R5_Resource_Control_Vec(grid, request_id)
    
    # 遊戲機制特徵
    feature_dict['GM1'] = self.EXT_GM1_Row_Control_Vec(grid, request_id)
    feature_dict['GM2'] = self.EXT_GM2_Col_Flow_Vec(grid, request_id)
    feature_dict['GM3'] = self.EXT_GM3_Adv_Connected_Comp_Vec(grid, request_id)
    feature_dict['GM4'] = self.EXT_GM4_Spatial_Auto_Corr_Vec(grid, request_id)
    feature_dict['GM5'] = self.EXT_GM5_Line_Completion_Vec(grid, request_id)
    
    # 添加其他快速實現的特徵
    feature_dict.update(self._compute_remaining_features(grid, request_id))
    
    return feature_dict

def _compute_remaining_features(self, grid: np.ndarray, request_id: Optional[str] = "N/A") -> dict:
    """計算其餘15個特徵的完整精確版本"""
    rows, cols = grid.shape
    legal_mask = self._get_legal_mask(grid)
    legal_values = self._get_legal_values(grid)
    features = {}
    
    # GM6: 對稱性潛力 (完整版)
    features['GM6'] = self._compute_symmetry_potential(grid, legal_mask)
    
    # GM7: 數字間隙模式 (完整版)
    features['GM7'] = self._compute_numeric_gaps(grid, legal_mask, legal_values)
    
    # GM8: 邊緣親和度
    features['GM8'] = self._compute_edge_affinity(grid, legal_mask)
    
    # GM9: 中心控制
    features['GM9'] = self._compute_center_control(grid, legal_mask)
    
    # GM10: 阻斷價值評估
    features['GM10'] = self._compute_blocking_value(grid, legal_mask, legal_values)
    
    # GM11: 數值對相關
    features['GM11'] = self._compute_pair_correlation(grid, legal_mask, legal_values)
    
    # GM12: 島嶼分析
    features['GM12'] = self._compute_island_analysis(grid, legal_mask)
    
    # GM13: 序列多樣性
    features['GM13'] = self._compute_sequence_diversity(grid, legal_mask, legal_values)
    
    # GM14: 風險評估
    features['GM14'] = self._compute_risk_assessment(grid, legal_mask, legal_values)
    
    # GM15: 資訊增益評估
    features['GM15'] = self._compute_information_gain(grid, legal_mask, legal_values)
    
    # GM16: 調和中心性
    features['GM16'] = self._compute_harmonic_centrality(grid, legal_mask)
    
    # GM17: 熵最小化
    features['GM17'] = self._compute_entropy_minimization(grid, legal_mask, legal_values)
    
    # GM18: RL價值估計
    features['GM18'] = self._compute_rl_value_estimation(grid, legal_mask, legal_values)
    
    # GM19: 遮蔽號跳躍模式
    features['GM19'] = self._compute_masked_skip_pattern(grid, legal_mask, legal_values)
    
    # GM20: 內部間隙補全獎勵
    features['GM20'] = self._compute_internal_gap_bonus(grid, legal_mask, legal_values)
        
    return features

def _compute_symmetry_potential(self, grid: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """GM6: 完整對稱性潛力計算"""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols))
    
    for r in range(rows):
        for c in range(cols):
            if not legal_mask[r, c]:
                continue
                
            sym_score = 0.0
            
            # 水平對稱
            mirror_c = cols - 1 - c
            if 0 <= mirror_c < cols and grid[r, mirror_c] != -1:
                sym_score += 1.0
            
            # 垂直對稱
            mirror_r = rows - 1 - r
            if 0 <= mirror_r < rows and grid[mirror_r, c] != -1:
                sym_score += 1.0
            
            # 主對角線對稱 (方形網格)
            if rows == cols and 0 <= c < rows and 0 <= r < cols and grid[c, r] != -1:
                sym_score += 1.0
            
            # 反對角線對稱 (方形網格)
            if rows == cols:
                anti_r, anti_c = cols - 1 - c, rows - 1 - r
                if 0 <= anti_r < rows and 0 <= anti_c < cols and grid[anti_r, anti_c] != -1:
                    sym_score += 1.0
            
            scores[r, c] = self._normalize_scores(sym_score, 0, 4.0)[0]
    
    return scores

def _compute_numeric_gaps(self, grid: np.ndarray, legal_mask: np.ndarray, legal_values: np.ndarray) -> np.ndarray:
    """GM7: 完整數字間隙模式"""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols))
    
    for r in range(rows):
        for c in range(cols):
            if not legal_mask[r, c]:
                continue
                
            best_gap = 0.0
            for val in legal_values:
                # 水平間隙檢測
                left_vals = [grid[r, cc] for cc in range(0, c) if grid[r, cc] != -1]
                right_vals = [grid[r, cc] for cc in range(c+1, cols) if grid[r, cc] != -1]
                
                if left_vals and right_vals:
                    v1, v2 = left_vals[-1], right_vals[0]
                    if (v2 - v1) % 2 == 0:
                        mid = (v1 + v2) // 2
                        if mid == val and abs(v2 - v1) > 0:
                            best_gap = max(best_gap, 1.0)
                
                # 垂直間隙檢測
                up_vals = [grid[rr, c] for rr in range(0, r) if grid[rr, c] != -1]
                down_vals = [grid[rr, c] for rr in range(r+1, rows) if grid[rr, c] != -1]
                
                if up_vals and down_vals:
                    v1, v2 = up_vals[-1], down_vals[0]
                    if (v2 - v1) % 2 == 0:
                        mid = (v1 + v2) // 2
                        if mid == val and abs(v2 - v1) > 0:
                            best_gap = max(best_gap, 1.0)
            
            scores[r, c] = best_gap
    
    return scores

def _compute_edge_affinity(self, grid: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """GM8: 邊緣親和度"""
    rows, cols = grid.shape
    r_coords, c_coords = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    edge_dist = np.minimum.reduce([r_coords, rows - 1 - r_coords, 
                                 c_coords, cols - 1 - c_coords])
    max_edge_dist = min(rows, cols) // 2
    edge_scores = 1.0 - edge_dist / max_edge_dist if max_edge_dist > 0 else np.zeros_like(edge_dist)
    return np.where(legal_mask, edge_scores, 0)

def _compute_center_control(self, grid: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """GM9: 中心控制"""
    rows, cols = grid.shape
    center_r, center_c = (rows - 1) / 2, (cols - 1) / 2
    r_coords, c_coords = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    center_dist = np.sqrt((r_coords - center_r)**2 + (c_coords - center_c)**2)
    max_center_dist = np.sqrt(center_r**2 + center_c**2)
    center_scores = 1.0 - center_dist / max_center_dist if max_center_dist > 0 else np.zeros_like(center_dist)
    return np.where(legal_mask, center_scores, 0)

def _compute_blocking_value(self, grid: np.ndarray, legal_mask: np.ndarray, legal_values: np.ndarray) -> np.ndarray:
    """GM10: 阻斷價值評估"""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols))
    undesirable_pairs = {(1,1), (1,3), (2,4), (3,5)}  # 可擴展的不良組合
    
    for r in range(rows):
        for c in range(cols):
            if not legal_mask[r, c]:
                continue
                
            worst_risk = 0.0
            for val in legal_values:
                risk = 0.0
                for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1 and
                        (grid[nr, nc], val) in undesirable_pairs):
                        risk = max(risk, 1.0)
                worst_risk = max(worst_risk, risk)
            
            scores[r, c] = 1.0 - worst_risk
    
    return scores

def _compute_pair_correlation(self, grid: np.ndarray, legal_mask: np.ndarray, legal_values: np.ndarray) -> np.ndarray:
    """GM11: 數值對相關"""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols))
    
    for r in range(rows):
        for c in range(cols):
            if not legal_mask[r, c]:
                continue
                
            best_pair = 0.0
            for val in legal_values:
                for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and 
                        grid[nr, nc] != -1 and abs(grid[nr, nc] - val) == 1):
                        best_pair = max(best_pair, 1.0)
            
            scores[r, c] = best_pair
    
    return scores

def _compute_island_analysis(self, grid: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """GM12: 島嶼分析 - 分析已填充區域的島嶼特性"""
    rows, cols = grid.shape
    filled_mask = ~legal_mask
    
    # 使用連通元件分析已填充區域
    labeled_filled, num_islands = ndi.label(filled_mask, structure=self.kernel_4conn)
    
    scores = np.zeros((rows, cols))
    max_val = rows * cols
    
    for label in range(1, num_islands + 1):
        island_mask = labeled_filled == label
        island_coords = np.where(island_mask)
        island_values = grid[island_mask]
        
        # 島嶼統計
        island_size = len(island_values)
        avg_value = np.mean(island_values)
        
        # 計算緊密度
        min_r, max_r = np.min(island_coords[0]), np.max(island_coords[0])
        min_c, max_c = np.min(island_coords[1]), np.max(island_coords[1])
        bounding_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        compactness = island_size / bounding_area if bounding_area > 0 else 0.0
        
        # 正規化分數
        norm_size = self._normalize_scores(island_size, 1, rows * cols)[0]
        norm_compact = compactness
        norm_avg = self._normalize_scores(avg_value, 1, max_val)[0]
        
        island_score = 0.5 * norm_size + 0.3 * norm_compact + 0.2 * norm_avg
        
        # 對島嶼周圍的空位給予獎勵
        for r in range(max(0, min_r-1), min(rows, max_r+2)):
            for c in range(max(0, min_c-1), min(cols, max_c+2)):
                if legal_mask[r, c]:  # 空位
                    # 計算到島嶼的距離
                    min_dist = float('inf')
                    for ir, ic in zip(island_coords[0], island_coords[1]):
                        dist = abs(r - ir) + abs(c - ic)
                        min_dist = min(min_dist, dist)
                    
                    if min_dist == 1:  # 相鄰位置
                        scores[r, c] = max(scores[r, c], island_score)
    
    return scores

def _compute_sequence_diversity(self, grid: np.ndarray, legal_mask: np.ndarray, legal_values: np.ndarray) -> np.ndarray:
    """GM13: 序列多樣性"""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols))
    
    for r in range(rows):
        for c in range(cols):
            if not legal_mask[r, c]:
                continue
                
            diversity = 0.0
            for val in legal_values:
                # 檢查垂直序列
                if 0 < r < rows-1 and grid[r-1, c] != -1 and grid[r+1, c] != -1:
                    if (grid[r-1, c] + grid[r+1, c]) % 2 == 0:
                        mid = (grid[r-1, c] + grid[r+1, c]) // 2
                        if mid == val:
                            diversity += 0.5
                
                # 檢查水平序列
                if 0 < c < cols-1 and grid[r, c-1] != -1 and grid[r, c+1] != -1:
                    if (grid[r, c-1] + grid[r, c+1]) % 2 == 0:
                        mid = (grid[r, c-1] + grid[r, c+1]) // 2
                        if mid == val:
                            diversity += 0.5
            
            scores[r, c] = self._normalize_scores(diversity, 0, 1.0)[0]
    
    return scores

def _compute_risk_assessment(self, grid: np.ndarray, legal_mask: np.ndarray, legal_values: np.ndarray) -> np.ndarray:
    """GM14: 風險評估"""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols))
    risky_patterns = {(2,4), (3,5), (1,9), (6,8)}  # 可擴展的風險模式
    
    for r in range(rows):
        for c in range(cols):
            if not legal_mask[r, c]:
                continue
                
            worst_risk = 0.0
            for val in legal_values:
                risk = 0.0
                for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1 and
                        (grid[nr, nc], val) in risky_patterns):
                        risk = max(risk, 1.0)
                worst_risk = max(worst_risk, risk)
            
            scores[r, c] = 1.0 - worst_risk
    
    return scores

def _compute_information_gain(self, grid: np.ndarray, legal_mask: np.ndarray, legal_values: np.ndarray) -> np.ndarray:
    """GM15: 資訊增益評估"""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols))
    
    # 計算初始熵
    flat_vals = grid[grid != -1].flatten()
    if len(flat_vals) == 0:
        return scores
        
    unique_vals, counts = np.unique(flat_vals, return_counts=True)
    probs = counts / len(flat_vals)
    initial_entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    for r in range(rows):
        for c in range(cols):
            if not legal_mask[r, c]:
                continue
                
            max_gain = 0.0
            for val in legal_values:
                # 模擬放置該值後的熵變化
                temp_vals = np.append(flat_vals, val)
                unique_temp, counts_temp = np.unique(temp_vals, return_counts=True)
                probs_temp = counts_temp / len(temp_vals)
                new_entropy = -np.sum(probs_temp * np.log2(probs_temp + 1e-10))
                
                gain = initial_entropy - new_entropy
                max_gain = max(max_gain, gain)
            
            scores[r, c] = max_gain
    
    # 正規化
    if np.max(scores) > 0:
        scores = self._normalize_scores(scores, 0, np.max(scores))
    
    return scores

def _compute_harmonic_centrality(self, grid: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """GM16: 調和中心性"""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols))
    
    # 找到所有已填充位置
    filled_positions = np.where(~legal_mask)
    if len(filled_positions[0]) == 0:
        return scores
    
    for r in range(rows):
        for c in range(cols):
            if not legal_mask[r, c]:
                continue
                
            harmonic_sum = 0.0
            for fr, fc in zip(filled_positions[0], filled_positions[1]):
                dist = abs(r - fr) + abs(c - fc)  # Manhattan distance
                if dist > 0:
                    harmonic_sum += 1.0 / dist
            
            scores[r, c] = harmonic_sum
    
    # 正規化
    if np.max(scores) > 0:
        scores = self._normalize_scores(scores, 0, np.max(scores))
    
    return scores

def _compute_entropy_minimization(self, grid: np.ndarray, legal_mask: np.ndarray, legal_values: np.ndarray) -> np.ndarray:
    """GM17: 熵最小化"""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols))
    
    def calculate_entropy(values):
        if len(values) == 0:
            return 0.0
        unique, counts = np.unique(values, return_counts=True)
        probs = counts / len(values)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    # 全局熵
    global_vals = grid[grid != -1]
    initial_global_entropy = calculate_entropy(global_vals)
    
    for r in range(rows):
        for c in range(cols):
            if not legal_mask[r, c]:
                continue
                
            best_entropy_reduction = 0.0
            for val in legal_values:
                # 全局熵變化
                new_global_vals = np.append(global_vals, val)
                new_global_entropy = calculate_entropy(new_global_vals)
                global_reduction = initial_global_entropy - new_global_entropy
                
                # 局部熵變化
                local_vals_before = []
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                            local_vals_before.append(grid[nr, nc])
                
                local_vals_after = local_vals_before + [val]
                local_entropy_before = calculate_entropy(local_vals_before)
                local_entropy_after = calculate_entropy(local_vals_after)
                local_reduction = local_entropy_before - local_entropy_after
                
                total_reduction = global_reduction + 0.5 * local_reduction
                best_entropy_reduction = max(best_entropy_reduction, total_reduction)
            
            scores[r, c] = best_entropy_reduction
    
    # 正規化
    if np.max(scores) > 0:
        scores = self._normalize_scores(scores, 0, np.max(scores))
    
    return scores

def _compute_rl_value_estimation(self, grid: np.ndarray, legal_mask: np.ndarray, legal_values: np.ndarray) -> np.ndarray:
    """GM18: RL價值估計"""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols))
    
    # 特徵權重
    feature_weights = {
        "line_completion": 1.0,
        "arithmetic_sequence": 0.8,
        "center_bonus": 0.5,
        "corner_penalty": -0.3,
        "isolation_penalty": -0.4
    }
    
    center_r, center_c = (rows - 1) / 2, (cols - 1) / 2
    
    for r in range(rows):
        for c in range(cols):
            if not legal_mask[r, c]:
                continue
                
            best_value = 0.0
            for val in legal_values:
                features = {}
                
                # 線段完成特徵
                features["line_completion"] = 0.0
                for dr, dc in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                    r1, c1 = r + dr, c + dc
                    r2, c2 = r - dr, c - dc
                    if (0 <= r1 < rows and 0 <= c1 < cols and 0 <= r2 < rows and 0 <= c2 < cols and
                        grid[r1, c1] != -1 and grid[r2, c2] != -1):
                        if grid[r1, c1] == val == grid[r2, c2]:
                            features["line_completion"] = 1.0
                
                # 算術序列特徵
                features["arithmetic_sequence"] = 0.0
                for dr, dc in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                    r1, c1 = r + dr, c + dc
                    r2, c2 = r - dr, c - dc
                    if (0 <= r1 < rows and 0 <= c1 < cols and 0 <= r2 < rows and 0 <= c2 < cols and
                        grid[r1, c1] != -1 and grid[r2, c2] != -1):
                        if ((grid[r1, c1] + grid[r2, c2]) % 2 == 0 and 
                            (grid[r1, c1] + grid[r2, c2]) // 2 == val and 
                            abs(grid[r1, c1] - grid[r2, c2]) > 0):
                            features["arithmetic_sequence"] = 1.0
                
                # 中心獎勵
                center_dist = abs(r - center_r) + abs(c - center_c)
                max_center_dist = center_r + center_c
                features["center_bonus"] = 1.0 - (center_dist / max_center_dist) if max_center_dist > 0 else 0.0
                
                # 角落懲罰
                is_corner = ((r == 0 or r == rows-1) and (c == 0 or c == cols-1))
                features["corner_penalty"] = 1.0 if is_corner else 0.0
                
                # 孤立懲罰
                neighbor_count = 0
                for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                        neighbor_count += 1
                features["isolation_penalty"] = 1.0 if neighbor_count == 0 else 0.0
                
                # 計算總價值
                total_value = sum(features[key] * feature_weights[key] for key in features)
                best_value = max(best_value, total_value)
            
            scores[r, c] = best_value
    
    # 正規化到 [0, 1]
    min_possible = sum(w for w in feature_weights.values() if w < 0)
    max_possible = sum(w for w in feature_weights.values() if w > 0)
    if max_possible != min_possible:
        scores = self._normalize_scores(scores, min_possible, max_possible)
    
    return scores

def _compute_masked_skip_pattern(self, grid: np.ndarray, legal_mask: np.ndarray, legal_values: np.ndarray) -> np.ndarray:
    """GM19: 遮蔽號跳躍模式"""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols))
    
    for r in range(rows):
        for c in range(cols):
            if not legal_mask[r, c]:
                continue
                
            best_skip = 0.0
            for val in legal_values:
                # 水平跳躍模式 (A, ?, B) 其中 ? = val, B-A = 2*(val-A)
                if 0 < c < cols-1:
                    left, right = grid[r, c-1], grid[r, c+1]
                    if left != -1 and right != -1:
                        # 檢查是否形成跳躍模式
                        if (right - left) == 2 * (val - left) and (val - left) != 0:
                            best_skip = max(best_skip, 1.0)
                
                # 垂直跳躍模式
                if 0 < r < rows-1:
                    up, down = grid[r-1, c], grid[r+1, c]
                    if up != -1 and down != -1:
                        if (down - up) == 2 * (val - up) and (val - up) != 0:
                            best_skip = max(best_skip, 1.0)
            
            scores[r, c] = best_skip
    
    return scores

def _compute_internal_gap_bonus(self, grid: np.ndarray, legal_mask: np.ndarray, legal_values: np.ndarray) -> np.ndarray:
    """GM20: 內部間隙補全獎勵"""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols))
    
    for r in range(rows):
        for c in range(cols):
            if not legal_mask[r, c]:
                continue
                
            best_bonus = 0.0
            for val in legal_values:
                for dr, dc in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                    r1, c1 = r + dr, c + dc
                    r2, c2 = r - dr, c - dc
                    if (0 <= r1 < rows and 0 <= c1 < cols and 0 <= r2 < rows and 0 <= c2 < cols and
                        grid[r1, c1] != -1 and grid[r2, c2] != -1):
                        v1, v2 = grid[r1, c1], grid[r2, c2]
                        # 檢查是否為內部間隙（中點插值）
                        if ((v1 + v2) % 2 == 0 and (v1 + v2) // 2 == val and 
                            abs(v1 - v2) != 0 and min(v1, v2) < val < max(v1, v2)):
                            # 額外檢查是否真的在內部（不是邊界）
                            if 0 < r < rows-1 and 0 < c < cols-1:
                                best_bonus = max(best_bonus, 0.3)  # 內部獎勵更高
                            else:
                                best_bonus = max(best_bonus, 0.1)  # 邊界獎勵較低
            
            scores[r, c] = best_bonus
    
    return scores
```

# 使用示例和性能測試

def performance_test():
“”“性能測試函數”””
import time

```
# 創建測試網格
test_grid = np.random.randint(-1, 10, (10, 10))
test_grid[test_grid == 0] = -1  # 將0設為空位

brain = VectorizedBrainModules()

# 測試單個功能
start_time = time.time()
scores = brain.EXT_A2_Weighted_Proximity_Vec(test_grid)
single_time = time.time() - start_time

# 測試批量計算
start_time = time.time()
all_features = brain.compute_all_features(test_grid)
batch_time = time.time() - start_time

print(f"單個特徵計算時間: {single_time:.4f}秒")
print(f"所有特徵批量計算時間: {batch_time:.4f}秒")
print(f"計算的特徵數量: {len(all_features)}")

return brain, all_features
```

if **name** == “**main**”:
brain, features = performance_test()
print(“向量化大腦模組載入完成！”)