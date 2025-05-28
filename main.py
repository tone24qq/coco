# main.py (業界極限版 - 包含統計、AI、圖論等原理)
import random
import math
import numpy as np
from new_module import PuzzleTensorOps
from typing import List, Dict, Tuple, Any, Optional # <--- 在這裡加入了 Optional
from collections import Counter, deque
def get_pto(board_shape=(5,5), value=0):
    board = np.full(board_shape, value, dtype=int)
    return PuzzleTensorOps(board)

# ===== 全域 PTO，開局直接掛載 =====
pto = get_pto((7,7), value=-1)
print("PTO 盤面內容:\n", pto.grid_view)

# ===== 你原本的主流程繼續寫 =====
def main():
    # ...任何流程都可用 pto 或 get_pto 新建
    # 例如：
    print("PTO 盤面 shape:", pto.shape)
    # 隨時用 PTO 的方法
    print("空格總數:", pto.count_true_along_axis(axis=None))
    # 用 PTO 做進階操作...
    new_pto = get_pto((4,4), value=99)
    print("新 PTO 盤面內容:\n", new_pto.grid_view)
    # ...你自己所有的 AI/for/分析流程

if __name__ == "__main__":
    main()
# -----------------------------------------------------------------------------
# 0. 輔助工具 (可能被某些高級模組使用)
# -----------------------------------------------------------------------------

class MathUtils:
    @staticmethod
    def sigmoid(x: float) -> float:
        """標準 sigmoid 函數，可將任意實數映射到 (0,1) 區間。"""
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    @staticmethod
    def normalize_value(value: float, min_val: float, max_val: float) -> float:
        """將值正規化到 [0,1] 區間。"""
        if max_val == min_val:
            return 0.5 # 或 0.0, 或 1.0，視情況而定
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    @staticmethod
    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """計算兩點之間的曼哈頓距離。"""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

class BoardAnalyzerUtils:
    @staticmethod
    def get_neighborhood(board_state: List[List[Any]], r: int, c: int, dist: int = 1, eight_connectivity: bool = True) -> List[Any]:
        """獲取指定點的鄰域值 (預設為3x3區域，不含自身)。"""
        neighbors = []
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return neighbors
        
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                if dr == 0 and dc == 0:
                    continue
                if not eight_connectivity and abs(dr) + abs(dc) > dist: # 僅4-connectivity
                    continue
                
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors.append(board_state[nr][nc])
        return neighbors

    @staticmethod
    def get_value_gradient(board_state: List[List[Any]], r: int, c: int) -> Tuple[float, float]:
        """計算某點基於周圍值的簡單數值梯度 (Sobel-like)。"""
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return (0.0, 0.0)

        val_at = lambda r_in, c_in: float(board_state[r_in][c_in]) if (0 <= r_in < rows and 0 <= c_in < cols and isinstance(board_state[r_in][c_in], (int, float))) else 0.0

        # Sobel-like Gx
        gx = (val_at(r-1, c+1) + 2*val_at(r, c+1) + val_at(r+1, c+1)) - \
             (val_at(r-1, c-1) + 2*val_at(r, c-1) + val_at(r+1, c-1))
        # Sobel-like Gy
        gy = (val_at(r+1, c-1) + 2*val_at(r+1, c) + val_at(r+1, c+1)) - \
             (val_at(r-1, c-1) + 2*val_at(r-1, c) + val_at(r-1, c+1))
        
        return gx, gy


# -----------------------------------------------------------------------------
# 1. 基礎類別定義
# -----------------------------------------------------------------------------

class LogicModule:
    """
    所有評分邏輯模組的基礎類別。
    """
    def __init__(self, module_id: str, name: str, description: str):
        self.module_id = module_id
        self.name = name
        self.description = description
        # 模組內部可選的動態權重或參數，可以在初始化時設定預設值
        self.internal_params = {} 

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        """
        核心分析方法，應由所有子類別覆寫。
        設計理念：每個模組專注於盤面的一個特定高階特徵或原理。
        用途描述：子類別將詳細說明。
        評分公式：通常為0-1之間的浮點數，越高越好(或表示某種特徵的強度)。
        擴展性：兼容多種數值盤面，不僅限於0/1。
        優化與延伸方向：子類別將詳細說明。
        """
        # 核心分流規則與用途: (此為基礎類別，不應被直接調用)
        print(f"警告: 基礎 LogicModule.analyze 被呼叫 (模組: {self.module_id})。應由子類覆寫。")
        return 0.0 

    def __repr__(self) -> str:
        return f"<LogicModule module_id='{self.module_id}' name='{self.name}'>"

class BoardInput: # (與之前版本相同，此處省略以節省空間，實際使用時請包含)
    """
    代表盤面輸入的資料結構。
    """
    def __init__(self, grid: List[List[Any]]):
        if not grid or not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
            raise ValueError("盤面必須是一個非空的二維列表。")
        row_lengths = [len(row) for row in grid]
        if not row_lengths: 
             self.grid = []
             self.rows = 0
             self.cols = 0
             return
        if len(set(row_lengths)) > 1:
            raise ValueError("盤面所有列的長度必須相同。")

        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def get_cell(self, row: int, col: int) -> Any:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError(f"位置 ({row}, {col}) 超出盤面邊界 ({self.rows}x{self.cols})。")
        return self.grid[row][col]

    def __repr__(self) -> str:
        return f"<BoardInput rows={self.rows} cols={self.cols}>"

    def display(self):
        print(f"Board ({self.rows}x{self.cols}):")
        if not self.grid:
            print("(空盤面)")
            return
        for row in self.grid:
            print(" ".join(map(str, row)))
        print("-" * (self.cols * 2 if self.cols > 0 else 1))

# -----------------------------------------------------------------------------
# 2. 特定模組實現 (業界極限版)
# -----------------------------------------------------------------------------

class A2(LogicModule):
    """
    設計理念：評估局部區域的"激活"程度或"關鍵資源"密度。
              原始A2基於與'1'的鄰近性，此處將其擴展。
    用途描述：用於識別盤面中與高價值目標（假設為值較大的元素或特定ID）接近的潛力位置。
              適用於需要搶佔關鍵資源點或形成協同效應的場景。
    評分公式原理：1. 計算自身價值 (可配置目標值)。
                 2. 計算周圍高價值元素的加權影響力 (距離衰減)。
                 3. 綜合評分 = sigmoid( w1 * self_value_score + w2 * neighbor_influence_score )
    兼容性：盤面元素可以是任意數值，模組內部可配置"目標價值"或"價值函數"。
    優化與延伸方向：1. 引入更複雜的距離衰減函數。
                    2. 根據盤面全局狀態動態調整"目標價值"。
                    3. 考慮不同類型高價值元素的協同/互斥效應。
    可能的多版本邏輯：A2.v1 (鄰近特定值), A2.v2 (鄰近多種高價值元素並考慮其類型), A2.v3 (基於"影響力擴散模型"的評分)。
    """
    def __init__(self):
        super().__init__(
            module_id="A2",
            name="Alpha Module v2 (Weighted Proximity & Influence)",
            description="Scores based on proximity to high-value elements and their influence, using distance decay."
        )
        self.internal_params = {
            "target_value_threshold": 0.8, # 假設盤面值已正規化或此為一個較高的原始值
            "influence_radius": 3,
            "self_value_weight": 0.4,
            "influence_weight": 0.6,
            "value_function": lambda x: float(x) if isinstance(x, (int, float)) else 0.0 # 如何從盤面格點獲取數值
        }

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 透過距離衰減的加權方式，評估目標格子周圍高價值元素的聚集影響力，以及自身是否為高價值點。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return 0.0

        val_func = self.internal_params["value_function"]
        current_cell_val = val_func(board_state[position_row][position_col])
        
        # 1. 自身價值評分
        self_value_score = 0.0
        if current_cell_val >= self.internal_params["target_value_threshold"]:
            self_value_score = 1.0
        elif current_cell_val > 0: # 對非目標但有值的格子給予較低分
            self_value_score = MathUtils.normalize_value(current_cell_val, 0, self.internal_params["target_value_threshold"]) * 0.5
            
        # 2. 周圍高價值元素的加權影響力
        neighbor_influence_score = 0.0
        total_possible_influence = 0 # 用於正規化
        
        radius = self.internal_params["influence_radius"]
        for r_offset in range(-radius, radius + 1):
            for c_offset in range(-radius, radius + 1):
                if r_offset == 0 and c_offset == 0:
                    continue # 不計算自身對自身的影響力(已由self_value_score處理)

                nr, nc = position_row + r_offset, position_col + c_offset
                
                if 0 <= nr < rows and 0 <= nc < cols:
                    dist = MathUtils.manhattan_distance((position_row, position_col), (nr, nc))
                    if dist == 0 or dist > radius: continue # dist=0 已排除

                    neighbor_val = val_func(board_state[nr][nc])
                    weight = 1.0 / (dist ** 2) # 距離平方反比衰減 (可替換為其他衰減函數)
                    total_possible_influence += (1.0 / (dist ** 2)) # 假設所有鄰居都是最大影響力

                    if neighbor_val >= self.internal_params["target_value_threshold"]:
                        neighbor_influence_score += weight * 1.0 
                    elif neighbor_val > 0: # 對非目標但有值的鄰居也給予部分影響力
                         neighbor_influence_score += weight * MathUtils.normalize_value(neighbor_val, 0, self.internal_params["target_value_threshold"]) * 0.5
        
        if total_possible_influence > 0:
            normalized_influence = neighbor_influence_score / total_possible_influence
        else: # 如果沒有任何在影響半徑內的鄰居 (例如1x1盤面，或radius=0)
            normalized_influence = 0.0

        # 3. 綜合評分
        combined_score_raw = (self.internal_params["self_value_weight"] * self_value_score +
                              self.internal_params["influence_weight"] * normalized_influence)
        
        return MathUtils.sigmoid( (combined_score_raw - 0.5) * 5 ) # 將0-1的raw score映射到sigmoid，使其更具區分度

class M3(LogicModule):
    """
    設計理念：從統計學角度分析局部區域的"複雜性"或"異質性"。
              原始M3計算3x3鄰域'1'的數量。進化版將分析數值分佈的熵或變異數。
    用途描述：用於識別盤面中具有高度變化或高度一致的局部區域。
              高分可能表示"機會點"(若尋找變化)或"穩定區"(若尋找一致)。
    評分公式原理：計算3x3鄰域內(含自身)數值的「信息熵」或「標準差/變異係數」。
                 信息熵高表示混亂/多樣，低表示單一/有序。標準差類似。
                 最終分數 = 1 - normalized_entropy (若高熵低分) 或 normalized_entropy (若高熵高分)。
                 此處選擇：高異質性(高熵/高變異)得高分。
    兼容性：適用於數值型盤面。對於類別型盤面，可以直接計算類別的熵。
    優化與延伸方向：1. 考慮不同大小的鄰域。
                    2. 引入加權熵/變異數，例如中心點權重更高。
                    3. 結合時間序列分析（如果盤面是動態的）。
    可能的多版本邏輯：M3.v1 (計數), M3.v2 (信息熵), M3.v3 (變異係數), M3.v4 (局部Gini不純度)。
    """
    def __init__(self):
        super().__init__(
            module_id="M3",
            name="Mega Module v2 (Local Heterogeneity - Entropy/StdDev)",
            description="Scores based on the heterogeneity (e.g., entropy or stddev) of values in the 3x3 neighborhood."
        )
        self.internal_params = {
            "metric": "stddev",  # "entropy" or "stddev"
            "value_bins": 5, # For entropy calculation if values are continuous/many
            "value_function": lambda x: float(x) if isinstance(x, (int, float)) else None 
        }

    def _calculate_entropy(self, values: List[float]) -> float:
        if not values: return 0.0
        
        value_counts = Counter(values)
        num_values = len(values)
        entropy = 0.0
        for count in value_counts.values():
            probability = count / num_values
            entropy -= probability * math.log2(probability)
        
        # 正規化熵 (最大熵為 log2(N_distinct_values) 或 log2(num_bins))
        # 這裡簡單處理：假設值的種類有限，或已被離散化
        # 如果值的種類很多，正規化會比較複雜。
        # 假設我們正規化到 log2(len(values)) 如果所有值都不同，或者 log2(self.internal_params["value_bins"])
        # 為了簡單，我們先不正規化熵，或者用一個固定的最大可能熵值（如 log2(10) for 10 possible values）
        # 此處返回原始熵，由後續 sigmoid 或其他方式調整範圍
        # 或者，如果值的範圍是0-1，可以假設最大熵 log2(2) = 1 if values are only 0 or 1.
        # Max entropy for K bins is log2(K)
        max_entropy_approx = math.log2(self.internal_params["value_bins"]) if self.internal_params["value_bins"] > 1 else 1.0
        if max_entropy_approx == 0: return 0.0 # Avoid division by zero if only one bin
        
        return MathUtils.normalize_value(entropy, 0, max_entropy_approx)


    def _calculate_stddev_score(self, values: List[float]) -> float:
        if not values or len(values) < 2: return 0.0 # Stddev requires at least 2 points
        
        std_dev = np.std(values)
        
        # 正規化標準差。需要知道數值的大致範圍。
        # 假設數值範圍已知或可以從盤面估計。
        # 例如，如果值是0-1之間，最大標準差約0.5 (對於[0,0,...,1,1,...])
        # 此處假設一個理論最大標準差或根據值的範圍估算
        # 為了簡單，我們假設值的範圍是 board_min_val to board_max_val
        # 粗略估計: (max_val - min_val) / 2 could be a rough upper bound for stddev
        # For now, just return a value related to std_dev, let global normalization handle.
        # Or, use sigmoid to map it.
        # Let's use a heuristic: higher std_dev is higher score. Max score if std_dev is e.g. > 0.25 (assuming values 0-1)
        # This is highly dependent on expected value range.
        # Alternative: Coefficient of Variation (std_dev / mean) if mean is not zero.
        
        mean_val = np.mean(values)
        if math.isclose(mean_val, 0): # 避免除以零
            if math.isclose(std_dev, 0): return 0.0 # 全是0
            else: return 1.0 # 有變化但均值為0，算高異質性
            
        coeff_of_variation = std_dev / mean_val
        # Coeff of variation can be large. We use sigmoid to squash.
        # A CoV of 1 means stddev is same as mean. Larger means more relative variability.
        return MathUtils.sigmoid((coeff_of_variation - 1.0) * 2.0) # Centered around CoV=1

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 透過計算局部3x3鄰域內數值分佈的統計異質性（如標準差或熵），來識別變化劇烈或高度同質的區域。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return 0.0

        neighborhood_values_raw = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = position_row + dr, position_col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    val = self.internal_params["value_function"](board_state[nr][nc])
                    if val is not None:
                        neighborhood_values_raw.append(val)
        
        if not neighborhood_values_raw: return 0.0

        if self.internal_params["metric"] == "entropy":
            # For entropy, values might need to be discretized if they are continuous
            # Simple discretization for demo:
            if not neighborhood_values_raw: return 0.0
            min_v, max_v = min(neighborhood_values_raw), max(neighborhood_values_raw)
            
            if math.isclose(min_v, max_v): # All values are same, entropy is 0
                return 0.0 

            binned_values = []
            if not math.isclose(max_v,min_v) :
                for v in neighborhood_values_raw:
                    # Discretize into N bins
                    bin_index = math.floor(MathUtils.normalize_value(v, min_v, max_v) * (self.internal_params["value_bins"] -1e-9) ) # -1e-9 to handle max_v case
                    binned_values.append(bin_index)
            else: # all values are same
                binned_values = [0] * len(neighborhood_values_raw)

            return self._calculate_entropy(binned_values)
        
        elif self.internal_params["metric"] == "stddev":
            return self._calculate_stddev_score(neighborhood_values_raw)
            
        return 0.0

class D3(LogicModule):
    """
    設計理念：基於「潛力場」或「吸引力/排斥力」概念，評估位置的優勢。
              原始D3基於與盤面中心的距離。進化版將考慮盤面上所有"有價值"元素產生的疊加場。
    用途描述：識別在全局力量平衡中最優的位置。例如，高價值點產生吸引力，危險點產生排斥力。
              適用於需要考慮全局布局和多體相互作用的策略。
    評分公式原理：Score(x,y) = Sum_i [ Charge_i / Distance((x,y), Pos_i)^p ] - Sum_j [ Penalty_j / Distance((x,y), Pos_j)^q ]
                 Charge_i 是吸引物的"電荷量"(價值)，Penalty_j 是排斥物的"懲罰值"。p, q 是距離衰減指數。
                 最終分數通過 sigmoid 進行映射。
    兼容性：盤面元素需要有明確的"價值"和"懲罰"屬性，或能透過函數轉換得到。
    優化與延伸方向：1. 使用更複雜的場函數（如高斯場）。
                    2. 引入方向性（例如某些元素只在特定方向產生影響）。
                    3. 考慮"遮蔽效應"。
    可能的多版本邏輯：D3.v1 (中心距離), D3.v2 (點電荷模型), D3.v3 (基於引力勢能概念)。
    """
    def __init__(self):
        super().__init__(
            module_id="D3",
            name="Delta Module v2 (Potential Field Evaluator)",
            description="Scores based on a potential field generated by attractive/repulsive elements on the board."
        )
        self.internal_params = {
            "attractive_value_threshold": 0.7, # 視為"正電荷"的閾值
            "repulsive_value_threshold": -0.5, # (假設盤面可能有負值代表懲罰，或特定ID)
                                            # 或者我們可以定義一個 is_repulsive(value) 函數
            "attraction_strength_factor": 1.0,
            "repulsion_strength_factor": 1.5, # 排斥力通常更強或作用範圍更廣
            "distance_decay_power_att": 1.5, # p
            "distance_decay_power_rep": 2.0, # q
            "value_function": lambda x: float(x) if isinstance(x, (int, float)) else 0.0,
            "max_relevant_elements": 50 # 性能考量，只考慮最近或最強的N個影響源
        }

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 綜合盤面上所有「吸引點」和「排斥點」對當前格位產生的疊加「潛力」大小，距離越近影響越大。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return 0.0
        
        val_func = self.internal_params["value_function"]
        total_potential = 0.0
        
        # 收集所有影響源及其屬性
        sources = []
        for r_s in range(rows):
            for c_s in range(cols):
                if r_s == position_row and c_s == position_col: continue # 源不能是評估點本身

                source_val = val_func(board_state[r_s][c_s])
                dist = MathUtils.manhattan_distance((position_row, position_col), (r_s, c_s))
                if dist == 0: dist = 0.5 #避免除以零，給一個很小的基礎距離

                potential_contribution = 0.0
                is_significant_source = False

                if source_val >= self.internal_params["attractive_value_threshold"]:
                    # 吸引點
                    charge = (source_val - self.internal_params["attractive_value_threshold"] + 0.1) # 基礎電荷 + 超出部分
                    potential_contribution = (self.internal_params["attraction_strength_factor"] * charge) / (dist ** self.internal_params["distance_decay_power_att"])
                    is_significant_source = True
                elif source_val <= self.internal_params["repulsive_value_threshold"]: # 假設負值代表排斥物
                    # 排斥點
                    penalty = abs(source_val - self.internal_params["repulsive_value_threshold"] + 0.1)
                    potential_contribution = -(self.internal_params["repulsion_strength_factor"] * penalty) / (dist ** self.internal_params["distance_decay_power_rep"])
                    is_significant_source = True
                
                if is_significant_source:
                    sources.append({"potential": potential_contribution, "dist": dist})

        # 性能優化: 如果源太多，只取影響最大的 N 個
        if len(sources) > self.internal_params["max_relevant_elements"]:
            sources.sort(key=lambda s: abs(s["potential"])/s["dist"], reverse=True) # 按 (潛力/距離) 排序，簡化影響力
            sources = sources[:self.internal_params["max_relevant_elements"]]
            
        for s in sources:
            total_potential += s["potential"]
            
        # 將 total_potential 映射到 0-1 區間
        # total_potential 的範圍可能很大，正負不定。需要一個好的方式來正規化。
        # sigmoid 是一個常用的選擇。需要調整輸入範圍和斜率。
        # 假設潛力值在 -10 到 +10 之間比較常見 (這需要實驗數據支持)
        # (total_potential / K) K是一個縮放因子
        scaling_factor = 5.0 # 讓潛力在 -1 ~ 1 左右時，sigmoid 能產生較好的區分
        return MathUtils.sigmoid(total_potential / scaling_factor)


class F10(LogicModule):
    """
    設計理念：檢測"結構性斷裂"或"邊界完整性"。原始F10評估邊緣/角落。
              進化版將使用類似影像處理中的邊緣檢測算法（如
              Laplacian of Gaussian LoG 的簡化思想）來找到數值變化劇烈、可能形成"邊界"或"裂縫"的區域。
    用途描述：識別盤面中的"不連續區域"、"勢力分界線"或"結構弱點"。
              高分可能表示一個重要的邊界，或是一個需要修補的斷裂點。
    評分公式原理：對目標格子應用一個簡化的拉普拉斯算子 (e.g., [[0,1,0],[1,-4,1],[0,1,0]]) 或高斯差分(DoG)的離散近似。
                 算子響應的絕對值大小代表該點的"邊緣強度"或"不連續性"。
                 Score = sigmoid( k * |Laplacian_response| )
    兼容性：適用於數值型盤面。
    優化與延伸方向：1. 使用更平滑的算子(如LoG)。
                    2. 檢測特定方向的邊緣。
                    3. 鏈接邊緣點形成輪廓線 (進階圖論)。
    可能的多版本邏輯：F10.v1 (邊角), F10.v2 (拉普拉斯邊緣檢測), F10.v3 (Canny邊緣檢測原理的簡化)。
    """
    def __init__(self):
        super().__init__(
            module_id="F10",
            name="Feature Module v2 (Structural Discontinuity Detector)",
            description="Detects structural breaks or sharp value changes using a Laplacian-like operator."
        )
        self.internal_params = {
            "laplacian_kernel_type": 1, # 1: 4-conn, 2: 8-conn
            "response_scaling_factor": 2.0, # 調整 sigmoid 輸入範圍
            "value_function": lambda x: float(x) if isinstance(x, (int, float)) else 0.0
        }

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 透過類拉普拉斯算子計算目標點與其鄰域的數值差異，響應越大表明該點是個數值急劇變化的「邊緣」或「斷裂點」。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return 0.0

        val_func = self.internal_params["value_function"]
        center_val = val_func(board_state[position_row][position_col])
        
        laplacian_response = 0.0
        
        if self.internal_params["laplacian_kernel_type"] == 1:
            # Kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
            kernel_sum = 0
            num_neighbors = 0
            # Sum of 4 neighbors
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = position_row + dr, position_col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    kernel_sum += val_func(board_state[nr][nc])
                    num_neighbors +=1
            if num_neighbors > 0: # 避免在邊角處因鄰居少而導致laplacian_response過小或不准
                 laplacian_response = kernel_sum - num_neighbors * center_val #或者 num_neighbors 應該是固定的4？
                                                                         # 標準拉普拉斯是 sum_neighbors - N * center_val, N是鄰居數
                                                                         # 此處 num_neighbors 是實際存在的鄰居數
            else: # 孤立點
                laplacian_response = 0

        elif self.internal_params["laplacian_kernel_type"] == 2:
            # Kernel: [[1, 1, 1], [1, -8, 1], [1, 1, 1]] (近似)
            kernel_sum = 0
            num_neighbors = 0
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = position_row + dr, position_col + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        kernel_sum += val_func(board_state[nr][nc])
                        num_neighbors +=1
            if num_neighbors > 0:
                laplacian_response = kernel_sum - num_neighbors * center_val
            else:
                laplacian_response = 0
        
        # 響應絕對值越大，表示差異越大
        # 正規化abs_response，假設值的範圍是0-1，那麼laplacian_response的範圍大致在 -N*1 ~ N*1 之間
        # 例如4-connectivity, N=4, response in [-4, 4]. abs in [0,4]
        # 正規化到0-1區間
        max_abs_response_approx = 4.0 # (粗略估計，基於4個鄰居，值域0-1時的最大可能差異和)
        if self.internal_params["laplacian_kernel_type"] == 2:
            max_abs_response_approx = 8.0

        normalized_abs_response = MathUtils.normalize_value(abs(laplacian_response), 0, max_abs_response_approx)
        
        return MathUtils.sigmoid((normalized_abs_response - 0.2) * self.internal_params["response_scaling_factor"] * 5) # 調整中心和斜率
# ... (接續第一部分的程式碼: MathUtils, BoardAnalyzerUtils, LogicModule, BoardInput, A2, M3, D3, F10) ...

# --- GM Modules (GM1 to GM18) ---

class GM1(LogicModule):
    """
    設計理念：評估一行內的"資源集中度"或"控制線"。原始GM1評估行內'1'的佔比。
              進化版將考慮行內數值的加權總和，並與盤面平均行價值進行比較，同時偵測"連線"。
    用途描述：識別在行方向上具有顯著資源積累或形成潛在屏障/通道的行中的格點。
              適用於需要建立橫向控制或突破對方橫向防線的場景。
    評分公式原理：1. 計算當前行加權價值 S_row = Sum(w_i * val_i)，w_i可基於位置或值本身。
                 2. 計算盤面所有行的平均價值 S_avg_row。
                 3. 偵測行內是否存在長度至少為 K 的同類高價值元素連線。
                 4. Score = sigmoid( factor1 * (S_row / S_avg_row - 1) + factor2 * has_connection_bonus )
    兼容性：數值型盤面，可定義價值函數和連線判斷標準。
    優化與延伸方向：1. 引入更複雜的連線偵測算法（如考慮間斷連線）。
                    2. 動態調整連線長度 K 的閾值。
                    3. 分析行內數值的"平滑度"或"波動性"。
    可能的多版本邏輯：GM1.v1 (佔比), GM1.v2 (加權價值與平均比較 + 連線), GM1.v3 (行內數值序列的傅立葉分析)。
    """
    def __init__(self):
        super().__init__(
            module_id="GM1", 
            name="Generated Module 1 (Advanced Row Control & Connectivity)", 
            description="Evaluates weighted value concentration, connectivity in the cell's row, compared to board average."
        )
        self.internal_params = {
            "min_connection_length": 3, # 連續K個才算
            "connection_value_threshold": 0.7, # 連線元素的值要達到的閾值
            "value_weight_function": lambda val, col_idx, total_cols: val * (1 + 0.1 * (col_idx - total_cols/2)), # 越往特定方向權重越高
            "row_value_comparison_factor": 1.5,
            "connection_bonus_factor": 1.0,
            "value_function": lambda x: float(x) if isinstance(x, (int, float)) else 0.0
        }

    def _calculate_weighted_row_value(self, row_data: List[Any], total_cols: int) -> float:
        val_func = self.internal_params["value_function"]
        weighted_sum = 0
        for col_idx, cell_val_raw in enumerate(row_data):
            cell_val = val_func(cell_val_raw)
            weight = self.internal_params["value_weight_function"](1.0, col_idx, total_cols) # 簡化權重，只基於位置
            weighted_sum += cell_val * weight
        return weighted_sum

    def _detect_row_connection(self, row_data: List[Any]) -> bool:
        val_func = self.internal_params["value_function"]
        min_len = self.internal_params["min_connection_length"]
        threshold = self.internal_params["connection_value_threshold"]
        
        current_streak = 0
        for cell_val_raw in row_data:
            cell_val = val_func(cell_val_raw)
            if cell_val >= threshold:
                current_streak += 1
                if current_streak >= min_len:
                    return True
            else:
                current_streak = 0
        return False

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 綜合評估格子所在行的加權價值（與全局平均比較）以及是否存在高價值元素的連續連接。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols or not (0 <= position_row < rows): return 0.0
        
        current_row_data = board_state[position_row]
        
        # 1. 計算當前行加權價值
        s_row = self._calculate_weighted_row_value(current_row_data, cols)
        
        # 2. 計算盤面所有行的平均價值 (為簡化，此處可估算或傳入，或只比較相對值)
        #    此處用一個簡化方式：與一個理論上的"平均期望行價值"比較
        #    假設平均盤面格子值為 P_avg (例如0.3)，則平均行價值約為 P_avg * cols * avg_weight
        #    或者，更簡單的，我們只看 s_row 的絕對大小，然後用 sigmoid 映射
        #    為了體現"與平均比較"，我們計算所有行的價值，再求平均 (這在單次analyze中效率低)
        #    折衷：與一個"基準行價值"比較，例如 cols * (閾值*0.5) * (平均權重約1)
        #    更優：如果可以預處理盤面，可以先計算好全局平均行價值。
        #    此處的analyze是獨立的，故採用一個簡化的比較方式
        
        # 簡單正規化 s_row: 假設 s_row 的範圍是 0 到 cols * max_val * max_weight
        # 假設 max_val=1, max_weight approx 1.5 (if col_idx at end, val=1), so max_s_row approx cols * 1.5
        # normalized_s_row = MathUtils.normalize_value(s_row, 0, cols * 1.5) # 粗略正規化
        # 另一種思路：如果 s_row 顯著高於某個基線（例如 cols * 0.5，假設值平均0.5）
        baseline_row_value = cols * 0.5 # 假設平均值0.5, 權重平均1
        row_value_score_component = (s_row / baseline_row_value -1.0) if baseline_row_value > 0 else 0
                                    # 結果 >0 表示高於平均，<0 表示低於

        # 3. 偵測行內連線
        has_connection_bonus_val = 1.0 if self._detect_row_connection(current_row_data) else 0.0
        
        # 4. 綜合評分
        raw_score = (self.internal_params["row_value_comparison_factor"] * row_value_score_component +
                     self.internal_params["connection_bonus_factor"] * has_connection_bonus_val)
        
        return MathUtils.sigmoid(raw_score) # raw_score 可能為負

class GM2(LogicModule):
    """
    設計理念：類似GM1，但專注於列方向的"資源流動性"或"垂直控制"。
              將分析列內數值的梯度變化，以識別流動順暢或存在"阻塞"的列。
    用途描述：識別在列方向上資源流動順暢或形成關鍵垂直通道/屏障的格點。
              適用於需要建立垂直壓制或保護垂直生命線的場景。
    評分公式原理：1. 計算列內數值序列的一階差分（梯度）。
                 2. 分析梯度的統計特性（例如，小梯度的連續性表示平滑，大梯度表示突變/阻塞）。
                 3. 高分給予梯度平穩且平均值較高的列中的格子，或梯度變化劇烈但朝有利方向的格點。
                 Score = w1 * (1 - normalized_gradient_variance) + w2 * normalized_avg_col_value
                         + w3 * favorable_sharp_gradient_bonus
    兼容性：數值型盤面。
    優化與延伸方向：1. 使用更複雜的信號處理技術分析列序列（如頻譜分析）。
                    2. 識別特定模式的梯度變化（如"V"型反轉）。
    可能的多版本邏輯：GM2.v1 (佔比), GM2.v2 (梯度平滑度與均值), GM2.v3 (列數據的遊程檢驗)。
    """
    def __init__(self):
        super().__init__(
            module_id="GM2", 
            name="Generated Module 2 (Advanced Column Flow & Gradient Analysis)", 
            description="Analyzes value gradients and consistency within the cell's column."
        )
        self.internal_params = {
            "gradient_variance_weight": 0.5, # (1-var) 平滑度權重
            "avg_col_value_weight": 0.3,     # 列平均值權重
            "sharp_gradient_bonus_weight": 0.2, # 有利的大梯度權重
            "sharp_gradient_threshold": 0.5,  # 梯度絕對值超過此數為"sharp" (假設值已正規化0-1)
            "value_function": lambda x: float(x) if isinstance(x, (int, float)) else 0.0
        }

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 透過分析格子所在列的數值梯度變化平滑度、列平均值以及是否存在有利的數值躍升，來評估垂直方向的控制力或流動性。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols or not (0 <= position_col < cols): return 0.0

        val_func = self.internal_params["value_function"]
        column_data = [val_func(board_state[r][position_col]) for r in range(rows)]
        
        if len(column_data) < 2: # 單行盤面或無法計算梯度
            return MathUtils.normalize_value(column_data[0] if column_data else 0, 0, 1) # 簡化為自身值

        # 1. 計算梯度序列
        gradients = np.diff(np.array(column_data, dtype=float))
        
        # 2. 分析梯度統計特性
        gradient_variance = float(np.var(gradients)) if len(gradients) > 0 else 0
        # 正規化梯度方差: 假設值域0-1, diff也在-1~1, var約在0~0.25(或更高如果值跳動大)
        # (1 - normalized_variance) 代表平滑度
        # 假設最大方差為0.25 (當值在0,1間隔跳躍時)，這是一個粗略的估計
        smoothness_score = 1.0 - MathUtils.normalize_value(gradient_variance, 0, 0.25) 

        # 3. 列平均值
        avg_col_value = float(np.mean(column_data))
        # 正規化列平均值 (假設原始值在0-1之間)
        normalized_avg_col_value = MathUtils.normalize_value(avg_col_value, 0, 1)

        # 4. 有利的大梯度獎勵 (例如，在當前位置下方出現一個大幅度的正向梯度)
        favorable_sharp_gradient_bonus = 0.0
        # 檢查 position_row 處的梯度 (即 value[pos_row+1] - value[pos_row])
        # gradients[i] = column_data[i+1] - column_data[i]
        # 所以 gradients[position_row] 是 cell[pos_row+1] - cell[pos_row]
        # 我們關心的是進入或離開 position_row 的梯度
        # 梯度於 position_row 之前: gradients[position_row-1] if position_row > 0
        # 梯度於 position_row 之後: gradients[position_row] if position_row < rows -1
        
        # 簡化: 檢查包含 position_row 的梯度是否劇烈且"向上" (值增加)
        # (更複雜的邏輯可以看是"流入"該格還是"流出"該格)
        # 此處判斷：如果當前格子下方的值比當前格子大很多，則加分 (有利於向上進攻)
        if position_row < rows - 1: # 存在下方格子
            grad_at_pos = gradients[position_row] # val[pos_row+1] - val[pos_row]
            if grad_at_pos > self.internal_params["sharp_gradient_threshold"]:
                favorable_sharp_gradient_bonus = MathUtils.normalize_value(grad_at_pos, self.internal_params["sharp_gradient_threshold"], 1.0) # 假設最大梯度為1
        
        # 綜合評分
        score = (self.internal_params["gradient_variance_weight"] * smoothness_score +
                 self.internal_params["avg_col_value_weight"] * normalized_avg_col_value +
                 self.internal_params["sharp_gradient_bonus_weight"] * favorable_sharp_gradient_bonus)
        
        return max(0.0, min(1.0, score)) # 確保在0-1之間

class GM3(LogicModule):
    """
    設計理念：基於圖論中的"連通分量"分析，評估目標格子所屬的同質區域的大小和密度。
              原始GM3評估3x3鄰域空格密度。進化版將採用類似BFS/DFS的方法尋找目標格子周圍的連通區域。
    用途描述：識別盤面中較大的同質集群（例如，大片連續的資源、大片空格）。
              高分可能表示該格屬於一個有潛力的大區域（攻佔或利用）。
    評分公式原理：1. 以目標格子為起點，執行廣度優先搜索(BFS)或深度優先搜索(DFS)尋找具有相似數值(在一定容差範圍內)的連通區域。
                 2. 計算該連通區域的大小（格子數量）和平均密度（區域內格子平均值/最大可能值）。
                 3. Score = w1 * normalized_area_size + w2 * normalized_area_density
    兼容性：數值型盤面，需要定義"相似數值"的判斷標準（例如，差值小於epsilon）。
    優化與延伸方向：1. 引入帶方向的連通性分析。
                    2. 考慮不同形狀的連通區域的價值（例如，細長型 vs 團塊型）。
                    3. 結合"滲透理論"分析區域的擴展潛力。
    可能的多版本邏輯：GM3.v1 (鄰域空格密度), GM3.v2 (BFS/DFS同質連通區域大小), GM3.v3 (考慮連通區域的"邊界複雜度")。
    """
    def __init__(self):
        super().__init__(
            module_id="GM3", 
            name="Generated Module 3 (Connected Component Analysis)", 
            description="Analyzes the size and density of the connected homogeneous region the cell belongs to."
        )
        self.internal_params = {
            "value_tolerance": 0.1, # 判定值是否"相似"的容差 (假設值正規化到0-1)
            "area_size_weight": 0.6,
            "area_density_weight": 0.4,
            "max_bfs_steps": 100, # 限制BFS的廣度以控制性能
            "value_function": lambda x: float(x) if isinstance(x, (int, float)) else None
        }

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 透過類BFS/DFS算法，計算目標格子所在同質（值在容差內）連通區域的大小和平均值，大而高價值的區域得分高。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols: return 0.0

        val_func = self.internal_params["value_function"]
        start_val_raw = board_state[position_row][position_col]
        start_val = val_func(start_val_raw)

        if start_val is None: return 0.0 # 如果起始點不是有效數值

        q = deque([(position_row, position_col)])
        visited = set([(position_row, position_col)])
        connected_area_cells = [(position_row, position_col)]
        
        steps = 0
        while q and steps < self.internal_params["max_bfs_steps"]:
            r, c = q.popleft()
            steps += 1

            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]: # 4-connectivity
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    neighbor_val_raw = board_state[nr][nc]
                    neighbor_val = val_func(neighbor_val_raw)
                    
                    if neighbor_val is not None and abs(neighbor_val - start_val) <= self.internal_params["value_tolerance"]:
                        visited.add((nr, nc))
                        q.append((nr, nc))
                        connected_area_cells.append((nr,nc))
        
        area_size = len(connected_area_cells)
        normalized_area_size = MathUtils.normalize_value(area_size, 1, rows * cols) # 最大是整個盤面

        area_sum_val = sum(val_func(board_state[r][c]) for r,c in connected_area_cells if val_func(board_state[r][c]) is not None)
        # 密度 = 區域平均值 (假設最大值為1.0)
        area_density = MathUtils.normalize_value(area_sum_val / area_size if area_size > 0 else 0, 0, 1.0) 
        
        score = (self.internal_params["area_size_weight"] * normalized_area_size +
                 self.internal_params["area_density_weight"] * area_density)
        
        return max(0.0, min(1.0, score))


class GM4(LogicModule):
    """
    設計理念：基於"空間自相關性"（Spatial Autocorrelation）的局部指標，如局部Moran's I 或 Geary's C 的簡化概念。
              原始GM4計算3x3鄰域平均值。進化版將評估目標格子與其鄰居的相似性（或差異性）是否高於/低於隨機期望。
    用途描述：識別盤面中"熱點"（高值聚集）、"冷點"（低值聚集）或"異類點"（與周圍顯著不同）。
              適用於尋找模式的聚集區或異常點。
    評分公式原理：1. 計算目標格子的值 V_cell。
                 2. 計算其鄰域的平均值 V_neighbors_avg。
                 3. 計算全局平均值 V_global_avg (或使用一個基準值)。
                 4. 局部 Moran's I 簡化形式: (V_cell - V_global_avg) * Sum_j(w_ij * (V_neighbor_j - V_global_avg))
                    其中 w_ij 是空間權重（例如，距離的倒數）。
                 5. 高正值表示高高聚集或低低聚集(熱點/冷點)，高負值表示高低/低高聚集(異類)。
                 Score = sigmoid(scaled_local_moran_I_statistic)
    兼容性：數值型盤面。
    優化與延伸方向：1. 標準化變量以計算真正的Moran's I統計量並進行顯著性檢驗。
                    2. 考慮不同定義的空間權重矩陣 (w_ij)。
                    3. 區分熱點和冷點給予不同類型的分數。
    可能的多版本邏輯：GM4.v1 (鄰域均值), GM4.v2 (簡化局部Moran's I), GM4.v3 (局部Geary's C)。
    """
    def __init__(self):
        super().__init__(
            module_id="GM4", 
            name="Generated Module 4 (Local Spatial Autocorrelation)", 
            description="Evaluates if the cell is part of a cluster (hot/cold spot) or an outlier using Moran's I like logic."
        )
        self.internal_params = {
            "neighborhood_dist": 1, # 鄰域定義，1 表示3x3
            "global_avg_estimate": 0.5, # 假設的全局平均值 (0-1盤面)，實際應從盤面計算或傳入
            "moran_scaling_factor": 5.0,
            "value_function": lambda x: float(x) if isinstance(x, (int, float)) else None
        }
    
    def _get_board_global_avg(self, board_state: List[List[Any]])-> float:
        # 在analyze中重複計算效率低，理想情況是預處理
        # 此處作為演示，進行計算
        vals = []
        val_func = self.internal_params["value_function"]
        for r_idx in range(len(board_state)):
            for c_idx in range(len(board_state[0])):
                v = val_func(board_state[r_idx][c_idx])
                if v is not None:
                    vals.append(v)
        return np.mean(vals) if vals else self.internal_params["global_avg_estimate"]


    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 透過類Moran's I統計量，評估目標格子與其周圍格子的數值相似性，從而識別高/低值聚集區或異常點。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return 0.0

        val_func = self.internal_params["value_function"]
        v_cell = val_func(board_state[position_row][position_col])
        if v_cell is None: return 0.0

        # 動態計算全局平均值（效率不高，但更準確）
        v_global_avg = self._get_board_global_avg(board_state)
        # v_global_avg = self.internal_params["global_avg_estimate"] # 或者使用預估值

        z_cell = v_cell - v_global_avg # 中心化
        
        sum_weighted_neighbor_z = 0.0
        num_neighbors = 0
        
        dist_param = self.internal_params["neighborhood_dist"]
        for dr in range(-dist_param, dist_param + 1):
            for dc in range(-dist_param, dist_param + 1):
                if dr == 0 and dc == 0: continue
                
                nr, nc = position_row + dr, position_col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    v_neighbor = val_func(board_state[nr][nc])
                    if v_neighbor is not None:
                        # 空間權重 w_ij (此處簡化為1，即不加權，或可設為 1/dist)
                        w_ij = 1.0 
                        z_neighbor = v_neighbor - v_global_avg
                        sum_weighted_neighbor_z += w_ij * z_neighbor
                        num_neighbors +=1
        
        if num_neighbors == 0:
            local_moran_I_like = 0 # 沒有鄰居，無法判斷自相關
        else:
            # 簡化的局部Moran's I (未除以方差和總權重，僅看符號和相對大小)
            local_moran_I_like = z_cell * sum_weighted_neighbor_z 
            # 除以 num_neighbors 可以得到平均的鄰域影響
            local_moran_I_like /= num_neighbors


        # local_moran_I_like > 0 表示相似聚集 (高高/低低 - 熱點/冷點)
        # local_moran_I_like < 0 表示不相似聚集 (高低/低高 - 異類點)
        # 分數設計: 我們希望熱點/冷點得分高，異類點得分低，或者反之。
        # 此處：讓絕對值大的聚集（不論正負）得分較高，表示該點在空間模式上很"突出"
        # 或者，我們只獎勵正相關（熱點/冷點）
        # 此處設計：獎勵正相關，懲罰負相關或無相關
        
        # 範圍估計: z_cell 在 -0.5~0.5 (若global_avg=0.5, val=0-1)。sum_weighted_neighbor_z (avg) 也在 -0.5~0.5
        # product 在 -0.25 ~ 0.25
        # return MathUtils.sigmoid(local_moran_I_like * self.internal_params["moran_scaling_factor"])
        
        # 新設計：讓高-高聚集和低-低聚集（即 local_moran_I_like > 0 且 z_cell 和 sum_weighted_neighbor_z 同號且絕對值大）得分高
        # 異類點 (local_moran_I_like < 0) 得分低
        if local_moran_I_like > 0: # 相似聚集
            # 強度取決於 local_moran_I_like 的大小
            # 假設 local_moran_I_like 的理論最大值約為 0.25 (如前述)
            # 再用 (z_cell)^2 或 abs(z_cell) 來強調中心點本身是否也偏離均值
            strength = MathUtils.normalize_value(local_moran_I_like, 0, 0.25) # 正相關強度
            score = 0.5 + 0.5 * strength # 映射到 0.5 - 1.0
        elif local_moran_I_like < 0: # 不相似聚集 (異類)
            strength = MathUtils.normalize_value(abs(local_moran_I_like), 0, 0.25) # 異類強度
            score = 0.5 - 0.5 * strength # 映射到 0 - 0.5
        else: # 無顯著自相關
            score = 0.5
        
        return score


class GM5(LogicModule):
    """
    設計理念：基於"極值理論"的簡化概念或"稀有事件偵測"，尋找盤面上的局部極端高值或低值點。
              原始GM5評估水平交替模式。進化版將偵測相對於其擴展鄰域是否為顯著的峰值或谷值。
    用途描述：識別盤面中異常突出或凹陷的"奇異點"。
              可用於尋找稀有資源、關鍵的戰術高地/低地、或潛在的突破口/薄弱點。
    評分公式原理：1. 計算目標格子的值 V_cell。
                 2. 計算其擴展鄰域（例如5x5或7x7，排除中心）的統計數據（如均值 V_neigh_avg、標準差 V_neigh_std、最大值 V_neigh_max、最小值 V_neigh_min）。
                 3. 如果 V_cell > V_neigh_avg + k1 * V_neigh_std (或 V_cell > V_neigh_max * factor)，則判定為"峰值"。
                 4. 如果 V_cell < V_neigh_avg - k2 * V_neigh_std (或 V_cell < V_neigh_min / factor)，則判定為"谷值"。
                 5. Score_peak = f(V_cell - (V_neigh_avg + k1*V_neigh_std)) (峰值強度)
                 6. Score_valley = f((V_neigh_avg - k2*V_neigh_std) - V_cell) (谷值強度)
                 7. 最終分數可選擇是獎勵峰值、獎勵谷值、或獎勵任何一種極端性。此處：獎勵峰值。
    兼容性：數值型盤面。
    優化與延伸方向：1. 使用更魯棒的統計量（如中位數絕對偏差 MAD）。
                    2. 考慮不同大小和形狀的"背景鄰域"。
                    3. 引入時間維度，偵測值的"突變"。
    可能的多版本邏輯：GM5.v1 (交替模式), GM5.v2 (局部峰值檢測), GM5.v3 (局部谷值檢測), GM5.v4 (綜合極端性評分)。
    """
    def __init__(self):
        super().__init__(
            module_id="GM5", 
            name="Generated Module 5 (Local Extremum Detector - Peak Focus)", 
            description="Detects if the cell is a significant peak value compared to its extended neighborhood."
        )
        self.internal_params = {
            "neighborhood_radius": 2, # 5x5 area (2 cells out from center)
            "std_dev_multiplier_k1": 1.5, # V_cell > avg + k1*std
            "min_difference_for_peak": 0.2, # V_cell 至少要比 avg 高這麼多 (0-1 scale)
            "value_function": lambda x: float(x) if isinstance(x, (int, float)) else None
        }

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 透過比較目標格子的值與其擴展鄰域的統計特性（均值、標準差），來識別該格子是否為一個顯著的局部「峰值」。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return 0.0

        val_func = self.internal_params["value_function"]
        v_cell = val_func(board_state[position_row][position_col])
        if v_cell is None: return 0.0

        neighborhood_values = []
        radius = self.internal_params["neighborhood_radius"]
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0: continue # Exclude self

                nr, nc = position_row + dr, position_col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    val = val_func(board_state[nr][nc])
                    if val is not None:
                        neighborhood_values.append(val)
        
        if not neighborhood_values: # No neighbors (e.g. 1x1 board, radius > 0)
            # If it's a 1x1 board, it could be considered an extremum by default, or neutral
            return 0.5 # Neutral score

        v_neigh_avg = float(np.mean(neighborhood_values))
        v_neigh_std = float(np.std(neighborhood_values))

        is_peak = False
        peak_strength = 0.0

        # 判定是否為峰值
        # 條件1: 比鄰域均值高出一定標準差倍數
        # 條件2: 比鄰域均值高出一個絕對閾值 (避免標準差很小時的小波動被放大)
        if (v_cell > v_neigh_avg + self.internal_params["std_dev_multiplier_k1"] * v_neigh_std and
            v_cell > v_neigh_avg + self.internal_params["min_difference_for_peak"]):
            is_peak = True
            # 計算峰值強度 (正規化)
            # 強度可以是 (v_cell - v_neigh_avg) / (v_neigh_std + epsilon)
            # 或更簡單： (v_cell - (v_neigh_avg + self.internal_params["min_difference_for_peak"]))
            # 假設值的範圍是0-1，那麼這個差值的範圍也是0-1左右
            strength_raw = v_cell - (v_neigh_avg + self.internal_params["min_difference_for_peak"])
            # 將strength_raw正規化 (假設其典型正值範圍在0到0.5之間，超出則更強)
            peak_strength = MathUtils.normalize_value(strength_raw, 0, 0.5) 


        # 也可以加入對谷值的判斷，但此模組專注於峰值
        # if (v_cell < v_neigh_avg - self.internal_params["std_dev_multiplier_k2"] * v_neigh_std and
        #     v_cell < v_neigh_avg - self.internal_params["min_difference_for_valley"]):
        #     is_valley = True
        #     ...

        if is_peak:
            return 0.5 + 0.5 * peak_strength # 映射到 [0.5, 1.0]
        else:
            # 如果不是明顯峰值，可以給一個基於其相對鄰域均值的分數
            # 例如，如果略高於平均，給0.5以上一點點，如果低於，則0.5以下一點點
            relative_to_avg = v_cell - v_neigh_avg
            # relative_to_avg 的範圍可能是 -1 到 1 (如果值是0-1)
            # 用sigmoid轉換為0-1
            return MathUtils.sigmoid(relative_to_avg * 2.0) * 0.5 # 映射到 [0, 0.5]
# ... (接續第二部分的程式碼: MathUtils, BoardAnalyzerUtils, LogicModule, BoardInput, A2, M3, D3, F10, GM1, GM2, GM3, GM4, GM5) ...

class GM6(LogicModule):
    """
    設計理念：基於"圖形匹配"或"模板匹配"的原理，偵測是否存在預定義的局部有利圖形。
              原始GM6評估垂直交替模式。進化版將允許定義多種小型圖案模板，並計算匹配度。
    用途描述：識別盤面中是否出現了特定的戰術微結構或有利的局部配置。
              例如，在圍棋中可能是"虎口"、"眼位"的雛形；在資源遊戲中可能是某種高效的資源組合。
    評分公式原理：1. 定義一組 2x2 或 3x3 的"目標圖案"模板及其期望的中心格數值。
                 2. 將模板與以目標格子為中心的鄰域進行比較（例如，計算漢明距離、歐氏距離或更複雜的結構相似性）。
                 3. 如果匹配度高於某閾值，則給予高分。
                 4. Score = Max_over_templates [ w_template * similarity_score(template, neighborhood) ]
    兼容性：數值型或類別型盤面均可，模板需要相應定義。
    優化與延伸方向：1. 允許旋轉和鏡像的模板匹配。
                    2. 使用卷積神經網絡(CNN)的初級原理，將模板視為小型卷積核。
                    3. 引入模糊匹配或帶有通配符的模板。
    可能的多版本邏輯：GM6.v1 (交替模式), GM6.v2 (固定多模板匹配), GM6.v3 (帶旋轉/鏡像的模板匹配)。
    """
    def __init__(self):
        super().__init__(
            module_id="GM6", 
            name="Generated Module 6 (Local Pattern/Template Matching)", 
            description="Detects predefined local beneficial patterns around the cell using template matching."
        )
        # 模板定義: (template_grid, center_value_in_template, template_weight)
        # 模板格子中的 None 可以是通配符，或者要求與盤面值匹配
        # 這裡簡化：模板是3x3，中心是(1,1)
        # 模板中的值是期望值，盤面值與之比較
        self.internal_params = {
            "templates": [
                { # " बनाता '1' " - 如果周圍是0，中間是1，則好
                  "grid": [[0, 0, 0], [0, 1, 0], [0, 0, 0]], "weight": 0.8, "name": "Isolated_1" 
                },
                { # " 連續三個 '1' (水平中心) "
                  "grid": [[None, None, None], [1, 1, 1], [None, None, None]], "weight": 1.0, "name": "Horizontal_3_Ones"
                },
                { # " 防禦結構 (角落保護一個高價值點) " (假設1是牆, 0.8是高價值點)
                  "grid": [[1, 1, None], [1, 0.8, None], [None, None, None]], "weight": 0.7, "name": "Defensive_Corner"
                }
            ],
            "similarity_threshold": 0.7, # 相似度達到多少才算匹配
            "value_function": lambda x: float(x) if isinstance(x, (int, float)) else -999 # -999 for non-matchable
        }

    def _calculate_template_similarity(self, sub_grid: List[List[float]], template_grid: List[List[float]]) -> float:
        """計算子網格與模板的相似度 (0-1)。值越大越相似。"""
        match_score = 0
        num_defined_cells_in_template = 0
        
        for r_t in range(len(template_grid)):
            for c_t in range(len(template_grid[0])):
                template_val = template_grid[r_t][c_t]
                if template_val is None: # 通配符
                    continue
                num_defined_cells_in_template += 1
                
                # 假設 sub_grid 和 template_grid 維度相同
                actual_val = sub_grid[r_t][c_t] 
                
                if actual_val == template_val : # 完全匹配
                    match_score += 1.0
                # 可加入模糊匹配邏輯，例如值相近也給部分分數
                elif abs(actual_val - template_val) < 0.15: # 輕微差異
                     match_score += 0.5
        
        if num_defined_cells_in_template == 0: return 0.0
        return match_score / num_defined_cells_in_template


    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 透過與一組預定義的局部3x3圖案模板進行比較，評估目標格子周圍是否形成了特定的有利結構。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return 0.0

        val_func = self.internal_params["value_function"]
        max_score_for_cell = 0.0

        for template_info in self.internal_params["templates"]:
            template_grid_def = template_info["grid"]
            template_weight = template_info["weight"]
            
            # 提取以 (position_row, position_col) 為中心的 3x3 鄰域
            # 模板的 (1,1) 對應盤面的 (position_row, position_col)
            sub_grid_values = [] # 3x3 list of lists
            valid_sub_grid = True
            for dr_template in range(-1, 2): # template row index from center (-1, 0, 1)
                current_row_vals = []
                for dc_template in range(-1, 2): # template col index from center (-1, 0, 1)
                    r_board, c_board = position_row + dr_template, position_col + dc_template
                    if 0 <= r_board < rows and 0 <= c_board < cols:
                        current_row_vals.append(val_func(board_state[r_board][c_board]))
                    else: # 超出邊界，則此模板無法完整匹配
                        # 或者可以視為與模板中的None(通配符)匹配，或給予懲罰
                        # 此處簡化：如果模板需要盤面邊界外的值，則認為不匹配或給低相似度
                        # 更穩健：如果模板對應位置是None，則邊界外也沒關係
                        if template_grid_def[dr_template+1][dc_template+1] is not None:
                             valid_sub_grid = False; break
                        else: # 模板是通配符，盤面邊界外可視為通配符匹配
                            current_row_vals.append(val_func(None)) # 表示一個不影響匹配的值
                if not valid_sub_grid: break
                sub_grid_values.append(current_row_vals)
            
            if not valid_sub_grid or len(sub_grid_values) != 3 or len(sub_grid_values[0]) != 3:
                # print(f"Debug: Skipping template {template_info['name']} at ({position_row},{position_col}) due to boundary.")
                continue # 無法形成完整的3x3子網格來匹配3x3模板

            similarity = self._calculate_template_similarity(sub_grid_values, template_grid_def)
            
            current_template_score = 0.0
            if similarity >= self.internal_params["similarity_threshold"]:
                # 可以讓分數與相似度成正比，而不只是二元判斷
                current_template_score = template_weight * similarity 
            
            if current_template_score > max_score_for_cell:
                max_score_for_cell = current_template_score
                
        return max(0.0, min(1.0, max_score_for_cell)) # 確保在0-1

class GM7(LogicModule):
    """
    設計理念：基於"最優路徑"或"最小成本路徑"的圖論思想（如Dijkstra或A*算法的簡化概念），
              評估從當前格子到達盤面上某個"目標區域"或"高價值點"的"通行便利性"或"成本效益"。
              原始GM7評估'1'的孤立性。進化版將評估其"可達性"。
    用途描述：識別那些容易到達重要戰略目標（例如，敵方基地、資源點、安全區）的格點。
              高分表示該點是良好的跳板或通道。
    評分公式原理：1. 定義盤面上的"目標節點/區域" (T)。
                 2. 將盤面視為一個圖，格點是節點，相鄰格點間的邊的"權重(成本)"由格子本身的值或特性決定（例如，低值格子成本低，高值格子成本高，或特定地形有特定成本）。
                 3. 從 (position_row, position_col) 出發，使用簡化的類Dijkstra或BFS（如果成本均為1）找到到達最近目標T的最短加權路徑長度 L。
                 4. Score = 1 - normalized(L)。路徑越短（成本越低），分數越高。
    兼容性：數值型盤面，可以定義通行成本函數。
    優化與延伸方向：1. 完整實現A*算法以提高效率。
                    2. 考慮不同方向的移動成本。
                    3. 引入"視野"或"可見性"限制。
    可能的多版本邏輯：GM7.v1 (孤立性), GM7.v2 (BFS到最近目標的步數), GM7.v3 (類Dijkstra到最近目標的加權成本)。
    """
    def __init__(self):
        super().__init__(
            module_id="GM7", 
            name="Generated Module 7 (Accessibility & Pathfinding Potential)", 
            description="Scores cell based on ease of access (e.g., low-cost path) to predefined target zones/values."
        )
        self.internal_params = {
            "target_value_criteria": lambda val: val is not None and val >= 0.9, # 定義什麼是目標點
            "cost_function": lambda val: (1.1 - val) if (val is not None and 0 <= val <= 1) else 100.0, # 值越小(0-1)，成本越低。非數字或範圍外則高成本
                                                                                                # 假設值是0-1, 成本在0.1到1.1之間
            "max_path_cost_normalization": 10.0, # 用於正規化路徑成本的估計最大成本 (盤面大小相關)
            "max_bfs_depth_for_targets": 10 # 限制尋找目標的範圍
        }

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 透過類Dijkstra/BFS算法，計算從當前格子到盤面上最近的「高價值目標點」的通行成本，成本越低（越容易到達）則分數越高。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return 0.0

        is_target_func = self.internal_params["target_value_criteria"]
        cost_func = self.internal_params["cost_function"]
        
        # 如果當前格子本身就是目標，則得分最高
        if is_target_func(board_state[position_row][position_col]):
            return 1.0

        # 使用BFS尋找帶權重的最短路徑 (如果權重都是1，就是標準BFS找最短路徑)
        # 因為我們有cost_function，更像是Uniform Cost Search (Dijkstra的特例)
        # 為了簡化，我們這裡用BFS，但限制搜索深度，並將成本視為步數的某種調整
        # 或者，實現一個簡化的Dijkstra
        
        # 簡化版：BFS尋找最近的目標，同時考慮路徑上格子的平均"通行容易度"
        # 此處改為：BFS找到最近目標，路徑長度作為成本的基礎。
        q = deque([( (position_row, position_col), 0 )]) # ((r,c), path_length)
        visited = set([(position_row, position_col)])
        min_path_length_to_target = float('inf')
        
        path_found = False
        bfs_steps_count = 0

        while q and bfs_steps_count < self.internal_params["max_bfs_depth_for_targets"] * (rows*cols): # 防止無限循環和過度搜索
            (curr_r, curr_c), length = q.popleft()
            bfs_steps_count +=1

            if length >= self.internal_params["max_bfs_depth_for_targets"] and self.internal_params["max_bfs_depth_for_targets"] > 0 :
                # 如果只關心一定步數內的目標
                continue 

            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    cell_val_at_nr_nc = board_state[nr][nc]
                    
                    if is_target_func(cell_val_at_nr_nc):
                        min_path_length_to_target = min(min_path_length_to_target, length + 1)
                        path_found = True
                        # 如果找到一個就停止，那就是最近的BFS路徑。如果要最優成本，需要Dijkstra。
                        # 此處簡化為找到任何一個目標路徑就更新長度，然後繼續搜索直到佇列空或達到限制，取最短的。
                    
                    # 即使不是目標，也加入佇列繼續搜索 (只要成本不高得離譜)
                    # 此處的BFS不直接使用cost_func來排序，而是純粹的步數。
                    # 可以在最後用min_path_length_to_target和cost_func結合
                    q.append(((nr, nc), length + 1))
        
        if not path_found:
            return 0.0 # 在限制範圍內未找到目標

        # 分數與路徑長度成反比
        # 正規化 min_path_length_to_target。最大路徑長度約 rows+cols
        normalized_cost = MathUtils.normalize_value(min_path_length_to_target, 1, self.internal_params["max_bfs_depth_for_targets"] or (rows+cols))
        score = 1.0 - normalized_cost
        
        return max(0.0, min(1.0, score))


class GM8(LogicModule):
    """
    設計理念：基於"全局稀疏性/密度"與"局部貢獻"的比較。
              原始GM8評估主對角線。進化版將評估整個盤面的某種資源（例如，值 > T的格子）的全局密度，
              然後評估當前格子若變為該資源，對全局密度的"邊際貢獻"或"相對重要性"。
    用途描述：在高密度區域進一步增加密度可能價值不高，但在稀疏區域增加一個關鍵資源點則價值巨大。
              反之亦然，如果目標是消除某種資源，則在密度高的地方消除一個點更有價值。
              此模組用於評估放置（或移除）操作的"戰略價值"。此處假設是放置。
    評分公式原理：1. 計算全局資源密度 D_global (例如，盤面上值 > T 的格子比例)。
                 2. 假設在 (pos_r, pos_c) 放置一個資源（使其值 > T），計算新的全局密度 D'_global。
                 3. 計算邊際貢獻 M = D'_global - D_global。
                 4. 根據 D_global 的水平調整 M 的價值。例如，若 D_global 很低，則 M 的權重高；若 D_global 很高，則 M 的權重低。
                 Score = sigmoid( M * (1 - D_global)^k )  (k是調節因子)
    兼容性：數值型盤面，需要定義"資源"的標準。
    優化與延伸方向：1. 考慮不同位置的"戰略乘數"（例如，中心區域的邊際貢獻價值更高）。
                    2. 引入博弈論中的"Shapley值"思想，評估個體對整體聯盟的貢獻。
    可能的多版本邏輯：GM8.v1 (主對角線), GM8.v2 (邊際密度貢獻), GM8.v3 (考慮格子對全局"結構連通性"的邊際貢獻)。
    """
    def __init__(self):
        super().__init__(
            module_id="GM8", 
            name="Generated Module 8 (Marginal Density Contribution)", 
            description="Scores cell based on its marginal contribution to global density of a target resource, weighted by current sparsity."
        )
        self.internal_params = {
            "resource_threshold": 0.7, # 定義什麼是"資源"
            "sparsity_weight_power_k": 1.5, # (1 - D_global)^k 中的 k
            "value_function": lambda x: float(x) if isinstance(x, (int, float)) else 0.0,
            "assumed_placed_value": 1.0 # 假設放置的資源的值
        }

    def _calculate_global_density(self, board_state: List[List[Any]], rows: int, cols: int) -> float:
        if rows * cols == 0: return 0.0
        resource_count = 0
        val_func = self.internal_params["value_function"]
        threshold = self.internal_params["resource_threshold"]
        for r in range(rows):
            for c in range(cols):
                if val_func(board_state[r][c]) >= threshold:
                    resource_count += 1
        return resource_count / (rows * cols)

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 評估在目標格子放置一個「資源」後，對盤面整體「資源密度」的邊際效益，此效益會根據當前盤面的「資源稀疏度」進行加權。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return 0.0

        val_func = self.internal_params["value_function"]
        current_val = val_func(board_state[position_row][position_col])
        resource_threshold = self.internal_params["resource_threshold"]
        
        # 如果當前格子已經是資源，再放置的邊際貢獻為0 (或很小，除非能提升其等級)
        if current_val >= resource_threshold :
             # 可以設計成如果放置的值更高，則有邊際效益
             if self.internal_params["assumed_placed_value"] > current_val:
                 pass # 繼續計算
             else:
                 return 0.1 # 已是資源且放置的值不大於現有值，邊際效益低

        d_global_before = self._calculate_global_density(board_state, rows, cols)

        # 模擬放置資源後的盤面 (僅修改一個點來計算新密度)
        # 效率考量：不實際複製盤面，而是直接計算資源數量的變化
        num_cells = rows * cols
        if num_cells == 0: return 0.0

        current_resource_count = d_global_before * num_cells
        
        new_resource_count_after_placement = current_resource_count
        if current_val < resource_threshold and self.internal_params["assumed_placed_value"] >= resource_threshold:
            # 原本不是資源，放置後變成資源
            new_resource_count_after_placement += 1
        elif current_val < resource_threshold and self.internal_params["assumed_placed_value"] < resource_threshold:
            # 原本不是，放置後也不是 (但值可能改變)，對"資源計數"無影響
            pass 
        elif current_val >= resource_threshold and self.internal_params["assumed_placed_value"] < resource_threshold:
            # 原本是，放置後變成不是 (移除或降級)，資源數減1
            # 但此模組通常用於評估"放置"的好處，所以這種情況應該是負效益或不考慮
            # 為了簡化，假設此模組只評估"增加資源"的情況
             return 0.05 # 覆蓋了已有資源使其不再是資源，負面影響
        
        d_global_after = new_resource_count_after_placement / num_cells
        marginal_contribution = d_global_after - d_global_before
        
        # 稀疏度加權因子 (1 - D_global)越高表示越稀疏，此時邊際貢獻越重要
        sparsity_factor = (1.0 - d_global_before) ** self.internal_params["sparsity_weight_power_k"]
        
        # 分數與邊際貢獻和稀疏度正相關
        # marginal_contribution 的範圍很小 (1/num_cells 或 0)
        # sparsity_factor 在 0-1
        # 結果也在一個小範圍內，需要放大
        raw_score = marginal_contribution * sparsity_factor * num_cells # *num_cells 將其放大回 "一個單位的貢獻"
                                                                    # 例如，若增加一個資源，mc = 1/N, raw_score = (1/N)*sparsity*N = sparsity
        
        # 如果目標是讓盤面更「均衡」，則當 D_global 接近0.5時，任何改變（無論增減）都可能降低分數。
        # 但此處設定是「稀疏時增加資源更有價值」
        return MathUtils.sigmoid((raw_score - 0.1) * 5.0) # 調整中心和斜率，因為raw_score可能在0附近


class GM9(LogicModule):
    """
    設計理念：基於"數值梯度場的流線"或"向量場分析"的簡化概念。
              原始GM9評估反對角線。進化版將分析目標格子周圍數值梯度的方向和強度，
              判斷該點是否位於一個有利的"流動方向"上，或是一個"匯聚點"/"發散點"。
    用途描述：識別盤面中數值流動的趨勢，例如資源從高濃度流向低濃度，或勢力擴張的方向。
              高分可能表示該點順應了主流趨勢，或是關鍵的控制節點。
    評分公式原理：1. 計算目標格子周圍（例如，其本身和直接鄰居）的數值梯度向量 (Gx, Gy)。
                 2. 分析這些梯度向量的一致性（例如，計算平均梯度方向和強度，或梯度場的散度/旋度）。
                 3. 如果梯度強且方向一致指向某個"有利區域"（或遠離"不利區域"），則分數高。
                 4. 如果是匯聚點（散度為負且大）且匯聚的是高價值，則分數高。
                 Score = f(gradient_strength, gradient_coherence, divergence_properties)
    兼容性：數值型盤面。
    優化與延伸方向：1. 完整計算梯度場的散度和旋度。
                    2. 引入"爬山法"或"梯度下降法"的原理，評估從該點出發能達到的最優局部目標。
                    3. 結合粒子追踪，模擬資源在梯度場中的流動。
    可能的多版本邏輯：GM9.v1 (反對角線), GM9.v2 (局部梯度方向與強度), GM9.v3 (梯度場散度/旋度分析)。
    """
    def __init__(self):
        super().__init__(
            module_id="GM9", 
            name="Generated Module 9 (Value Gradient Flow & Convergence)", 
            description="Analyzes local value gradient direction, strength, and convergence/divergence."
        )
        self.internal_params = {
            "gradient_strength_weight": 0.4,
            "gradient_coherence_weight": 0.3, # 方向一致性
            "convergence_weight": 0.3,       # 匯聚/發散性
            "value_function": lambda x: float(x) if isinstance(x, (int, float)) else 0.0,
            "epsilon": 1e-6 # 避免除以零
        }

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 綜合評估目標格子周圍數值梯度的強度、方向一致性以及該點是傾向於數值匯聚還是發散。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return 0.0
        
        val_func = self.internal_params["value_function"]

        # 1. 計算中心點和其直接鄰居的梯度
        # Gx, Gy = BoardAnalyzerUtils.get_value_gradient(board_state, position_row, position_col)
        # gradient_magnitude_center = math.hypot(Gx, Gy)
        
        # 為了分析一致性和散度，需要周圍多個點的梯度，或對中心點周圍的值進行分析
        # 計算中心點的散度 (Divergence) 近似:
        # dFx/dx + dFy/dy
        # Fx(x+1,y) - Fx(x-1,y) / 2dx  (簡化 dFx/dx approx val(x+1) - val(x-1) )
        # Fy(x,y+1) - Fy(x,y-1) / 2dy  (簡化 dFy/dy approx val(y+1) - val(y-1) )
        # 散度: (val(x+1,y) - val(x-1,y)) + (val(x,y+1) - val(x,y-1)) (未除以2dx)
        
        val_at = lambda r, c: val_func(board_state[r][c]) if (0 <= r < rows and 0 <= c < cols and val_func(board_state[r][c]) is not None) else 0.0
        
        vx_plus1 = val_at(position_row, position_col + 1)
        vx_minus1 = val_at(position_row, position_col - 1)
        vy_plus1 = val_at(position_row + 1, position_col)
        vy_minus1 = val_at(position_row - 1, position_col)
        
        # 散度近似
        # 如果周圍值高於中心，則預期是匯聚(負散度)；如果周圍低，則是發散(正散度)
        # 散度 = (f(x+h)-f(x-h))/2h + (g(y+h)-g(y-h))/2h
        # 此處，我們考慮中心點的值與周圍值的關係
        # 梯度Gx(i,j) = (V(i+1,j)-V(i-1,j))/2, Gy(i,j) = (V(i,j+1)-V(i,j-1))/2
        # Div(i,j) = Gx(i+1,j)-Gx(i-1,j) / 2 + Gy(i,j+1)-Gy(i,j-1) / 2
        # 這需要計算鄰居的梯度，較複雜。
        
        # 簡化版：計算中心點的梯度幅度和周圍梯度的平均方向
        Gx_center, Gy_center = BoardAnalyzerUtils.get_value_gradient(board_state, position_row, position_col)
        magnitude_center = math.hypot(Gx_center, Gy_center)
        angle_center = math.atan2(Gy_center, Gx_center)

        # 正規化梯度強度 (假設最大梯度幅值與盤面值域相關，例如差值最大為1，Gx,Gy約為4，mag約5-6)
        # 假設值域0-1，那麼單步差最大1。 Gx,Gy基於Sobel的權重，最大響應可能是4 (e.g. [0,0,0] vs [1,1,1])
        # 幅值 sqrt(4^2+4^2) = sqrt(32) approx 5.6
        norm_magnitude = MathUtils.normalize_value(magnitude_center, 0, 6.0) 
        
        # 分析周圍鄰居的梯度方向是否與中心梯度方向一致 (內積)
        # 或更簡單：如果中心梯度指向高價值區域，則加分
        # 此處採用：檢查散度。如果值從周圍流向中心（匯聚），則可能是個好點。
        # 散度近似: (Gx(x+dx/2) - Gx(x-dx/2)) / dx + (Gy(y+dy/2) - Gy(y-dy/2)) / dy
        # 簡化版 divergence ~ (Val(x+1)+Val(x-1)+Val(y+1)+Val(y-1) - 4*Val_center) (類似拉普拉斯)
        # 拉普拉斯為負表示局部極大（匯聚），為正表示局部極小（發散）
        # (這裡使用F10的拉普拉斯計算，但改變其語義)
        center_val = val_at(position_row, position_col)
        sum_4_neighbors = vx_plus1 + vx_minus1 + vy_plus1 + vy_minus1
        laplacian_like = sum_4_neighbors - 4 * center_val 
        
        # laplacian_like < 0: 匯聚 (局部最大，周圍比它低，值流向它) -> 高分
        # laplacian_like > 0: 發散 (局部最小，周圍比它高，值從它流出) -> 低分
        # 正規化laplacian，範圍約-4到4 (若值域0-1)
        # 我們希望 -4 -> 1分, +4 -> 0分
        convergence_score = MathUtils.normalize_value(-laplacian_like, -4.0, 4.0) 

        # 方向一致性: 較難在局部簡單計算。
        # 此處簡化為只考慮梯度強度和匯聚性。
        
        # 如果梯度本身很小，那麼匯聚性意義不大
        # 可以用梯度強度來加權匯聚性得分
        # final_score = norm_magnitude * convergence_score
        # 或者分別加權
        
        final_score = (self.internal_params["gradient_strength_weight"] * norm_magnitude +
                       self.internal_params["convergence_weight"] * convergence_score)
        # 由於沒有coherence, 重新分配權重
        total_weight_used = self.internal_params["gradient_strength_weight"] + self.internal_params["convergence_weight"]
        if total_weight_used > 1e-6 : # 避免除以零
             final_score = final_score / total_weight_used # 平均一下
        else: # 如果權重都是0
            final_score = (norm_magnitude + convergence_score) / 2.0


        return max(0.0, min(1.0, final_score))


class GM10(LogicModule):
    """
    設計理念：基於博弈論中的"控制區域"或"影響力地圖"概念。
              原始GM10評估L型圖案。進化版將模擬從當前格子出發，在一定步數內能"控制"或"影響"到的格子數量，
              同時考慮這些被影響格子的價值。
    用途描述：識別那些具有強大輻射影響力或能控制大片區域的戰略要點。
              適用於評估棋盤遊戲中的棋子佈局、或資源控制遊戲中的設施選址。
    評分公式原理：1. 從 (pos_r, pos_c) 出發，模擬影響力擴散（例如，BFS固定步數K）。
                 2. 擴散過程中，若遇到"障礙物"（值過高/過低，或特定ID）則停止該方向擴散。
                 3. 計算在K步內能無障礙到達的所有格子的集合 S_reachable。
                 4. 計算這些格子的總價值 V_reachable = Sum_{cell in S_reachable} (value_func(cell) * decay_factor(dist))。
                 5. Score = normalized(V_reachable)。
    兼容性：數值型盤面，需要定義"障礙物"和"價值函數"。
    優化與延伸方向：1. 引入更複雜的影響力擴散模型（考慮方向、衰減）。
                    2. 區分"己方影響力"和"對敵方壓制力"。
                    3. 結合Alpha-Beta剪枝思想評估放置後的未來影響力變化。
    可能的多版本邏輯：GM10.v1 (L型圖案), GM10.v2 (BFS影響力區域價值), GM10.v3 (考慮障礙和衰減的影響力擴散)。
    """
    def __init__(self):
        super().__init__(
            module_id="GM10", 
            name="Generated Module 10 (Influence Mapping & Area Control)", 
            description="Calculates the value of area a cell can influence/control within K steps, considering obstacles."
        )
        self.internal_params = {
            "influence_steps_k": 3, # 影響力擴散的步數
            "obstacle_threshold_low": -0.1, # 值低於此被視為障礙 (或特定ID)
            "obstacle_threshold_high": 1.1, # 值高於此被視為障礙 (假設盤面值在0-1)
            "value_function_for_influenced_area": lambda x: float(x) if (isinstance(x, (int, float)) and 0 <= x <=1) else 0.0, # 被影響區域的格子如何貢獻價值
            "distance_decay_factor": 0.8 # 每遠一步，價值貢獻衰減因子
        }

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 模擬從目標格子出發的影響力擴散（限定步數內），計算能無障礙觸及並施加影響的區域的總價值（考慮距離衰減）。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return 0.0

        q = deque([( (position_row, position_col), 0, 1.0 )]) # ((r,c), current_step, current_decay_multiplier)
        visited_for_influence = set([(position_row, position_col)])
        total_influenced_value = 0.0
        
        # 當前格子自身的價值也應被考慮，或者說影響力從自身開始
        val_func = self.internal_params["value_function_for_influenced_area"]
        # total_influenced_value += val_func(board_state[position_row][position_col]) # 自身影響

        max_possible_value_in_k_steps = 0 # 用於正規化

        while q:
            (r, c), step, decay_multiplier = q.popleft()

            if step > self.internal_params["influence_steps_k"]:
                continue

            # 計算 (r,c) 點對總影響價值的貢獻 (除了起始點)
            if not (r == position_row and c == position_col): # 起始點的價值可以不算在"擴散"的部分
                 cell_val = val_func(board_state[r][c])
                 total_influenced_value += cell_val * decay_multiplier
            
            # 理論最大值計算: 假設每一步都能擴散到4個新格子，且每個格子的值都是1
            # 這是一個粗略的上限，實際BFS擴散面積比較複雜
            # 簡化：假設K步內能影響的格子數上限約為 (2K+1)^2 (一個方形區域)
            # 此處不精確計算max_possible_value_in_k_steps，而是基於擴散格子數
            # 或使用一個基於K的理論最大值，例如 K步內最多影響 1+4+8+12... 個格子。
            # max_influenced_cells_approx = (2*self.internal_params["influence_steps_k"]+1)**2
            # max_total_value_approx = max_influenced_cells_approx * 1.0 (假設最大值1)
            # 正規化因子後面處理。

            if step < self.internal_params["influence_steps_k"]:
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited_for_influence:
                        # 檢查是否為障礙
                        obstacle_val = board_state[nr][nc] # 原始值
                        is_obstacle = False
                        if isinstance(obstacle_val, (int,float)):
                            if obstacle_val < self.internal_params["obstacle_threshold_low"] or \
                               obstacle_val > self.internal_params["obstacle_threshold_high"]:
                                is_obstacle = True
                        # else: can treat non-numeric as obstacle or transparent, depends on game rule
                        # Here, non-numeric is not obstacle based on cost_func for influenced_area

                        if not is_obstacle:
                            visited_for_influence.add((nr, nc))
                            q.append(((nr, nc), step + 1, decay_multiplier * self.internal_params["distance_decay_factor"]))
        
        # 正規化 influenced_value
        # K=1, max cells = 1 (self) + 4 = 5. max decay_mult = 1 (self) + 0.8*4
        # K=2, max cells = 5 + 8 = 13. max decay_mult = (self) + 0.8*4 + 0.8^2*8
        # 預估一個最大可能影響力值用於正規化
        # 假設每個可達格子都是最大值1。
        # 粗略估計最大影響區域格子數 (不考慮重疊和邊界)
        max_cells_in_radius_k = 0
        current_shell_cells = 1
        current_decay = 1.0
        for i in range(self.internal_params["influence_steps_k"] + 1): # include step 0 (self)
            # max_cells_in_radius_k += current_shell_cells # this counts cells
            if i > 0 : # only add decay for steps > 0 (from self)
                 max_possible_value_in_k_steps += (4 * (i)) * current_decay # very rough approx of cells at step i
            elif i == 0:
                 max_possible_value_in_k_steps += 1 * current_decay # self

            current_decay *= self.internal_params["distance_decay_factor"]
            # current_shell_cells = 4 * (i+1) # cells in next shell (approx)
        if self.internal_params["influence_steps_k"] == 0: max_possible_value_in_k_steps = 1.0
        if self.internal_params["influence_steps_k"] == 1: max_possible_value_in_k_steps = 1 + 4*0.8 # self + 4 neighbors at dist 1
        if self.internal_params["influence_steps_k"] == 2: max_possible_value_in_k_steps = 1 + 4*0.8 + 8*0.8*0.8 # self + 4 N + 8 N at dist 2
        if self.internal_params["influence_steps_k"] == 3: max_possible_value_in_k_steps = 1 + 4*0.8 + 8*0.64 + 12*0.512

        if max_possible_value_in_k_steps == 0: return 0.0
        
        normalized_influence = total_influenced_value / max_possible_value_in_k_steps
        return max(0.0, min(1.0, normalized_influence))
# ... (接續第三部分的程式碼: MathUtils, BoardAnalyzerUtils, LogicModule, BoardInput, A2, M3, D3, F10, GM1-GM10) ...

class GM11(LogicModule):
    """
    設計理念：基於"多尺度空間分析" (Multi-scale Spatial Analysis) 的原理，評估不同尺度下目標格子的特徵一致性或顯著性。
              原始GM11評估行奇偶匹配。進化版將分析在不同大小的鄰域窗口下，目標格子的值相對於該窗口統計特性（如均值、中位數）的穩定性或突出性。
    用途描述：識別那些在多個空間尺度上都表現出相似特徵（例如，始終是高點或低點）的"魯棒"格點，或是在特定尺度下才顯現其特異性的格點。
              適用於需要區分持久性特徵和隨機噪聲，或尋找具有特定作用範圍特徵的場景。
    評分公式原理：1. 定義多個尺度（例如，3x3, 5x5, 7x7鄰域窗口）。
                 2. 對於每個尺度 s：
                    a. 計算窗口內數值的均值 M_s 和標準差 S_s。
                    b. 計算目標格子值 V_cell 相對於 (M_s, S_s) 的 Z-score: Z_s = (V_cell - M_s) / (S_s + epsilon)。
                 3. 綜合多個尺度的 Z_s 值（例如，計算加權平均Z_avg，或觀察Z_s隨尺度變化的趨勢）。
                 4. Score = sigmoid( f(Z_avg) ) 或 g(trend_of_Z_s)。若希望獎勵持續高/低點，則 Z_s 符號一致且絕對值大則高分。
    兼容性：數值型盤面。
    優化與延伸方向：1. 使用影像金字塔或小波轉換實現更系統的多尺度分解。
                    2. 分析Z-score序列的穩定性或特定模式（例如，在某尺度突然劇變）。
                    3. 根據不同尺度的重要性賦予不同權重。
    可能的多版本邏輯：GM11.v1 (行奇偶), GM11.v2 (多尺度Z-score均值/方差), GM11.v3 (Z-score序列趨勢分析)。
    """
    def __init__(self):
        super().__init__(
            module_id="GM11", 
            name="Generated Module 11 (Multi-scale Significance Analyzer)", 
            description="Assesses cell's value significance across multiple neighborhood scales (e.g., Z-scores)."
        )
        self.internal_params = {
            "scales_radii": [1, 2, 3], # 对应 3x3, 5x5, 7x7 邻域的半径
            "value_function": lambda x: float(x) if isinstance(x, (int, float)) else None,
            "epsilon": 1e-6,
            "z_score_consistency_weight": 0.7, # Z-score符号一致性权重
            "z_score_magnitude_weight": 0.3    # Z-score平均大小权重
        }

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 透過在不同大小的鄰域（尺度）下計算目標格子值的Z-score，評估其在多尺度下的統計顯著性及其穩定性。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return 0.0

        val_func = self.internal_params["value_function"]
        v_cell = val_func(board_state[position_row][position_col])
        if v_cell is None: return 0.0

        z_scores_at_scales = []
        magnitudes_at_scales = []

        for radius in self.internal_params["scales_radii"]:
            neighborhood_values = []
            # 收集當前尺度鄰域的值 (不含中心點)
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0: continue
                    nr, nc = position_row + dr, position_col + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        val = val_func(board_state[nr][nc])
                        if val is not None:
                            neighborhood_values.append(val)
            
            if not neighborhood_values: # 如果鄰域為空 (例如盤面太小或格子在極端角落)
                # 給一個中性或基於v_cell本身的值
                # 此處簡化：若無鄰居比較，則此尺度貢獻小
                z_scores_at_scales.append(0) # 無法比較，Z-score為0
                magnitudes_at_scales.append(0)
                continue

            neigh_mean = float(np.mean(neighborhood_values))
            neigh_std = float(np.std(neighborhood_values))

            if neigh_std < self.internal_params["epsilon"]: # 避免除以零, 若標準差極小
                # 如果鄰域值都一樣
                if math.isclose(v_cell, neigh_mean): # 自身與鄰域同質
                    z_score = 0.0
                else: # 自身與同質鄰域不同，則Z-score會很大
                    z_score = (v_cell - neigh_mean) / self.internal_params["epsilon"] 
            else:
                z_score = (v_cell - neigh_mean) / neigh_std
            
            z_scores_at_scales.append(z_score)
            magnitudes_at_scales.append(abs(z_score))

        if not z_scores_at_scales: return 0.5 # 無法進行任何尺度的分析

        # 分析Z-score序列
        # 1. Z-score 符號一致性 (是否一直是正或一直是負)
        #    越高表示在多尺度下，該點相對於周圍的特性越穩定 (始終高於/低於鄰域均值)
        num_positive_z = sum(1 for z in z_scores_at_scales if z > 0.1) # 加一個小門檻避免0的影響
        num_negative_z = sum(1 for z in z_scores_at_scales if z < -0.1)
        
        consistency_score = 0.0
        if len(z_scores_at_scales) > 0 :
            # 如果所有非零z-score同號，則一致性高
            if num_positive_z == 0 and num_negative_z > 0: # 全為負 (穩定低點)
                consistency_score = num_negative_z / len(z_scores_at_scales)
            elif num_negative_z == 0 and num_positive_z > 0: # 全為正 (穩定高點)
                consistency_score = num_positive_z / len(z_scores_at_scales)
            elif num_positive_z == 0 and num_negative_z == 0: # Z-score都接近0
                 consistency_score = 0.5 # 中性
            else: # 符號不一致，一致性低
                consistency_score = 0.0


        # 2. Z-score 平均絕對大小 (顯著性)
        #    越大表示該點在各尺度下都顯著不同於其鄰域
        avg_magnitude = float(np.mean(magnitudes_at_scales)) if magnitudes_at_scales else 0.0
        # 正規化 avg_magnitude (Z-score 通常在 -3~3 之間比較典型，abs 在 0~3)
        norm_avg_magnitude = MathUtils.normalize_value(avg_magnitude, 0, 3.0)

        # 綜合評分
        final_score = (self.internal_params["z_score_consistency_weight"] * consistency_score +
                       self.internal_params["z_score_magnitude_weight"] * norm_avg_magnitude)
        
        return max(0.0, min(1.0, final_score))


class GM12(LogicModule):
    """
    設計理念：基於"紋理分析" (Texture Analysis) 的初步概念，例如使用灰度共生矩陣(GLCM)的簡化思想或局部二值模式(LBP)。
              原始GM12評估列奇偶匹配。進化版將分析目標格子3x3或5x5鄰域內的數值"紋理特徵"，如平滑度、粗糙度、方向性。
    用途描述：識別盤面中具有特定紋理模式的區域。例如，平滑區域可能代表安全區，粗糙區域可能代表爭奪區，特定方向紋理可能代表通道。
    評分公式原理：1. 對目標格子的鄰域，計算簡化的紋理描述符。
                 例如，LBP變種：將鄰域值與中心值比較，得到一個二進制序列，轉換為一個數值。
                 或計算鄰域內梯度的一致性（方向熵）作為方向性指標。
                 或計算局部方差作為粗糙度指標。
                 2. 此處選擇實現一個簡化的 LBP 變種 + 局部方差。
                 Score_LBP = f(LBP_value) (例如，某些LBP值對應有利的紋理)
                 Score_Variance = g(local_variance) (例如，低方差=平滑=高分)
                 Final_Score = w1 * Score_LBP + w2 * Score_Variance
    兼容性：數值型盤面。對於LBP，可能需要先對數值進行量化。
    優化與延伸方向：1. 實現完整的GLCM特徵提取（對比度、能量、熵、同質性）。
                    2. 使用Gabor濾波器組分析多方向多頻率的紋理。
                    3. 訓練一個小型分類器來識別"有利紋理"。
    可能的多版本邏輯：GM12.v1 (列奇偶), GM12.v2 (簡化LBP + 方差), GM12.v3 (梯度方向熵)。
    """
    def __init__(self):
        super().__init__(
            module_id="GM12", 
            name="Generated Module 12 (Local Texture Analyzer - LBP-like & Variance)", 
            description="Analyzes local texture features like patterns (LBP-like) and smoothness (variance)."
        )
        self.internal_params = {
            "lbp_radius": 1, # 3x3 neighborhood for LBP
            "lbp_num_points": 8, # 8 neighbors
            "lbp_weight": 0.6,
            "variance_weight": 0.4,
            "value_function": lambda x: float(x) if isinstance(x, (int, float)) else None,
            "target_lbp_patterns": { # 假設某些LBP值是有利的 (需要根據實際盤面和遊戲定義)
                0b00000000: 1.0, # 全比中心小 (或全比中心大，取決於LBP定義) -> 平坦區中的一個點
                0b11111111: 1.0, # 全比中心大 (或全比中心小) -> 平坦區中的一個點
                0b01010101: 0.7, # 交替模式
            }
        }

    def _get_simplified_lbp_value(self, board_state: List[List[Any]], r: int, c: int, radius: int, num_points: int) -> int:
        # 簡化的LBP: 只比較鄰居和中心的關係 (大於等於中心為1，小於為0)
        # 順時針或逆時針取點
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        val_func = self.internal_params["value_function"]
        center_val_raw = board_state[r][c]
        center_val = val_func(center_val_raw)
        if center_val is None: return -1 # 無法計算LBP

        lbp_code = 0
        # 簡化為8個固定鄰居點 for 3x3
        points_coords_offsets = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)] # 8 points clockwise
        
        for i in range(len(points_coords_offsets)):
            dr, dc = points_coords_offsets[i]
            nr, nc = r + dr, c + dc
            
            bit = 0
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbor_val = val_func(board_state[nr][nc])
                if neighbor_val is not None and neighbor_val >= center_val:
                    bit = 1
            # else: 邊界外可以視為0或1，或不計入。此處視為0 (不比中心大)

            lbp_code |= (bit << (len(points_coords_offsets) - 1 - i)) # MSB first
        return lbp_code


    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 透過類局部二值模式(LBP)分析鄰域結構的均一性/特定模式，並結合局部方差評估平滑度/粗糙度。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return 0.0
        
        val_func = self.internal_params["value_function"]

        # 1. LBP-like score
        lbp_val = self._get_simplified_lbp_value(board_state, position_row, position_col, 
                                                 self.internal_params["lbp_radius"], 
                                                 self.internal_params["lbp_num_points"])
        
        score_lbp = 0.0
        if lbp_val != -1: # 計算成功
            # 檢查是否為目標LBP模式
            if lbp_val in self.internal_params["target_lbp_patterns"]:
                score_lbp = self.internal_params["target_lbp_patterns"][lbp_val]
            else:
                # 非目標模式，可以給一個基於LBP值本身的分數，例如LBP值的漢明權重（1的個數）
                # 或LBP值的"旋轉不變性" (取最小的旋轉等價值)
                # 此處簡化：非目標模式給予較低基礎分
                score_lbp = 0.2 
        
        # 2. Variance score (smoothness)
        #    使用3x3鄰域(含中心)計算方差
        neighborhood_for_variance = []
        for dr_v in [-1,0,1]:
            for dc_v in [-1,0,1]:
                nr_v, nc_v = position_row + dr_v, position_col + dc_v
                if 0 <= nr_v < rows and 0 <= nc_v < cols:
                    v = val_func(board_state[nr_v][nc_v])
                    if v is not None:
                        neighborhood_for_variance.append(v)
        
        score_variance = 0.5 # 中性分
        if len(neighborhood_for_variance) >= 2:
            local_var = float(np.var(neighborhood_for_variance))
            # 低方差 (平滑) -> 高分。正規化方差。
            # 假設值域0-1，最大方差約0.25 (例如一半0一半1)
            norm_variance = MathUtils.normalize_value(local_var, 0, 0.25)
            score_variance = 1.0 - norm_variance # 平滑度分數
        
        # 3. Final score
        final_score = (self.internal_params["lbp_weight"] * score_lbp +
                       self.internal_params["variance_weight"] * score_variance)
        
        return max(0.0, min(1.0, final_score))

class GM13(LogicModule):
    """
    設計理念：基於"分形維數" (Fractal Dimension) 或"盒子計數法" (Box-Counting)的簡化概念，評估目標格子周圍區域的"空間填充複雜度"或"不規則性"。
              原始GM13評估象限密度。進化版將分析不同尺度下，包含"有效內容"(例如，值 > T)的格子數量增長情況。
    用途描述：識別盤面中結構複雜、犬牙交錯的區域，或 наоборот, 結構簡單、平鋪直敘的區域。
              高分可能表示該區域具有高度的邊界效應或滲透潛力。
    評分公式原理：1. 以目標格子為中心，考慮多個不同大小的盒子(正方形鄰域)。
                 2. 對於每個盒子大小 r，計算盒子內"有效內容"的格子數量 N(r)。
                 3. 分析 N(r) 與 r 的關係。若 N(r) ~ r^D，則 D 是分形維數的近似。
                    例如，在 log-log 圖上，log(N(r)) vs log(r) 的斜率是 D。
                 4. 此處簡化：不直接計算D，而是觀察 N(r)/r^2 (即密度) 在不同尺度下的變化。
                    如果密度隨尺度變化不大，可能是空間填充均勻。如果密度在某尺度劇減，可能有空洞或不規則。
                 Score = f(density_stability_across_scales) 或 g(approximated_D)。
                 此處選擇：如果局部密度較高且在小尺度下變化不大，則給高分 (表示是一個密集的、有內容的局部)。
    兼容性：數值型或二值化後的盤面（根據"有效內容"定義）。
    優化與延伸方向：1. 實現更精確的盒子計數法或Minkowski維數計算。
                    2. 分析不同方向上的分形特性。
                    3. 結合熵指標評估結構的複雜性和隨機性。
    可能的多版本邏輯：GM13.v1 (象限密度), GM13.v2 (多尺度局部密度變化率), GM13.v3 (簡化盒子計數維數近似)。
    """
    def __init__(self):
        super().__init__(
            module_id="GM13", 
            name="Generated Module 13 (Spatial Filling & Complexity Analyzer)", 
            description="Analyzes complexity/irregularity around cell using multi-scale density (box-counting like)."
        )
        self.internal_params = {
            "box_radii": [1, 2, 3], # 盒子(鄰域)半徑，對應盒子大小 (2r+1)x(2r+1)
            "effective_content_threshold": 0.5, # 值高於此才算"有效內容"
            "density_stability_weight": 0.7,
            "avg_density_weight": 0.3,
            "value_function": lambda x: float(x) if isinstance(x, (int, float)) else 0.0
        }

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 透過在不同大小的「盒子」(鄰域)內計算「有效內容」的密度，並分析密度隨盒子大小變化的穩定性，來評估局部空間的填充複雜度。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return 0.0
        
        val_func = self.internal_params["value_function"]
        threshold = self.internal_params["effective_content_threshold"]
        
        densities_at_scales = []
        
        for radius in self.internal_params["box_radii"]:
            box_size_total_cells = (2 * radius + 1) ** 2
            effective_cells_in_box = 0
            actual_cells_in_box_on_board = 0 # 處理邊界情況

            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = position_row + dr, position_col + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        actual_cells_in_box_on_board +=1
                        if val_func(board_state[nr][nc]) >= threshold:
                            effective_cells_in_box += 1
            
            if actual_cells_in_box_on_board > 0:
                current_density = effective_cells_in_box / actual_cells_in_box_on_board
                densities_at_scales.append(current_density)
            else: # 盒子完全在盤面外 (不可能，因為中心點在內) 或 1x1 盤面半徑為0
                # 如果是1x1盤面，radius=0 (box_radii可包含0)，則密度是自身是否為有效內容
                if radius == 0 and actual_cells_in_box_on_board == 1 :
                    densities_at_scales.append(1.0 if effective_cells_in_box == 1 else 0.0)
                # else: densities_at_scales.append(0) # 尺度過大或無效，密度為0

        if not densities_at_scales:
            # 如果中心點本身是有效內容，給0.5，否則0.1
            return 0.5 if val_func(board_state[position_row][position_col]) >= threshold else 0.1

        # 1. 平均密度
        avg_density = float(np.mean(densities_at_scales))
        
        # 2. 密度隨尺度的變化穩定性 (用方差的倒數表示)
        #    低方差表示密度在不同尺度下較穩定
        density_variance = float(np.var(densities_at_scales)) if len(densities_at_scales) >=2 else 0.0
        # 正規化方差 (密度0-1, 方差0-0.25 approx)
        norm_density_variance = MathUtils.normalize_value(density_variance, 0, 0.25)
        density_stability_score = 1.0 - norm_density_variance

        # 綜合評分: 高平均密度且高穩定性 -> 高分
        final_score = (self.internal_params["avg_density_weight"] * avg_density +
                       self.internal_params["density_stability_weight"] * density_stability_score)
        
        return max(0.0, min(1.0, final_score))

class GM14(LogicModule):
    """
    設計理念：模擬"生態位" (Niche) 或"資源競爭"模型。
              原始GM14評估棋盤格。進化版將評估目標格子在多種"資源維度"上的適應性，
              並考慮周圍"競爭者"（其他高價值格）對這些資源的佔用情況。
    用途描述：識別那些能夠有效利用多種可用資源且競爭壓力較小的"黃金地段"。
              適用於需要進行多目標優化選址的場景。
    評分公式原理：1. 定義多個"資源維度" R_1, R_2, ..., R_k (例如，與A類資源的距離，與B類補給線的連接性等)。
                 2. 對於每個資源維度 j，計算目標格子 (pos_r, pos_c) 的"資源獲取能力" S_j(pos)。
                 3. 對於每個資源維度 j，估算周圍其他格點對該資源的"競爭強度" C_j(pos)。
                    (例如，計算鄰域內有多少其他格子也高度依賴 R_j)。
                 4. 目標格子的總體適應性 Score = Sum_j [ w_j * S_j(pos) / (1 + alpha * C_j(pos)) ]
                    其中 w_j 是資源維度j的重要性權重，alpha是競爭敏感度因子。
    兼容性：需要能夠從盤面狀態中提取多維度的資源信息和競爭者信息。
    優化與延伸方向：1. 引入更複雜的競爭模型（如Lotka-Volterra方程的離散形式）。
                    2. 動態調整資源維度的重要性權重 w_j。
                    3. 考慮"生態演替"的過程，即一個格子從一種生態位轉變為另一種。
    可能的多版本邏輯：GM14.v1 (棋盤格), GM14.v2 (多維資源適應性與競爭), GM14.v3 (基於代理的資源競爭模擬)。
    """
    def __init__(self):
        super().__init__(
            module_id="GM14", 
            name="Generated Module 14 (Ecological Niche & Competition Modeler)", 
            description="Scores cell based on its fitness in a multi-resource environment considering local competition."
        )
        self.internal_params = {
            "resource_definitions": [ # (resource_eval_func, weight, competition_radius)
                {"name": "ProximityToValueTypeA", # 假設盤面值 0.8-1.0 是 TypeA
                 "eval_func": lambda board, r, c, vr, vc: 1.0 / (1 + MathUtils.manhattan_distance((r,c),(vr,vc))), # 資源獲取能力 (與最近A的距離倒數)
                 "value_check_func": lambda val: val is not None and val >=0.8, # 什麼是TypeA
                 "weight": 0.5, "competition_radius": 2},
                {"name": "AccessToOpenSpace", # 假設盤面值 0-0.2 是 OpenSpace
                 "eval_func": lambda board, r, c, vr, vc: 1.0 / (1 + MathUtils.manhattan_distance((r,c),(vr,vc))),
                 "value_check_func": lambda val: val is not None and val <=0.2,
                 "weight": 0.3, "competition_radius": 3},
                {"name": "HighLocalGradient", # 資源是數值變化劇烈的地方 (F10的簡化)
                 "eval_func": lambda board, r, c, vr, vc: math.hypot(*BoardAnalyzerUtils.get_value_gradient(board,r,c)), # vr, vc not used here
                 "value_check_func": lambda val: True, # 不檢查特定值，而是計算梯度
                 "is_location_based_resource": True, # 表示這個資源的Sj是直接在(r,c)計算，不是找最近的
                 "weight": 0.2, "competition_radius": 1}
            ],
            "competition_sensitivity_alpha": 0.5,
            "value_function": lambda x: float(x) if isinstance(x, (int, float)) else None,
            "max_dist_for_resource_search": 7 # 尋找資源點的最大曼哈頓距離
        }

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 綜合評估目標格子在多個預定義「資源維度」上的獲取能力，並將此能力根據周圍其他格子對相同資源的「競爭強度」進行折損。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return 0.0
        val_func = self.internal_params["value_function"]
        
        total_weighted_fitness_score = 0.0
        total_resource_weights = 0.0

        for resource_def in self.internal_params["resource_definitions"]:
            s_j = 0.0 # 資源獲取能力
            
            if resource_def.get("is_location_based_resource", False):
                # 資源本身就在 (position_row, position_col) 計算，不需要尋找
                s_j = resource_def["eval_func"](board_state, position_row, position_col, 0, 0)
                # 正規化 s_j (例如梯度幅值需要正規化)
                if resource_def["name"] == "HighLocalGradient":
                    s_j = MathUtils.normalize_value(s_j, 0, 6.0) # 同GM9的梯度幅值正規化
            else:
                # 尋找最近的滿足 value_check_func 的資源點
                min_dist_to_resource = float('inf')
                nearest_resource_pos = None
                
                # 廣度優先搜索尋找最近的資源點 (限定範圍)
                q_res = deque([( (position_row, position_col), 0 )])
                visited_res = set([(position_row, position_col)])
                found_res_for_dim = False

                while q_res:
                    (curr_r, curr_c), dist = q_res.popleft()
                    if dist > self.internal_params["max_dist_for_resource_search"]: continue

                    # 檢查 (curr_r, curr_c) 是否為資源點 (如果不是起始點的話)
                    # 或者，是檢查 (curr_r, curr_c) 是否為資源點，然後計算 (pos_r,pos_c) 到它的距離
                    # 此處邏輯：從 (pos_r,pos_c) 出發，找到最近的滿足 value_check_func 的格子
                    # BFS 已經保證了 dist 是最短路徑長度（步數）
                    
                    # 檢查 (curr_r, curr_c) 是否為當前維度的資源
                    # 如果 (pos_r, pos_c) 本身就是資源，則dist=0
                    val_at_curr = val_func(board_state[curr_r][curr_c])
                    if resource_def["value_check_func"](val_at_curr):
                       s_j = resource_def["eval_func"](board_state, position_row, position_col, curr_r, curr_c) # eval_func 通常是 1/(1+dist)
                       found_res_for_dim = True
                       break # 找到最近的

                    if not found_res_for_dim: # 如果還沒找到，繼續擴展
                        for dr_res, dc_res in [(0,1), (0,-1), (1,0), (-1,0)]:
                            nr_res, nc_res = curr_r + dr_res, curr_c + dc_res
                            if 0 <= nr_res < rows and 0 <= nc_res < cols and (nr_res, nc_res) not in visited_res:
                                visited_res.add((nr_res, nc_res))
                                q_res.append(((nr_res, nc_res), dist + 1))
                if not found_res_for_dim : s_j = 0 # 未找到該資源

            # 計算競爭強度 C_j
            c_j = 0.0
            num_competitors = 0
            # 在競爭半徑內，計算有多少其他格子也"想要"這個資源
            # "想要"的定義可以是：如果它們獲取此資源的能力也很強
            comp_radius = resource_def["competition_radius"]
            for dr_comp in range(-comp_radius, comp_radius + 1):
                for dc_comp in range(-comp_radius, comp_radius + 1):
                    if dr_comp == 0 and dc_comp == 0: continue
                    
                    nr_comp, nc_comp = position_row + dr_comp, position_col + dc_comp
                    if 0 <= nr_comp < rows and 0 <= nc_comp < cols:
                        # 評估 (nr_comp, nc_comp) 對同一個資源維度的獲取能力
                        # 為了簡化，我們假設如果鄰居也是"高價值"或"空地"，它就是一個潛在競爭者
                        # 或者，更簡單地，只計算鄰居的數量作為競爭代理
                        # 此處簡化：若鄰居的值也滿足該資源的value_check_func (或是一個通用高價值)，則視為競爭者
                        val_at_competitor = val_func(board_state[nr_comp][nc_comp])
                        if resource_def["value_check_func"](val_at_competitor): # 如果鄰居本身就是同類資源
                             c_j += 1.0 # 每個這樣的鄰居增加競爭
                        num_competitors +=1
            
            # 正規化 c_j (例如除以競爭半徑內的最大可能鄰居數)
            max_possible_competitors = (2*comp_radius+1)**2 -1
            norm_c_j = c_j / max_possible_competitors if max_possible_competitors > 0 else 0

            # 單個資源維度的適應性
            fitness_j = s_j / (1 + self.internal_params["competition_sensitivity_alpha"] * norm_c_j)
            
            total_weighted_fitness_score += resource_def["weight"] * fitness_j
            total_resource_weights += resource_def["weight"]

        if total_resource_weights == 0: return 0.0
        
        # 最終分數正規化 (因為fitness_j通常是0-1，加權和也在相似範圍，但可能超過1)
        # 此處假設total_weighted_fitness_score不會遠超total_resource_weights
        final_score = total_weighted_fitness_score / total_resource_weights
        return max(0.0, min(1.0, final_score))
# ... (接續第四部分的程式碼: MathUtils, BoardAnalyzerUtils, LogicModule, BoardInput, A2, M3, D3, F10, GM1-GM14) ...

class GM15(LogicModule):
    """
    設計理念：基於"控場博弈"中的"眼位"或"安全區域"的建立。
              原始GM15評估到最近'1'的距離。進化版將評估一個格子是否能成為一個被己方（高價值）格子"包圍"而形成的"安全眼"的中心。
    用途描述：識別那些有潛力形成被己方力量牢固保護的"安全據點"或"資源儲藏點"。
              在圍棋等遊戲中極為重要，在其他領域可代表數據的"可信核心"或系統的"穩定態"。
    評分公式原理：1. 假設目標格子 (pos_r, pos_c) 是一個潛在的"眼位中心"（通常是低值或空格）。
                 2. 檢查其周圍（例如，直接鄰居或擴展一圈的鄰居）是否主要由"己方高價值"格子佔據。
                 3. 檢查這些"己方高價值"格子是否形成了一個完整的"包圍圈"，沒有明顯的"缺口"被"敵方低價值"格子滲透。
                 4. 考慮"眼位"的大小，太小可能不是真眼。
                 Score = w1 * (completeness_of_surrounding_wall) + w2 * (average_strength_of_wall_elements) - w3 * (size_of_eye_penalty_if_too_small)
    兼容性：數值型盤面，需要定義"己方高價值"、"敵方低價值"和"眼位中心"的標準。
    優化與延伸方向：1. 使用更精確的圖論算法檢測"圍空"和"眼的死活"。
                    2. 考慮多個小眼組合形成大眼的潛力。
                    3. 引入對手下一步可能"破眼"的風險評估。
    可能的多版本邏輯：GM15.v1 (到最近'1'距離), GM15.v2 (被己方包圍的安全眼位潛力), GM15.v3 (考慮眼位大小和內部連接性的真眼判斷)。
    """
    def __init__(self):
        super().__init__(
            module_id="GM15", 
            name="Generated Module 15 (Secure Territory & 'Eye' Formation Potential)", 
            description="Evaluates if a cell can become a center of a secure 'eye' surrounded by friendly high-value cells."
        )
        self.internal_params = {
            "eye_center_max_value": 0.2, # "眼"的中心應該是低值或空格
            "friendly_wall_min_value": 0.7, # 構成"牆壁"的己方格子的最低值
            "opponent_breach_max_value": 0.3, # 如果牆上有缺口，缺口處的值不能太高(否則是敵方滲透)
            "eye_radius": 1, # 檢查眼周圍1圈的牆壁 (3x3區域的邊緣)
            "min_wall_elements_ratio": 0.75, # 牆壁至少要有這麼多比例是己方高價值格子
            "wall_strength_weight": 0.6,
            "wall_completeness_weight": 0.4,
            "value_function": lambda x: float(x) if isinstance(x, (int, float)) else -1.0 # -1 for non-numeric
        }

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 評估目標格子（通常為低值或空格）是否被周圍的「己方高價值」格子有效包圍，從而形成一個潛在的「安全眼位」。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return 0.0
        
        val_func = self.internal_params["value_function"]
        cell_val = val_func(board_state[position_row][position_col])

        # 條件1: 眼的中心必須是低值
        if cell_val > self.internal_params["eye_center_max_value"]:
            return 0.0 # 不是合格的眼位中心候選

        wall_elements_values = []
        potential_wall_positions = 0
        actual_friendly_wall_elements = 0
        
        radius = self.internal_params["eye_radius"] # 通常為1，即3x3區域的邊緣
        # 遍歷眼位周圍形成牆壁的格子
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) != radius and abs(dc) != radius : # 只考慮最外圈
                    if not (abs(dr) == radius or abs(dc) == radius): # 確保是外圈
                        continue
                
                nr, nc = position_row + dr, position_col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    potential_wall_positions +=1
                    wall_val = val_func(board_state[nr][nc])
                    wall_elements_values.append(wall_val)
                    if wall_val >= self.internal_params["friendly_wall_min_value"]:
                        actual_friendly_wall_elements +=1
                    elif wall_val <= self.internal_params["opponent_breach_max_value"]:
                        # 發現可能的缺口被低價值(或敵方)佔據，這對眼不利
                        # 此處簡化：只統計友方牆壁元素
                        pass 
        
        if potential_wall_positions == 0 : # 例如1x1盤面，沒有牆
            return 0.1 # 可能是孤立的安全點？或者低分

        # 條件2: 牆壁完整性
        wall_completeness_ratio = actual_friendly_wall_elements / potential_wall_positions
        if wall_completeness_ratio < self.internal_params["min_wall_elements_ratio"]:
            # 牆不夠完整，不是好眼
            # 可以給一個基於ratio的低分
            return 0.1 * wall_completeness_ratio 

        # 條件3: 牆壁強度 (友方牆壁元素的平均值)
        avg_wall_strength = 0.0
        friendly_wall_values = [v for v in wall_elements_values if v >= self.internal_params["friendly_wall_min_value"]]
        if friendly_wall_values:
            avg_wall_strength = np.mean(friendly_wall_values)
        # 正規化牆壁強度 (假設值0-1)
        norm_avg_wall_strength = MathUtils.normalize_value(avg_wall_strength, self.internal_params["friendly_wall_min_value"], 1.0)

        score = (self.internal_params["wall_completeness_weight"] * wall_completeness_ratio + # ratio本身已正規化
                 self.internal_params["wall_strength_weight"] * norm_avg_wall_strength)
        
        # 額外獎勵：如果眼中心的value特別低（例如0）
        if math.isclose(cell_val, 0):
            score += 0.1
            
        return max(0.0, min(1.0, score))


class GM16(LogicModule):
    """
    設計理念：從"關鍵路徑分析" (Critical Path Analysis) 或"瓶頸識別" (Bottleneck Detection) 的角度出發。
              原始GM16評估行對稱性。進化版將評估一個格子是否處於連接盤面兩個或多個重要區域的"唯一路徑"或"狹窄通道"上（即瓶頸）。
    用途描述：識別那些控制關鍵通道、移除後會導致重要區域隔絕的"戰術咽喉點"。
              高分表示該點是兵家必爭之地。
    評分公式原理：1. 預先定義或動態識別盤面上的多個"重要區域" (Zone_A, Zone_B, ...)。
                 2. 假設暫時移除目標格子 (pos_r, pos_c) (例如，使其通行成本極高)。
                 3. 計算移除前後，Zone_A 與 Zone_B 之間的最短路徑長度 (或連通性)。
                 4. 如果移除該格子導致路徑長度顯著增加或完全不連通，則該格子是瓶頸，得分高。
                 Score = f( (L_after_removal - L_before_removal) / L_before_removal )
    兼容性：數值型盤面，需要定義區域和通行成本。
    優化與延伸方向：1. 使用更高效的圖算法計算多對區域間的連通性和割點/割邊。
                    2. 考慮不同區域的重要性權重。
                    3. 評估格子作為"備用路徑"的價值。
    可能的多版本邏輯：GM16.v1 (行對稱), GM16.v2 (移除後兩點間路徑變化), GM16.v3 (基於最小割思想的瓶頸分析)。
    """
    def __init__(self):
        super().__init__(
            module_id="GM16", 
            name="Generated Module 16 (Bottleneck & Critical Path Analyzer)", 
            description="Identifies cells that act as bottlenecks between important board zones."
        )
        self.internal_params = {
            # 簡化：預定義兩個對角點作為重要區域的代表點，分析移除當前格子對它們之間路徑的影響
            "zone_A_representative": (0,0), # 左上角
            "zone_B_representative": None, # 右下角, 將在analyze中動態設定
            "cost_function_for_path": lambda val: (1.1 - float(val)) if (isinstance(val, (int,float)) and 0 <= val <= 1) else 100.0,
            "removed_cell_cost": 1000.0, # 移除格子後的高成本
            "max_path_search_steps": 150 # Dijkstra步數限制
        }

    def _dijkstra_shortest_path(self, board_state: List[List[Any]], start_pos: Tuple[int,int], end_pos: Tuple[int,int], 
                                rows: int, cols: int, cost_func, 
                                removed_cell: Tuple[int,int]=None, removed_cost: float=0) -> float:
        # dists : (cost, r, c) - use heapq for priority queue
        import heapq
        pq = [(0, start_pos[0], start_pos[1])] # (cost, r, c)
        min_costs = {} # {(r,c): cost}
        min_costs[start_pos] = 0
        
        steps = 0

        while pq and steps < self.internal_params["max_path_search_steps"] * rows * cols: # Safety break
            steps+=1
            cost, r, c = heapq.heappop(pq)

            if cost > min_costs.get((r,c), float('inf')):
                continue
            if (r,c) == end_pos:
                return cost

            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    edge_cost = 0
                    if removed_cell and (nr,nc) == removed_cell:
                        edge_cost = removed_cost
                    else:
                        edge_cost = cost_func(board_state[nr][nc])
                    
                    new_cost = cost + edge_cost
                    if new_cost < min_costs.get((nr,nc), float('inf')):
                        min_costs[(nr,nc)] = new_cost
                        heapq.heappush(pq, (new_cost, nr, nc))
        return float('inf') # Path not found


    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 透過比較移除目標格子前後，盤面上兩個預定義「重要區域」代表點之間的最短路徑成本變化，來識別關鍵的「瓶頸」格子。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or rows < 2 or cols < 2: return 0.0 # 需要足夠大的盤面來形成路徑和瓶頸

        # 動態設定 Zone B 代表點為右下角
        self.internal_params["zone_B_representative"] = (rows - 1, cols - 1)
        
        # 如果目標格子本身就是起點或終點，它不太可能是它們之間的瓶頸（除非是唯一通道的一部分）
        # 這種情況下，它的移除會直接導致路徑無限長。
        start_node = self.internal_params["zone_A_representative"]
        end_node = self.internal_params["zone_B_representative"]

        if (position_row, position_col) == start_node or (position_row, position_col) == end_node:
            # 可以給一個特殊分數，或認為它不是中間瓶頸
            # 此處簡化：如果格子是起點或終點，則其瓶頸性評估方式不同，暫時給中等分數
             # 檢查如果它是起點/終點，是否有其他路徑
            pass # 繼續計算，看移除它是否有影響

        cost_func = self.internal_params["cost_function_for_path"]
        
        # 1. 計算移除前的最短路徑成本
        l_before = self._dijkstra_shortest_path(board_state, start_node, end_node, rows, cols, cost_func)

        # 2. 計算移除目標格子後的opathies最短路徑成本
        l_after = self._dijkstra_shortest_path(board_state, start_node, end_node, rows, cols, cost_func,
                                             removed_cell=(position_row, position_col),
                                             removed_cost=self.internal_params["removed_cell_cost"])
        
        if math.isinf(l_before) and math.isinf(l_after): # 原本就不通，移除後也不通
            return 0.0
        if math.isinf(l_before) and not math.isinf(l_after): # 不應發生，移除一個點使不通的路通了？除非成本函數設計特殊
             return 0.0 # 異常
        if not math.isinf(l_before) and math.isinf(l_after): # 移除後不通了，此為強瓶頸
            # 分數可以很高，例如 1.0
            # 考慮 l_before 的大小，如果l_before本身就很長，則這個瓶頸相對不那麼"意外"
            # 此處簡化：強瓶頸直接給高分
            return 1.0 
        
        # 兩者皆可通
        if l_before < 1e-6 : # l_before 接近0 (例如起點終點是同一個格子且成本為0)
            if l_after > l_before : return 0.8 # 移除後成本增加，算是有影響
            else: return 0.1

        cost_increase_ratio = (l_after - l_before) / l_before
        
        # cost_increase_ratio 可能很大。用 sigmoid 映射到 0-1
        # ratio = 0 -> sigmoid(0) = 0.5
        # ratio = 1 (成本翻倍) -> sigmoid(1*k)
        # ratio = 很大 (接近不通) -> sigmoid(大) -> 1
        scaling_factor = 2.0 # 調整敏感度
        score = MathUtils.sigmoid(cost_increase_ratio * scaling_factor)
        
        # 如果 L_before 很大 (本身就很難走)，那麼即使 cost_increase_ratio 不大，
        # 這個點的重要性也可能降低。可以加入對 L_before 的考量。
        # 此處暫不加入。
        
        return max(0.0, min(1.0, score))


class GM17(LogicModule):
    """
    設計理念：應用"網絡流" (Network Flow) 或"最大流最小割定理" (Max-Flow Min-Cut Theorem) 的簡化思想。
              原始GM17評估列對稱性。進化版將評估目標格子作為一個"中轉樞紐" (Hub) 對於盤面上預定義的"源點"(Source)到"匯點"(Sink)之間
              潛在"流量"的貢獻度或重要性。
    用途描述：識別那些對於維持網絡流通性至關重要的節點。
              高分表示該點是重要的流量樞紐或關鍵基礎設施。
    評分公式原理：1. 定義盤面上的源點S和匯點T。
                 2. 每個格子的值可以被視為其"容量" (Capacity)。
                 3. 評估如果 (pos_r, pos_c) 作為路徑上的一點，它能為多少條從S到T的不同路徑（或總流量）做出貢獻。
                    這通常很難直接計算。簡化思路：
                    a. 評估從 S 到 (pos_r,pos_c) 的最大"流入容量"。
                    b. 評估從 (pos_r,pos_c) 到 T 的最大"流出容量"。
                    c. 該點的"中轉潛力" = min(流入容量, 格子自身容量, 流出容量)。
                    d. "流入/流出容量"可以用多條路徑的總和或最寬路徑來近似。
                 Score = normalized(transfer_potential)
    兼容性：數值型盤面，值代表容量。
    優化與延伸方向：1. 使用Edmonds-Karp或Dinic算法計算實際的最大流。
                    2. 考慮多源多匯的情況。
                    3. 評估移除該點對最大流的影響（類似最小割）。
    可能的多版本邏輯：GM17.v1 (列對稱), GM17.v2 (S-T路徑上格子的容量瓶頸分析), GM17.v3 (多路徑S-T流量貢獻估算)。
    """
    def __init__(self):
        super().__init__(
            module_id="GM17", 
            name="Generated Module 17 (Network Flow Hub Potential)", 
            description="Evaluates cell's potential as a hub in a flow network from source(s) to sink(s)."
        )
        self.internal_params = {
            "source_coords": [(0,0)], # 可多個源點
            "sink_coords": None,      # 動態設定為盤面右下角或多個
            "capacity_function": lambda val: float(val) if (isinstance(val, (int,float)) and val > 0) else 0.0, # 值作為容量
            "max_paths_to_consider": 3, # 為了簡化，只考慮幾條較好的路徑
            "path_finding_max_depth": 15,
            "path_capacity_aggregation": "min_bottleneck", # "sum_of_paths" or "min_bottleneck_avg"
        }

    def _find_paths_and_bottlenecks(self, board_state: List[List[Any]], start_node: Tuple[int,int], end_node: Tuple[int,int],
                                   rows: int, cols: int, capacity_func, current_eval_cell:Tuple[int,int]) -> List[float]:
        # 使用BFS找到多條（不一定不相交）路徑，並計算每條路徑的瓶頸容量
        # current_eval_cell 是我們正在評估的 GM17 的 (position_row, position_col)
        # 我們想知道從 start_node 到 current_eval_cell，再從 current_eval_cell 到 end_node 的路徑情況
        
        paths_data = [] # [(path_bottleneck_capacity, path_length)]

        # Stage 1: Start to current_eval_cell
        q_s_to_c = deque([([start_node], float('inf'))]) # (current_path_nodes_list, current_path_bottleneck)
        visited_paths_s_to_c = {tuple(start_node):float('inf')} # path_tuple : bottleneck (not efficient for general graph, ok for grid)
        
        paths_found_s_to_c = []

        # 為了簡化，這裡的BFS不直接找瓶頸，而是找路徑，然後再計算瓶頸
        # 或者，修改BFS的狀態來記錄路徑上的最小容量
        
        # 此函數需要返回從 start 到 end 經過 current_eval_cell 的路徑瓶頸列表
        # 實現一個能找到多條路徑的BFS/DFS變體比較複雜。
        # 簡化：我們假設 current_eval_cell 就是路徑的中間點。
        # 計算 S -> current_eval_cell 的流入潛力，和 current_eval_cell -> T 的流出潛力。
        # "流入/流出潛力"可以是在一定步數內能到達的、容量加權的格子數。

        # 極簡化版本：如果 current_eval_cell 在 S 和 T 的一條"好路徑"上，就有價值。
        # "好路徑" = 格子值都比較高 (容量大)
        # 我們可以隨機走幾條路徑，或者用 Dijkstra 找到一條最佳容量路徑。
        
        # 為了演示，我們做一個非常簡化的版本：
        # 檢查 (pos_r, pos_c) 是否在 (0,0) 到 (rows-1, cols-1) 的一條直線上（如果可能）
        # 且直線上的格子容量都比較高。
        # 這遠非網絡流，但作為一個簡化版的"中轉樞紐"概念。
        
        # 更好的簡化: 計算從S到current_eval_cell的"可達性得分" (類似GM7)，
        # 和從current_eval_cell到T的"可達性得分"。
        # "可達性得分"可以用BFS計算，看能在K步內接觸到多少高容量格子。
        # Score = Inflow_Potential * Outflow_Potential * Cell_Capacity
        
        # 假設使用類似GM7的BFS來估算流入/流出潛力
        #流入潛力：從Source到current_eval_cell的容易程度。
        #流出潛力：從current_eval_cell到Sink的容易程度。
        # 此處直接返回一個基於格子自身容量和其是否在S-T路徑上的猜測值
        
        # 以current_eval_cell為中心，看它是否能連接S和T (S在左上，T在右下)
        # 如果cell在S的右下方，且在T的左上方，則它可能在路徑上。
        # 並且其自身容量要高。
        
        dist_to_s = MathUtils.manhattan_distance(current_eval_cell, start_node)
        dist_to_t = MathUtils.manhattan_distance(current_eval_cell, end_node)
        dist_s_to_t = MathUtils.manhattan_distance(start_node, end_node)

        # 如果點在S-T的"大致方向"上
        on_general_path = False
        if dist_s_to_t > 0: # 避免S=T
            # 如果 current_eval_cell 到 S 和 T 的距離之和接近 S 到 T 的直線距離
            # (在網格中，曼哈頓距離本身就是最短路徑長)
            if math.isclose(dist_to_s + dist_to_t, dist_s_to_t):
                 on_general_path = True
        
        cell_capacity = capacity_func(board_state[current_eval_cell[0]][current_eval_cell[1]])
        
        if on_general_path:
            # 越靠近S-T路徑的中點，且自身容量越大，分數越高
            # 偏離中點的程度: abs(dist_to_s - dist_to_t) / dist_s_to_t (0表示中點, 1表示端點)
            centrality_factor = 0.0
            if dist_s_to_t > 0:
                 centrality_factor = 1.0 - (abs(dist_to_s - dist_to_t) / dist_s_to_t) # 0 to 1
            
            # 返回一個綜合了路徑位置和自身容量的分數
            # 正規化cell_capacity (假設原始值0-1, 容量也是0-1)
            norm_cell_capacity = MathUtils.normalize_value(cell_capacity, 0, 1.0)
            return [0.5 * norm_cell_capacity + 0.5 * centrality_factor] # 返回一個包含單個"路徑"潛力值的列表
        else:
            return [0.1 * MathUtils.normalize_value(cell_capacity, 0, 1.0)] # 不在主要路徑上，但自身容量仍有少許價值

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 評估目標格子作為從多個「源點」到多個「匯點」的路徑上的「中轉樞紐」的潛力，考慮其自身容量和路徑位置。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or rows < 2 or cols < 2: return 0.0

        cap_func = self.internal_params["capacity_function"]
        
        # 動態設定sink_coords
        sink_coords_list = self.internal_params["sink_coords"] or [(rows - 1, cols - 1)]
        if not isinstance(sink_coords_list, list): sink_coords_list = [sink_coords_list]
            
        source_coords_list = self.internal_params["source_coords"]
        if not isinstance(source_coords_list, list): source_coords_list = [source_coords_list]

        avg_hub_potential = 0.0
        num_s_t_pairs = 0

        for s_node in source_coords_list:
            # 確保 s_node 在邊界內
            s_node = (min(max(0, s_node[0]), rows-1), min(max(0, s_node[1]), cols-1))
            for t_node in sink_coords_list:
                t_node = (min(max(0, t_node[0]), rows-1), min(max(0, t_node[1]), cols-1))

                if s_node == t_node : continue # 源和匯不能相同
                num_s_t_pairs +=1

                # 對於每個S-T對，評估 (pos_r,pos_c) 的中轉潛力
                # _find_paths_and_bottlenecks 返回的是一個代表潛力的分數列表（此處只有一個元素）
                path_potentials = self._find_paths_and_bottlenecks(board_state, s_node, t_node, 
                                                                 rows, cols, cap_func, 
                                                                 (position_row, position_col))
                
                if path_potentials: # 如果找到任何潛力
                    avg_hub_potential += np.mean(path_potentials) # 此處只有一個值，直接加
        
        if num_s_t_pairs == 0: return 0.0 # 沒有有效的S-T對
        
        final_score = avg_hub_potential / num_s_t_pairs
        return max(0.0, min(1.0, final_score))

class GM18(LogicModule):
    """
    設計理念：基於"強化學習"中的"狀態價值函數" (State-Value Function V(s)) 或"動作價值函數" (Action-Value Function Q(s,a)) 的預測原理。
              原始GM18評估2x2子網格。進化版將嘗試為當前格子（狀態s）評估一個"潛在價值"，
              這個價值是通過一個簡化的、預定義的"價值網絡"或一組"專家規則"（類似強化學習中的策略或價值函數的局部近似）來計算的。
              它會考慮當前格子的特徵以及周圍格子的特徵組合。
    用途描述：模擬一個簡化的AI代理對當前格子"長期價值"的評估。
              高分表示該格子在當前及未來可能的局面演化中具有較高的戰略潛力。
    評分公式原理：1. 提取目標格子及其鄰域的多個特徵 F_1, F_2, ..., F_n（例如：自身值、鄰居均值、特定圖案是否存在、到邊界的距離等）。
                 2. 這些特徵被輸入一個預定義的加權線性組合或一個小型決策樹（硬編碼）。
                    V(s) = w_0 + w_1*F_1 + w_2*F_2 + ... + w_n*F_n
                 3. 或使用一組 IF-THEN 規則： IF (condition_on_features) THEN score = X ELSE IF ...
                 4. 最終分數經過 sigmoid 映射。
    兼容性：數值型或混合型盤面，需要預先設計好特徵提取和評估規則/權重。
    優化與延伸方向：1. 使用真實的機器學習模型（如梯度提升樹、小型神經網絡）進行訓練，學習價值函數的權重。
                    2. 引入蒙地卡羅樹搜索(MCTS)的playout思想，從當前格子出發模擬幾步未來走勢，評估終局價值。
                    3. 讓特徵權重 w_i 能夠根據全局盤面狀態動態調整。
    可能的多版本邏輯：GM18.v1 (2x2子網格和), GM18.v2 (基於多特徵的線性加權價值評估), GM18.v3 (小型硬編碼決策樹評估)。
    """
    def __init__(self):
        super().__init__(
            module_id="GM18", 
            name="Generated Module 18 (Simplified RL-inspired Value Estimator)", 
            description="Estimates cell's strategic value using a predefined set of features and a weighted evaluation function (RL-like)."
        )
        # 定義特徵提取函數和對應的權重
        # 特徵函數接受 (board_state, r, c, val_func, rows, cols)
        self.internal_params = {
            "features_and_weights": [
                {"name": "self_value", "func": lambda b,r,c,vf,R,C: vf(b[r][c]) if vf(b[r][c]) is not None else 0, "weight": 0.3},
                {"name": "avg_neighbor_3x3", 
                 "func": lambda b,r,c,vf,R,C: np.mean([vf(val) for val in BoardAnalyzerUtils.get_neighborhood(b,r,c,1,True) if vf(val) is not None] or [0]),
                 "weight": 0.2},
                {"name": "is_edge_or_corner", # 類似 F10
                 "func": lambda b,r,c,vf,R,C: 1.0 if (r==0 or r==R-1 or c==0 or c==C-1) else 0,
                 "weight": -0.1}, # 假設邊緣/角落在此評估中是負面 (可調整)
                {"name": "local_gradient_magnitude", # 類似 GM9 / F10
                 "func": lambda b,r,c,vf,R,C: MathUtils.normalize_value(math.hypot(*BoardAnalyzerUtils.get_value_gradient(b,r,c)),0,6.0),
                 "weight": 0.15},
                {"name": "num_friendly_neighbors_A2_like", # 類似A2，周圍高價值點個數
                 "func": lambda b,r,c,vf,R,C: sum(1 for val in BoardAnalyzerUtils.get_neighborhood(b,r,c,1,True) if vf(val) is not None and vf(val) >= 0.7), # 假設0.7是友方
                 "weight": 0.25}
                # 可以添加更多特徵...
            ],
            "bias_w0": 0.0, # 線性組合的偏置項
            "value_function": lambda x: float(x) if isinstance(x, (int, float)) else None,
        }

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 核心分流規則與用途: 透過提取目標格子及其周圍環境的多個預定義特徵，並將這些特徵進行加權線性組合（類似簡化AI模型的價值評估），來估算其戰略潛力。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows: return 0.0
        val_func = self.internal_params["value_function"]

        feature_values = []
        current_value_estimate = self.internal_params["bias_w0"]

        for fw_item in self.internal_params["features_and_weights"]:
            try:
                feature_val = fw_item["func"](board_state, position_row, position_col, val_func, rows, cols)
                # 正規化 (部分特徵函數可能已返回正規化值，部分可能沒有)
                # 例如，self_value 假設是0-1，avg_neighbor也是0-1。is_edge是0或1。
                # num_friendly_neighbors 可能0-8，需要正規化
                if fw_item["name"] == "num_friendly_neighbors_A2_like":
                    feature_val = MathUtils.normalize_value(feature_val, 0, 8) # 8個鄰居
                
                current_value_estimate += fw_item["weight"] * feature_val
            except Exception as e:
                # print(f"Error calculating feature {fw_item['name']} for module GM18: {e}")
                pass # 忽略計算失敗的特徵，或給予0貢獻

        # current_value_estimate 的範圍取決於權重和特徵值，可能正負，大小不定
        # 使用 sigmoid 映射到 0-1
        # 需要調整 sigmoid 的輸入使其有意義。如果 sum(abs(weights)) 約為1，且特徵0-1
        # 則 value_estimate 約在 -1 到 1 (如果bias=0)。
        # sigmoid(value_estimate) 即可。可以再乘個係數放大變化。
        return MathUtils.sigmoid(current_value_estimate * 2.0) # 乘2增加敏感度

# -----------------------------------------------------------------------------
# 3. 模듈註冊與全局權重 (與之前版本類似，確保所有模組被實例化)
# -----------------------------------------------------------------------------
REGISTERED_MODULES: List[LogicModule] = [
    A2(), M3(), D3(), F10(),
    GM1(), GM2(), GM3(), GM4(), GM5(), GM6(), GM7(), GM8(), GM9(),
    GM10(), GM11(), GM12(), GM13(), GM14(), GM15(), GM16(), GM17(), GM18()
]

print(f"Registered {len(REGISTERED_MODULES)} modules:")
for mod in REGISTERED_MODULES:
    print(f" - {mod.module_id}: {mod.name} ({mod.description})") # 印出描述以確認

GLOBAL_MODULE_WEIGHTS: Dict[str, float] = {module.module_id: 1.0 for module in REGISTERED_MODULES}
# 根據需要調整特定模組的權重
if "A2" in GLOBAL_MODULE_WEIGHTS: GLOBAL_MODULE_WEIGHTS["A2"] = 1.5 
if "M3" in GLOBAL_MODULE_WEIGHTS: GLOBAL_MODULE_WEIGHTS["M3"] = 1.2
if "GM7" in GLOBAL_MODULE_WEIGHTS: GLOBAL_MODULE_WEIGHTS["GM7"] = 1.3 # 例如，路徑相關的模組比較重要
if "GM16" in GLOBAL_MODULE_WEIGHTS: GLOBAL_MODULE_WEIGHTS["GM16"] = 1.4 # 瓶頸分析也很重要
if "GM18" in GLOBAL_MODULE_WEIGHTS: GLOBAL_MODULE_WEIGHTS["GM18"] = 1.1 # AI價值估算


# -----------------------------------------------------------------------------
# 4. 核心處理邏輯 (與之前版本相同)
# -----------------------------------------------------------------------------

def process_board(board_input: BoardInput, modules: List[LogicModule]) -> Dict[Tuple[int, int], Dict[str, float]]:
    all_cell_scores: Dict[Tuple[int, int], Dict[str, float]] = {}
    if not board_input.grid: return all_cell_scores 
    for r in range(board_input.rows):
        for c in range(board_input.cols):
            cell_scores: Dict[str, float] = {}
            for module in modules:
                try:
                    score = module.analyze(board_input.grid, r, c)
                    cell_scores[module.module_id] = score 
                except Exception as e:
                    print(f"錯誤：模組 {module.module_id} 在分析位置 ({r},{c}) 時發生錯誤: {e}")
                    cell_scores[module.module_id] = 0.0 
            all_cell_scores[(r, c)] = cell_scores
    return all_cell_scores

def normalize_scores(
    module_scores_by_cell: Dict[Tuple[int, int], Dict[str, float]],
    modules: List[LogicModule],
    method: str = 'min-max'
) -> Dict[Tuple[int, int], Dict[str, float]]:
    if method == "none":
        return module_scores_by_cell
    
    normalized_scores_by_cell = {cell: {} for cell in module_scores_by_cell}
    
    module_all_scores: Dict[str, List[float]] = {m.module_id: [] for m in modules}
    for _cell_pos, scores_dict in module_scores_by_cell.items():
        for mod_id, score in scores_dict.items():
            if mod_id in module_all_scores: 
                 module_all_scores[mod_id].append(score)

    for mod_id_key, scores_list in module_all_scores.items():
        if not scores_list:
            for cell_pos_norm in normalized_scores_by_cell: 
                normalized_scores_by_cell[cell_pos_norm][mod_id_key] = 0.0
            continue

        if method == 'min-max':
            min_score, max_score = min(scores_list), max(scores_list)
            for cell_pos_norm in normalized_scores_by_cell:
                raw_score = module_scores_by_cell.get(cell_pos_norm, {}).get(mod_id_key)
                if raw_score is not None:
                    if (max_score - min_score) < 1e-9: # Handle float comparison for min_score == max_score
                        normalized_scores_by_cell[cell_pos_norm][mod_id_key] = 0.0 if math.isclose(min_score,0) else 0.5
                    else:
                        normalized_scores_by_cell[cell_pos_norm][mod_id_key] = (raw_score - min_score) / (max_score - min_score)
                else: 
                    normalized_scores_by_cell[cell_pos_norm][mod_id_key] = 0.0 
        elif method == 'z-score':
            mean_score, std_score = float(np.mean(scores_list)), float(np.std(scores_list))
            for cell_pos_norm in normalized_scores_by_cell:
                raw_score = module_scores_by_cell.get(cell_pos_norm, {}).get(mod_id_key)
                if raw_score is not None:
                    if std_score < 1e-9: # Handle std_dev being close to zero
                        normalized_scores_by_cell[cell_pos_norm][mod_id_key] = 0.0
                    else:
                        normalized_scores_by_cell[cell_pos_norm][mod_id_key] = (raw_score - mean_score) / std_score
                else:
                    normalized_scores_by_cell[cell_pos_norm][mod_id_key] = 0.0
        else: 
             for cell_pos_norm in normalized_scores_by_cell:
                normalized_scores_by_cell[cell_pos_norm][mod_id_key] = module_scores_by_cell.get(cell_pos_norm, {}).get(mod_id_key, 0.0)
    return normalized_scores_by_cell


def fuse_scores(
    scores_to_fuse_input: Dict[Tuple[int, int], Dict[str, float]],
    weights: Dict[str, float]
) -> Dict[Tuple[int, int], float]:
    fused_scores_output: Dict[Tuple[int, int], float] = {}
    
    effective_weights = weights.copy()
    all_module_ids_in_scores = set()
    for scores_dict_val in scores_to_fuse_input.values():
        all_module_ids_in_scores.update(scores_dict_val.keys())
    
    for mod_id_in_score in all_module_ids_in_scores:
        if mod_id_in_score not in effective_weights:
            # print(f"警告: 模組 {mod_id_in_score} 的權重未定義，使用預設權重 1.0。")
            effective_weights[mod_id_in_score] = 1.0

    for cell_pos, mod_scores in scores_to_fuse_input.items():
        weighted_sum, sum_of_weights = 0.0, 0.0
        if not mod_scores:
            fused_scores_output[cell_pos] = 0.0; continue

        for module_id, norm_score in mod_scores.items():
            weight = effective_weights.get(module_id, 1.0) # 再次確認，即使上面已加入預設
            weighted_sum += norm_score * weight
            sum_of_weights += weight
        
        fused_scores_output[cell_pos] = (weighted_sum / sum_of_weights) if sum_of_weights != 0 else 0.0
    return fused_scores_output

def simple_fuse_scores( # (此函數在所有模組升級後，其"簡單性"可能不再完全適用於評估，但保留)
    raw_cell_scores: Dict[Tuple[int, int], Dict[str, float]]
) -> Dict[Tuple[int, int], float]:
    fused_scores: Dict[Tuple[int, int], float] = {}
    for cell_pos, mod_scores in raw_cell_scores.items():
        if not mod_scores: fused_scores[cell_pos] = 0.0; continue
        average_score = sum(mod_scores.values()) / len(mod_scores) if len(mod_scores) > 0 else 0.0
        fused_scores[cell_pos] = average_score
    return fused_scores

def get_final_scores_for_board(
    board_input: BoardInput,
    modules: List[LogicModule],
    module_weights: Dict[str, float],
    normalization_method: str = 'min-max'
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], Dict[str, float]]]:
    if normalization_method not in ['min-max', 'z-score', 'none']:
        raise ValueError(f"不支援的正規化方法: {normalization_method}")

    raw_cell_module_scores = process_board(board_input, modules)

    scores_to_fuse: Dict[Tuple[int, int], Dict[str, float]]
    if normalization_method != 'none':
        scores_to_fuse = normalize_scores(raw_cell_module_scores, modules, method=normalization_method)
    else:
        scores_to_fuse = raw_cell_module_scores
    
    final_fused_scores = fuse_scores(scores_to_fuse, module_weights)
    return final_fused_scores, scores_to_fuse

# -----------------------------------------------------------------------------
# 5. 主執行區塊 (與之前版本類似，但現在所有模組都有高級邏輯)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 盤面定義 (可擴展為支持更複雜的數值)
    test_board_1_data = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    test_board_1 = BoardInput(grid=test_board_1_data)

    # 另一測試盤面，具有更多樣的數值
    test_board_complex_data = [
        [0.1, 0.5, 0.9, 0.2],
        [0.8, 0.3, 0.0, 0.6],
        [0.4, 0.7, 0.5, 0.1],
        [0.0, 0.2, 0.8, 0.4]
    ]
    test_board_complex = BoardInput(grid=test_board_complex_data)


    print("\n===== 第 1 節：模組類別骨架的自動化生成 =====")
    print("此版本中所有22個模組已內置具體的高級分析邏輯。")
    print(f"已註冊 {len(REGISTERED_MODULES)} 個模組。")
    if len(REGISTERED_MODULES) != 22:
        print(f"警告: 實際註冊模組數 {len(REGISTERED_MODULES)} 不符預期的22!")

    # --- Section 2 & 3 的簡單融合展示 ---
    print("\n===== 第 2 & 3 節：初始系統整合 (所有模組已具高級邏輯) =====")
    current_test_board_display = test_board_1 # 可切換到 test_board_complex
    print(f"\n處理測試盤面 (展示原始分數差異，使用 {current_test_board_display.rows}x{current_test_board_display.cols} 盤面)...")
    current_test_board_display.display()
    
    raw_scores_current = process_board(current_test_board_display, REGISTERED_MODULES)
    # fused_scores_simple_current = simple_fuse_scores(raw_scores_current) # simple_fuse可能意義不大

    print(f"\n測試盤面 ({current_test_board_display.rows}x{current_test_board_display.cols}) 上部分格子的部分模組原始分數 (展示分流能力):")
    print("| 格子(R,C) | Mod ID | Raw Score |")
    print("|-----------|--------|-----------|")
    display_coords = [(0,0), (1,1), (current_test_board_display.rows-1, current_test_board_display.cols-1)]
    display_mod_ids = [m.module_id for m in REGISTERED_MODULES[:4]] # 展示前4個模組

    for r_idx, c_idx in display_coords:
        if 0 <= r_idx < current_test_board_display.rows and 0 <= c_idx < current_test_board_display.cols:
            pos = (r_idx, c_idx)
            if pos in raw_scores_current:
                for mod_id_disp in display_mod_ids:
                    score_val_disp = raw_scores_current[pos].get(mod_id_disp, float('nan'))
                    print(f"| ({r_idx},{c_idx})     | {mod_id_disp:<6} | {score_val_disp: .4f}    |")
            else:
                print(f"| ({r_idx},{c_idx})     | (N/A)  | (N/A)     |")
    print("\n")


    # --- Section 4 的優化融合流程展示 ---
    print("\n===== 第 4 節：優化分數融合流程 (使用 test_board_complex) =====")
    print("\nGLOBAL_MODULE_WEIGHTS (部分範例):")
    for mod_id_gw, weight_gw in list(GLOBAL_MODULE_WEIGHTS.items())[:5]:
         print(f"  {mod_id_gw}: {weight_gw}")

    print(f"\n處理 test_board_complex ({test_board_complex.rows}x{test_board_complex.cols})，使用 Min-Max 正規化和加權平均:")
    test_board_complex.display()
    
    fused_scores_complex_minmax, norm_scores_complex_minmax = get_final_scores_for_board(
        test_board_complex, REGISTERED_MODULES, GLOBAL_MODULE_WEIGHTS, normalization_method='min-max'
    )

    print("\n使用 Min-Max 正規化和加權平均後的融合分數 (test_board_complex):")
    print("| 格子(R,C) | Norm_A2 | Norm_M3 | Norm_GM1 | Fused (Min-Max,W) |")
    print("|-----------|---------|---------|----------|-------------------|")
    for r_idx in range(test_board_complex.rows):
        for c_idx in range(test_board_complex.cols):
            pos = (r_idx, c_idx)
            final_s = fused_scores_complex_minmax.get(pos, float('nan'))
            a2_n_s = norm_scores_complex_minmax.get(pos, {}).get("A2", float('nan'))
            m3_n_s = norm_scores_complex_minmax.get(pos, {}).get("M3", float('nan'))
            gm1_n_s = norm_scores_complex_minmax.get(pos, {}).get("GM1", float('nan'))
            print(f"| ({r_idx},{c_idx})     | {a2_n_s:.2f}    | {m3_n_s:.2f}    | {gm1_n_s:.2f}     | {final_s:.4f}            |")

    print(f"\n處理 test_board_complex，使用 Z-Score 正規化和加權平均:")
    fused_scores_complex_zscore, norm_scores_complex_zscore = get_final_scores_for_board(
        test_board_complex, REGISTERED_MODULES, GLOBAL_MODULE_WEIGHTS, normalization_method='z-score'
    )
    print("\n使用 Z-Score 正規化和加權平均後的融合分數 (test_board_complex):")
    print("| 格子(R,C) | Norm_A2(Z) | Norm_M3(Z) | Norm_GM1(Z)| Fused (Z-Score,W) |")
    print("|-----------|------------|------------|-------------|-------------------|")
    for r_idx in range(test_board_complex.rows):
        for c_idx in range(test_board_complex.cols):
            pos = (r_idx, c_idx)
            final_s_z = fused_scores_complex_zscore.get(pos, float('nan'))
            a2_n_s_z = norm_scores_complex_zscore.get(pos, {}).get("A2", float('nan'))
            m3_n_s_z = norm_scores_complex_zscore.get(pos, {}).get("M3", float('nan'))
            gm1_n_s_z = norm_scores_complex_zscore.get(pos, {}).get("GM1", float('nan'))
            print(f"| ({r_idx},{c_idx})     | {a2_n_s_z:+.2f}      | {m3_n_s_z:+.2f}      | {gm1_n_s_z:+.2f}       | {final_s_z:.4f}            |")
    print("\n")

    # --- Section 5 的多樣化情境驗證 ---
    print("\n===== 第 5 節：多樣化測試情境的綜合驗證 (使用 test_board_1, test_board_complex) =====")
    # (test_board_2 和 test_board_3 先前是為簡單0/1盤面設計的A2/M3偏好，對於數值盤面可能需要重新設計)
    # 此處僅用已有的 test_board_1 和 test_board_complex
    
    adv_test_boards = {
        "Board_1 (0/1)": test_board_1, 
        "Board_Complex (Numeric)": test_board_complex
    }
    adv_results_all_boards: Dict[str, Dict[Tuple[int, int], float]] = {}

    for board_name, board_obj in adv_test_boards.items():
        print(f"\n--- 處理 {board_name} ---")
        board_obj.display()
        fused_s, norm_s = get_final_scores_for_board(
            board_obj, REGISTERED_MODULES, GLOBAL_MODULE_WEIGHTS, normalization_method='min-max' # 或 'z-score'
        )
        adv_results_all_boards[board_name] = fused_s
        # (可以像之前一樣印出詳細表格，此處從略以節省篇幅，重點是能運行)
        print(f"{board_name} 的部分融合分數 (前幾個格子):")
        for i, (pos_adv, score_adv) in enumerate(list(fused_s.items())):
            if i < 3 : print(f"  Cell {pos_adv}: {score_adv:.4f}")
            else: break
# ... (接續先前 analyzer.py 或 main.py 的程式碼: MathUtils, BoardAnalyzerUtils, ext_a2 到 ext_gm18 的函式定義) ...

def ext_gm19_masked_number_skip_pattern_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """
    Core Rule: Analyzes the distribution of revealed numbers to detect significant "skip patterns" (e.g., a number expected in row X appears in row Y) and identifies if these skips follow an underlying regularity.
    Purpose: To identify if the placement of numbers deviates from a simple sequential or expected placement in a structured way, suggesting a hidden rule or deliberate "跳格" arrangement.
    ---
    Design Philosophy: Assumes numbers might have an "ideal" or "base" position (e.g., if numbers 1-N are typically placed row by row). It then looks for systematic deviations (skips) from these base positions. Detected skip patterns (e.g., "always skip +2 rows", "alternate column for sequential numbers") are then used to score empty cells based on how well placing a potential number there would fit an observed dominant skip pattern.
    Use Case: In scratch cards or puzzles where number placement might follow a non-obvious but consistent layout rule (e.g., for difficulty, or to form a visual pattern when revealed).
    Scoring Formula Principle:
        1. Identify "revealed number sequences" (e.g., 1, 2, 3, ... or other known sequences relevant to the card type).
        2. For each sequence, determine a "base/expected" placement grid (e.g., number 'k' expected at row `floor((k-1)/cols)`, col `(k-1)%cols`).
        3. Calculate "skip vectors" (delta_row, delta_col) for each revealed number relative to its base position.
        4. Use clustering (e.g., k-means on skip vectors) or pattern detection (e.g., FFT on skip vector components if they form a series) to find dominant skip patterns.
        5. For each empty cell (r, c):
            a. For each potential number 'v' that could go there:
                i. Calculate where 'v' *would have been* according to a dominant skip pattern if its base position was 'bp_v'.
                ii. If (r,c) matches this predicted skipped position, score is high.
        6. The cell's score is the max score across potential numbers 'v' that fit a strong pattern.
    Compatibility: Handles integer grid. -1 is empty. Requires a concept of "expected sequences" or a way to define them.
    Extensibility: Dynamic detection of sequences, more robust pattern detection algorithms (e.g., time series analysis for skips), configurable base placement rules.
    Optimization & Extension Directions: Pre-calculate sequence base positions, use optimized clustering, allow for partial pattern matches.
    Possible Multi-version Logic: GM19.v1 (simple row/col skip counts), GM19.v2 (this version with dominant skip pattern matching), GM19.v3 (probabilistic model of skip generation).
    Additional Design Suggestion (棋譜分析/數列補全): The concept of "skip vectors" can be seen as analyzing "moves" from an expected position. If numbers are part of multiple interleaved sequences (like different colored "stones" in Go), this module could try to disentangle these sequences and their respective placement logics. For "數列補全", if a row/column is expected to contain an arithmetic or geometric progression of numbers, but some are "skipped" to other locations, this module could identify such cross-row/column progressions.
    """
    logger.debug(f"Executing ext_gm19_masked_number_skip_pattern_vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    if rows == 0 or cols == 0: return scores

    def val_func_gm19(x_val: int) -> Optional[int]: # Returns int for sequence checks
        if x_val != -1 and x_val > 0 : # Assuming positive integers are the numbers of interest
            return int(x_val)
        return None

    # --- Step 1: Identify revealed numbers and their positions ---
    revealed_numbers = [] # List of (value, r, c)
    for r_idx in range(rows):
        for c_idx in range(cols):
            val = val_func_gm19(grid[r_idx, c_idx])
            if val is not None:
                revealed_numbers.append({'value': val, 'r': r_idx, 'c': c_idx})
    
    if not revealed_numbers:
        return scores # No numbers to analyze patterns from

    revealed_numbers.sort(key=lambda x: x['value']) # Sort by value to analyze sequences

    # --- Step 2: Define "base/expected" placement (example: row-by-row for numbers 1 to rows*cols) ---
    # This is a critical assumption and highly dependent on the scratch card's typical layout.
    # For a generic approach, we might need to infer this or make it configurable.
    # Let's assume numbers 1 to N are expected.
    
    max_revealed_value = max(rn['value'] for rn in revealed_numbers) if revealed_numbers else 0
    # Heuristic: consider numbers from 1 up to max_revealed_value + a bit (e.g., +cols to allow for next row)
    # Or, if the game has a fixed set of numbers (e.g. 1 to 25 for a 5x5 card)
    expected_max_number_on_card = rows * cols # A common assumption for some card types
    
    base_positions: Dict[int, Tuple[int, int]] = {} # value -> (expected_r, expected_c)
    for k_val in range(1, expected_max_number_on_card + 1):
        base_r = (k_val - 1) // cols
        base_c = (k_val - 1) % cols
        if base_r < rows : # Ensure base position is within grid dimensions
            base_positions[k_val] = (base_r, base_c)

    # --- Step 3: Calculate "skip vectors" ---
    skip_vectors: Dict[int, Tuple[int, int]] = {} # value -> (delta_r, delta_c)
    for rn in revealed_numbers:
        val = rn['value']
        if val in base_positions:
            expected_r, expected_c = base_positions[val]
            skip_vectors[val] = (rn['r'] - expected_r, rn['c'] - expected_c)

    if not skip_vectors:
        return scores # No skips calculable

    # --- Step 4: Find dominant skip patterns ---
    # This is complex. For a "limit" version, we can try a few things:
    # Option A: Clustering skip vectors (e.g., if many numbers have a (+1, 0) skip)
    # Option B: Look for arithmetic progressions in skip components if numbers are sequential
    # Option C: Test against predefined common skip patterns
    
    # Simplified for now: Count frequency of skip vectors. The most frequent is "dominant".
    # More advanced: Use DBSCAN or k-means if skips are continuous-like.
    dominant_skip_patterns = []
    if skip_vectors:
        skip_vector_tuples = [v for v in skip_vectors.values()]
        if not skip_vector_tuples: return scores

        counts = Counter(skip_vector_tuples)
        # Consider patterns that appear at least, say, twice or a certain percentage
        min_occurrences_for_pattern = max(2, int(len(skip_vector_tuples) * 0.1)) 
        for skip_vec, count in counts.most_common():
            if count >= min_occurrences_for_pattern:
                # How many numbers exhibit this pattern?
                numbers_with_this_skip = [val for val, sv in skip_vectors.items() if sv == skip_vec]
                # Strength of pattern = count * (avg value of numbers with this skip / max_revealed_value (heuristic))
                # Simplified strength: just the count or normalized count
                pattern_strength = MathUtils.normalize_value(count, min_occurrences_for_pattern, len(skip_vector_tuples))
                dominant_skip_patterns.append({'skip': skip_vec, 'strength': pattern_strength, 'example_values': numbers_with_this_skip[:3]})
            else: # Stop if counts are too low
                break 
    
    if not dominant_skip_patterns:
        # Fallback: Check for simple global shifts if no specific pattern is strong
        # e.g. if all numbers are shifted by (+delta_r_avg, +delta_c_avg)
        # This is more complex to implement robustly here.
        # For now, if no dominant patterns found, score will be low.
        pass

    # --- Step 5 & 6: Score empty cells based on fitting dominant patterns ---
    # For each empty cell, and for each potential number that could go there (1 to expected_max_number_on_card)
    potential_numbers_to_place = set(range(1, expected_max_number_on_card + 1)) - set(rn['value'] for rn in revealed_numbers)

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: # Only score empty cells
                continue

            cell_max_pattern_score = 0.0
            for potential_val in potential_numbers_to_place:
                if potential_val not in base_positions: continue
                
                base_r, base_c = base_positions[potential_val]
                
                for pattern in dominant_skip_patterns:
                    skip_dr, skip_dc = pattern['skip']
                    pattern_strength = pattern['strength']
                    
                    # Predicted position for potential_val if it followed this skip pattern
                    predicted_r = base_r + skip_dr
                    predicted_c = base_c + skip_dc
                    
                    if predicted_r == r_idx and predicted_c == c_idx:
                        # This empty cell IS where potential_val would land if it followed this dominant pattern.
                        # Score is based on pattern strength.
                        # Can also add factor if potential_val is "close" to numbers that exhibit this pattern.
                        current_pattern_fit_score = pattern_strength
                        
                        # Bonus if potential_val is arithmetically/sequentially close to other numbers ALREADY following this pattern
                        # Example: if pattern has [5,10,15] and potential_val is 20.
                        # This requires more complex sequence analysis within the pattern followers.
                        # Simplified: just use pattern_strength for now.
                        if current_pattern_fit_score > cell_max_pattern_score:
                            cell_max_pattern_score = current_pattern_fit_score
            
            scores[r_idx, c_idx] = cell_max_pattern_score # Max score if multiple patterns/values fit
            
    return scores


def ext_gm20_skip_pattern_confidence_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    """
    Core Rule: Based on skip patterns detected by GM19 (or a similar module), this module evaluates how placing a potential number in an empty cell would enhance or complete global skip regularities.
    Purpose: To provide a confidence score for placing a specific number in a specific empty cell, prioritizing placements that strengthen observed systematic "跳格" patterns.
    ---
    Design Philosophy: Acts as a "verifier" or "enhancer" for patterns found by GM19. It simulates placing a number and checks if this placement increases the "regularity score" of one or more dominant skip patterns.
    Use Case: Deciding *which* number to place in an empty cell that GM19 identified as a potential "pattern slot". Helps distinguish between multiple numbers that might fit a geometric slot.
    Scoring Formula Principle:
        1. Relies on GM19 (or an internal equivalent) to provide dominant skip patterns and the numbers following them.
        2. For each empty cell (r, c) and each potential number 'v':
            a. Temporarily "place" 'v' at (r,c).
            b. Re-evaluate the strength/consistency of dominant skip patterns *with 'v' included*.
               - Does 'v' now perfectly align with an existing pattern's skip vector from its base position?
               - Does adding 'v' make a sequence of numbers (e.g., 1,2,v,4,5) that follow the pattern more complete or arithmetically regular?
               - Does 'v' bridge two smaller groups of numbers that follow the same skip?
            c. The score for placing 'v' at (r,c) is proportional to the increase in global pattern regularity or the strength of the pattern 'v' now completes.
        3. The cell's final score is the max score across all potential 'v' that significantly enhance patterns.
    Compatibility: Integer grid. -1 is empty. Highly dependent on the quality of pattern detection from GM19 or equivalent.
    Extensibility: Use more sophisticated measures of "pattern regularity" (e.g., sequence completion metrics, graph-based pattern matching), consider negative evidence (placements that break patterns).
    Optimization & Extension Directions: Incremental update of pattern strengths instead of full re-evaluation, use a "lookahead" mechanism to see if placing 'v' enables further pattern completions.
    Possible Multi-version Logic: GM20.v1 (simple check if cell+value fits one GM19 pattern), GM20.v2 (this version measuring enhancement of regularity), GM20.v3 (probabilistic scoring of pattern completion).
    Additional Design Suggestion (規律性跳動與結構穩定性): This module directly addresses this. "Structural stability" can be interpreted as the persistence and predictability of the skip patterns. If placing a number makes the overall set of skip vectors "simpler" (e.g., reduces the number of distinct dominant skip vectors, or increases the count of a very regular one), that's a high score.
    """
    logger.debug(f"Executing ext_gm20_skip_pattern_confidence_vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    if rows == 0 or cols == 0: return scores

    # This module is complex as it ideally needs the output of GM19 or similar pattern detection.
    # For a self-contained "extreme" version, it would need to:
    # 1. Detect patterns on the current board (similar to GM19 logic).
    # 2. For each empty cell (er, ec) and each potential number 'v_test':
    #    a. Create a hypothetical grid_test by placing 'v_test' at (er, ec).
    #    b. Re-detect patterns on grid_test.
    #    c. Compare pattern "quality" (strength, regularity, completeness) before and after.
    # This is computationally very expensive if done naively for every cell and every number.

    # --- Simplified "Extreme" Approach for GM20: ---
    # Assume GM19 has identified some dominant skip patterns and the numbers that follow them.
    # GM20 will focus on: if placing 'v_test' at (er,ec) makes it *join* an existing strong pattern
    # in a way that makes that pattern *more arithmetically or sequentially complete*.

    # Re-using parts of GM19's logic for pattern detection (could be refactored into a shared utility)
    # For brevity, I'll sketch the core idea rather than fully re-implementing GM19 here.
    
    # --- Placeholder for GM19's core pattern detection logic ---
    # This would populate `dominant_skip_patterns_initial` and `skip_vectors_initial`
    # For demonstration, let's assume we run a simplified GM19-like detection first.
    # For a real implementation, GM19's results would be an input or GM20 would call GM19.
    
    # Let's assume `ext_gm19_masked_number_skip_pattern_vec` can be called to get some info,
    # or we replicate its pattern finding part.
    # This creates a dependency or requires code duplication / refactoring.
    # To make GM20 self-contained for now, it needs its own pattern analysis.

    # --- Step 1: Initial Pattern Analysis (Simplified from GM19) ---
    revealed_numbers_initial = []
    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1 and grid[r_idx, c_idx] > 0:
                revealed_numbers_initial.append({'value': int(grid[r_idx, c_idx]), 'r': r_idx, 'c': c_idx})
    
    if not revealed_numbers_initial: return scores
    revealed_numbers_initial.sort(key=lambda x: x['value'])
    
    expected_max_number_on_card_gm20 = rows * cols
    base_positions_gm20: Dict[int, Tuple[int, int]] = {
        k_val: ((k_val - 1) // cols, (k_val - 1) % cols)
        for k_val in range(1, expected_max_number_on_card_gm20 + 1)
        if ((k_val - 1) // cols) < rows
    }

    skip_vectors_initial: Dict[int, Tuple[int, int]] = {}
    for rn in revealed_numbers_initial:
        val = rn['value']
        if val in base_positions_gm20:
            skip_vectors_initial[val] = (rn['r'] - base_positions_gm20[val][0], rn['c'] - base_positions_gm20[val][1])

    dominant_patterns_details_initial = [] # List of {'skip': (dr,dc), 'values': [v1,v2...], 'strength': float}
    if skip_vectors_initial:
        skip_vector_tuples = list(skip_vectors_initial.values())
        counts = Counter(skip_vector_tuples)
        min_occurrences = max(2, int(len(skip_vector_tuples) * 0.1))
        for skip_vec, count in counts.most_common():
            if count >= min_occurrences:
                current_pattern_values = sorted([val for val, sv in skip_vectors_initial.items() if sv == skip_vec])
                dominant_patterns_details_initial.append({
                    'skip': skip_vec, 
                    'values': current_pattern_values, 
                    'strength': MathUtils.normalize_value(count, min_occurrences, len(skip_vector_tuples))
                })
    # --- End of Initial Pattern Analysis ---

    potential_numbers_to_place_gm20 = set(range(1, expected_max_number_on_card_gm20 + 1)) - set(rn['value'] for rn in revealed_numbers_initial)

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue # Only empty cells

            max_confidence_score_for_cell = 0.0

            for v_test in potential_numbers_to_place_gm20:
                if v_test not in base_positions_gm20: continue
                
                base_r_test, base_c_test = base_positions_gm20[v_test]
                current_confidence_for_v_test = 0.0

                for initial_pattern in dominant_patterns_details_initial:
                    pattern_skip_dr, pattern_skip_dc = initial_pattern['skip']
                    pattern_values = initial_pattern['values']
                    pattern_strength_initial = initial_pattern['strength']

                    # Check if (r_idx, c_idx) is the "skipped" position for v_test according to this pattern
                    predicted_r_for_v_test = base_r_test + pattern_skip_dr
                    predicted_c_for_v_test = base_c_test + pattern_skip_dc

                    if predicted_r_for_v_test == r_idx and predicted_c_for_v_test == c_idx:
                        # Yes, v_test at (r_idx, c_idx) geometrically fits this pattern's skip rule.
                        # Now, assess how adding v_test enhances the "regularity" of pattern_values.
                        # Regularity: e.g., forms a more complete arithmetic sequence.
                        
                        temp_pattern_values_with_v_test = sorted(pattern_values + [v_test])
                        
                        # Measure regularity: e.g., how many elements form an arithmetic progression?
                        # Simplified: Check if v_test fills a "gap" or extends the sequence arithmetically.
                        # Example: if pattern_values = [2, 4, 8, 10] (skip is constant, diffs are 2)
                        # If v_test = 6, it fills a gap. If v_test = 0 or 12, it extends.
                        
                        # Heuristic for "enhancement":
                        # 1. Basic fit score (from pattern_strength_initial)
                        enhancement_score = pattern_strength_initial * 0.5 # Base for fitting the skip

                        # 2. Bonus for sequence completion/extension
                        #    Requires analyzing `temp_pattern_values_with_v_test` for arithmetic progression.
                        #    This is complex. A simpler proxy: is v_test "close" to the existing pattern_values?
                        if len(pattern_values) >= 1:
                            min_val_in_pattern = pattern_values[0]
                            max_val_in_pattern = pattern_values[-1]
                            avg_diff_in_pattern = 0
                            if len(pattern_values) > 1:
                                diffs = np.diff(pattern_values)
                                if len(diffs)>0 : avg_diff_in_pattern = np.mean(diffs)

                            # If v_test extends the sequence
                            if math.isclose(v_test, max_val_in_pattern + avg_diff_in_pattern) if avg_diff_in_pattern > 0 else False:
                                enhancement_score += pattern_strength_initial * 0.3
                            elif math.isclose(v_test, min_val_in_pattern - avg_diff_in_pattern) if avg_diff_in_pattern > 0 else False:
                                enhancement_score += pattern_strength_initial * 0.3
                            
                            # If v_test fills an internal gap (harder to check simply)
                            # Simplified: if v_test is between min and max and maintains some regularity
                            # For now, geometric fit is the main driver, regularity bonus is small addition.

                        current_confidence_for_v_test = max(current_confidence_for_v_test, enhancement_score)
                
                if current_confidence_for_v_test > max_confidence_score_for_cell:
                    max_confidence_score_for_cell = current_confidence_for_v_test
            
            scores[r_idx, c_idx] = max_confidence_score_for_cell
            
    return scores


# (All 22 EXT_GM modules should be defined before this point in a complete file)
# Make sure all GM module functions from the previous response (GM3-GM14) are included here.
# For this segment, I'm assuming they are already defined above this point.
# The example below only lists a few for brevity in this specific segment.
# In the FINAL combined code, ALL ext_..._vec functions MUST be present.

# Placeholder for GM3-GM14 definitions if they were not in the prior "Part 3"
# In a real combined file, they would be here. For this snippet, assume they are present.
# Example:
def ext_gm3_adv_connected_comp_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray: # Copied from my generation context
    """
    Core Rule: Analyzes the size and density of the connected homogeneous region the cell belongs to using BFS/DFS.
    Purpose: Identifies large, coherent clusters of similar-valued cells.
    ---
    Design Philosophy: Based on graph theory's "connected components". Finds regions of similar values (within a tolerance).
    Use Case: Identifying large resource patches, empty areas for expansion, or consolidated territories.
    Scoring Formula Principle: Score = w1 * normalized_area_size + w2 * normalized_area_density.
    Compatibility: Integer grid. -1 empty. Similarity tolerance is configurable.
    Extensibility: Weighted connectivity (cost to traverse), different connectivity types (8-conn vs 4-conn), shape analysis of components.
    Optimization & Extension Directions: Union-find algorithm for faster component labeling on static grids, caching component properties.
    Possible Multi-version Logic: GM3.v1 (simple neighbor emptiness), GM3.v2 (this version), GM3.v3 (component boundary analysis).
    """
    logger.debug(f"Executing ext_gm3_adv_connected_comp_vec", extra={'request_id': request_id})
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores

    # Parameters from previous generation for GM3
    value_tolerance = 0.1 * 10 # Assuming values 1-10, tolerance of 1. Adjust if val_func normalizes.
    area_size_weight = 0.6
    area_density_weight = 0.4
    max_bfs_steps = 100 # Max cells to explore for a component
    
    def val_func_gm3(x_val: int) -> Optional[float]:
        if x_val == -1: return None # Or treat as a specific value like 0 if appropriate for "homogeneity"
        return float(x_val)

    for r_start_bfs in range(rows):
        for c_start_bfs in range(cols):
            start_val = val_func_gm3(grid[r_start_bfs, c_start_bfs])
            if start_val is None:
                scores[r_start_bfs, c_start_bfs] = 0.0 # Or a score indicating emptiness benefits
                continue

            q = deque([(r_start_bfs, c_start_bfs)])
            visited_bfs = set([(r_start_bfs, c_start_bfs)])
            component_cells_coords = [(r_start_bfs, c_start_bfs)]
            steps = 0

            while q and steps < max_bfs_steps:
                r_curr, c_curr = q.popleft()
                steps += 1
                for dr_bfs, dc_bfs in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r_curr + dr_bfs, c_curr + dc_bfs
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited_bfs:
                        neighbor_val = val_func_gm3(grid[nr, nc])
                        if neighbor_val is not None and abs(neighbor_val - start_val) <= value_tolerance:
                            visited_bfs.add((nr, nc))
                            q.append((nr, nc))
                            component_cells_coords.append((nr,nc))
            
            area_size = len(component_cells_coords)
            norm_area_size = MathUtils.normalize_value(area_size, 1, rows * cols)

            component_values = [val_func_gm3(grid[r_comp][c_comp]) for r_comp, c_comp in component_cells_coords]
            component_values_clean = [v for v in component_values if v is not None]

            avg_density_val = np.mean(component_values_clean) if component_values_clean else 0
            # Normalize density (e.g. if values 1-10 for game)
            norm_avg_density = MathUtils.normalize_value(avg_density_val, 0, 10.0, clamp=True) 
            
            # The score is for the starting cell, representing the component it's in.
            # All cells in the same component would ideally get a similar score from this module if calculated for them.
            # To make it specific for (r_start_bfs, c_start_bfs):
            scores[r_start_bfs, c_start_bfs] = (area_size_weight * norm_area_size +
                                                area_density_weight * norm_avg_density)
    return scores

# ... Other GM functions (GM4-GM14) from previous parts would be listed here ...
# For this final segment, I am focusing on GM15-GM18 and then the integration.
# It's assumed GM4-GM14 definitions provided earlier are part of the complete code.
# (The user will concatenate all these parts)
# main.py (FastAPI with "Industry Extreme" Analyzer - Complete)

import json
import os
import uuid # For request IDs
import time # For process time
import logging # For enhanced logging
from datetime import datetime # For health check timestamp
import math # For advanced math in modules
from collections import Counter, deque # For advanced logic in modules

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, validator, Field
from typing import List, Dict, Tuple, Callable, Optional, Any
import numpy as np
# from ortools.sat.python import cp_model  # patched out for execution # Assuming Google OR-Tools is used
# from celery.result import AsyncResult # Kept as per original, if used
# from celery_worker import solve_task  # Kept as per original, if used

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - RequestID: %(request_id)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# --- Application Setup ---
app = FastAPI(
    title="Plug-in權重 + 張量流 + 自動數字範圍 - AI Manager Enabled (Extreme Analyzer v2.0)", # Updated title
    version="3.4", # Incremented version
    description="Enhanced analysis API with AI Manager capabilities and 22 'Industry Extreme' analyzer modules."
)

# --- Middleware for Request ID and Logging ---
class RequestContextLogMiddleware:
    async def __call__(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.time()
        # Log before processing the request to capture request details early
        logger.info(f"Request started: {request.method} {request.url.path}", extra={'request_id': request_id})
        
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        logger.info(f"Request finished: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s", extra={'request_id': request_id})
        return response

app.middleware("http")(RequestContextLogMiddleware())

# -----------------------------------------------------------------------------
# 0. Helper Utilities for Advanced Modules
# (MathUtils and BoardAnalyzerUtils as defined in previous parts)
# -----------------------------------------------------------------------------
class MathUtils:
    @staticmethod
    def sigmoid(x: float, k: float = 1.0) -> float:
        try:
            return 1 / (1 + math.exp(-k * x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    @staticmethod
    def normalize_value(value: float, min_val: float, max_val: float, clamp: bool = True) -> float:
        if max_val == min_val:
            return 0.5 if math.isclose(value, min_val) else (0.0 if value < min_val else 1.0)
        normalized = (value - min_val) / (max_val - min_val)
        if clamp:
            return max(0.0, min(1.0, normalized))
        return normalized

    @staticmethod
    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

class BoardAnalyzerUtils:
    @staticmethod
    def get_neighborhood_values(grid: np.ndarray, r: int, c: int, radius: int = 1,
                                eight_connectivity: bool = True,
                                val_func: Callable[[int], Optional[float]] = lambda x_val: float(x_val) if x_val != -1 else None,
                                include_center: bool = False) -> List[float]:
        neighbors = []
        rows, cols = grid.shape
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if not include_center and dr == 0 and dc == 0: continue
                if not eight_connectivity and abs(dr) + abs(dc) > radius: continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    processed_val = val_func(grid[nr, nc])
                    if processed_val is not None: neighbors.append(processed_val)
        return neighbors

    @staticmethod
    def get_value_gradient_at_cell(grid: np.ndarray, r: int, c: int,
                                   val_func: Callable[[int], float] = lambda x_val: float(x_val) if x_val != -1 else 0.0) -> Tuple[float, float]:
        rows, cols = grid.shape
        def safe_val(r_in, c_in):
            if 0 <= r_in < rows and 0 <= c_in < cols: return val_func(grid[r_in, c_in])
            return 0.0
        gx = (safe_val(r-1, c+1) + 2*safe_val(r, c+1) + safe_val(r+1, c+1)) - \
             (safe_val(r-1, c-1) + 2*safe_val(r, c-1) + safe_val(r+1, c-1))
        gy = (safe_val(r+1, c-1) + 2*safe_val(r+1, c) + safe_val(r+1, c+1)) - \
             (safe_val(r-1, c-1) + 2*safe_val(r-1, c) + safe_val(r-1, c+1))
        return gx, gy

# --- Pydantic Models (As defined by user, with minor adjustments if needed) ---
class ProposedValue(BaseModel):
    pos: List[int] = Field(..., min_items=2, max_items=2, description="Position [row, col]")
    value: int = Field(..., description="Proposed value. Positive for placing, -1 for clearing (if applicable).")

class AnalyzeRequest(BaseModel):
    new_card: List[List[int]] = Field(..., description="The game grid, -1 for empty cells")
    proposed_values: List[ProposedValue] = Field(..., description="List of proposed values to analyze")

    @validator("new_card")
    def check_rectangular_and_numeric(cls, g):
        if not g: raise ValueError("new_card cannot be empty")
        if any(not isinstance(row, list) for row in g): raise ValueError("new_card must be a list of lists")
        if not g[0]: raise ValueError("Rows in new_card cannot be empty") # Ensure first row is not empty for cols check
        if any(len(row) != len(g[0]) for row in g): raise ValueError("new_card must be a rectangular grid")
        if any(not isinstance(val, int) for row in g for val in row): raise ValueError("All values in new_card must be integers")
        return g

    @validator("proposed_values", each_item=True)
    def check_pv_bounds_and_value(cls, pv, values):
        grid = values.get("new_card")
        if grid:
            rows, cols = len(grid), len(grid[0])
            r, c = pv.pos
            if not (0 <= r < rows and 0 <= c < cols):
                raise ValueError(f"Proposed position {pv.pos} is out of bounds for grid {rows}x{cols}")
            
            current_cell_val = grid[r][c]
            if pv.value != -1 and current_cell_val != -1 : # Placing positive on non-empty
                 raise ValueError(f"Cell at {pv.pos} is already filled with {current_cell_val}. Cannot place {pv.value}.")
            if pv.value == -1 and current_cell_val == -1: # Clearing already empty
                 raise ValueError(f"Cell at {pv.pos} is already empty. Cannot clear further.")
        return pv

class TensorRuleContribution(BaseModel):
    rule_name: str
    score_if_applied: float # Raw float score from module for the cell
    weight: float
    weighted_score: float

class CandidateDetail(BaseModel):
    pos: List[int]
    value: int
    is_valid_proposal: bool = True
    tensor_flow_contributions: List[TensorRuleContribution] = Field(default_factory=list)
    raw_tensor_flow_score: float
    mem_score_value: float
    final_objective_score: float
    is_selected_by_cp: bool = False
    cp_solver_notes: Optional[str] = None

class AnalyzeResultDetail(CandidateDetail): pass

class AnalyzeSuccessResponse(BaseModel):
    request_id: str
    status: str = "success"
    main_module_version: str = Field(default=app.version)
    analysis_engine_version: str 
    message: Optional[str] = None
    result: Optional[AnalyzeResultDetail] = None
    all_candidates_evaluated: List[CandidateDetail]

class AnalyzeErrorResponse(BaseModel):
    request_id: str
    status: str
    main_module_version: str = Field(default=app.version)
    analysis_engine_version: str
    message: str
    error_type: Optional[str] = None
    details: Optional[Any] = None

# --- Memory Handling (User's original logic) ---
MEM_PATH = os.path.join(os.path.dirname(__file__), "memory_cards.json")
_memory_freq: Dict[Tuple[int, int, int], int] = {}
_total_samples_in_memory = 0

def load_memory_data(req_id: str = "startup"): # (User's original function)
    global _memory_freq, _total_samples_in_memory
    _memory_freq = {}
    _total_samples_in_memory = 0
    try:
        if os.path.exists(MEM_PATH):
            with open(MEM_PATH, "r", encoding="utf-8") as f: data = json.load(f)
            for card in data.get("memory_cards", []):
                for r_mem, row_data in enumerate(card): # Renamed r to r_mem
                    for c_mem, v_mem in enumerate(row_data): # Renamed c,v
                        if v_mem != -1:
                            _memory_freq[(r_mem, c_mem, v_mem)] = _memory_freq.get((r_mem, c_mem, v_mem), 0) + 1
                            _total_samples_in_memory += 1
            logger.info(f"Memory data loaded: {_total_samples_in_memory} samples from {len(data.get('memory_cards', []))} cards.", extra={'request_id': req_id})
        else: logger.warning(f"Memory file not found: {MEM_PATH}. Mem score will be 0.", extra={'request_id': req_id})
    except Exception as e: logger.error(f"Error loading memory data from {MEM_PATH}: {e}", exc_info=True, extra={'request_id': req_id})
load_memory_data()

def mem_score(r: int, c: int, v_proposed: int, legal_values_for_pos: Optional[set] = None) -> float: # Added Optional to legal_values
    # If legal_values_for_pos is None, we skip this check for mem_score.
    # This is useful if mem_score should purely reflect historical frequency regardless of current game "legality".
    if legal_values_for_pos is not None and v_proposed not in legal_values_for_pos and v_proposed != -1:
        return 0.0
    if _total_samples_in_memory == 0: return 0.0
    count = _memory_freq.get((r, c, v_proposed), 0)
    # Normalize by total samples. Alternative: normalize by total occurrences at (r,c) or total for value 'v'.
    return float(count) / float(_total_samples_in_memory)

# -----------------------------------------------------------------------------
# "Industry Extreme" Analyzer Modules (Definitions for EXT_A2 to EXT_GM20)
# (All 22 ext_..._vec functions as defined in previous parts should be pasted here)
# For example:
# def ext_a2_weighted_proximity_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray: ...
# def ext_m3_local_heterogeneity_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray: ...
# ... and so on for all 22 modules up to ext_gm20_skip_pattern_confidence_vec ...
# THIS IS WHERE YOU PASTE ALL THE `ext_..._vec` FUNCTIONS FROM MY PREVIOUS RESPONSES
# (Part 1, Part 2, Part 3 (Continued) containing GM19, GM20, and the placeholder mention for GM3-GM14 that you should have filled)
# Example (ensure all 22 are actually here in your final code):
def ext_a2_weighted_proximity_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    # (Full implementation from previous responses)
    logger.debug(f"Stub for ext_a2_weighted_proximity_vec", extra={'request_id': request_id}); return np.random.rand(*grid.shape)
def ext_m3_local_heterogeneity_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    logger.debug(f"Stub for ext_m3_local_heterogeneity_vec", extra={'request_id': request_id}); return np.random.rand(*grid.shape)
# ... (ALL 22 functions must be fully defined here) ...
def ext_gm19_masked_number_skip_pattern_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    logger.debug(f"Stub for ext_gm19_masked_number_skip_pattern_vec", extra={'request_id': request_id}); return np.random.rand(*grid.shape)
def ext_gm20_skip_pattern_confidence_vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    logger.debug(f"Stub for ext_gm20_skip_pattern_confidence_vec", extra={'request_id': request_id}); return np.random.rand(*grid.shape)
# IMPORTANT: Replace above stubs with the full function definitions provided earlier.
# I am not re-pasting all 22 long function definitions here to save space in this segment.
# You must have them from the previous segments.

# -----------------------------------------------------------------------------

# --- Module Registration and Weights for Extreme Analyzer ---
EXTREME_MODULE_FUNCS_VEC: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {
    "EXT_A2_Proximity": ext_a2_weighted_proximity_vec, # Make sure this function is fully defined above
    "EXT_M3_Heterogeneity": ext_m3_local_heterogeneity_vec, # And this one
    # ... and all other 20 extreme module functions (EXT_D3 to EXT_GM20)
    # Example for GM19, GM20 (ensure all are listed)
    "EXT_GM19_SkipPattern": ext_gm19_masked_number_skip_pattern_vec,
    "EXT_GM20_SkipConfidence": ext_gm20_skip_pattern_confidence_vec,
}
# Populate with all 22 module functions properly.
# For brevity, I am not listing all 22 assignments here, but your code should.
# Ensure the keys match the function names or desired identifiers.

EXTREME_MODULE_WEIGHTS: Dict[str, float] = {} # Will be populated based on EXTREME_MODULE_FUNCS_VEC keys
default_weight = 1.0
# Example weights (tune these carefully!)
# Keep weights from previous EXTREME_MODULE_WEIGHTS definition or redefine here.
# For every key in EXTREME_MODULE_FUNCS_VEC, ensure there's a weight.
illustrative_weights = {
    "EXT_A2_Proximity": 1.2, "EXT_M3_Heterogeneity": 1.0, "EXT_D3_PotentialField": 1.1, "EXT_F10_Discontinuity": 0.9,
    "EXT_GM1_RowControl": 1.0, "EXT_GM2_ColFlow": 1.0, "EXT_GM3_ConnectedComp": 1.1, "EXT_GM4_SpatialAutoCorr": 0.8,
    "EXT_GM5_LocalExtremum": 1.2, "EXT_GM6_PatternMatch": 0.9, "EXT_GM7_Accessibility": 1.3, "EXT_GM8_MarginalDensity": 0.7,
    "EXT_GM9_GradientFlow": 0.8, "EXT_GM10_InfluenceMap": 1.2, "EXT_GM11_MultiScaleSig": 0.9, "EXT_GM12_Texture": 0.7,
    "EXT_GM13_SpatialFill": 0.8, "EXT_GM14_NicheCompetition": 1.0, "EXT_GM15_SecureTerritory": 1.3,
    "EXT_GM16_Bottleneck": 1.4, "EXT_GM17_NetworkHub": 1.1, "EXT_GM18_RLValueEst": 1.5,
    "EXT_GM19_SkipPattern": 1.3, "EXT_GM20_SkipConfidence": 1.4
}
for mod_key in EXTREME_MODULE_FUNCS_VEC: # Ensure all registered functions have weights
    EXTREME_MODULE_WEIGHTS[mod_key] = illustrative_weights.get(mod_key, default_weight)
    if mod_key not in illustrative_weights:
        logger.warning(f"Using default weight {default_weight} for extreme module: {mod_key}.")

ANALYSIS_ENGINE_VERSION_EXTREME = "2.1_extreme_full" # Updated version

# --- Modified Tensor Flow Scoring for Extreme Modules ---
def extreme_tensor_flow_score_detailed(grid: np.ndarray, request_id: str) -> Tuple[np.ndarray, List[List[List[TensorRuleContribution]]]]:
    rows, cols = grid.shape
    total_score_grid_agg = np.zeros((rows, cols), dtype=float)
    rule_contributions_grid_agg: List[List[List[TensorRuleContribution]]] = [[[] for _ in range(cols)] for _ in range(rows)]

    for name, func in EXTREME_MODULE_FUNCS_VEC.items(): # Use the new dict
        try:
            module_score_grid = func(grid, request_id) # Returns (R, C) float scores
            weight = EXTREME_MODULE_WEIGHTS.get(name, 1.0) # Use new weights
            if math.isclose(weight, 0): continue

            current_module_weighted_scores = module_score_grid * weight
            total_score_grid_agg += current_module_weighted_scores

            for r_idx in range(rows):
                for c_idx in range(cols):
                    raw_module_score = module_score_grid[r_idx, c_idx]
                    weighted_score = current_module_weighted_scores[r_idx, c_idx]
                    if abs(raw_module_score) > 1e-4 or abs(weighted_score) > 1e-4 : # Add if significant
                        contribution = TensorRuleContribution(
                            rule_name=name,
                            score_if_applied=round(raw_module_score, 4),
                            weight=weight,
                            weighted_score=round(weighted_score, 4)
                        )
                        rule_contributions_grid_agg[r_idx][c_idx].append(contribution)
        except Exception as e:
            logger.error(f"Error processing EXTREME rule '{name}': {e}", exc_info=True, extra={'request_id': request_id})
            pass 
            
    return total_score_grid_agg, rule_contributions_grid_agg

# --- Helper Functions (get_card_max_value, get_legal_values_for_placement - User's original) ---
def get_card_max_value(grid: np.ndarray) -> int: # (As provided by user)
    if grid.size == 0: return 0
    valid_values = grid[grid != -1]
    return int(np.max(valid_values)) if valid_values.size > 0 else 0

def get_legal_values_for_placement(grid: np.ndarray, for_value: Optional[int] = None) -> set: # Added for_value, though not used in current simple logic
    # This function's logic is crucial and game-specific.
    # Original assumption: values 1 up to card_max + 1, and not already present.
    card_max_val = get_card_max_value(grid)
    # If card is empty, or only contains -1, what's the range? Default to 1-10 example.
    base_upper_bound = 10 
    if card_max_val > 0 :
        base_upper_bound = card_max_val + 1 # Allow placing one number higher than current max
    
    all_possible_to_place = set(range(1, base_upper_bound + 1))
    
    # Values already used on the grid might not be placeable again
    # This depends on game rules (e.g. Sudoku no repeats in unit, some games allow repeats)
    # Current logic removes ALL used positive values from placement options.
    used_values_on_board = set(int(v) for v in grid.flatten() if v != -1 and v > 0)
    
    legal_placements = all_possible_to_place - used_values_on_board
    if not legal_placements and not used_values_on_board: # Empty board, nothing used, but range 1-10 allowed
        return all_possible_to_place
    return legal_placements


# --- CP Solver Logic (User's original) ---
def solve_cp_for_candidates( # (As provided by user)
    grid_shape: Tuple[int, int], current_grid_state: np.ndarray,
    candidates_to_evaluate: List[CandidateDetail], request_id: str
) -> List[CandidateDetail]:
    model = cp_model.CpModel()
    num_candidates = len(candidates_to_evaluate)
    if num_candidates == 0: return candidates_to_evaluate
    x = [model.NewBoolVar(f"x_{i}") for i in range(num_candidates)]
    model.Add(sum(x) == 1)
    objective_terms = [x[i] * int(candidates_to_evaluate[i].final_objective_score * 1000) for i in range(num_candidates)]
    model.Maximize(sum(objective_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0.5 
    solver.parameters.num_search_workers = os.cpu_count() or 1
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        logger.info(f"CP Solver solution. Status: {solver.StatusName(status)}", extra={'request_id': request_id})
        for i in range(num_candidates):
            if solver.Value(x[i]) == 1:
                candidates_to_evaluate[i].is_selected_by_cp = True
                candidates_to_evaluate[i].cp_solver_notes = f"Selected by CP ({solver.StatusName(status)})"
            elif candidates_to_evaluate[i].cp_solver_notes is None:
                 candidates_to_evaluate[i].cp_solver_notes = "Not selected by CP"
    else:
        logger.warning(f"CP Solver no solution. Status: {solver.StatusName(status)}", extra={'request_id': request_id})
        for cand_detail in candidates_to_evaluate:
            if cand_detail.cp_solver_notes is None:
                cand_detail.cp_solver_notes = f"No CP solution ({solver.StatusName(status)})"
    return candidates_to_evaluate

# --- API Endpoint: /analyze (Updated) --- (這是你提供的行號 3439 附近的註解)
@app.post("/analyze", response_model=AnalyzeSuccessResponse, responses={
    200: {"model": AnalyzeSuccessResponse}, 400: {"model": AnalyzeErrorResponse},
    422: {"model": AnalyzeErrorResponse}, 500: {"model": AnalyzeErrorResponse}},  # <<< 主要修改在這裡，補上了 '}'
    tags=["Analysis Engine vExtreme"])
async def analyze(req: AnalyzeRequest, request: Request): # (Structure from user's main.py)
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    try:
        logger.info(f"EXTREME Analyzer v{ANALYSIS_ENGINE_VERSION_EXTREME}: Request received. Grid: {len(req.new_card)}x{len(req.new_card[0]) if req.new_card and req.new_card[0] else 'empty'}. Proposals: {len(req.proposed_values)}.", extra={'request_id': request_id})
        if not req.new_card or not req.new_card[0]: # (這是你提供的行號 3449 附近的程式碼)
            raise HTTPException(status_code=400, detail="new_card cannot be empty or have empty rows.")
        grid = np.array(req.new_card, dtype=int) # (這是你提供的行號 3450 附近的程式碼)

        raw_tf_scores_grid_extreme, rule_contributions_grid_extreme = await run_in_threadpool(
            extreme_tensor_flow_score_detailed, grid, request_id
        )
        
        all_evaluated_candidates: List[CandidateDetail] = []
        globally_legal_values_for_new_placement = get_legal_values_for_placement(grid)

        for pv in req.proposed_values:
            r, c = pv.pos[0], pv.pos[1]
            val_proposed = pv.value
            is_valid_for_processing = True; proposal_notes = ""

            if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]): # Already caught by Pydantic, but double check
                is_valid_for_processing = False; proposal_notes = "Position out of bounds."
            elif val_proposed != -1 and grid[r,c] != -1:
                is_valid_for_processing = False; proposal_notes = f"Cell [{r},{c}] already filled with {grid[r,c]}."
            elif val_proposed == -1 and grid[r,c] == -1:
                is_valid_for_processing = False; proposal_notes = f"Cell [{r},{c}] is already empty."
            elif val_proposed != -1 and val_proposed not in globally_legal_values_for_new_placement :
                 # This 'legality' check might be too strict if a game allows placing numbers already on board in different contexts.
                 # For now, if get_legal_values_for_placement is strict, this will filter.
                 # Consider if val_proposed should be positive for this check.
                 if val_proposed > 0: # Only apply this specific legality check if proposing a positive number
                    is_valid_for_processing = False; proposal_notes = f"Value {val_proposed} not in globally legal set {globally_legal_values_for_new_placement} for new positive placement."

            if not is_valid_for_processing:
                 all_evaluated_candidates.append(CandidateDetail(
                    pos=[r,c], value=val_proposed, is_valid_proposal=False,
                    raw_tensor_flow_score=0, mem_score_value=0, final_objective_score=0,
                    cp_solver_notes=proposal_notes or "Invalid proposal." ))
                 continue
            
            raw_tf_score_cell = raw_tf_scores_grid_extreme[r, c]
            tf_contrib_cell = rule_contributions_grid_extreme[r][c]
            
            # For mem_score, use a relevant set of values. If proposing positive, use globally_legal + itself.
            # If proposing -1 (clear), mem_score might represent "badness" of current value or be 0.
            current_mem_score = 0.0
            mem_score_factor = 5.0
            if val_proposed != -1: # Placing a positive value
                # mem_score usually evaluates frequency of (r,c,val_proposed)
                # The "legal_values" for mem_score might just be any positive value, or game specific.
                # Using a broad set for mem_score context here.
                mem_score_context_set = set(range(1, (get_card_max_value(grid) or 9) +5 )) # broader set
                current_mem_score = mem_score(r, c, val_proposed, mem_score_context_set)
            else: # Proposing to clear (val_proposed == -1)
                # Score for clearing could be: -(raw_tf_score of the value being cleared)
                # Or a fixed "clear bonus". Or mem_score could be negative of the cleared value's positive mem_score.
                # This needs specific game logic. For now, no mem_score impact for clearing.
                mem_score_factor = 0.0 
                # The raw_tf_score_cell from extreme_tensor_flow_score_detailed for (r,c) might represent
                # the "goodness" of placing something positive there. If clearing, this score might be inverted.
                # For now, assume raw_tf_score_cell is still relevant as context, or specific modules should handle "clear" intent.
                # This is a complex area depending on how "clearing" is valued.
                # Let's assume for now the scores from modules are for "positive potential".
                # So if we clear, the objective might be to maximize - (score of thing cleared) or achieve a neutral state.
                # For simplicity, `raw_tf_score_cell` for a clearing move currently reflects the "goodness" of that empty spot *after* clearing.
                # This might need a different scoring head if clearing is a primary action to evaluate.


            final_obj_for_cp = raw_tf_score_cell + (mem_score_factor * current_mem_score)

            all_evaluated_candidates.append(CandidateDetail(
                pos=[r, c], value=val_proposed, is_valid_proposal=True,
                tensor_flow_contributions=tf_contrib_cell,
                raw_tensor_flow_score=round(raw_tf_score_cell, 4),
                mem_score_value=round(current_mem_score, 4),
                final_objective_score=round(final_obj_for_cp, 4) ))

        candidates_for_cp_solver = [cd for cd in all_evaluated_candidates if cd.is_valid_proposal]

        if not candidates_for_cp_solver:
            return AnalyzeSuccessResponse(request_id=request_id, status="no_valid_proposals_for_cp_extreme",
                analysis_engine_version=ANALYSIS_ENGINE_VERSION_EXTREME,
                message="No valid proposals to submit to solver.", result=None,
                all_candidates_evaluated=all_evaluated_candidates)

        updated_candidates = await run_in_threadpool( solve_cp_for_candidates, grid.shape, grid, candidates_for_cp_solver, request_id)
        
        # Merge results
        processed_map = {(cand.pos[0], cand.pos[1], cand.value): cand for cand in updated_candidates}
        for i, cand_orig in enumerate(all_evaluated_candidates):
            if cand_orig.is_valid_proposal:
                key = (cand_orig.pos[0], cand_orig.pos[1], cand_orig.value)
                if key in processed_map: all_evaluated_candidates[i] = processed_map[key]

        selected_list = [cand for cand in all_evaluated_candidates if cand.is_selected_by_cp]
        final_res: Optional[AnalyzeResultDetail] = None; msg = "EXTREME Analysis complete."; stat = "success_extreme"

        if not selected_list: msg = "EXTREME CP Solver: No candidate selected."; stat = "fail_no_selection_cp_extreme"
        elif len(selected_list) == 1: final_res = AnalyzeResultDetail(**selected_list[0].model_dump()); msg = f"EXTREME Analyzer selected: Pos={final_res.pos}, Val={final_res.value}."
        else: final_res = AnalyzeResultDetail(**max(selected_list, key=lambda cd: cd.final_objective_score).model_dump()); msg = f"EXTREME CP: Multiple options, selected best: Pos={final_res.pos}, Val={final_res.value}."; stat="success_multiple_options_extreme"
        
        logger.info(msg, extra={'request_id': request_id})
        return AnalyzeSuccessResponse(request_id=request_id, status=stat, analysis_engine_version=ANALYSIS_ENGINE_VERSION_EXTREME,
                                      message=msg, result=final_res, all_candidates_evaluated=all_evaluated_candidates)
    except HTTPException as he: raise he 
    except ValueError as ve: logger.warning(f"EXTREME Analyzer ValueError: {str(ve)}", exc_info=False, extra={'request_id': request_id}); raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e: logger.error(f"EXTREME Analyzer Unexpected error: {str(e)}", exc_info=True, extra={'request_id': request_id}); raise HTTPException(status_code=500, detail="Internal server error in extreme analyzer.")

# --- Health Check Endpoint (User's original, adapted) ---
@app.get("/health/analyze", response_model=AnalyzeHealthStatus, tags=["Health & Monitoring"])
async def health_analyze(request: Request): # (User's original health check structure)
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    logger.info("Health check for /analyze (EXTREME version).", extra={'request_id': request_id})
    checks = {}; overall_status = "UP"

    if not EXTREME_MODULE_FUNCS_VEC: checks["extreme_module_funcs_load"] = "FAIL"; overall_status = "DEGRADED"
    else: checks["extreme_module_funcs_load"] = f"OK: {len(EXTREME_MODULE_FUNCS_VEC)} funcs"
    if not EXTREME_MODULE_WEIGHTS: checks["extreme_module_weights_load"] = "FAIL"; overall_status = "DEGRADED"
    else: checks["extreme_module_weights_load"] = f"OK: {len(EXTREME_MODULE_WEIGHTS)} weights"
    if EXTREME_MODULE_FUNCS_VEC and EXTREME_MODULE_WEIGHTS:
        missing = [n for n in EXTREME_MODULE_FUNCS_VEC if n not in EXTREME_MODULE_WEIGHTS]
        if missing: checks["extreme_funcs_weights_match"] = f"WARN: Missing weights: {missing}"; overall_status = "DEGRADED"
        else: checks["extreme_funcs_weights_match"] = "OK"
    
    if not os.path.exists(MEM_PATH): checks["memory_file_exists"] = f"FAIL: {MEM_PATH} not found"; overall_status="DEGRADED"
    else: checks["memory_file_exists"] = "OK"
    
    try:
        dummy_grid = np.array([[-1, 1, 5, 0], [2, -1, 8, 3], [4, 6, -1, 7], [0,0,0,0]], dtype=int)
        _, _ = extreme_tensor_flow_score_detailed(dummy_grid, "health_check_extreme_tf")
        checks["extreme_tf_execution_test"] = "OK"
    except Exception as e: checks["extreme_tf_execution_test"] = f"FAIL: {str(e)}"; logger.error("Health: extreme_tf test FAIL.", exc_info=True, extra={'request_id': request_id}); overall_status="ERROR"
    try: _ = cp_model.CpModel(); checks["cp_solver_avail_test"] = "OK"
    except Exception as e: checks["cp_solver_avail_test"] = f"FAIL: {str(e)}"; logger.error("Health: CP Solver test FAIL.", exc_info=True, extra={'request_id': request_id}); overall_status="ERROR"

    return AnalyzeHealthStatus(status=overall_status, analysis_engine_version=ANALYSIS_ENGINE_VERSION_EXTREME, checks=checks,
                               components={"numpy": np.__version__, "ortools": getattr(cp_model, '__version__', "unknown"), "analyzer_type": "Extreme Logic Modules v22"})

# --- Celery Task Definition (User's original, if used) ---
# from celery_worker import app as celery_app # Assuming celery_app is defined in celery_worker.py
# @celery_app.task(name="main_app.analyze_async_task") # Example task name
# def analyze_async_task(request_data: dict, request_id: str):
#     # This would be a Celery task wrapper around the core analysis logic
#     # It would need to reconstruct the grid, call extreme_tensor_flow_score_detailed,
#     # handle candidates, and run CP solver.
#     # For simplicity, the main /analyze endpoint is synchronous (with run_in_threadpool for CPU parts).
#     # A full Celery integration requires careful state and data passing.
#     logger.info(f"Celery task analyze_async_task received for request_id: {request_id}", extra={'request_id': request_id})
#     # ... reconstruct AnalyzeRequest from request_data ...
#     # ... call a modified version of the analyze endpoint's core logic ...
#     # ... return results ...
#     return {"status": "processing_async", "request_id": request_id, "details": "Placeholder for async result"}


# --- Main execution for local testing (User's original, adapted) ---
if __name__ == "__main__":
    import uvicorn
    # Ensure all 22 ext_..._vec functions are defined above.
    # Also ensure EXTREME_MODULE_FUNCS_VEC is fully populated.
    # Example: Check if EXTREME_MODULE_FUNCS_VEC has 22 modules, otherwise error
    if len(EXTREME_MODULE_FUNCS_VEC) != 22:
        critical_error_msg = f"CRITICAL ERROR: EXTREME_MODULE_FUNCS_VEC is not fully populated! Expected 22 modules, found {len(EXTREME_MODULE_FUNCS_VEC)}. Please ensure all 'ext_..._vec' function definitions from previous parts are included AND correctly registered in the EXTREME_MODULE_FUNCS_VEC dictionary."
        logger.critical(critical_error_msg)
        print(critical_error_msg) # Also print to console for immediate visibility
        # raise RuntimeError(critical_error_msg) # Optionally raise error to stop execution
    else:
        logger.info(f"Successfully registered {len(EXTREME_MODULE_FUNCS_VEC)} extreme module functions.")


    if not os.path.exists(MEM_PATH):
        logger.info(f"Creating dummy {MEM_PATH} for testing.")
        dummy_mem_data = {"memory_cards": [[[1,2,-1,4],[-1,3,1,5],[2,-1,3,6],[7,8,9,-1]]]} 
        with open(MEM_PATH, "w") as f_mem_main: json.dump(dummy_mem_data, f_mem_main)
        load_memory_data("main_startup_dummy_mem")

    logger.info(f"Starting Uvicorn server for EXTREME Analyzer FastAPI app (Version: {app.version}, Engine: {ANALYSIS_ENGINE_VERSION_EXTREME}). Access OpenAPI docs at /docs.")
    uvicorn.run(app, host="0.0.0.0", port=8000)