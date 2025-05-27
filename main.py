# main.py (集中式完整版)
import random
import os # os 模組仍然可能用於未來擴展，但在此版本中非核心
import numpy as np
from typing import List, Dict, Tuple, Any

# -----------------------------------------------------------------------------
# 1. 基礎類別定義 (原 logic_module_base.py 和 board_input.py)
# -----------------------------------------------------------------------------

class LogicModule:
    """
    所有評分邏輯模組的基礎類別。
    """
    def __init__(self, module_id: str, name: str, description: str):
        """
        初始化邏輯模組。

        :param module_id: 模組的唯一識別碼。
        :param name: 模組的可讀名稱。
        :param description: 模組功能的簡要描述。
        """
        self.module_id = module_id
        self.name = name
        self.description = description

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        """
        分析給定盤面位置並回傳一個分數。
        此方法應由子類別覆寫以實現具體的評分邏輯。
        預設回傳一個隨機假分數，供初期測試使用。

        :param board_state: 當前整個盤面的狀態 (二維列表)。
        :param position_row: 正在分析的格子的列索引。
        :param position_col: 正在分析的格子的行索引。
        :return: 一個代表該位置評分的浮點數。
        """
        return random.uniform(0.0, 1.0)

    def __repr__(self) -> str:
        return f"<LogicModule module_id='{self.module_id}' name='{self.name}'>"

class BoardInput:
    """
    代表盤面輸入的資料結構。
    """
    def __init__(self, grid: List[List[Any]]):
        """
        初始化盤面輸入。

        :param grid: 一個二維列表，代表盤面格子。
        """
        if not grid or not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
            raise ValueError("盤面必須是一個非空的二維列表。")
        
        row_lengths = [len(row) for row in grid]
        if len(set(row_lengths)) > 1:
            raise ValueError("盤面所有列的長度必須相同。")

        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def get_cell(self, row: int, col: int) -> Any:
        """
        獲取指定位置格子的內容。
        """
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError(f"位置 ({row}, {col}) 超出盤面邊界 ({self.rows}x{self.cols})。")
        return self.grid[row][col]

    def __repr__(self) -> str:
        return f"<BoardInput rows={self.rows} cols={self.cols}>"

    def display(self):
        """
        以可讀格式顯示盤面。
        """
        print(f"Board ({self.rows}x{self.cols}):")
        for row in self.grid:
            print(" ".join(map(str, row)))
        print("-" * (self.cols * 2 if self.cols > 0 else 1))

# -----------------------------------------------------------------------------
# 2. 特定模組實現 (原 modules/a2.py, modules/m3.py 等)
# -----------------------------------------------------------------------------

class A2(LogicModule): #
    def __init__(self):
        super().__init__(
            module_id="A2",
            name="Alpha Module 2 (Proximity Scorer)",
            description="Scores based on proximity to '1' tiles (higher if adjacent)."
        )

    def analyze(self, board_state: list[list[any]], position_row: int, position_col: int) -> float: #
        score = 0.1  # 基本分數 #
        is_one_itself = False #
        is_adjacent_to_one = False #
        rows = len(board_state) #
        cols = len(board_state[0]) if rows > 0 else 0 #

        try:
            if board_state[position_row][position_col] == 1: #
                is_one_itself = True #
        except (TypeError, ValueError, IndexError): 
            pass

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: #
            nr, nc = position_row + dr, position_col + dc #
            if 0 <= nr < rows and 0 <= nc < cols: #
                try:
                    if board_state[nr][nc] == 1: #
                        is_adjacent_to_one = True #
                        break #
                except (TypeError, ValueError):
                    pass #
        
        if is_adjacent_to_one: #
            score = 0.9 #
        
        if is_one_itself: #
            score = max(score, 0.5) #
            
        return score #

class M3(LogicModule): #
    def __init__(self):
        super().__init__(
            module_id="M3",
            name="Mega Module 3 (Neighborhood Counter)",
            description="Scores based on count of '1's in 3x3 neighborhood."
        )

    def analyze(self, board_state: list[list[any]], position_row: int, position_col: int) -> float: #
        count_of_ones = 0 #
        rows = len(board_state) #
        cols = len(board_state[0]) if rows > 0 else 0 #

        for dr in [-1, 0, 1]: #
            for dc in [-1, 0, 1]: #
                nr, nc = position_row + dr, position_col + dc #
                if 0 <= nr < rows and 0 <= nc < cols: #
                    try:
                        if board_state[nr][nc] == 1: #
                            count_of_ones += 1 #
                    except (TypeError, ValueError):
                        pass #
        
        max_possible_ones = 9 #
        score = count_of_ones / max_possible_ones if max_possible_ones > 0 else 0.0 #
        
        return score #

# --- 其餘佔位符模組 (D3, F10, GenMod1-18) ---
# 根據文件 表1 和 generate_modules.py 的 MODULE_CONFIGURATIONS
# 這些模組將使用 LogicModule 的預設 analyze() 方法 (回傳隨機值) 或自訂的隨機值。

class D3_Placeholder(LogicModule):
    def __init__(self):
        super().__init__(module_id="D3", name="Delta Module 3", description="專注於 delta 計算，版本 3。") #
    # analyze 方法將使用基類的隨機實現

class F10_Placeholder(LogicModule):
    def __init__(self):
        super().__init__(module_id="F10", name="Feature Module 10", description="識別進階特徵，版本 10。") #

# GenMod 佔位符
# 我們將為 GenMod1 到 GenMod18 建立佔位符類別
# 這些類別的 module_id 會是 GM1, GM2, ... GM18
GEN_MOD_PLACEHOLDERS = []
for i in range(1, 19):
    class_name = f"GenMod{i}_Placeholder"
    module_id = f"GM{i}"
    name = f"Generated Module {i}"
    description = f"GM{i} 邏輯的佔位符。" # (GM1 描述作為範例)
    
    # 動態創建類別
    # setattr(module, name, type(name, (LogicModule,), { init_method_name: init_method }))
    # 為了簡單和明確，我們直接定義它們，或者在 REGISTERED_MODULES 中直接實例化基礎 LogicModule
    # 此處選擇後者，在註冊時處理

# -----------------------------------------------------------------------------
# 3. 模組註冊與全局權重 (Section 1.3, 4.3)
# -----------------------------------------------------------------------------
REGISTERED_MODULES: List[LogicModule] = []

REGISTERED_MODULES.append(A2())
REGISTERED_MODULES.append(M3())
REGISTERED_MODULES.append(D3_Placeholder())
REGISTERED_MODULES.append(F10_Placeholder())

# 添加 GenMod1 到 GenMod18 的佔位符實例
for i in range(1, 19):
    module_id = f"GM{i}"
    name = f"Generated Module {i}"
    description = f"GM{i} 邏輯的佔位符。" #
    # 使用不同的隨機範圍以符合文件中的 PlaceholderGeneratedModule
    class TempGenMod(LogicModule):
        def __init__(self, mid, mname, mdesc):
            super().__init__(mid, mname, mdesc)
        def analyze(self, board_state: list[list[any]], position_row: int, position_col: int) -> float:
            return random.uniform(0.1, 0.3) # (類似文件中 PlaceholderGeneratedModule 的行為)
    REGISTERED_MODULES.append(TempGenMod(module_id, name, description))


# 確保總共22個模組 (如果上面加起來不足)
# 文件中有一段邏輯是用 PlaceholderGeneratedModule 補齊到22個
# 目前 A2, M3, D3, F10 (4個) + 18個 GenMod = 22個，應該剛好。
if len(REGISTERED_MODULES) < 22:
    print(f"警告: 註冊的模組數 {len(REGISTERED_MODULES)} 少於22個，將用更多佔位符補齊。")
    for i in range(len(REGISTERED_MODULES), 22):
        class FinalPaddingModule(LogicModule):
            def __init__(self, idx):
                super().__init__(f"PAD{idx+1}", f"Padding Module {idx+1}", "Extra padding module")
                self.val_min = random.uniform(0.0,0.2)
                self.val_max = random.uniform(0.3,0.5)
            def analyze(self, board_state: list[list[any]], position_row: int, position_col: int) -> float:
                return random.uniform(self.val_min, self.val_max)
        REGISTERED_MODULES.append(FinalPaddingModule(i))


print(f"Registered {len(REGISTERED_MODULES)} modules:")
for mod in REGISTERED_MODULES:
    print(f" - {mod.module_id}: {mod.name}") #

GLOBAL_MODULE_WEIGHTS: Dict[str, float] = {module.module_id: 1.0 for module in REGISTERED_MODULES} #

# 範例權重調整
if "A2" in GLOBAL_MODULE_WEIGHTS: GLOBAL_MODULE_WEIGHTS["A2"] = 2.0 #
if "M3" in GLOBAL_MODULE_WEIGHTS: GLOBAL_MODULE_WEIGHTS["M3"] = 1.5 #
if "D3" in GLOBAL_MODULE_WEIGHTS: GLOBAL_MODULE_WEIGHTS["D3"] = 0.5 #


# -----------------------------------------------------------------------------
# 4. 核心處理邏輯 (Section 2.3, 2.4, 4)
# -----------------------------------------------------------------------------

def process_board(board_input: BoardInput, modules: List[LogicModule]) -> Dict[Tuple[int, int], Dict[str, float]]: #
    all_cell_scores: Dict[Tuple[int, int], Dict[str, float]] = {}
    for r in range(board_input.rows):
        for c in range(board_input.cols):
            cell_scores: Dict[str, float] = {} #
            for module in modules:
                try:
                    score = module.analyze(board_input.grid, r, c) #
                    cell_scores[module.module_id] = score 
                except Exception as e:
                    print(f"錯誤：模組 {module.module_id} 在分析位置 ({r},{c}) 時發生錯誤: {e}") #
                    cell_scores[module.module_id] = 0.0 
            all_cell_scores[(r, c)] = cell_scores
    return all_cell_scores

def normalize_scores(
    module_scores_by_cell: Dict[Tuple[int, int], Dict[str, float]],
    modules: List[LogicModule], # 需要 modules 列表來獲取所有 module_id
    method: str = 'min-max'
) -> Dict[Tuple[int, int], Dict[str, float]]: #
    if method == "none":
        return module_scores_by_cell #
    
    normalized_scores_by_cell = {cell: {} for cell in module_scores_by_cell} #
    
    module_all_scores: Dict[str, List[float]] = {m.module_id: [] for m in modules} #
    for _cell_pos, scores_dict in module_scores_by_cell.items():
        for mod_id, score in scores_dict.items():
            if mod_id in module_all_scores:
                 module_all_scores[mod_id].append(score) #

    for mod_id_key, scores_list in module_all_scores.items(): #
        if not scores_list: #
            # 如果此模組沒有分數，則所有格子上的此模組正規化分數為0
            for cell_pos_norm in normalized_scores_by_cell:
                normalized_scores_by_cell[cell_pos_norm][mod_id_key] = 0.0
            continue

        if method == 'min-max': #
            min_score = min(scores_list) #
            max_score = max(scores_list) #
            for cell_pos_norm in normalized_scores_by_cell: # Iterate all cells for this module
                if mod_id_key in module_scores_by_cell.get(cell_pos_norm, {}):
                    raw_score = module_scores_by_cell[cell_pos_norm][mod_id_key] #
                    if (max_score - min_score) == 0: #
                        normalized_scores_by_cell[cell_pos_norm][mod_id_key] = 0.0 if min_score == 0 else 0.5 #
                    else:
                        normalized_scores_by_cell[cell_pos_norm][mod_id_key] = (raw_score - min_score) / (max_score - min_score) #
                else: # If a cell didn't have this module's score (e.g. error in process_board)
                    normalized_scores_by_cell[cell_pos_norm][mod_id_key] = 0.0


        elif method == 'z-score': #
            mean_score = float(np.mean(scores_list)) #
            std_score = float(np.std(scores_list))   #
            for cell_pos_norm in normalized_scores_by_cell:
                 if mod_id_key in module_scores_by_cell.get(cell_pos_norm, {}):
                    raw_score = module_scores_by_cell[cell_pos_norm][mod_id_key] #
                    if std_score == 0: #
                        normalized_scores_by_cell[cell_pos_norm][mod_id_key] = 0.0 #
                    else:
                        normalized_scores_by_cell[cell_pos_norm][mod_id_key] = (raw_score - mean_score) / std_score #
                 else:
                    normalized_scores_by_cell[cell_pos_norm][mod_id_key] = 0.0
        else: 
            for cell_pos_norm in normalized_scores_by_cell: #
                if mod_id_key in module_scores_by_cell.get(cell_pos_norm, {}):
                    normalized_scores_by_cell[cell_pos_norm][mod_id_key] = module_scores_by_cell[cell_pos_norm][mod_id_key] #
                else:
                    normalized_scores_by_cell[cell_pos_norm][mod_id_key] = 0.0
    return normalized_scores_by_cell


def fuse_scores(
    scores_to_fuse_input: Dict[Tuple[int, int], Dict[str, float]], # Renamed to avoid clash
    weights: Dict[str, float]
) -> Dict[Tuple[int, int], float]: #
    fused_scores_output: Dict[Tuple[int, int], float] = {} # Renamed
    
    effective_weights = weights.copy() #
    all_module_ids_in_scores = set() #
    for scores_dict_val in scores_to_fuse_input.values(): #
        all_module_ids_in_scores.update(scores_dict_val.keys()) #
    
    for mod_id_in_score in all_module_ids_in_scores: #
        if mod_id_in_score not in effective_weights: #
            effective_weights[mod_id_in_score] = 1.0 #

    for cell_pos, mod_scores in scores_to_fuse_input.items(): #
        weighted_sum = 0.0 #
        sum_of_weights = 0.0 #
        
        if not mod_scores: #
            fused_scores_output[cell_pos] = 0.0 #
            continue

        for module_id, norm_score in mod_scores.items(): #
            weight = effective_weights.get(module_id, 1.0) #
            weighted_sum += norm_score * weight #
            sum_of_weights += weight #
        
        if sum_of_weights == 0: #
            fused_scores_output[cell_pos] = 0.0 #
        else:
            fused_scores_output[cell_pos] = weighted_sum / sum_of_weights #
            
    return fused_scores_output

def simple_fuse_scores(
    raw_cell_scores: Dict[Tuple[int, int], Dict[str, float]]
) -> Dict[Tuple[int, int], float]: #
    fused_scores: Dict[Tuple[int, int], float] = {}
    for cell_pos, mod_scores in raw_cell_scores.items():
        if not mod_scores: #
            fused_scores[cell_pos] = 0.0  
            continue #
        
        average_score = sum(mod_scores.values()) / len(mod_scores) if len(mod_scores) > 0 else 0.0 #
        fused_scores[cell_pos] = average_score #
    return fused_scores


def get_final_scores_for_board(
    board_input: BoardInput,
    modules: List[LogicModule],
    module_weights: Dict[str, float],
    normalization_method: str = 'min-max'
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], Dict[str, float]]]: #
    """
    處理盤面並回傳每個格子的最終融合分數以及各模組的(正規化後)分數。
    此函式是評分系統的核心API，整合了原始分數計算、正規化和加權融合。

    :param board_input: BoardInput 實例，代表要評分的盤面。
    :param modules: LogicModule 實例的列表，包含所有啟用的評分模組。
    :param module_weights: 一個字典，將 module_id 對應到其權重，用於加權平均。
    :param normalization_method: 字串，指定正規化方法。
                                 可選值: 'min-max', 'z-score', 'none'。
                                 'none' 表示不進行正規化，直接使用原始分數進行融合。
    :return: 一個元組 (final_fused_scores, cell_module_scores):
             - final_fused_scores: 字典 {(row, col): final_score}，每個格子的最終融合分數。
             - cell_module_scores: 字典 {(row, col): {module_id: score}}，
                                   每個格子中各模組的分數 (如果進行了正規化，則為正規化後的分數)。
    :raises ValueError: 如果 normalization_method 不是支援的類型。
    """
    if normalization_method not in ['min-max', 'z-score', 'none']: #
        raise ValueError(f"不支援的正規化方法: {normalization_method}") #

    raw_cell_module_scores = process_board(board_input, modules) #

    scores_to_fuse: Dict[Tuple[int, int], Dict[str, float]]
    if normalization_method != 'none': #
        scores_to_fuse = normalize_scores(raw_cell_module_scores, modules, method=normalization_method) #
    else:
        scores_to_fuse = raw_cell_module_scores #
    
    # 文件在 get_final_scores_for_board 中提到:
    # "if normalization_method == 'none' and not module_weights: # 假設第2節的簡單情境
    #     final_fused_scores = simple_fuse_scores(scores_to_fuse)
    # else:
    #     final_fused_scores = fuse_scores(scores_to_fuse, module_weights)"
    # 然而，GLOBAL_MODULE_WEIGHTS 通常會被傳入，所以 module_weights 不太會是 None 或空字典。
    # 為了更貼近 Section 2 的 simple_fuse_scores (它不使用權重)，我們做一個判斷。
    # 但嚴格來說，Section 2 的主邏輯是直接調用 simple_fuse_scores。
    # 此處我們遵循 get_final_scores_for_board 應總是使用 fuse_scores (它能處理權重，即便權重全為1)
    final_fused_scores = fuse_scores(scores_to_fuse, module_weights) #
        
    return final_fused_scores, scores_to_fuse #

# -----------------------------------------------------------------------------
# 5. 主執行區塊 (依照文件各 Section 進行)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Section 2.2: 創建一個最小測試盤面
    test_board_1_data = [ #
        [0, 0, 1], #
        [0, 1, 0], #
        [1, 0, 0]  #
    ] #
    test_board_1 = BoardInput(grid=test_board_1_data) #

    print("\n===== 第 1 節：模組類別骨架的自動化生成 =====")
    print("此部分在此版本中已將類別定義和實例化整合到 main.py。")
    print(f"已註冊 {len(REGISTERED_MODULES)} 個模組。")
    if len(REGISTERED_MODULES) != 22:
        print(f"警告: 註冊的模組數量 ({len(REGISTERED_MODULES)}) 不等於預期的 22。")


    print("\n===== 第 2 節：初始系統整合與最小盤面測試 =====") #
    print(f"\n處理最小測試盤面 (test_board_1) 使用 {len(REGISTERED_MODULES)} 個模組 (A2, M3 真實邏輯, 其餘虛設)...") #
    test_board_1.display() #
    
    raw_scores_board1_sec2 = process_board(test_board_1, REGISTERED_MODULES) #
    fused_scores_board1_simple = simple_fuse_scores(raw_scores_board1_sec2) #

    print("\n最小測試盤面 (test_board_1) 的初步融合分數 (簡單平均):") #
    print("| 格子座標 (列, 行) | 初步融合分數 (簡單平均) |") #
    print("|---|---|") #
    for r_idx in range(test_board_1.rows):
        for c_idx in range(test_board_1.cols):
            pos = (r_idx, c_idx)
            score_val = fused_scores_board1_simple.get(pos, float('nan'))
            print(f"| ({r_idx},{c_idx}) | {score_val:.4f} |") #
    print("(註：由於部分模組回傳隨機分數，實際值可能每次運行不同。)\n") #


    print("\n===== 第 3 節：實作特定專長評分邏輯 =====") #
    print(f"\n在 test_board_1 上運行，A2 和 M3 使用真實邏輯，其餘 {len(REGISTERED_MODULES)-2} 個模組使用虛設邏輯:") #
    test_board_1.display() #

    raw_scores_board1_mixed_logic = process_board(test_board_1, REGISTERED_MODULES) #
    fused_scores_board1_mixed_simple = simple_fuse_scores(raw_scores_board1_mixed_logic) #

    print("\n使用混合邏輯 (A2, M3 真實, 其餘虛設) 的初步融合分數 (簡單平均):") #
    print("以及 A2, M3 的原始分數 (基於修正後A2邏輯):") #
    print("| 格子 (R,C) | A2 (真實) | M3 (真實) | Fused (簡單平均) |") #
    print("|---|---|---|---|") #
    for r_idx in range(test_board_1.rows):
        for c_idx in range(test_board_1.cols):
            pos = (r_idx, c_idx)
            # 使用 .get(module_id, float('nan')) 以處理可能的模組缺失情況 (雖然此處應都存在)
            a2_score = raw_scores_board1_mixed_logic.get(pos, {}).get("A2", float('nan')) #
            m3_score = raw_scores_board1_mixed_logic.get(pos, {}).get("M3", float('nan')) #
            fused_score = fused_scores_board1_mixed_simple.get(pos, float('nan'))
            print(f"| ({r_idx},{c_idx}) | {a2_score:.2f} | {m3_score:.2f} | {fused_score:.4f} |") #
    print("(註：A2, M3 分數應與文件 Table 3 (修正後) 一致，其餘模組隨機影響 Fused 分數)\n") #


    print("\n===== 第 4 節：優化分數融合流程 =====") #
    print("\nGLOBAL_MODULE_WEIGHTS (部分範例):") #
    limited_weights_to_show = ["A2", "M3", "D3", "F10", "GM1", "GM2"] # 只顯示部分
    for mod_id_gw, weight_gw in GLOBAL_MODULE_WEIGHTS.items():
        if mod_id_gw in limited_weights_to_show:
             print(f"  {mod_id_gw}: {weight_gw}") #
    print("  ... (其餘模組權重按GLOBAL_MODULE_WEIGHTS中的定義)")

    print(f"\n處理 test_board_1，使用 Min-Max 正規化和加權平均:") #
    test_board_1.display() #
    
    fused_scores_b1_adv_minmax, norm_scores_b1_minmax = get_final_scores_for_board( #
        test_board_1,
        REGISTERED_MODULES,
        GLOBAL_MODULE_WEIGHTS,
        normalization_method='min-max' #
    )

    print("\n使用 Min-Max 正規化和加權平均後的融合分數 (test_board_1):") #
    print("| 格子 (R,C) | Norm_A2 | Norm_M3 | Fused (Min-Max, Weighted) |") #
    print("|---|---|---|---|") #
    for r_idx in range(test_board_1.rows):
        for c_idx in range(test_board_1.cols):
            pos = (r_idx, c_idx)
            final_s = fused_scores_b1_adv_minmax.get(pos, float('nan')) #
            a2_n_s = norm_scores_b1_minmax.get(pos, {}).get("A2", float('nan')) #
            m3_n_s = norm_scores_b1_minmax.get(pos, {}).get("M3", float('nan')) #
            
            a2_s_str = f"{a2_n_s:.2f}" if isinstance(a2_n_s, (float, np.floating)) else str(a2_n_s) #
            m3_s_str = f"{m3_n_s:.2f}" if isinstance(m3_n_s, (float, np.floating)) else str(m3_n_s) #
            final_s_str = f"{final_s:.4f}" if isinstance(final_s, (float, np.floating)) else str(final_s) #
            print(f"| ({r_idx},{c_idx}) | {a2_s_str} | {m3_s_str} | {final_s_str} |") #

    print(f"\n處理 test_board_1，使用 Z-Score 正規化和加權平均:") #
    fused_scores_b1_adv_zscore, norm_scores_b1_zscore = get_final_scores_for_board( #
        test_board_1,
        REGISTERED_MODULES, #
        GLOBAL_MODULE_WEIGHTS, #
        normalization_method='z-score' #
    )
    print("\n使用 Z-Score 正規化和加權平均後的融合分數 (test_board_1):") #
    print("| 格子 (R,C) | Norm_A2 (Z) | Norm_M3 (Z) | Fused (Z-Score, Weighted) |")
    print("|---|---|---|---|")
    for r_idx in range(test_board_1.rows):
        for c_idx in range(test_board_1.cols):
            pos = (r_idx, c_idx)
            final_s_z = fused_scores_b1_adv_zscore.get(pos, float('nan'))
            a2_n_s_z = norm_scores_b1_zscore.get(pos, {}).get("A2", float('nan'))
            m3_n_s_z = norm_scores_b1_zscore.get(pos, {}).get("M3", float('nan'))

            a2_s_str_z = f"{a2_n_s_z:.2f}" if isinstance(a2_n_s_z, (float, np.floating)) else str(a2_n_s_z)
            m3_s_str_z = f"{m3_n_s_z:.2f}" if isinstance(m3_n_s_z, (float, np.floating)) else str(m3_n_s_z)
            final_s_str_z = f"{final_s_z:.4f}" if isinstance(final_s_z, (float, np.floating)) else str(final_s_z)
            print(f"| ({r_idx},{c_idx}) | {a2_s_str_z} | {m3_s_str_z} | {final_s_str_z} |") #


    print("\n\n===== 第 5 節：多樣化測試情境的綜合驗證 =====") #
    
    test_board_2_data = [ #
        [1, 1, 1], #
        [1, 0, 1], #
        [1, 1, 1]  #
    ] #
    test_board_2 = BoardInput(grid=test_board_2_data) #

    test_board_3_data = [ #
        [1, 1, 0], #
        [1, 0, 1], #
        [0, 1, 1]  #
    ] #
    test_board_3 = BoardInput(grid=test_board_3_data) #
    
    test_boards = { #
        "Board 1 (Minimal)": test_board_1, #
        "Board 2 (A2 Favored)": test_board_2, #
        "Board 3 (M3 Favored/Complex)": test_board_3 #
    }

    results_all_boards: Dict[str, Dict[Tuple[int, int], float]] = {} #
    normalized_module_scores_all_boards: Dict[str, Dict[Tuple[int, int], Dict[str, float]]] = {} #

    for board_name, board_obj in test_boards.items(): #
        print(f"\n--- 處理 {board_name} ---") #
        board_obj.display() #
        
        fused_scores, normalized_scores = get_final_scores_for_board( #
            board_obj,
            REGISTERED_MODULES,
            GLOBAL_MODULE_WEIGHTS,
            normalization_method='min-max' 
        )
        results_all_boards[board_name] = fused_scores #
        normalized_module_scores_all_boards[board_name] = normalized_scores #

        print(f"\n{board_name} 的融合分數 (Min-Max, 加權):") #
        print("| 格子 (R,C) | Norm_A2 | Norm_M3 | Fused (Min-Max, Weighted) |")
        print("|---|---|---|---|")
        for r_idx in range(board_obj.rows):
            for c_idx in range(board_obj.cols):
                pos = (r_idx, c_idx)
                final_score_5 = fused_scores.get(pos, float('nan')) #
                a2_norm_s_5 = normalized_scores.get(pos, {}).get("A2", float('nan')) #
                m3_norm_s_5 = normalized_scores.get(pos, {}).get("M3", float('nan')) #

                a2_str_5 = f"{a2_norm_s_5:.2f}" if isinstance(a2_norm_s_5, (float, np.floating)) else str(a2_norm_s_5) #
                m3_str_5 = f"{m3_norm_s_5:.2f}" if isinstance(m3_norm_s_5, (float, np.floating)) else str(m3_norm_s_5) #
                final_str_5 = f"{final_score_5:.4f}" if isinstance(final_score_5, (float, np.floating)) else str(final_score_5) #
                
                print(f"| ({r_idx},{c_idx}) | {a2_str_5} | {m3_str_5} | {final_str_5} |") #

    print("\nAPI 輸出範例 (Board 2 的融合分數 - results_all_boards['Board 2 (A2 Favored)'] ):") #
    example_api_output_board2 = results_all_boards.get("Board 2 (A2 Favored)", {}) #
    if example_api_output_board2:
        for pos_api, score_api in example_api_output_board2.items(): #
            print(f"  Cell {pos_api}: {score_api:.4f}") #
    else:
        print("  (Board 2 結果未找到)")

    print("\n表 5：跨多樣化測試盤面的代表性格子融合分數比較 (Min-Max 正規化, 加權平均)") #
    print("| 格子座標 (列, 行) | 融合分數 (盤面 1) | 融合分數 (盤面 2 - A2 偏好) | 融合分數 (盤面 3 - M3 偏好) |") #
    print("|---|---|---|---|") #
    example_coords = [(0,0), (1,1), (2,2)] # (範例座標)
    for r_ex, c_ex in example_coords: #
        pos_ex = (r_ex, c_ex)
        s1 = results_all_boards.get("Board 1 (Minimal)", {}).get(pos_ex, float('nan'))
        s2 = results_all_boards.get("Board 2 (A2 Favored)", {}).get(pos_ex, float('nan'))
        s3 = results_all_boards.get("Board 3 (M3 Favored/Complex)", {}).get(pos_ex, float('nan'))
        s1_str = f"{s1:.4f}" if isinstance(s1, (float, np.floating)) else "N/A" #
        s2_str = f"{s2:.4f}" if isinstance(s2, (float, np.floating)) else "N/A" #
        s3_str = f"{s3:.4f}" if isinstance(s3, (float, np.floating)) else "N/A" #
        print(f"| ({r_ex},{c_ex}) | {s1_str} | {s2_str} | {s3_str} |") #
    print("(註：此處數值為實際運行結果，應觀察到不同盤面之間分數的顯著變化。)\n") #

    print("\n===== 第 6 節：最終程式碼結構、註釋與執行指南 =====") #
    print("程式碼結構：所有內容已整合至此單一 main.py 檔案。") # (概念性引用)
    print("註釋與文檔字串已加入主要函式與類別中。") #
    print("執行指南：") #
    print("1. 確認 Python 版本 (建議 3.8+)。") #
    print("2. 安裝必要函式庫: pip install numpy (如果尚未安裝)。建議建立 requirements.txt 並寫入 'numpy>=1.20.0'") #
    print("3. 將此完整程式碼儲存為 main.py。")
    print("4. 執行主程式: python main.py") #
    print("5. 預期輸出將依照 Section 2 至 5 的內容逐步顯示。") #

    print("\n結論與後續步驟：系統已依照文件逐步建構並驗證。")
    print("後續建議：逐步完成其餘模組邏輯、細化權重、擴展測試案例、建立單元測試等。") #

