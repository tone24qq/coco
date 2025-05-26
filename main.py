import logging
from typing import List, Optional, Dict, Tuple, Any, Type
from abc import ABC, abstractmethod
import math
import time
import os # Added for file paths
import json # Added for JSON operations

from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# --- Logging configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 0. Configuration & File Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_CONTRIBUTION_WEIGHTS_PATH = os.path.join(BASE_DIR, "module_contribution_weights.json")
LOCAL_HISTORICAL_MEMORY_PATH = os.path.join(BASE_DIR, "local_historical_memory.json")

# --- Global Configuration for Weights ---
# Default module contribution weights (used if file not found or key missing)
DEFAULT_MODULE_CONTRIBUTION_WEIGHTS: Dict[str, float] = {
    "M1_BaseScore": 1.0,
    "M2_VectorAP": 1.0, # Increased default weight for implemented module
    "M3_VectorGP": 0.7,
    "M4_SymmetryAxial": 0.6,
    "M5_SegmentDiff": 0.5,
    "M6_Historical": 1.2,
    "M10_BridgeChain": 0.7,
    "M11_GridLinearGrowth": 0.7,
    "M14_SymmetryFill": 0.6,
    "M17_CenterCompletion": 0.7,
    "M21_EndSegmentMean": 0.6,
    "M6_MemorySimilarity": 1.0,
}

# Current module contribution weights (loaded from file or defaults)
CURRENT_MODULE_CONTRIBUTION_WEIGHTS: Dict[str, float] = {}

DEFAULT_LOGIC_CODE_WEIGHT_IF_MISSING: float = 0.1
DEFAULT_MODULE_CONTRIBUTION_WEIGHT_IF_MISSING: float = 0.5

# --- Weight Management Functions ---
def load_module_contribution_weights() -> None:
    global CURRENT_MODULE_CONTRIBUTION_WEIGHTS
    if os.path.exists(MODULE_CONTRIBUTION_WEIGHTS_PATH):
        try:
            with open(MODULE_CONTRIBUTION_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                loaded_weights = json.load(f)
            # Merge with defaults: defaults provide baseline, file overrides
            CURRENT_MODULE_CONTRIBUTION_WEIGHTS = {**DEFAULT_MODULE_CONTRIBUTION_WEIGHTS, **loaded_weights}
            logger.info(f"Loaded module contribution weights from {MODULE_CONTRIBUTION_WEIGHTS_PATH}")
        except Exception as e:
            logger.error(f"Failed to load module contribution weights: {e}. Using defaults.", exc_info=True)
            CURRENT_MODULE_CONTRIBUTION_WEIGHTS = DEFAULT_MODULE_CONTRIBUTION_WEIGHTS.copy()
    else:
        logger.info(f"Module contribution weights file not found at {MODULE_CONTRIBUTION_WEIGHTS_PATH}. Using defaults and creating file.")
        CURRENT_MODULE_CONTRIBUTION_WEIGHTS = DEFAULT_MODULE_CONTRIBUTION_WEIGHTS.copy()
        save_module_contribution_weights() # Create the file with defaults

def save_module_contribution_weights() -> None:
    try:
        with open(MODULE_CONTRIBUTION_WEIGHTS_PATH, "w", encoding="utf-8") as f:
            json.dump(CURRENT_MODULE_CONTRIBUTION_WEIGHTS, f, indent=2, sort_keys=True)
        logger.info(f"Saved module contribution weights to {MODULE_CONTRIBUTION_WEIGHTS_PATH}")
    except Exception as e:
        logger.error(f"Failed to save module contribution weights: {e}", exc_info=True)

# Load weights at startup
load_module_contribution_weights()

# -----------------------------------------------------------------------------
# 1. Pydantic Models for API Request/Response (Unchanged from previous)
# -----------------------------------------------------------------------------
class BoardInput(BaseModel):
    new_card: List[Optional[int]] = Field(
        ...,
        description="一維列表，盤面內容 (null 代表遮蔽格)。Row-major order.",
        example=[None, 2, None, 5, None, 6, 7, 8, None, 10, 11, None]
    )
    proposed_values: List[int] = Field(
        ...,
        description="要推理的目標數字列表。",
        example=[3, 9]
    )
    cols: int = Field(
        ...,
        gt=0,
        description="盤面的欄數，用於還原盤面形狀並推導行數。"
    )
    position_codes: Optional[List[str]] = Field(
        default=None,
        description="可選，每格代號列表，長度需等於 new_card。若不提供，系統將自動生成。"
    )
    logic_code_weights: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="將邏輯代碼映射到其基礎權重的字典 (M1_BaseScore 的輸入)。"
    )
    active_modules: Optional[List[str]] = Field(
        default=None,
        description="要啟用的特定邏輯模組ID列表。若為None或空，則啟用所有已註冊模組。"
    )
    module_weights: Optional[Dict[str, float]] = Field( # User can override loaded/default weights per request
        default=None,
        description="各邏輯模組的貢獻權重。若提供，將覆蓋從文件加載的當前請求的權重。"
    )
    top_n_count: int = Field(
        default=3,
        gt=0,
        description="每個提議數值返回的最佳位置數量上限。"
    )
    historical_api_endpoint: Optional[str] = Field(
        default=None,
        description="外部歷史數據API的端點URL (用於 M6_Historical)。"
    )

    @validator('position_codes')
    def validate_position_codes_length(cls, v, values):
        if v is not None and 'new_card' in values:
            if len(v) != len(values['new_card']):
                raise ValueError("position_codes 長度必須與 new_card 長度一致。")
        return v

    @validator('cols')
    def validate_cols_and_new_card_length(cls, v, values):
        if 'new_card' in values and v > 0: # v is cols
            if len(values['new_card']) > 0 and len(values['new_card']) % v != 0:
                raise ValueError(f"new_card 的長度 ({len(values['new_card'])}) 必須能被 cols ({v}) 整除。")
        return v

class PositionScore(BaseModel):
    position_code: str = Field(..., description="盤格的邏輯代碼。")
    score: float = Field(..., description="該位置對於提議數值的綜合評分。")

class ValuePrediction(BaseModel):
    proposed_value: int = Field(..., description="提議的數值。")
    top_n_positions: List[PositionScore] = Field(..., description="該數值最可能的Top-N位置及其評分。")

class InferenceResponse(BaseModel):
    predictions: List[ValuePrediction] = Field(..., description="所有提議數值的預測結果列表。")
    processing_time_ms: Optional[float] = Field(default=None, description="處理請求耗時（毫秒）。")
    warnings: Optional[List[str]] = Field(default=None, description="處理過程中的警告信息。")

class ModuleInfo(BaseModel):
    module_id: str
    name: str
    description: str

# -----------------------------------------------------------------------------
# 2. Internal Data Structures (Unchanged from previous)
# -----------------------------------------------------------------------------
class InternalBoardCell:
    def __init__(self, row: int, col: int, logic_code: str, value: Optional[int], is_fixed: bool, base_score: float):
        self.row = row
        self.col = col
        self.logic_code = logic_code
        self.value = value
        self.is_fixed = is_fixed
        self.base_score = base_score

class InternalBoardState:
    def __init__(self, board_input: BoardInput):
        self.source_input = board_input
        self.rows: int
        self.cols: int = board_input.cols
        
        if self.cols <= 0: self.rows = 0
        elif not board_input.new_card: self.rows = 0
        else:
            if len(board_input.new_card) % self.cols != 0:
                raise ValueError("InternalBoardState: new_card length not divisible by cols.")
            self.rows = len(board_input.new_card) // self.cols

        self.grid_1d: List[Optional[int]] = board_input.new_card
        self.logic_codes_1d: List[str] = []
        self.board: List[List[InternalBoardCell]] = []
        self._initialize_board()

    def _initialize_board(self):
        if not self.grid_1d and self.cols > 0 and self.rows == 0:
            self.logic_codes_1d = []; self.board = []; return

        if self.source_input.position_codes:
            if len(self.source_input.position_codes) != len(self.grid_1d):
                 raise ValueError("Provided position_codes length does not match new_card length.")
            self.logic_codes_1d = self.source_input.position_codes
        else:
            self.logic_codes_1d = self._generate_default_logic_codes(self.rows, self.cols)
        
        if self.grid_1d and len(self.logic_codes_1d) != len(self.grid_1d):
             raise ValueError("Critical error in logic code and grid alignment.")

        idx_1d = 0
        for r in range(self.rows):
            row_cells: List[InternalBoardCell] = []
            for c in range(self.cols):
                value = self.grid_1d[idx_1d] if self.grid_1d else None
                logic_code = self.logic_codes_1d[idx_1d] if self.logic_codes_1d else f"ERR{r}{c}"
                is_fixed = value is not None
                
                base_cell_score = (self.source_input.logic_code_weights.get(logic_code, DEFAULT_LOGIC_CODE_WEIGHT_IF_MISSING)
                                   if self.source_input.logic_code_weights
                                   else DEFAULT_LOGIC_CODE_WEIGHT_IF_MISSING)

                cell_state = InternalBoardCell(r, c, logic_code, value, is_fixed, base_cell_score)
                row_cells.append(cell_state)
                idx_1d += 1
            self.board.append(row_cells)

    def _get_col_letter(self, n: int) -> str:
        string = ""; temp_n = n
        while temp_n >= 0:
            string = chr(ord('A') + temp_n % 26) + string
            temp_n = temp_n // 26 - 1
        return string if string else ("A" if n==0 else "")

    def _generate_default_logic_codes(self, num_rows: int, num_cols: int) -> List[str]:
        codes = []
        if num_rows == 0 or num_cols == 0: return []
        for r_idx in range(num_rows):
            for c_idx in range(num_cols):
                codes.append(f"{self._get_col_letter(c_idx)}{r_idx + 1}")
        return codes

    def get_cell(self, row: int, col: int) -> Optional[InternalBoardCell]:
        if 0 <= row < self.rows and 0 <= col < self.cols: return self.board[row][col]
        return None

    def get_board_id(self) -> str:
        """Creates a unique identifier for the current board state (excluding proposed values)."""
        # Simplified version inspired by user's _make_board_id
        # Uses the 1D grid and dimensions for hashing.
        # None is replaced with a special marker for hashing consistency.
        # Ensure that the order of elements in grid_1d is fixed (row-major).
        filled_part_tuple = tuple(val if val is not None else -999 for val in self.grid_1d)
        grid_hash = hash((self.rows, self.cols, filled_part_tuple))
        empty_count = self.grid_1d.count(None)
        return f"{self.rows}x{self.cols}_empty{empty_count}_hash{grid_hash}"

# -----------------------------------------------------------------------------
# 3. Logic Module Framework (Strategy Pattern)
# -----------------------------------------------------------------------------
class LogicModule(ABC): # (Unchanged from previous)
    def __init__(self, module_id: str, name: str, description: str):
        self.module_id = module_id; self.name = name; self.description = description
    @abstractmethod
    def analyze(self, board_state: InternalBoardState, cell_to_evaluate: Tuple[int, int], proposed_value: int) -> float: pass
    def get_info(self) -> ModuleInfo: return ModuleInfo(module_id=self.module_id, name=self.name, description=self.description)

class M1_BaseScoreModule(LogicModule): # (Unchanged from previous)
    def __init__(self): super().__init__("M1_BaseScore", "基礎位置權重模組", "...")
    def analyze(self, board_state: InternalBoardState, cell_to_evaluate: Tuple[int, int], pv: int) -> float:
        cell = board_state.get_cell(cell_to_evaluate[0], cell_to_evaluate[1])
        return cell.base_score if cell else DEFAULT_LOGIC_CODE_WEIGHT_IF_MISSING

class M2_VectorAPModule(LogicModule): # (Unchanged from previous, using the implemented version)
    def __init__(self): super().__init__("M2_VectorAP", "等差向量推理", "判斷新值是否與周圍形成等差數列（橫、直、斜）。")
    def _get_value_at(self, r: int, c: int, board_state: InternalBoardState) -> Optional[int]:
        if 0 <= r < board_state.rows and 0 <= c < board_state.cols:
            cell = board_state.board[r][c]
            if cell and cell.is_fixed: return cell.value
        return None
    def _check_ap(self, Nums: List[Optional[int]]) -> bool:
        actual_values = [v for v in Nums if v is not None]
        if len(actual_values) < 3: return False
        diff = actual_values[1] - actual_values[0]
        for i in range(2, len(actual_values)):
            if actual_values[i] - actual_values[i-1] != diff: return False
        return True
    def analyze(self, board_state: InternalBoardState, cell_eval: Tuple[int, int], pv: int) -> float:
        r_eval, c_eval = cell_eval; max_ap_score = 0.0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        score_map_bridge = {3: 0.7}; score_map_extend = {3: 0.6, 4: 0.85, 5: 1.0}
        for dr, dc in directions:
            val_m1 = self._get_value_at(r_eval - dr, c_eval - dc, board_state)
            val_p1 = self._get_value_at(r_eval + dr, c_eval + dc, board_state)
            if val_m1 is not None and val_p1 is not None:
                if self._check_ap([val_m1, pv, val_p1]): max_ap_score = max(max_ap_score, score_map_bridge[3])
            val_m2 = self._get_value_at(r_eval - 2*dr, c_eval - 2*dc, board_state)
            if val_m2 is not None and val_m1 is not None:
                if self._check_ap([val_m2, val_m1, pv]): max_ap_score = max(max_ap_score, score_map_extend[3])
            val_p2 = self._get_value_at(r_eval + 2*dr, c_eval + 2*dc, board_state)
            if val_p1 is not None and val_p2 is not None:
                if self._check_ap([pv, val_p1, val_p2]): max_ap_score = max(max_ap_score, score_map_extend[3])
            if val_m2 is not None and val_m1 is not None and val_p1 is not None:
                if self._check_ap([val_m2, val_m1, pv, val_p1]): max_ap_score = max(max_ap_score, score_map_extend[4])
            if val_m1 is not None and val_p1 is not None and val_p2 is not None:
                if self._check_ap([val_m1, pv, val_p1, val_p2]): max_ap_score = max(max_ap_score, score_map_extend[4])
            if val_m2 is not None and val_m1 is not None and val_p1 is not None and val_p2 is not None:
                 if self._check_ap([val_m2, val_m1, pv, val_p1, val_p2]): max_ap_score = max(max_ap_score, score_map_extend[5])
        logger.debug(f"M2_VectorAPModule for cell ({r_eval},{c_eval}), PV {pv}: final score={max_ap_score:.4f}")
        return max_ap_score

class M3_VectorGPModule(LogicModule):
    def __init__(self): super().__init__("M3_VectorGP", "等比向量", "判斷新值是否與周圍形成等比數列")
    def analyze(self, board_state: InternalBoardState, cell_eval: Tuple[int, int], pv: int) -> float:
        logger.debug(f"{self.module_id} @ ({cell_eval[0]},{cell_eval[1]}) for PV {pv} - Placeholder. Algo ideas in user's file.")
        return 0.3

class M4_SymmetryAxialModule(LogicModule):
    def __init__(self): super().__init__("M4_SymmetryAxial", "軸對稱性", "判斷盤面左右/上下/對角是否因新值更對稱")
    def analyze(self, board_state: InternalBoardState, cell_eval: Tuple[int, int], pv: int) -> float:
        logger.debug(f"{self.module_id} @ ({cell_eval[0]},{cell_eval[1]}) for PV {pv} - Placeholder. Algo ideas (a3,a4,a8_symmetry_vec) in user's file.")
        return 0.3

class M5_SegmentDiffModule(LogicModule):
    def __init__(self): super().__init__("M5_SegmentDiff", "段差推理", "分析局部區塊的數值變化趨勢，一致性強加分")
    def analyze(self, board_state: InternalBoardState, cell_eval: Tuple[int, int], pv: int) -> float:
        logger.debug(f"{self.module_id} @ ({cell_eval[0]},{cell_eval[1]}) for PV {pv} - Placeholder. Algo ideas (m1,m2_seq_pattern_vec) in user's file.")
        return 0.3

class M6_HistoricalModule(LogicModule):
    def __init__(self):
        super().__init__("M6_Historical", "歷史記憶", "呼叫 API 或本地JSON查詢歷史卡片相似格")
        self._local_memory_cache: Dict[str, Any] = {}
        self._load_local_memory()

    def _load_local_memory(self):
        if os.path.exists(LOCAL_HISTORICAL_MEMORY_PATH):
            try:
                with open(LOCAL_HISTORICAL_MEMORY_PATH, "r", encoding="utf-8") as f:
                    self._local_memory_cache = json.load(f)
                logger.info(f"Loaded local historical memory from {LOCAL_HISTORICAL_MEMORY_PATH}")
            except Exception as e:
                logger.error(f"Failed to load local historical memory: {e}", exc_info=True)
                self._local_memory_cache = {}
        else:
            logger.info(f"Local historical memory file not found at {LOCAL_HISTORICAL_MEMORY_PATH}. Will be empty.")
            self._local_memory_cache = {}
            # Optionally create an empty file:
            # with open(LOCAL_HISTORICAL_MEMORY_PATH, "w", encoding="utf-8") as f: json.dump({}, f)

    def analyze(self, board_state: InternalBoardState, cell_to_evaluate: Tuple[int, int], proposed_value: int) -> float:
        r, c = cell_to_evaluate
        api_endpoint = board_state.source_input.historical_api_endpoint
        
        if api_endpoint:
            logger.debug(f"{self.module_id} @ ({r},{c}) for PV {proposed_value} - Placeholder for API call to {api_endpoint}")
            # Actual API call logic (e.g., using httpx) would go here
            # try:
            #     response = httpx.post(api_endpoint, json={...})
            #     return response.json().get("score", 0.5)
            # except Exception as e:
            #     logger.error(f"{self.module_id} API call failed: {e}")
            #     return 0.2 # Low score on error
            return 0.5  # Placeholder if API call is not implemented
        else:
            # Fallback to local JSON memory
            board_id = board_state.get_board_id()
            cell = board_state.get_cell(r, c)
            if not cell: return 0.0 # Should not happen
            
            memory_key = f"{cell.logic_code}_{proposed_value}"
            board_memory = self._local_memory_cache.get(board_id, {})
            entry = board_memory.get(memory_key)

            if entry and isinstance(entry, dict) and "score" in entry:
                score = entry["score"]
                count = entry.get("count", 1) # Assume count if not present
                logger.debug(f"{self.module_id} @ ({r},{c}) for PV {proposed_value} - Found in local memory: score={score}, count={count} (BoardID: {board_id}, Key: {memory_key})")
                return max(0.0, min(1.0, float(score))) # Ensure normalization
            else:
                logger.debug(f"{self.module_id} @ ({r},{c}) for PV {proposed_value} - Not found in local memory (BoardID: {board_id}, Key: {memory_key})")
                return 0.4 # Neutral-low score if not in local memory

# Placeholder modules from user's list (with hints to user's file)
class M10_BridgeChainModule(LogicModule):
    def __init__(self): super().__init__("M10_BridgeChain", "搭橋邏輯", "...")
    def analyze(self, board_state: InternalBoardState, c_eval: Tuple[int, int], pv: int) -> float:
        logger.debug(f"{self.module_id} @ ({c_eval[0]},{c_eval[1]}) for PV {pv} - Placeholder. Consider adapting logic for bridging gaps.")
        return 0.3
# ... (Other placeholder modules M11, M14, M17, M21, M6_MemorySimilarity should be similarly updated with logging and comments) ...
class M11_GridLinearGrowthModule(LogicModule):
    def __init__(self): super().__init__("M11_GridLinearGrowth", "格位遞增", "檢查格位編號與值是否呈線性關係")
    def analyze(self, board_state: InternalBoardState, cell_eval: Tuple[int, int], pv: int) -> float:
        logger.debug(f"{self.module_id} @ ({cell_eval[0]},{cell_eval[1]}) for PV {pv} - Placeholder. Your file has 'M10_seq_order_match_vec' which is related to sequence order.")
        return 0.3
class M14_SymmetryFillModule(LogicModule):
    def __init__(self): super().__init__("M14_SymmetryFill", "對稱填補", "對稱位置有數時預測缺值位置")
    def analyze(self, board_state: InternalBoardState, cell_eval: Tuple[int, int], pv: int) -> float:
        logger.debug(f"{self.module_id} @ ({cell_eval[0]},{cell_eval[1]}) for PV {pv} - Placeholder. Your file's a3, a4, a8 are related.")
        return 0.3
class M17_CenterCompletionModule(LogicModule):
    def __init__(self): super().__init__("M17_CenterCompletion", "中央延伸", "中央已知數值向兩邊推理延伸數列")
    def analyze(self, board_state: InternalBoardState, cell_eval: Tuple[int, int], pv: int) -> float:
        logger.debug(f"{self.module_id} @ ({cell_eval[0]},{cell_eval[1]}) for PV {pv} - Placeholder.")
        return 0.3
class M21_EndSegmentMeanModule(LogicModule):
    def __init__(self): super().__init__("M21_EndSegmentMean", "行尾均差補格", "行尾/列尾的數列趨勢延伸補值")
    def analyze(self, board_state: InternalBoardState, cell_eval: Tuple[int, int], pv: int) -> float:
        logger.debug(f"{self.module_id} @ ({cell_eval[0]},{cell_eval[1]}) for PV {pv} - Placeholder. Your file's m1, m2, m8 are related to sequence/gap analysis.")
        return 0.3
class M6_MemorySimilarityModule(LogicModule):
    def __init__(self): super().__init__("M6_MemorySimilarity", "本地樣本記憶", "根據已記錄的樣本卡片比對相似盤面來推補值")
    def analyze(self, board_state: InternalBoardState, cell_eval: Tuple[int, int], pv: int) -> float:
        logger.debug(f"{self.module_id} @ ({cell_eval[0]},{cell_eval[1]}) for PV {pv} - Placeholder. This could use similar local JSON memory as M6_Historical or your file's 'mem_score' logic.")
        # This might be very similar to M6_Historical's local memory path if not distinguished.
        # For now, let it be a placeholder. It could use a different JSON file or a different section in the same file.
        return 0.3

# -----------------------------------------------------------------------------
# 4. Module Registration and Management (Unchanged from previous)
# -----------------------------------------------------------------------------
class ModuleRegistry:
    def __init__(self):
        self._modules: Dict[str, LogicModule] = {}; self._register_default_modules()
    def _register_default_modules(self):
        self.register_module(M1_BaseScoreModule()); self.register_module(M2_VectorAPModule())
        self.register_module(M3_VectorGPModule()); self.register_module(M4_SymmetryAxialModule())
        self.register_module(M5_SegmentDiffModule()); self.register_module(M6_HistoricalModule())
        self.register_module(M10_BridgeChainModule()); self.register_module(M11_GridLinearGrowthModule())
        self.register_module(M14_SymmetryFillModule()); self.register_module(M17_CenterCompletionModule())
        self.register_module(M21_EndSegmentMeanModule()); self.register_module(M6_MemorySimilarityModule())
    def register_module(self, module_instance: LogicModule):
        if module_instance.module_id in self._modules: logger.warning(f"Module ID '{module_instance.module_id}' already registered.")
        self._modules[module_instance.module_id] = module_instance
    def get_module(self, module_id: str) -> Optional[LogicModule]: return self._modules.get(module_id)
    def get_all_modules(self) -> List[LogicModule]: return list(self._modules.values())
    def get_module_infos(self) -> List[ModuleInfo]: return [m.get_info() for m in self._modules.values()]

module_registry = ModuleRegistry()

# -----------------------------------------------------------------------------
# 5. Inference Engine
# -----------------------------------------------------------------------------
class InferenceEngine: # Modified to use CURRENT_MODULE_CONTRIBUTION_WEIGHTS and request overrides
    def __init__(self, registry: ModuleRegistry = Depends(lambda: module_registry)):
        self.registry = registry

    def run_inference(self, board_state: InternalBoardState) -> Tuple[List[ValuePrediction], List[str]]:
        all_value_predictions: List[ValuePrediction] = []
        warnings_log: List[str] = []
        active_module_ids_input = board_state.source_input.active_modules
        
        active_logic_modules: List[LogicModule] = []
        if active_module_ids_input is None or not active_module_ids_input:
            active_logic_modules = self.registry.get_all_modules()
            logger.info("No active_modules specified, using all registered modules.")
        else:
            for mod_id in active_module_ids_input:
                module = self.registry.get_module(mod_id)
                if module: active_logic_modules.append(module)
                else: warnings_log.append(f"Requested active module ID '{mod_id}' not found.")
            logger.info(f"Using specified active modules: {[m.module_id for m in active_logic_modules]}")
        
        is_only_m1_active_explicitly = (active_module_ids_input and "M1_BaseScore" in active_module_ids_input and len(active_module_ids_input) == 1)

        if not active_logic_modules and not (is_only_m1_active_explicitly and board_state.source_input.logic_code_weights):
            warnings_log.append("No effective logic modules selected or found. Inference results may be uniform or zero.")

        # Use global weights as base, override with request-specific weights if provided
        effective_module_contrib_weights = CURRENT_MODULE_CONTRIBUTION_WEIGHTS.copy()
        if board_state.source_input.module_weights is not None: # Check for None explicitly
            effective_module_contrib_weights.update(board_state.source_input.module_weights)
            logger.info("Overriding global module contribution weights with request-specific weights.")

        for proposed_val in board_state.source_input.proposed_values:
            position_candidates: List[Tuple[InternalBoardCell, float]] = []
            is_pv_on_board = any(c.value == proposed_val for r_list in board_state.board for c in r_list if c.is_fixed)
            if is_pv_on_board:
                warnings_log.append(f"Proposed value {proposed_val} is already on the board. Skipping.")
                all_value_predictions.append(ValuePrediction(proposed_value=proposed_val, top_n_positions=[]))
                continue

            for r in range(board_state.rows):
                for c_idx in range(board_state.cols):
                    cell_state = board_state.get_cell(r, c_idx)
                    if not cell_state or cell_state.is_fixed: continue
                    numerator, denominator = 0.0, 0.0
                    
                    m1_module = self.registry.get_module("M1_BaseScore")
                    is_m1_active = (active_module_ids_input is None or not active_module_ids_input or "M1_BaseScore" in active_module_ids_input)

                    if m1_module and is_m1_active:
                        raw_m1 = m1_module.analyze(board_state, (r,c_idx), proposed_val)
                        norm_m1 = max(0.0, min(1.0, raw_m1))
                        m1_w = effective_module_contrib_weights.get("M1_BaseScore", DEFAULT_MODULE_CONTRIBUTION_WEIGHT_IF_MISSING)
                        numerator += norm_m1 * m1_w; denominator += m1_w
                        logger.debug(f"Cell ({r},{c_idx})='{cell_state.logic_code}', PV {proposed_val}: M1_BaseScore score={norm_m1:.2f}, weight={m1_w:.2f}")

                    for module_inst in active_logic_modules:
                        if module_inst.module_id == "M1_BaseScore": continue
                        raw_mod = module_inst.analyze(board_state, (r, c_idx), proposed_val)
                        norm_mod = max(0.0, min(1.0, raw_mod))
                        mod_w = effective_module_contrib_weights.get(module_inst.module_id, DEFAULT_MODULE_CONTRIBUTION_WEIGHT_IF_MISSING)
                        numerator += norm_mod * mod_w; denominator += mod_w
                        logger.debug(f"Cell({r},{c_idx})='{cell_state.logic_code}',PV {proposed_val}:{module_inst.module_id} score={norm_mod:.2f},weight={mod_w:.2f}")

                    final_score = (numerator / denominator) if denominator > 0 else 0.0
                    logger.info(f"Cell ({r},{c_idx})='{cell_state.logic_code}', PV {proposed_val}: Final Score = {final_score:.4f}")
                    position_candidates.append((cell_state, final_score))

            position_candidates.sort(key=lambda item: item[1], reverse=True)
            top_n = [PositionScore(position_code=cs.logic_code, score=round(s, 4)) for cs, s in position_candidates[:board_state.source_input.top_n_count]]
            all_value_predictions.append(ValuePrediction(proposed_value=proposed_val, top_n_positions=top_n))
        return all_value_predictions, warnings_log

# -----------------------------------------------------------------------------
# 6. FastAPI Application Setup
# -----------------------------------------------------------------------------
app = FastAPI(title="可自適應盤面補格系統 V1.2", version="1.2.0", description="...", openapi_tags=[...]) # Details omitted for brevity
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
def get_inference_engine(): return InferenceEngine(module_registry)

@app.post("/analyze", response_model=InferenceResponse, summary="...", tags=["Inference"])
async def analyze_board(board_input: BoardInput = Body(...), engine: InferenceEngine = Depends(get_inference_engine)):
    start_proc_time = time.perf_counter()
    try: internal_board = InternalBoardState(board_input)
    except ValueError as e: logger.error(f"Error init board: {e}"); raise HTTPException(status_code=422, detail=str(e))
    if internal_board.rows == 0 and internal_board.cols > 0:
        proc_time = round((time.perf_counter() - start_proc_time) * 1000, 2)
        return InferenceResponse(predictions=[], processing_time_ms=proc_time, warnings=["Board has 0 cells."])
    predictions, warnings = engine.run_inference(internal_board)
    proc_time = round((time.perf_counter() - start_proc_time) * 1000, 2)
    return InferenceResponse(predictions=predictions, processing_time_ms=proc_time, warnings=warnings if warnings else None)

@app.get("/config/logic_modules", response_model=List[ModuleInfo], summary="...", tags=["Configuration"])
async def get_available_logic_modules(registry: ModuleRegistry = Depends(lambda: module_registry)): return registry.get_module_infos()

@app.get("/config/module_weights", response_model=Dict[str, float], summary="獲取當前模組貢獻權重", tags=["Configuration"])
async def get_current_module_weights():
    """返回當前服務端加載的模組貢獻權重。"""
    return CURRENT_MODULE_CONTRIBUTION_WEIGHTS

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down: saving module contribution weights...")
    save_module_contribution_weights()
    # Note: Local historical memory is read-only for now, so no save needed on shutdown for it.
    # If it were updated, it would be saved elsewhere (e.g., after specific operations or via a feedback mechanism).

# -----------------------------------------------------------------------------
# 7. Main execution for local development & Test Case (Unchanged from previous)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Test Case for M2_VectorAPModule (and now checks M6 local memory fallback) ---
    print("--- Running M2_VectorAPModule & M6_Historical (Local Mem) Test Case ---")
    
    # Create a dummy local_historical_memory.json for testing M6 fallback
    dummy_local_mem = {
        "4x5_empty15_hash-1234567890": { # Replace with actual hash if known, or generate one
            "A2_3": {"score": 0.9, "count": 5}, # High score for A2 with value 3 on this board
            "B1_8": {"score": 0.75, "count": 2}
        }
    }
    # Create a board_id for our test_card_data to use in dummy_local_mem
    # For simplicity, we'll manually construct a plausible one. In real use, InternalBoardState.get_board_id() would be used.
    # Our test_card_data has 15 empty cells in a 4x5 grid.
    # Hash depends on content, let's use a placeholder string.
    test_board_id_placeholder = "4x5_empty15_hashSAMPLE" # You'd generate this properly from test_card_data

    dummy_local_mem[test_board_id_placeholder] = {
         "A2_3": {"score": 0.92, "count": 3}, # Test score for M6 local memory
    }

    try:
        with open(LOCAL_HISTORICAL_MEMORY_PATH, "w", encoding="utf-8") as f_mem:
            json.dump(dummy_local_mem, f_mem, indent=2)
        logger.info(f"Created dummy local historical memory at {LOCAL_HISTORICAL_MEMORY_PATH} for test.")
    except Exception as e:
        logger.error(f"Could not create dummy local memory for test: {e}")


    test_card_data = [1, None, 5, None, None, None, 4, None, None, None, None, None, 7, None, 13, None, None, None, None, None]
    test_cols = 5
    test_logic_code_weights = {"A1": 0.5, "A2": 0.8, "A3": 0.5, "B2": 0.5, "C3": 0.5, "C5": 0.5}

    test_board_input_m2 = BoardInput(
        new_card=test_card_data, proposed_values=[3], cols=test_cols,
        logic_code_weights=test_logic_code_weights,
        active_modules=["M1_BaseScore", "M2_VectorAP"],
        module_weights={"M1_BaseScore": 1.0, "M2_VectorAP": 1.0}
    )
    test_board_input_m6_local = BoardInput(
        new_card=test_card_data, proposed_values=[3], cols=test_cols,
        logic_code_weights=test_logic_code_weights, # Need this for M1
        active_modules=["M1_BaseScore", "M6_Historical"], # Test M6 local fallback
        module_weights={"M1_BaseScore": 0.2, "M6_Historical": 1.0}, # Give M6 more weight here
        historical_api_endpoint=None # Ensure local memory is used
    )
    
    try:
        # Test M2
        print("\n--- Testing M2_VectorAP Module Logic ---")
        test_internal_board_m2 = InternalBoardState(test_board_input_m2)
        m2_module_instance = module_registry.get_module("M2_VectorAP")
        cell_to_eval_coords = (0, 1); pv_to_test = 3
        print(f"Board ID for M2 test: {test_internal_board_m2.get_board_id()} (Note: this might differ from placeholder if content hash changes)")
        if m2_module_instance:
            m2_score = m2_module_instance.analyze(test_internal_board_m2, cell_to_eval_coords, pv_to_test)
            print(f"M2_VectorAPModule score for A2 with PV={pv_to_test}: {m2_score:.4f} (Expected around 0.7)")
        
        # Test M6 Local Memory Fallback
        print("\n--- Testing M6_Historical Module (Local Memory Fallback) ---")
        test_internal_board_m6 = InternalBoardState(test_board_input_m6_local)
        # IMPORTANT: To match the dummy_local_mem key, we need the *exact* board_id.
        # Let's overwrite the generated one with our placeholder for this specific test.
        # This is a hack for testing; in reality, the board_id would be consistently generated.
        original_board_id_m6 = test_internal_board_m6.get_board_id()
        print(f"Generated Board ID for M6 test: {original_board_id_m6}")
        print(f"Manually created local_historical_memory.json uses key: {test_board_id_placeholder}")
        print("To make test work, M6 will look for this key in the dummy JSON.")
        
        # Hack: We cannot easily change the board_id *inside* InternalBoardState after creation.
        # For a robust test, the dummy_local_mem should use the *actual* generated board_id.
        # Let's log what the actual ID is, and assume the dummy file was created with it.
        # For this run, we will simulate M6 looking up the pre-set key from dummy_local_mem.
        
        # Re-create dummy_local_mem with the ACTUAL board_id generated by this test run
        actual_test_board_id = test_internal_board_m6.get_board_id()
        dummy_local_mem_corrected = {
            actual_test_board_id: { # Use the actual generated board_id
                 "A2_3": {"score": 0.92, "count": 3}, 
            }
        }
        with open(LOCAL_HISTORICAL_MEMORY_PATH, "w", encoding="utf-8") as f_mem_corr:
            json.dump(dummy_local_mem_corrected, f_mem_corr, indent=2)
        logger.info(f"Re-created dummy local historical memory with actual board_id: {actual_test_board_id}")


        m6_module_instance = module_registry.get_module("M6_Historical")
        if m6_module_instance and isinstance(m6_module_instance, M6_HistoricalModule):
            m6_module_instance._load_local_memory() # Ensure it re-reads the corrected dummy file
            m6_score = m6_module_instance.analyze(test_internal_board_m6, cell_to_eval_coords, pv_to_test)
            print(f"M6_Historical (local) score for A2 with PV={pv_to_test}: {m6_score:.4f} (Expected from dummy file: 0.92)")

        # Test full inference engine with M6 local
        print("\n--- Running Inference Engine with Test Case (M1 & M6 local active) ---")
        engine = InferenceEngine(module_registry)
        predictions_test_m6, warnings_test_m6 = engine.run_inference(test_internal_board_m6)
        for pred in predictions_test_m6:
            print(f"Proposed Value: {pred.proposed_value}")
            for pos_score in pred.top_n_positions:
                print(f"  - Position: {pos_score.position_code}, Score: {pos_score.score}")
        if warnings_test_m6: print(f"Warnings: {warnings_test_m6}")

    except ValueError as e: print(f"Error during test case setup: {e}")
    except Exception as e: print(f"Unexpected error: {e}"); import traceback; traceback.print_exc()

    print("\n--- Starting Uvicorn server for API ---")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

