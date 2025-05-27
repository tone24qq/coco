# analyzer.py
# pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements

"""
核心調度器 (analyzer.py): 專責處理來自 API 的分析請求。

一句話總結：
「所有棋盤邏輯、分析規則、分數運算都只許在 main.py/模組寫，
analyzer.py 永遠只做協調與公平融合，任何越界皆屬大忌。」

本檔案完全遵循高內聚低耦合原則，禁止硬編寫任何業務邏輯、規則、演算法、
模組分數計算與棋盤分析。所有分析行為均應由 main.py 之官方註冊模組執行。
"""

import base64
import io
import logging
from typing import List, Dict, Any, Tuple, Optional, Union # Union is used implicitly by Optional

import matplotlib
matplotlib.use('Agg') # Ensure Matplotlib works in a headless environment
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np

# --- Configuration ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Custom Exceptions ---
class AnalyzerError(Exception):
    """Base class for exceptions in the Analyzer."""
    pass

class InitializationError(AnalyzerError):
    """Error during Analyzer initialization."""
    pass

class InvalidInputError(AnalyzerError):
    """Error due to invalid input parameters."""
    pass

class ModuleError(AnalyzerError):
    """Base class for errors related to main.py modules."""
    pass

class ModuleNotFoundError(ModuleError):
    """Error when a requested module is not found or registered."""
    pass

class ModuleExecutionError(ModuleError):
    """Error during the execution of a module in main.py."""
    pass

class VisualizationError(AnalyzerError):
    """Error during the generation of the visualization."""
    pass

# NoActionableCellError was previously defined but not explicitly raised often.
# Instead, warnings are logged and empty suggestions are returned, which is usually preferred.

class Analyzer:
    """
    智慧評分系統的核心調度器。
    負責接收分析請求，調用 main.py 中的邏輯模組，融合結果，並返回建議。
    嚴格遵守不干涉分析邏輯、僅做協調與公平融合的原則。
    """

    # Define a list of distinct colors for visualizing different proposed values
    PV_COLORS = list(mcolors.TABLEAU_COLORS.values()) + \
                list(mcolors.CSS4_COLORS.values()) # More colors if needed

    def __init__(self, main_module: Any, default_top_n: int = 3):
        """
        初始化 Analyzer。

        Args:
            main_module: 已載入的 main.py 模組。
                         必須提供 `get_module_score(module_name, new_card, proposed_value)` 方法
                         和 `registered_modules` (dict) 屬性。
            default_top_n: 預設回傳的 Top-N 建議數量。

        Raises:
            InitializationError: 若 main_module 不符合要求。
        """
        if not hasattr(main_module, 'get_module_score') or \
           not callable(main_module.get_module_score):
            raise InitializationError(
                "main_module 必須提供可呼叫的 'get_module_score' 方法。"
            )
        if not hasattr(main_module, 'registered_modules') or \
           not isinstance(main_module.registered_modules, dict):
            raise InitializationError(
                "main_module 必須提供 'registered_modules' (字典) 屬性。"
            )

        self.main_module = main_module
        self.default_top_n = default_top_n
        logger.info("Analyzer initialized with default_top_n=%d. Registered modules from main_module: %s",
                    self.default_top_n, list(main_module.registered_modules.keys()))

    def _validate_inputs(
        self,
        new_card: List[List[int]],
        proposed_values: List[int],
        active_modules: Optional[List[str]],
        module_weights: Optional[Dict[str, float]],
        top_n: Optional[int]
    ) -> Tuple[int, int, List[int], Optional[List[str]], Optional[Dict[str, float]], int]:
        """
        驗證輸入參數的合法性。
        此方法僅驗證數據結構和基本類型，不涉及業務邏輯。

        Raises:
            InvalidInputError: 若輸入參數不合法。
        Returns:
            Tuple containing validated (row_count, col_count, proposed_values,
            active_modules, module_weights, top_n).
        """
        if not new_card or not isinstance(new_card, list):
            raise InvalidInputError("盤面 (new_card) 不得為空且必須是列表。")
        if not all(isinstance(row, list) for row in new_card): # Check all elements are lists
            raise InvalidInputError("盤面 (new_card) 的每一行必須是列表。")

        rows = len(new_card)
        if rows == 0:
            raise InvalidInputError("盤面 (new_card) 不得為空 (沒有行)。")
        
        # Assuming all rows should exist before checking cols of the first row.
        cols = len(new_card[0]) if new_card[0] is not None else 0 # handle case of new_card = [None] potentially
        if cols == 0 and rows > 0 : # if first row is empty list
             if not all(len(row) == 0 for row in new_card): # if other rows are not empty
                  raise InvalidInputError("盤面 (new_card) 的列定義不一致或首行為空但其他行非空。")
             # if all rows are empty lists, it's a 0-column board.
        elif cols == 0 and rows == 0: # Already handled by "rows == 0"
            pass


        if not all(len(row) == cols for row in new_card):
            raise InvalidInputError("盤面 (new_card) 必須是矩形 (所有行的列數需一致)。")
        if not all(isinstance(val, int) for row in new_card for val in row):
            raise InvalidInputError("盤面 (new_card) 中的所有值必須是整數。")

        # The presence or absence of -1 cells is a board state, not an error for analyzer.
        # The suggestion logic (_get_top_n_for_pv) will handle it if no -1 cells are found.
        has_negative_one = any(val == -1 for row in new_card for val in row)
        if not has_negative_one and rows > 0 and cols > 0 : # Only warn if board is not empty
            logger.warning("盤面 (new_card) 中沒有 -1 (未開) 的格子。可能無法提供「填入」建議。")


        if not proposed_values or not isinstance(proposed_values, list): # Allow empty list if API spec permits
            # Assuming proposed_values must be non-empty based on "欲找出每個 proposed_value 的最佳填入格子"
            raise InvalidInputError("候選值 (proposed_values) 必須是非空列表。")
        if not all(isinstance(pv, int) for pv in proposed_values):
            raise InvalidInputError("候選值 (proposed_values) 中的所有值必須是整數。")

        if active_modules is not None: # Optional, can be None
            if not isinstance(active_modules, list) or \
               not all(isinstance(m, str) for m in active_modules):
                raise InvalidInputError("啟用模組 (active_modules) 若提供，必須是字串列表。")

        if module_weights is not None: # Optional, can be None
            if not isinstance(module_weights, dict) or \
               not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in module_weights.items()):
                raise InvalidInputError("模組權重 (module_weights) 若提供，必須是 {str: float/int} 格式的字典。")

        final_top_n = top_n if top_n is not None else self.default_top_n
        if not isinstance(final_top_n, int) or final_top_n <= 0:
            raise InvalidInputError(f"Top-N 數量 ({final_top_n}) 必須是正整數。")

        logger.debug("Input validation successful. Rows: %d, Cols: %d, Top_N: %d", rows, cols, final_top_n)
        return rows, cols, proposed_values, active_modules, module_weights, final_top_n

    def _get_effective_modules_and_weights(
        self,
        requested_active_modules: Optional[List[str]],
        requested_module_weights: Optional[Dict[str, float]]
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        根據請求和已註冊模組，確定實際運行的模組及其權重。
        嚴格僅使用 `main_module.registered_modules` 作為模組來源。
        """
        registered_module_names = list(self.main_module.registered_modules.keys())
        effective_module_names: List[str]

        if requested_active_modules is None:
            effective_module_names = registered_module_names # Use all registered if not specified
            logger.info("未指定 active_modules，將使用所有已註冊模組: %s", effective_module_names)
        else:
            effective_module_names = []
            for module_name in requested_active_modules:
                if module_name not in registered_module_names:
                    # 核心原則：任何非 main.py 註冊的模組一律忽略
                    logger.warning("請求的模組 '%s' 未在 main.py 中註冊。將被忽略。", module_name)
                else:
                    effective_module_names.append(module_name)
            if not effective_module_names and requested_active_modules: # requested but none were valid
                 logger.warning("指定的 active_modules (%s) 均未在 main.py 註冊，無模組可執行。", requested_active_modules)
            elif not effective_module_names: # requested_active_modules was empty list
                 logger.warning("active_modules 列表為空，無模組可執行。")


        # 核心原則：公平融合，分數融合嚴守 module_weights 設定
        final_module_weights: Dict[str, float] = {
            name: 1.0 for name in effective_module_names # Default weight is 1.0 for active modules
        }

        if requested_module_weights:
            for name, weight in requested_module_weights.items():
                if name in final_module_weights: # Only apply weights to *effective* modules
                    final_module_weights[name] = float(weight)
                else:
                    # 核心原則：不得額外干預，不允許有權重作弊空間
                    logger.warning(
                        "權重配置中的模組 '%s' 未在生效模組列表 (%s) 中，其權重將被忽略。",
                        name, effective_module_names
                    )
        logger.info("生效模組 (Effective Modules): %s", effective_module_names)
        logger.info("最終模組權重 (Final Module Weights): %s", final_module_weights)
        return effective_module_names, final_module_weights

    def _fuse_scores(
        self,
        module_scores_map: Dict[str, np.ndarray],
        module_weights_map: Dict[str, float],
        rows: int,
        cols: int
    ) -> np.ndarray:
        """
        公平融合來自不同模組的評分。
        融合過程僅依賴傳入的評分和權重，不引入額外邏輯。
        最終結果將正規化至 0-1 範圍。
        """
        # Initialize with zeros. If no modules provide scores, this remains zero.
        fused_scores = np.zeros((rows, cols), dtype=float)

        if not module_scores_map:
            logger.warning("沒有從模組獲取到任何評分，融合結果將為零矩陣。")
            return fused_scores # Return a zero matrix (already normalized in a way)

        active_module_names_with_scores = list(module_scores_map.keys())
        logger.debug("開始融合 %d 個模組的評分: %s", len(active_module_names_with_scores), active_module_names_with_scores)

        for module_name, scores_array in module_scores_map.items():
            # Weight is fetched from the map which only contains effective modules and their final weights
            weight = module_weights_map.get(module_name)
            if weight is None: # Should not happen if _get_effective_modules_and_weights is correct
                logger.error("嚴重內部錯誤：模組 '%s' 在評分融合階段缺少權重。將使用預設值 1.0。", module_name)
                weight = 1.0
            
            if not isinstance(scores_array, np.ndarray) or scores_array.shape != (rows, cols):
                logger.error("模組 '%s' 的評分格式不符 (期望 %dx%d np.ndarray, 得到 %s %s)。此模組分數將被忽略。",
                             module_name, rows, cols, type(scores_array),
                             scores_array.shape if isinstance(scores_array, np.ndarray) else "N/A")
                continue # Skip this module's score in fusion

            logger.debug("融合模組 '%s' 的評分 (權重: %.2f)。", module_name, weight)
            fused_scores += scores_array * weight

        # Normalize the final fused scores to a 0-1 range for consistent interpretation
        # This is a data presentation step, not an alteration of relative analytical merit
        min_score = np.min(fused_scores)
        max_score = np.max(fused_scores)

        if max_score == min_score:
            # All fused scores are identical. Avoid division by zero.
            # If all scores are 0, normalized is 0. If all are X > 0, normalized could be 0.5 or 1.
            # For consistency, if all are same, map to 0 if min_score is 0, else 0.5.
            # Or simply, if range is zero, all normalized scores are 0 (unless min_score > 0, then maybe 1?)
            # Simplest: if range is zero, set all to 0. Or if a single point, it's 0 if it's the min, 1 if it's the max.
            # If all are same, it means no discrimination, so 0.0 might be safest.
            normalized_fused_scores = np.zeros_like(fused_scores)
            if min_score != 0 : # e.g. if all scores are 5, normalized to 0 still.
                 logger.debug("Fused scores are all identical (%.4f), normalized to 0.0.", min_score)

        else:
            normalized_fused_scores = (fused_scores - min_score) / (max_score - min_score)
            logger.debug("Fused scores normalized from range [%.4f, %.4f] to [0, 1].", min_score, max_score)

        return normalized_fused_scores

    def _get_top_n_for_pv(
        self,
        fused_scores_board: np.ndarray,
        board_state: List[List[int]],
        top_n: int
    ) -> List[Dict[str, Any]]:
        """
        從融合後的評分中，為單個 proposed_value 找出 Top-N 建議。
        嚴格僅考慮盤面中為 -1 (未開) 的格子作為候選。
        排序依據為 `fused_scores_board`，不進行額外評分。
        """
        suggestions = []
        rows, cols = fused_scores_board.shape

        candidate_cells: List[Tuple[float, int, int]] = [] # List of (score, row, col)
        has_fillable_cells = False
        for r in range(rows):
            for c in range(cols):
                if board_state[r][c] == -1: # 核心原則：僅在 main.py 指定的目標上操作 (-1 格)
                    has_fillable_cells = True
                    candidate_cells.append((fused_scores_board[r, c], r, c))

        if not has_fillable_cells: # No -1 cells on the board
            logger.info("盤面上沒有值為 -1 的可填入格子。無法為此 proposed_value 提供建議。")
            return []
        if not candidate_cells: # Should be caught by has_fillable_cells, but as a safeguard.
            logger.info("候選格子列表為空 (可能所有格子都不是 -1)。")
            return []


        # Sort candidates by score in descending order.
        # This is a data processing step, not an analytical one.
        candidate_cells.sort(key=lambda x: x[0], reverse=True)

        for score, r, c in candidate_cells[:top_n]:
            suggestions.append({
                'position': [r, c], # 0-indexed for programmatic use
                'score': round(float(score), 6) # Standardized score (0-1), higher precision
            })
        return suggestions

    def analyze_board(
        self,
        new_card: List[List[int]],
        proposed_values: List[int],
        active_modules: Optional[List[str]] = None,
        module_weights: Optional[Dict[str, float]] = None,
        top_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        執行棋盤分析的核心方法。
        此方法負責協調流程、調用模組、融合分數並準備回傳結果。
        它自身不包含任何棋盤分析、規則判斷或評分邏輯。
        """
        logger.info(
            "接收分析請求: %d 個候選值, 盤面尺寸 %dx%d (approx). Active modules hint: %s",
            len(proposed_values) if proposed_values else 0,
            len(new_card) if new_card else 0,
            len(new_card[0]) if new_card and new_card[0] else 0,
            str(active_modules) if active_modules else "ALL"
        )

        # 1. 驗證輸入 (結構性，非業務邏輯)
        rows, cols, validated_pvs, val_active_modules, val_module_weights, final_top_n = \
            self._validate_inputs(new_card, proposed_values, active_modules, module_weights, top_n)

        # 2. 確定生效模組與權重 (基於註冊和輸入，無自創)
        effective_modules, final_weights = self._get_effective_modules_and_weights(
            val_active_modules, val_module_weights
        )

        all_suggestions: Dict[Union[int, str], List[Dict[str, Any]]] = {} # Key can be int or str if PVs are such
        all_fused_scores_for_pvs: Dict[Union[int, str], np.ndarray] = {}

        if not effective_modules:
            logger.warning("沒有任何生效的分析模組。分析將產生空建議和零分盤面。")
            # Still proceed to generate visualization for the board state if possible.
            # And return empty suggestions for all PVs.
            for pv in validated_pvs:
                 all_suggestions[pv] = []
                 all_fused_scores_for_pvs[pv] = np.zeros((rows, cols), dtype=float)

        else: # Modules are available
            for pv_idx, pv in enumerate(validated_pvs):
                logger.info("開始處理候選值: %s (編號 %d / 总数 %d)", pv, pv_idx + 1, len(validated_pvs))
                module_scores_for_pv: Dict[str, np.ndarray] = {} # Scores from valid modules for this PV

                for module_name in effective_modules:
                    try:
                        logger.debug("調用 main.py 模組 '%s' 處理盤面及候選值 '%s'", module_name, pv)
                        # 核心原則：所有分析、推理、評分、圖樣辨識等計算 皆交由 main.py 的模組處理。
                        # analyzer.py 僅負責呼叫 main 中的模組並傳入參數。
                        raw_scores = self.main_module.get_module_score(module_name, new_card, pv)

                        # Validate structure of returned scores from module
                        if not isinstance(raw_scores, (list, np.ndarray)):
                            raise ModuleExecutionError(
                                f"模組 '{module_name}' 為 PV '{pv}' 返回的評分類型不正確 "
                                f"(期望 list 或 np.ndarray, 實際得到: {type(raw_scores).__name__})."
                            )
                        
                        scores_np = np.array(raw_scores, dtype=float) # Convert to NumPy array

                        if scores_np.shape != (rows, cols):
                            raise ModuleExecutionError(
                                f"模組 '{module_name}' 為 PV '{pv}' 返回的評分維度不正確 "
                                f"(期望 {rows}x{cols}, 實際得到: {scores_np.shape})."
                            )
                        module_scores_for_pv[module_name] = scores_np
                        logger.debug("模組 '%s' 為候選值 '%s' 成功返回評分 (維度: %s)。",
                                     module_name, pv, str(scores_np.shape))

                    except ModuleNotFoundError as e: # Should not happen if effective_modules is correctly derived
                        logger.error("嚴重內部錯誤: 模組 '%s' 在調用時未找到，儘管它在生效列表中: %s", module_name, e, exc_info=True)
                    except ModuleExecutionError as e: # Custom error from module or validation fail
                        logger.error("模組 '%s' 為 PV '%s' 執行時發生錯誤或返回不合規數據: %s. 此模組評分將被忽略。",
                                     module_name, pv, e, exc_info=True) # Log full exception for module dev
                    except Exception as e: # Catch any other unexpected errors from module call
                        logger.error("調用模組 '%s' (for PV '%s') 時發生未預期嚴重錯誤: %s. 此模組評分將被忽略。",
                                     module_name, pv, e, exc_info=True)
                        # This ensures analyzer robustness against faulty modules.

                # 3. 蒐集模組分析結果並公平融合 (僅加權和正規化)
                if not module_scores_for_pv and effective_modules: # No module succeeded for this PV
                    logger.warning("對於候選值 '%s', 所有生效模組 (%s) 均未能成功提供評分。", pv, effective_modules)
                    fused_scores_pv = np.zeros((rows, cols), dtype=float) # Default to zero scores
                elif not effective_modules : # Should be covered by the top-level check already
                     fused_scores_pv = np.zeros((rows, cols), dtype=float)
                else:
                    fused_scores_pv = self._fuse_scores(module_scores_for_pv, final_weights, rows, cols)

                all_fused_scores_for_pvs[pv] = fused_scores_pv

                # 4. 根據融合分數回傳 Top-N 建議 (僅排序和篩選)
                all_suggestions[pv] = self._get_top_n_for_pv(fused_scores_pv, new_card, final_top_n)
                logger.info("候選值 '%s': 找到 %d 個建議 (請求 Top-%d)。",
                            pv, len(all_suggestions[pv]), final_top_n)

        # 5. 產生視覺化圖 (僅做展示，不引入業務推理)
        try:
            visualization_b64 = self._generate_visualization(
                new_card,
                validated_pvs,
                all_suggestions,
                all_fused_scores_for_pvs,
                final_top_n # This is the N for suggestions, visualization might show more/less based on its logic
            )
        except Exception as e:
            logger.error("生成視覺化圖像時發生嚴重錯誤: %s", e, exc_info=True)
            # Do not let visualization failure stop data return if analysis was successful.
            # Return placeholder or error message for visualization.
            visualization_b64 = self._generate_error_visualization(rows, cols, f"視覺化失敗: {type(e).__name__}")
            # Or, re-raise if visualization is critical: raise VisualizationError(f"生成視覺化圖像時發生錯誤: {e}") from e

        return {
            'suggestions': all_suggestions,
            'visualization': visualization_b64,
            'board_dimensions': {'rows': rows, 'cols': cols},
            'processed_params': { # Information about how the request was processed
                'requested_top_n': top_n if top_n is not None else f"default ({self.default_top_n})",
                'actual_top_n': final_top_n,
                'requested_active_modules': val_active_modules if val_active_modules is not None else "ALL (default)",
                'effective_active_modules': effective_modules,
                'requested_module_weights': val_module_weights if val_module_weights is not None else "None (default 1.0)",
                'final_module_weights': final_weights
            }
        }

    def _generate_error_visualization(self, rows: int, cols: int, error_message: str) -> str:
        """Generates a placeholder image indicating a visualization error."""
        try:
            fig_width = max(cols * 0.5, 5)
            fig_height = max(rows * 0.5, 3)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.text(0.5, 0.5, f"錯誤:\n{error_message}",
                    ha='center', va='center', fontsize=10, color='red', wrap=True)
            ax.axis('off')
            img_base64 = self._fig_to_base64(fig)
            plt.close(fig)
            return img_base64
        except Exception as e:
            logger.error("生成錯誤提示圖像本身也失敗了: %s", e)
            return "Error generating error visualization."


    def _generate_visualization(
        self,
        board_state: List[List[int]],
        proposed_values: List[int],
        all_suggestions: Dict[Union[int, str], List[Dict[str, Any]]],
        all_fused_scores_for_pvs: Dict[Union[int, str], np.ndarray],
        top_n_suggestion_count: int # The N value used for suggestions
    ) -> str:
        """
        產生盤面視覺化圖像。
        此方法嚴格只展示已分析的結果 (盤面、分數、建議)，不引入新的分析或推理。
        """
        rows = len(board_state)
        cols = len(board_state[0]) if rows > 0 else 0

        if rows == 0 or cols == 0:
            logger.warning("無法生成視覺化：盤面為空。")
            return self._generate_error_visualization(5,1, "盤面為空")


        cell_size_inch = max(0.5, min(1.0, 10.0 / max(rows, cols, 1))) # Dynamic cell size
        fig_width = max(cols * cell_size_inch, 6)
        fig_height = max(rows * cell_size_inch, 4)
        # Add extra space for legend if many PVs
        if len(proposed_values) > 3 : fig_width +=2


        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.set_xticklabels(np.arange(1, cols + 1), fontsize=max(6, cell_size_inch * 10))
        ax.set_yticklabels(np.arange(1, rows + 1), fontsize=max(6, cell_size_inch * 10))
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xlabel("列 (Col)", fontsize=max(7,cell_size_inch * 12))
        ax.set_ylabel("行 (Row)", fontsize=max(7,cell_size_inch * 12))

        ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')


        # 1. 熱力圖 (Heatmap) - 針對第一個 proposed_value 的 -1 格子
        # 視覺化原則：只做展示，根據分析結果做標註、熱力圖
        heatmap_data = np.full((rows, cols), np.nan)
        first_pv_for_heatmap = None
        if proposed_values and proposed_values[0] in all_fused_scores_for_pvs:
            first_pv_for_heatmap = proposed_values[0]
            scores_for_first_pv = all_fused_scores_for_pvs[first_pv_for_heatmap]
            for r_idx in range(rows):
                for c_idx in range(cols):
                    if board_state[r_idx][c_idx] == -1: # Only show scores for fillable cells
                         heatmap_data[r_idx, c_idx] = scores_for_first_pv[r_idx, c_idx]
            
            # Normalize for consistent colormap usage (scores are already 0-1 from fusion)
            # The scores from fusion are already normalized, so heatmap_data for -1 cells is 0-1.
            # No need for re-normalization here if _fuse_scores guarantees 0-1.
            cmap = plt.cm.viridis
            cmap.set_bad(color='white', alpha=0) # Make NaN cells (non -1 cells) transparent
            # Plotting with vmin/vmax ensures consistent color scale if scores are already normalized
            cax = ax.imshow(heatmap_data, cmap=cmap, alpha=0.6, aspect='auto', vmin=0, vmax=1)
            # Add a colorbar for the heatmap
            # Position colorbar to avoid overlap, might need adjustment based on fig_width/height
            # fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04, label=f"PV {first_pv_for_heatmap} 歸一化分數")


        # 2. 標記盤面數字 和 建議位置
        suggestion_texts_on_cells: Dict[Tuple[int, int], List[str]] = {}
        cell_highlights: List[Dict[str,Any]] = [] # Store patches to add later

        for pv_idx, pv in enumerate(proposed_values):
            pv_color = self.PV_COLORS[pv_idx % len(self.PV_COLORS)]
            if pv in all_suggestions:
                # Display up to top_n_display_limit (e.g. 3) suggestions on graph
                # This could be different from `top_n_suggestion_count` used for data
                top_n_to_display_on_graph = min(top_n_suggestion_count, 3) # Max 3 ranks on graph for clarity

                for rank_idx, suggestion in enumerate(all_suggestions[pv][:top_n_to_display_on_graph]):
                    r, c = suggestion['position']
                    rank = rank_idx + 1
                    # score_val = suggestion['score'] # Score available if needed for text
                    
                    text_for_cell = f"{pv}(R{rank})"
                    if (r,c) not in suggestion_texts_on_cells:
                        suggestion_texts_on_cells[(r,c)] = []
                    suggestion_texts_on_cells[(r,c)].append(text_for_cell)

                    rect_line_width = 2.0 if rank == 1 else (1.5 if rank == 2 else 1.0)
                    cell_highlights.append({
                        'coords': (c - 0.5, r - 0.5), 'width': 1, 'height': 1,
                        'linewidth': rect_line_width, 'edgecolor': pv_color,
                        'facecolor': mcolors.to_rgba(pv_color, alpha=0.10 if rank == 1 else 0.05) # Light fill for suggestions
                    })
        
        # Add highlight patches (drawn underneath text)
        for highlight in cell_highlights:
            rect = patches.Rectangle(**{k:v for k,v in highlight.items() if k!= 'coords'},
                                     xy=highlight['coords'])
            ax.add_patch(rect)


        # Add text for board numbers and aggregated suggestion texts
        font_size = max(6, cell_size_inch * 10) # Base font size
        for r_idx in range(rows):
            for c_idx in range(cols):
                cell_value = board_state[r_idx][c_idx]
                
                current_cell_texts = []
                # Original board number or masked cell marker
                if cell_value != -1:
                    current_cell_texts.append(str(cell_value))
                else: # Masked cell
                    # If suggestions exist for this masked cell, they become primary text
                    if (r_idx, c_idx) in suggestion_texts_on_cells:
                        current_cell_texts.extend(suggestion_texts_on_cells[(r_idx, c_idx)])
                    else: # Masked cell, no suggestion
                        current_cell_texts.append("●") # Dot for masked, empty cells

                final_display_text = "\n".join(current_cell_texts)
                
                # Adjust font size if text is too long for the cell
                num_lines = final_display_text.count('\n') + 1
                dynamic_font_size = font_size / num_lines if num_lines > 1 else font_size
                dynamic_font_size = max(4, dynamic_font_size * min(1, (cell_size_inch*10) / (len(final_display_text)/max(num_lines,1) +1) ))


                ax.text(c_idx, r_idx, final_display_text,
                        ha='center', va='center', fontsize=dynamic_font_size, color='black',
                        wrap=True) # Removed bbox for cleaner look, highlights handle visual cue

        # 3. 圖標題和圖例
        pv_str = ", ".join(map(str, proposed_values)) if proposed_values else "無"
        title_str = f"盤面分析 ({rows}x{cols}) - 候選值: [{pv_str}]"
        if not any(sugg_list for sugg_list in all_suggestions.values()): # If all suggestion lists are empty
            title_str += "\n(盤面無 -1 格或模組未提供有效建議)"
        
        plt.title(title_str, fontsize=max(8, cell_size_inch * 14), pad=20) # Add padding for title

        # Create legend for PV colors
        if proposed_values and any(s for pv_suggs in all_suggestions.values() for s in pv_suggs):
            legend_elements = []
            added_pvs_to_legend = set()
            for pv_idx, pv in enumerate(proposed_values):
                 # Only add to legend if this PV has suggestions and hasn't been added
                 if pv not in added_pvs_to_legend and any(s for s in all_suggestions.get(pv,[])):
                    color = self.PV_COLORS[pv_idx % len(self.PV_COLORS)]
                    legend_elements.append(patches.Patch(facecolor=color, edgecolor=color, label=f'候選值 {pv} 建議'))
                    added_pvs_to_legend.add(pv)
            if legend_elements:
                # Position legend outside, adjust bbox_to_anchor and loc as needed
                ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.03, 0.5),
                          fontsize=max(7, cell_size_inch * 10), title="圖例")

        plt.tight_layout(rect=[0, 0, 0.9 if legend_elements else 1, 1]) # Adjust layout to make space for legend if present

        img_base64 = self._fig_to_base64(fig)
        plt.close(fig) # Essential for releasing memory
        return img_base64

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Converts a Matplotlib figure to a base64 encoded PNG string."""
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format='png', dpi=max(75, int(cell_size_inch * 20))) # Dynamic DPI
        except Exception as e:
            plt.close(fig) # ensure closure even on savefig error
            logger.error("fig.savefig failed: %s", e, exc_info=True)
            raise VisualizationError(f"Failed to save figure to buffer: {e}") from e
        
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        return img_base64

# --- Example Usage and Mocking (for testing and demonstration) ---
if __name__ == '__main__':
    # This part is for demonstration and basic testing.
    # In a real scenario, `main_module` would be imported and passed.

    class MockMainModule:
        """
        一個 Mock 的 main.py 模組，用於 Analyzer 的測試和演示。
        此 Mock 模組自身不進行複雜分析，僅返回符合格式的隨機或預定分數。
        """
        def __init__(self):
            # Simulating registered modules in main.py
            self.registered_modules = {
                "GM1_Random": self._gm1_random_scorer,
                "GM2_TargetTopLeft": self._gm2_target_top_left,
                "GM3_PV_Bonus": self._gm3_pv_bonus,
                "GM_ErrorModule": self._gm_error_module,
                "GM_WrongShape": self._gm_wrong_shape,
            }
            logger.info("MockMainModule initialized with modules: %s", list(self.registered_modules.keys()))

        def _gm1_random_scorer(self, board: List[List[int]], proposed_value: int) -> np.ndarray:
            rows, cols = len(board), len(board[0]) if board else (0,0)
            logger.debug("[Mock GM1_Random] PV %d. Board %dx%d. Returning random scores.", proposed_value, rows, cols)
            return np.random.rand(rows, cols)

        def _gm2_target_top_left(self, board: List[List[int]], proposed_value: int) -> np.ndarray:
            rows, cols = len(board), len(board[0]) if board else (0,0)
            scores = np.zeros((rows, cols))
            if rows > 0 and cols > 0:
                scores[0, 0] = 0.9 # High score for top-left
            if rows > 1 and cols > 1:
                scores[1, 1] = 0.7
            logger.debug("[Mock GM2_TargetTopLeft] PV %d. Board %dx%d. Prioritizing top-left.", proposed_value, rows, cols)
            return scores

        def _gm3_pv_bonus(self, board: List[List[int]], proposed_value: int) -> np.ndarray:
            rows, cols = len(board), len(board[0]) if board else (0,0)
            scores = np.full((rows, cols), 0.1) # Base score
            if proposed_value == 5: # Bonus for PV 5
                if rows > 0 and cols > 0 and board[rows-1][cols-1] == -1 : # If bottom-right is available
                    scores[rows-1, cols-1] = 1.0
            logger.debug("[Mock GM3_PV_Bonus] PV %d. Board %dx%d. Bonus for PV 5 at bottom-right.", proposed_value, rows, cols)
            return scores

        def _gm_error_module(self, board: List[List[int]], proposed_value: int) -> np.ndarray:
            logger.debug("[Mock GM_ErrorModule] Intentionally raising ModuleExecutionError for PV %d.", proposed_value)
            raise ModuleExecutionError("GM_ErrorModule: Simulated controlled module failure.")

        def _gm_wrong_shape(self, board: List[List[int]], proposed_value: int) -> np.ndarray:
            rows, cols = len(board), len(board[0]) if board else (0,0)
            logger.debug("[Mock GM_WrongShape] Returning scores with wrong shape for PV %d.", proposed_value)
            return np.random.rand(rows + 1, cols) # Incorrect shape

        def get_module_score(self, module_name: str, new_card: List[List[int]], pv: int) -> np.ndarray:
            if module_name not in self.registered_modules:
                # This case should ideally be caught by _get_effective_modules_and_weights
                # but as a safeguard in the mock:
                raise ModuleNotFoundError(f"Mock module '{module_name}' not found during get_module_score call.")
            
            # Simulate a module taking some time
            # import time
            # time.sleep(0.01)

            return self.registered_modules[module_name](new_card, pv)

    logger.info("--- analyzer.py Self-Test/Demonstration START ---")

    # Initialize Analyzer with the Mock
    analyzer_instance = Analyzer(main_module=MockMainModule(), default_top_n=3)

    # Test Case 1: Standard board
    board1 = [
        [1,  -1,  3, -1],
        [-1, 5,  -1, 8],
        [9, 10, 11, -1],
    ]
    pvs1 = [2, 4, 7]
    logger.info("\n--- Test Case 1: Standard Board (3x4), PVs: %s ---", pvs1)
    try:
        results1 = analyzer_instance.analyze_board(
            new_card=board1,
            proposed_values=pvs1,
            active_modules=["GM1_Random", "GM2_TargetTopLeft", "GM3_PV_Bonus"],
            module_weights={"GM2_TargetTopLeft": 2.0, "GM_NonExistent": 0.5} # GM_NonExistent will be ignored
        )
        logger.info("Test Case 1 Processed Params: %s", results1['processed_params'])
        for pv, suggestions in results1['suggestions'].items():
            logger.info(f"  Suggestions for PV {pv}:")
            for sugg in suggestions:
                logger.info(f"    Pos: {sugg['position']}, Score: {sugg['score']:.4f}")
        with open("analyzer_test_case_1.png", "wb") as f:
            f.write(base64.b64decode(results1['visualization']))
        logger.info("Test Case 1 Visualization: analyzer_test_case_1.png")
    except AnalyzerError as e:
        logger.error("Test Case 1 FAILED: %s", e, exc_info=True)


    # Test Case 2: Including modules that will error or return wrong shape
    board2 = [[-1, -1], [-1, 2]]
    pvs2 = [1]
    logger.info("\n--- Test Case 2: Small Board (2x2), PVs: %s, with faulty modules ---", pvs2)
    try:
        results2 = analyzer_instance.analyze_board(
            new_card=board2,
            proposed_values=pvs2,
            active_modules=["GM1_Random", "GM_ErrorModule", "GM_WrongShape", "GM2_TargetTopLeft"],
            top_n=2
        )
        logger.info("Test Case 2 Processed Params: %s", results2['processed_params'])
        logger.info("Test Case 2 results (GM_ErrorModule, GM_WrongShape contributions should be gracefully ignored):")
        for pv, suggestions in results2['suggestions'].items():
            logger.info(f"  Suggestions for PV {pv}: {suggestions}")
        with open("analyzer_test_case_2.png", "wb") as f:
            f.write(base64.b64decode(results2['visualization']))
        logger.info("Test Case 2 Visualization: analyzer_test_case_2.png")
    except AnalyzerError as e:
        logger.error("Test Case 2 FAILED: %s", e, exc_info=True)

    # Test Case 3: Board with no -1 cells
    board3 = [[1, 2], [3, 4]]
    pvs3 = [5]
    logger.info("\n--- Test Case 3: Full Board (2x2, no -1), PVs: %s ---", pvs3)
    try:
        results3 = analyzer_instance.analyze_board(new_card=board3, proposed_values=pvs3)
        logger.info("Test Case 3 Processed Params: %s", results3['processed_params'])
        logger.info("Test Case 3 results (expect no suggestions):")
        for pv, suggestions in results3['suggestions'].items():
            assert not suggestions, f"PV {pv} should have no suggestions on a full board."
            logger.info(f"  Suggestions for PV {pv}: {suggestions}")
        with open("analyzer_test_case_3.png", "wb") as f:
            f.write(base64.b64decode(results3['visualization']))
        logger.info("Test Case 3 Visualization: analyzer_test_case_3.png")
    except AnalyzerError as e:
        logger.error("Test Case 3 FAILED: %s", e, exc_info=True)


    # Test Case 4: No effective modules (all requested are non-existent)
    board4 = [[-1,-1],[-1,-1]]
    pvs4 = [10]
    logger.info("\n--- Test Case 4: No effective modules, PVs: %s ---", pvs4)
    try:
        results4 = analyzer_instance.analyze_board(
            new_card=board4,
            proposed_values=pvs4,
            active_modules=["NonExistent_A", "NonExistent_B"]
        )
        logger.info("Test Case 4 Processed Params: %s", results4['processed_params'])
        logger.info("Test Case 4 results (expect no suggestions from modules, scores should be 0):")
        for pv, suggestions in results4['suggestions'].items():
            # If scores are all 0 for -1 cells, suggestions will have score 0.0
            logger.info(f"  Suggestions for PV {pv}: {suggestions}")
            for sugg in suggestions:
                assert sugg['score'] == 0.0, "Scores should be 0.0 if no modules ran or all returned 0."
        with open("analyzer_test_case_4.png", "wb") as f:
            f.write(base64.b64decode(results4['visualization']))
        logger.info("Test Case 4 Visualization: analyzer_test_case_4.png")
    except AnalyzerError as e:
        logger.error("Test Case 4 FAILED: %s", e, exc_info=True)

    # Test Case 5: Empty board
    board5 = [] # type: List[List[int]]
    pvs5 = [1]
    logger.info("\n--- Test Case 5: Empty board input ---")
    try:
        results5 = analyzer_instance.analyze_board(new_card=board5, proposed_values=pvs5)
        # This should raise InvalidInputError in _validate_inputs
    except InvalidInputError as e:
        logger.info("Test Case 5 Correctly caught InvalidInputError for empty board: %s", e)
    except Exception as e: # Catch any other exception for diagnostics
        logger.error("Test Case 5 FAILED with unexpected error: %s", e, exc_info=True)

    board6 = [[]] # type: List[List[int]]
    pvs6 = [1]
    logger.info("\n--- Test Case 6: Board with empty row input ---")
    try:
        results6 = analyzer_instance.analyze_board(new_card=board6, proposed_values=pvs6)
        # This should also be caught by validation (0 cols)
    except InvalidInputError as e:
        logger.info("Test Case 6 Correctly caught InvalidInputError for board with empty row: %s", e)
    except Exception as e:
        logger.error("Test Case 6 FAILED with unexpected error: %s", e, exc_info=True)


    logger.info("\n--- analyzer.py Self-Test/Demonstration COMPLETE ---")
