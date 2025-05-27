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
import random # 新增: 用於生成臨時 request_id (如果沒有從外部傳入)

import matplotlib
matplotlib.use('Agg') # Ensure Matplotlib works in a headless environment
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np

# --- Configuration ---
# Configure logging
# 建議將此basicConfig移至應用程式入口 (如 main_api.py) 以避免重複配置，
# 但如果 analyzer.py 可能被獨立運行測試，保留也無妨。
# 確保 level 至少是 INFO 或 DEBUG 才能看到新加的日誌。
logging.basicConfig(
    level=logging.INFO, # 修改這裡為 logging.DEBUG 可以看到更詳細的日誌
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

class Analyzer:
    """
    智慧評分系統的核心調度器。
    負責接收分析請求，調用 main.py 中的邏輯模組，融合結果，並返回建議。
    嚴格遵守不干涉分析邏輯、僅做協調與公平融合的原則。
    """

    PV_COLORS = list(mcolors.TABLEAU_COLORS.values()) + \
                list(mcolors.CSS4_COLORS.values())

    def __init__(self, main_module: Any, default_top_n: int = 3):
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
        if not new_card or not isinstance(new_card, list):
            raise InvalidInputError("盤面 (new_card) 不得為空且必須是列表。")
        if not all(isinstance(row, list) for row in new_card):
            raise InvalidInputError("盤面 (new_card) 的每一行必須是列表。")

        rows = len(new_card)
        if rows == 0:
            raise InvalidInputError("盤面 (new_card) 不得為空 (沒有行)。")
        
        cols = len(new_card[0]) if new_card and new_card[0] is not None else 0
        if cols == 0 and rows > 0 :
             if not all(len(row) == 0 for row in new_card):
                  raise InvalidInputError("盤面 (new_card) 的列定義不一致或首行為空但其他行非空。")

        if not all(len(row) == cols for row in new_card):
            raise InvalidInputError("盤面 (new_card) 必須是矩形 (所有行的列數需一致)。")
        if not all(isinstance(val, int) for row in new_card for val in row):
            raise InvalidInputError("盤面 (new_card) 中的所有值必須是整數。")

        has_negative_one = any(val == -1 for row in new_card for val in row)
        if not has_negative_one and rows > 0 and cols > 0 :
            logger.warning("盤面 (new_card) 中沒有 -1 (未開) 的格子。可能無法提供「填入」建議。")

        if not proposed_values or not isinstance(proposed_values, list):
            raise InvalidInputError("候選值 (proposed_values) 必須是非空列表。")
        if not all(isinstance(pv, int) for pv in proposed_values):
            raise InvalidInputError("候選值 (proposed_values) 中的所有值必須是整數。")

        if active_modules is not None:
            if not isinstance(active_modules, list) or \
               not all(isinstance(m, str) for m in active_modules):
                raise InvalidInputError("啟用模組 (active_modules) 若提供，必須是字串列表。")

        if module_weights is not None:
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
        registered_module_names = list(self.main_module.registered_modules.keys())
        effective_module_names: List[str]

        if requested_active_modules is None:
            effective_module_names = registered_module_names
            logger.info("未指定 active_modules，將使用所有已註冊模組: %s", effective_module_names)
        else:
            effective_module_names = []
            for module_name in requested_active_modules:
                if module_name not in registered_module_names:
                    logger.warning("請求的模組 '%s' 未在 main.py 中註冊。將被忽略。", module_name)
                else:
                    effective_module_names.append(module_name)
            if not effective_module_names and requested_active_modules:
                 logger.warning("指定的 active_modules (%s) 均未在 main.py 註冊，無模組可執行。", requested_active_modules)
            elif not effective_module_names:
                 logger.warning("active_modules 列表為空，無模組可執行。")

        final_module_weights: Dict[str, float] = {
            name: 1.0 for name in effective_module_names
        }

        if requested_module_weights:
            for name, weight in requested_module_weights.items():
                if name in final_module_weights:
                    final_module_weights[name] = float(weight)
                else:
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
        cols: int,
        request_id: Optional[str] = "N/A_REQ_ID" # 新增 request_id 參數
    ) -> np.ndarray:
        fused_scores = np.zeros((rows, cols), dtype=float)

        if not module_scores_map:
            logger.warning(f"RequestID: {request_id} - 沒有從模組獲取到任何評分，融合結果將為零矩陣。")
            return fused_scores

        active_module_names_with_scores = list(module_scores_map.keys())
        logger.debug(f"RequestID: {request_id} - 開始融合 {len(active_module_names_with_scores)} 個模組的評分: {active_module_names_with_scores}")

        for module_name, scores_array in module_scores_map.items():
            weight = module_weights_map.get(module_name)
            if weight is None:
                logger.error(f"RequestID: {request_id} - 嚴重內部錯誤：模組 '{module_name}' 在評分融合階段缺少權重。將使用預設值 1.0。")
                weight = 1.0
            
            if not isinstance(scores_array, np.ndarray) or scores_array.shape != (rows, cols):
                logger.error(f"RequestID: {request_id} - 模組 '{module_name}' 的評分格式不符 (期望 {rows}x{cols} np.ndarray, 得到 {type(scores_array)} {scores_array.shape if isinstance(scores_array, np.ndarray) else 'N/A'})。此模組分數將被忽略。")
                continue

            logger.debug(f"RequestID: {request_id} - 融合模組 '{module_name}' 的評分 (權重: {weight:.2f})。")
            fused_scores += scores_array * weight

        min_score_val = np.min(fused_scores)
        max_score_val = np.max(fused_scores)

        if max_score_val == min_score_val:
            normalized_fused_scores = np.zeros_like(fused_scores)
            if min_score_val != 0 :
                 logger.debug(f"RequestID: {request_id} - Fused scores are all identical ({min_score_val:.4f}), normalized to 0.0.")
        else:
            normalized_fused_scores = (fused_scores - min_score_val) / (max_score_val - min_score_val)
            logger.debug(f"RequestID: {request_id} - Fused scores normalized from range [{min_score_val:.4f}, {max_score_val:.4f}] to [0, 1].")

        return normalized_fused_scores

    def _get_top_n_for_pv(
        self,
        fused_scores_board: np.ndarray,
        board_state: List[List[int]],
        top_n: int,
        request_id: Optional[str] = "N/A_REQ_ID" # 新增 request_id 參數
    ) -> List[Dict[str, Any]]:
        suggestions = []
        rows, cols = fused_scores_board.shape
        candidate_cells: List[Tuple[float, int, int]] = []
        has_fillable_cells = False
        for r in range(rows):
            for c in range(cols):
                if board_state[r][c] == -1:
                    has_fillable_cells = True
                    candidate_cells.append((fused_scores_board[r, c], r, c))

        if not has_fillable_cells:
            logger.info(f"RequestID: {request_id} - 盤面上沒有值為 -1 的可填入格子。無法為此 proposed_value 提供建議。")
            return []
        if not candidate_cells: # Should be caught by has_fillable_cells
            logger.info(f"RequestID: {request_id} - 候選格子列表為空 (可能所有格子都不是 -1)。")
            return []

        candidate_cells.sort(key=lambda x: x[0], reverse=True)
        for score, r, c in candidate_cells[:top_n]:
            suggestions.append({
                'position': [r, c],
                'score': round(float(score), 6)
            })
        return suggestions

    def analyze_board(
        self,
        new_card: List[List[int]],
        proposed_values: List[int],
        active_modules: Optional[List[str]] = None,
        module_weights: Optional[Dict[str, float]] = None,
        top_n: Optional[int] = None,
        # 考慮從 API 層傳入 request_id，或者在 Analyzer 實例中管理
        request_id_for_logging: Optional[str] = None 
    ) -> Dict[str, Any]:
        """
        執行棋盤分析的核心方法。
        """
        # 如果外部沒有傳入 request_id_for_logging，則生成一個臨時的
        if request_id_for_logging is None:
            request_id_for_logging = "analyzer-req-" + str(random.randint(10000, 99999))
            logger.info(f"Generated temporary RequestID for logging: {request_id_for_logging}")


        logger.info(
            f"RequestID: {request_id_for_logging} - 接收分析請求: {len(proposed_values) if proposed_values else 0} 個候選值, "
            f"盤面尺寸 {len(new_card)}x{len(new_card[0]) if new_card and new_card[0] else 'empty'} (approx). "
            f"Active modules hint: {str(active_modules) if active_modules else 'ALL'}"
        )

        try:
            rows, cols, validated_pvs, val_active_modules, val_module_weights, final_top_n = \
                self._validate_inputs(new_card, proposed_values, active_modules, module_weights, top_n)
        except InvalidInputError as e:
            logger.error(f"RequestID: {request_id_for_logging} - 輸入參數驗證失敗: {e}", exc_info=True)
            # 根據 API 設計，這裡應該返回一個錯誤響應或重新拋出異常
            # 為了演示完整性，這裡返回一個錯誤結構
            return {
                'error': f"Invalid input: {e}", 
                'suggestions': {}, 
                'visualization': self._generate_error_visualization(0,0, f"Invalid Input: {e}"), # rows,cols 未定義
                'board_dimensions': {'rows':0, 'cols':0}, 
                'processed_params': {'request_id': request_id_for_logging, 'error': True}
            }

        effective_modules, final_weights = self._get_effective_modules_and_weights(
            val_active_modules, val_module_weights
        )

        all_suggestions: Dict[Union[int, str], List[Dict[str, Any]]] = {}
        all_fused_scores_for_pvs: Dict[Union[int, str], np.ndarray] = {}

        if not effective_modules:
            logger.warning(f"RequestID: {request_id_for_logging} - 沒有任何生效的分析模組。分析將產生空建議和零分盤面。")
            for pv in validated_pvs:
                 all_suggestions[pv] = []
                 # 確保 rows, cols 在此處可用
                 all_fused_scores_for_pvs[pv] = np.zeros((rows, cols) if rows > 0 and cols > 0 else (0,0), dtype=float)
        else:
            for pv_idx, pv in enumerate(validated_pvs):
                logger.info(f"RequestID: {request_id_for_logging} - Analyzer: Processing PV {pv} ({pv_idx + 1}/{len(validated_pvs)})")
                module_scores_for_pv: Dict[str, np.ndarray] = {}

                # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
                # vvv 這就是主要加入日誌的地方 vvv
                for module_name in effective_modules:
                    try:
                        logger.debug(f"RequestID: {request_id_for_logging} - Analyzer: Calling module '{module_name}' for PV '{pv}'")
                        raw_scores = self.main_module.get_module_score(module_name, new_card, pv)
                        
                        if raw_scores is None:
                            logger.warning(f"RequestID: {request_id_for_logging} - Module '{module_name}' for PV '{pv}' returned None. Skipping.")
                            continue

                        scores_np = np.array(raw_scores, dtype=float)

                        if scores_np.shape != (rows, cols):
                            logger.error(
                                f"RequestID: {request_id_for_logging} - Module '{module_name}' for PV '{pv}' returned incorrect score shape. "
                                f"Expected {rows}x{cols}, got {scores_np.shape}. Skipping this module's scores."
                            )
                            continue
                        
                        non_zero_count = np.count_nonzero(scores_np)
                        sum_of_scores = np.sum(scores_np)
                        if scores_np.size > 0:
                            min_score_val = np.min(scores_np)
                            max_score_val = np.max(scores_np)
                            mean_score_val = np.mean(scores_np)
                        else:
                            min_score_val, max_score_val, mean_score_val = 0.0, 0.0, 0.0
                            logger.warning(f"RequestID: {request_id_for_logging} - Module '{module_name}' for PV '{pv}' returned an empty scores_np after shape check (unexpected).")

                        logger.info(
                            f"RequestID: {request_id_for_logging} - PV: {pv} - Module: [{module_name}] - "
                            f"Raw scores stats: Shape={scores_np.shape}, Non-zero={non_zero_count}, "
                            f"Sum={sum_of_scores:.4f}, Min={min_score_val:.4f}, Max={max_score_val:.4f}, Mean={mean_score_val:.4f}"
                        )

                        if rows <= 5 and cols <= 5: # 盤面較小時打印完整分數
                           logger.debug(f"RequestID: {request_id_for_logging} - PV: {pv} - Module: [{module_name}] - Raw scores board:\n{scores_np}")
                        elif scores_np.size > 0: # 盤面較大時打印片段
                           logger.debug(f"RequestID: {request_id_for_logging} - PV: {pv} - Module: [{module_name}] - Raw scores board (first 3x3 snippet if available):\n{scores_np[:min(3,rows),:min(3,cols)]}")

                        module_scores_for_pv[module_name] = scores_np
                        logger.debug(f"RequestID: {request_id_for_logging} - Analyzer: Successfully got scores from module '{module_name}' for PV '{pv}'")

                    except Exception as e_module:
                        logger.error(
                            f"RequestID: {request_id_for_logging} - Analyzer: Error calling or processing scores from module '{module_name}' for PV '{pv}': {e_module}. "
                            "This module's scores will be skipped.",
                            exc_info=True
                        )
                # ^^^ 日誌加入結束 ^^^
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                if not module_scores_for_pv:
                    logger.warning(f"RequestID: {request_id_for_logging} - PV: {pv} - No valid scores obtained from any module.")
                    fused_scores_pv = np.zeros((rows, cols), dtype=float)
                else:
                    fused_scores_pv = self._fuse_scores(module_scores_for_pv, final_weights, rows, cols, request_id=request_id_for_logging)

                all_fused_scores_for_pvs[pv] = fused_scores_pv
                all_suggestions[pv] = self._get_top_n_for_pv(fused_scores_pv, new_card, final_top_n, request_id=request_id_for_logging)
                logger.info(f"RequestID: {request_id_for_logging} - PV: {pv} - Found {len(all_suggestions[pv])} suggestions (Top-{final_top_n}).")

        try:
            visualization_b64 = self._generate_visualization(
                new_card,
                validated_pvs,
                all_suggestions,
                all_fused_scores_for_pvs,
                final_top_n,
                request_id=request_id_for_logging # 新增 request_id 參數
            )
        except Exception as e_viz:
            logger.error(f"RequestID: {request_id_for_logging} - 生成視覺化圖像時發生嚴重錯誤: {e_viz}", exc_info=True)
            visualization_b64 = self._generate_error_visualization(rows if 'rows' in locals() else 0, cols if 'cols' in locals() else 0, f"視覺化失敗: {type(e_viz).__name__}")


        return {
            'suggestions': all_suggestions,
            'visualization': visualization_b64,
            'board_dimensions': {'rows': rows if 'rows' in locals() else 0, 'cols': cols if 'cols' in locals() else 0},
            'processed_params': {
                'requested_top_n': top_n if top_n is not None else f"default ({self.default_top_n})",
                'actual_top_n': final_top_n if 'final_top_n' in locals() else self.default_top_n, # defensive
                'requested_active_modules': val_active_modules if 'val_active_modules' in locals() else "N/A",
                'effective_active_modules': effective_modules if 'effective_modules' in locals() else [],
                'requested_module_weights': val_module_weights if 'val_module_weights' in locals() else "N/A",
                'final_module_weights': final_weights if 'final_weights' in locals() else {},
                'request_id': request_id_for_logging
            }
        }

    def _generate_error_visualization(self, rows: int, cols: int, error_message: str) -> str:
        try:
            fig_width = max(cols * 0.5 if cols > 0 else 1, 5)
            fig_height = max(rows * 0.5 if rows > 0 else 1, 3)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.text(0.5, 0.5, f"錯誤:\n{error_message}",
                    ha='center', va='center', fontsize=10, color='red', wrap=True)
            ax.axis('off')
            img_base64 = self._fig_to_base64(fig) # Use self._fig_to_base64
            plt.close(fig)
            return img_base64
        except Exception as e:
            logger.error("生成錯誤提示圖像本身也失敗了: %s", e, exc_info=True) # Add exc_info
            return "Error generating error visualization."


    def _generate_visualization(
        self,
        board_state: List[List[int]],
        proposed_values: List[int],
        all_suggestions: Dict[Union[int, str], List[Dict[str, Any]]],
        all_fused_scores_for_pvs: Dict[Union[int, str], np.ndarray],
        top_n_suggestion_count: int,
        request_id: Optional[str] = "N/A_REQ_ID" # 新增 request_id 參數
    ) -> str:
        logger.debug(f"RequestID: {request_id} - Generating visualization...")
        rows = len(board_state)
        cols = len(board_state[0]) if rows > 0 else 0

        if rows == 0 or cols == 0:
            logger.warning(f"RequestID: {request_id} - 無法生成視覺化：盤面為空。")
            return self._generate_error_visualization(0,0, "盤面為空") # Pass 0,0 for rows,cols

        cell_size_inch = max(0.5, min(1.0, 10.0 / max(rows, cols, 1)))
        fig_width = max(cols * cell_size_inch, 6)
        fig_height = max(rows * cell_size_inch, 4)
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

        heatmap_data = np.full((rows, cols), np.nan)
        first_pv_for_heatmap = None
        if proposed_values and proposed_values[0] in all_fused_scores_for_pvs:
            first_pv_for_heatmap = proposed_values[0]
            scores_for_first_pv = all_fused_scores_for_pvs[first_pv_for_heatmap]
            for r_idx in range(rows):
                for c_idx in range(cols):
                    if board_state[r_idx][c_idx] == -1:
                         heatmap_data[r_idx, c_idx] = scores_for_first_pv[r_idx, c_idx]
            
            if not np.all(np.isnan(heatmap_data)):
                cmap = plt.cm.viridis
                cmap.set_bad(color='white', alpha=0)
                ax.imshow(heatmap_data, cmap=cmap, alpha=0.6, aspect='auto', vmin=0, vmax=1)

        suggestion_texts_on_cells: Dict[Tuple[int, int], List[str]] = {}
        cell_highlights: List[Dict[str,Any]] = []

        for pv_idx, pv in enumerate(proposed_values):
            pv_color = self.PV_COLORS[pv_idx % len(self.PV_COLORS)]
            if pv in all_suggestions:
                top_n_to_display_on_graph = min(top_n_suggestion_count, 3)
                for rank_idx, suggestion in enumerate(all_suggestions[pv][:top_n_to_display_on_graph]):
                    r, c = suggestion['position']
                    rank = rank_idx + 1
                    text_for_cell = f"{pv}(R{rank})"
                    if (r,c) not in suggestion_texts_on_cells:
                        suggestion_texts_on_cells[(r,c)] = []
                    suggestion_texts_on_cells[(r,c)].append(text_for_cell)
                    rect_line_width = 2.0 if rank == 1 else (1.5 if rank == 2 else 1.0)
                    cell_highlights.append({
                        'coords': (c - 0.5, r - 0.5), 'width': 1, 'height': 1,
                        'linewidth': rect_line_width, 'edgecolor': pv_color,
                        'facecolor': mcolors.to_rgba(pv_color, alpha=0.10 if rank == 1 else 0.05)
                    })
        
        for highlight in cell_highlights:
            rect = patches.Rectangle(**{k:v for k,v in highlight.items() if k!= 'coords'},
                                     xy=highlight['coords'])
            ax.add_patch(rect)

        font_size = max(6, cell_size_inch * 10)
        for r_idx in range(rows):
            for c_idx in range(cols):
                cell_value = board_state[r_idx][c_idx]
                current_cell_texts = []
                if cell_value != -1:
                    current_cell_texts.append(str(cell_value))
                else:
                    if (r_idx, c_idx) in suggestion_texts_on_cells:
                        current_cell_texts.extend(suggestion_texts_on_cells[(r_idx, c_idx)])
                    else:
                        current_cell_texts.append("●")
                final_display_text = "\n".join(current_cell_texts)
                num_lines = final_display_text.count('\n') + 1
                dynamic_font_size = font_size / num_lines if num_lines > 1 else font_size
                dynamic_font_size = max(4, dynamic_font_size * min(1, (cell_size_inch*10) / (len(final_display_text)/max(num_lines,1) +1) ))
                ax.text(c_idx, r_idx, final_display_text,
                        ha='center', va='center', fontsize=dynamic_font_size, color='black',
                        wrap=True)

        pv_str = ", ".join(map(str, proposed_values)) if proposed_values else "無"
        title_str = f"盤面分析 ({rows}x{cols}) - 候選值: [{pv_str}]"
        if not any(sugg_list for sugg_list in all_suggestions.values()):
            title_str += "\n(盤面無 -1 格或模組未提供有效建議)"
        
        plt.title(title_str, fontsize=max(8, cell_size_inch * 14), pad=20)

        legend_elements = []
        if proposed_values and any(s for pv_suggs in all_suggestions.values() for s in pv_suggs):
            added_pvs_to_legend = set()
            for pv_idx, pv in enumerate(proposed_values):
                 if pv not in added_pvs_to_legend and any(s for s in all_suggestions.get(pv,[])):
                    color = self.PV_COLORS[pv_idx % len(self.PV_COLORS)]
                    legend_elements.append(patches.Patch(facecolor=color, edgecolor=color, label=f'候選值 {pv} 建議'))
                    added_pvs_to_legend.add(pv)
            if legend_elements:
                ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.03, 0.5),
                          fontsize=max(7, cell_size_inch * 10), title="圖例")

        plt.tight_layout(rect=[0, 0, 0.9 if legend_elements else 1, 1])

        img_base64 = self._fig_to_base64(fig) # Use self._fig_to_base64
        plt.close(fig)
        return img_base64

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        buf = io.BytesIO()
        try:
            # Attempt to get cell_size_inch if available, otherwise use a default for dpi
            # This is a bit hacky, ideally cell_size_inch would be available or dpi fixed
            current_cell_size_inch = getattr(self, '_current_cell_size_inch_for_dpi', 0.75) # Default if not set
            fig.savefig(buf, format='png', dpi=max(75, int(current_cell_size_inch * 20)))
        except Exception as e:
            plt.close(fig)
            logger.error("fig.savefig failed: %s", e, exc_info=True) # Add exc_info
            raise VisualizationError(f"Failed to save figure to buffer: {e}") from e
        
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        return img_base64

# --- Example Usage and Mocking (for testing and demonstration) ---
# (The if __name__ == '__main__': block remains the same as your original, no changes needed there)
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
            rows, cols = len(board), len(board[0]) if board and board[0] else (0,0) # defensive
            logger.debug("[Mock GM1_Random] PV %d. Board %dx%d. Returning random scores.", proposed_value, rows, cols)
            if rows == 0 or cols == 0: return np.array([[]]) # Handle empty board case for mock
            return np.random.rand(rows, cols)

        def _gm2_target_top_left(self, board: List[List[int]], proposed_value: int) -> np.ndarray:
            rows, cols = len(board), len(board[0]) if board and board[0] else (0,0)
            scores = np.zeros((rows, cols))
            if rows > 0 and cols > 0:
                scores[0, 0] = 0.9 # High score for top-left
            if rows > 1 and cols > 1:
                scores[1, 1] = 0.7
            logger.debug("[Mock GM2_TargetTopLeft] PV %d. Board %dx%d. Prioritizing top-left.", proposed_value, rows, cols)
            return scores

        def _gm3_pv_bonus(self, board: List[List[int]], proposed_value: int) -> np.ndarray:
            rows, cols = len(board), len(board[0]) if board and board[0] else (0,0)
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
            rows, cols = len(board), len(board[0]) if board and board[0] else (0,0)
            logger.debug("[Mock GM_WrongShape] Returning scores with wrong shape for PV %d.", proposed_value)
            if rows == 0 or cols == 0: return np.array([[]]) # Handle empty board for mock
            return np.random.rand(rows + 1, cols) # Incorrect shape

        def get_module_score(self, module_name: str, new_card: List[List[int]], pv: int) -> np.ndarray:
            if module_name not in self.registered_modules:
                raise ModuleNotFoundError(f"Mock module '{module_name}' not found during get_module_score call.")
            
            return self.registered_modules[module_name](new_card, pv)

    logger.info("--- analyzer.py Self-Test/Demonstration START ---")
    analyzer_instance = Analyzer(main_module=MockMainModule(), default_top_n=3)
    # Test Case 1
    board1 = [[1, -1, 3, -1], [-1, 5, -1, 8], [9, 10, 11, -1]]
    pvs1 = [2, 4, 7]
    logger.info("\n--- Test Case 1: Standard Board (3x4), PVs: %s ---", pvs1)
    try:
        results1 = analyzer_instance.analyze_board(
            new_card=board1, proposed_values=pvs1,
            active_modules=["GM1_Random", "GM2_TargetTopLeft", "GM3_PV_Bonus"],
            module_weights={"GM2_TargetTopLeft": 2.0, "GM_NonExistent": 0.5},
            request_id_for_logging="test_req_001" # Example request_id
        )
        logger.info("Test Case 1 Processed Params: %s", results1['processed_params'])
        for pv, suggestions in results1['suggestions'].items():
            logger.info(f"  Suggestions for PV {pv}:")
            for sugg in suggestions: logger.info(f"    Pos: {sugg['position']}, Score: {sugg['score']:.4f}")
        if results1.get('visualization') and isinstance(results1['visualization'], str) and results1['visualization'].startswith('Error') == False :
            with open("analyzer_test_case_1.png", "wb") as f: f.write(base64.b64decode(results1['visualization']))
            logger.info("Test Case 1 Visualization: analyzer_test_case_1.png")
        else: logger.warning("Test Case 1 Visualization data missing or indicates error.")
    except AnalyzerError as e: logger.error("Test Case 1 FAILED: %s", e, exc_info=True)
    # (Rest of the test cases remain the same, ensure to pass request_id_for_logging if you added it to analyze_board signature)
    # ... (Your original test cases 2, 3, 4, 5, 6 can follow here) ...
    board2 = [[-1, -1], [-1, 2]]
    pvs2 = [1]
    logger.info("\n--- Test Case 2: Small Board (2x2), PVs: %s, with faulty modules ---", pvs2)
    try:
        results2 = analyzer_instance.analyze_board(
            new_card=board2, proposed_values=pvs2,
            active_modules=["GM1_Random", "GM_ErrorModule", "GM_WrongShape", "GM2_TargetTopLeft"],
            top_n=2, request_id_for_logging="test_req_002"
        )
        logger.info("Test Case 2 Processed Params: %s", results2['processed_params'])
        logger.info("Test Case 2 results (GM_ErrorModule, GM_WrongShape contributions should be gracefully ignored):")
        for pv, suggestions in results2['suggestions'].items(): logger.info(f"  Suggestions for PV {pv}: {suggestions}")
        if results2.get('visualization') and isinstance(results2['visualization'], str) and results2['visualization'].startswith('Error') == False :
            with open("analyzer_test_case_2.png", "wb") as f: f.write(base64.b64decode(results2['visualization']))
            logger.info("Test Case 2 Visualization: analyzer_test_case_2.png")
        else: logger.warning("Test Case 2 Visualization data missing or indicates error.")
    except AnalyzerError as e: logger.error("Test Case 2 FAILED: %s", e, exc_info=True)

    board3 = [[1, 2], [3, 4]]
    pvs3 = [5]
    logger.info("\n--- Test Case 3: Full Board (2x2, no -1), PVs: %s ---", pvs3)
    try:
        results3 = analyzer_instance.analyze_board(new_card=board3, proposed_values=pvs3, request_id_for_logging="test_req_003")
        logger.info("Test Case 3 Processed Params: %s", results3['processed_params'])
        logger.info("Test Case 3 results (expect no suggestions):")
        for pv, suggestions in results3['suggestions'].items():
            assert not suggestions, f"PV {pv} should have no suggestions on a full board."
            logger.info(f"  Suggestions for PV {pv}: {suggestions}")
        if results3.get('visualization') and isinstance(results3['visualization'], str) and results3['visualization'].startswith('Error') == False:
            with open("analyzer_test_case_3.png", "wb") as f: f.write(base64.b64decode(results3['visualization']))
            logger.info("Test Case 3 Visualization: analyzer_test_case_3.png")
        else: logger.warning("Test Case 3 Visualization data missing or indicates error.")
    except AnalyzerError as e: logger.error("Test Case 3 FAILED: %s", e, exc_info=True)

    board4 = [[-1,-1],[-1,-1]]
    pvs4 = [10]
    logger.info("\n--- Test Case 4: No effective modules, PVs: %s ---", pvs4)
    try:
        results4 = analyzer_instance.analyze_board(
            new_card=board4, proposed_values=pvs4,
            active_modules=["NonExistent_A", "NonExistent_B"], request_id_for_logging="test_req_004"
        )
        logger.info("Test Case 4 Processed Params: %s", results4['processed_params'])
        logger.info("Test Case 4 results (expect no suggestions from modules, scores should be 0):")
        for pv, suggestions in results4['suggestions'].items():
            logger.info(f"  Suggestions for PV {pv}: {suggestions}")
            for sugg in suggestions: assert sugg['score'] == 0.0, "Scores should be 0.0 if no modules ran or all returned 0."
        if results4.get('visualization') and isinstance(results4['visualization'], str) and results4['visualization'].startswith('Error') == False:
            with open("analyzer_test_case_4.png", "wb") as f: f.write(base64.b64decode(results4['visualization']))
            logger.info("Test Case 4 Visualization: analyzer_test_case_4.png")
        else: logger.warning("Test Case 4 Visualization data missing or indicates error.")
    except AnalyzerError as e: logger.error("Test Case 4 FAILED: %s", e, exc_info=True)

    board5: List[List[int]] = []
    pvs5 = [1]
    logger.info("\n--- Test Case 5: Empty board input ---")
    try:
        results5 = analyzer_instance.analyze_board(new_card=board5, proposed_values=pvs5, request_id_for_logging="test_req_005")
    except InvalidInputError as e: logger.info("Test Case 5 Correctly caught InvalidInputError for empty board: %s", e)
    except Exception as e: logger.error("Test Case 5 FAILED with unexpected error: %s", e, exc_info=True)

    board6: List[List[int]] = [[]]
    pvs6 = [1]
    logger.info("\n--- Test Case 6: Board with empty row input ---")
    try:
        results6 = analyzer_instance.analyze_board(new_card=board6, proposed_values=pvs6, request_id_for_logging="test_req_006")
    except InvalidInputError as e: logger.info("Test Case 6 Correctly caught InvalidInputError for board with empty row: %s", e)
    except Exception as e: logger.error("Test Case 6 FAILED with unexpected error: %s", e, exc_info=True)

    logger.info("\n--- analyzer.py Self-Test/Demonstration COMPLETE ---")