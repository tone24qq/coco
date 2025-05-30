# analyzer.py
# pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements

"""
核心調度器(analyzer.py): 專責處理來自API的分析請求。

一句話總結:
「所有棋盤邏輯、分析規則、分數運算都只許在 main.py/模組寫,
analyzer.py 永遠只做協調與公平融合,任何越界皆屬大忌。」

本檔案完全遵循高內聚低耦合原則,禁止硬編寫任何業務邏輯、規則、演算法、
模組分數計算與棋盤分析。所有分析行為均應由 main.py 之官方註冊模組執行。
"""

import base64
import io
import logging
import brain
from typing import Any, Callable, Protocol, runtime_checkable
import random # 新增:用於生成臨時 request_id(如果沒有從外部傳入)

import matplotlib
matplotlib.use('Agg') # Ensure Matplotlib works in a headless environment
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np

# --- Configuration ---
# Configure logging
# 建議將此basicConfig移至應用程式入口(如main_api.py) 以避免重複配置,
# 但如果 analyzer.py 可能被獨立運行測試,保留也無妨。
# 確保 level 至少是INFO或DEBUG才能看到新加的日誌。
logging.basicConfig(
    level=logging.INFO, # 修改這裡為 logging.DEBUG 可以看到更詳細的日誌
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Type Definitions for main_module Protocol ---
@runtime_checkable
class MainModuleProtocol(Protocol):
    """
    Protocol defining the expected interface for the main_module.
    """
    registered_modules: dict[str, Callable[[list[list[int]], int], np.ndarray]]

    def get_module_score(self, module_name: str, new_card: list[list[int]], pv: int) -> np.ndarray:
        ...

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
    負責接收分析請求,調用 main.py中的邏輯模組,融合結果,並返回建議。
    嚴格遵守不干涉分析邏輯、僅做協調與公平融合的原則。
    """
    PV_COLORS: list[str] = list(mcolors.TABLEAU_COLORS.values()) + \
                           list(mcolors.CSS4_COLORS.values())
    _current_cell_size_inch_for_dpi: float # For _fig_to_base64, managed internally

    def __init__(self, main_module: MainModuleProtocol, default_top_n: int = 3):
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
        self._current_cell_size_inch_for_dpi = 0.75 # Default, can be updated before visualization
        logger.info("Analyzer initialized with default_top_n=%d. Registered modules from main_module: %s",
                    self.default_top_n, list(main_module.registered_modules.keys()))

    def _validate_inputs(
        self,
        new_card: list[list[int]],
        proposed_values: list[int],
        active_modules: list[str] | None,
        module_weights: dict[str, float] | None,
        top_n: int | None
    ) -> tuple[int, int, list[int], list[str] | None, dict[str, float] | None, int]:
        if not new_card or not isinstance(new_card, list):
            raise InvalidInputError("盤面 (new_card) 不得為空且必須是列表。")
        if not all(isinstance(row, list) for row in new_card):
            raise InvalidInputError("盤面 (new_card) 的每一行必須是列表。")

        rows = len(new_card)
        if rows == 0:
            # If rows is 0, it implies an empty list was passed, which is caught by "not new_card"
            # However, if new_card = [[]], rows = 1. This case means no actual cells.
            # Let's adjust to check if there's any data effectively.
             if not any(new_card): # check if new_card is not just [[]] or [[], []]
                raise InvalidInputError("盤面 (new_card) 不得為空 (沒有實際資料列)。")


        cols = len(new_card[0]) if rows > 0 and isinstance(new_card[0], list) else 0

        if rows > 0 and cols == 0: # e.g. new_card = [[]] or [[], []]
            if not all(len(row) == 0 for row in new_card): # This should not happen if cols is 0
                 raise InvalidInputError("盤面 (new_card) 的列定義不一致或首行為空但其他行非空。")
        elif rows > 0 and not all(len(row) == cols for row in new_card):
             raise InvalidInputError("盤面 (new_card) 必須是矩形 (所有行的列數需一致)。")


        if not all(isinstance(val, int) for row in new_card for val in row):
            raise InvalidInputError("盤面 (new_card) 中的所有值必須是整數。")

        has_negative_one = any(val == -1 for row in new_card for val in row)
        if not has_negative_one and rows > 0 and cols > 0: # Only log if board has cells
            logger.warning("盤面 (new_card) 中沒有-1 (未開) 的格子。可能無法提供「填入」建議。")

        if not proposed_values or not isinstance(proposed_values, list):
            raise InvalidInputError("候選值 (proposed_values) 必須是非空列表。")
        if not all(isinstance(pv, int) for pv in proposed_values):
            raise InvalidInputError("候選值 (proposed_values) 中的所有值必須是整數。")

        if active_modules is not None:
            if not isinstance(active_modules, list) or \
               not all(isinstance(m, str) for m in active_modules):
                raise InvalidInputError("啟用模組 (active_modules) 若提供, 必須是字串列表。")

        if module_weights is not None:
            if not isinstance(module_weights, dict) or \
               not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in module_weights.items()):
                raise InvalidInputError("模組權重 (module_weights) 若提供, 必須是 {str: float/int} 格式的字典。")

        final_top_n = top_n if top_n is not None else self.default_top_n
        if not isinstance(final_top_n, int) or final_top_n <= 0:
            raise InvalidInputError(f"Top-N 數量 ({final_top_n}) 必須是正整數。")

        logger.debug("Input validation successful. Rows: %d, Cols: %d, Top_N: %d", rows, cols, final_top_n)
        return rows, cols, proposed_values, active_modules, module_weights, final_top_n

    def _get_effective_modules_and_weights(
        self,
        requested_active_modules: list[str] | None,
        requested_module_weights: dict[str, float] | None
    ) -> tuple[list[str], dict[str, float]]:
        registered_module_names = list(self.main_module.registered_modules.keys())
        effective_module_names: list[str]

        if requested_active_modules is None:
            effective_module_names = registered_module_names
            logger.info("未指定 active_modules, 將使用所有已註冊模組: %s", effective_module_names)
        else:
            effective_module_names = []
            for module_name in requested_active_modules:
                if module_name not in registered_module_names:
                    logger.warning("請求的模組 '%s' 未在 main_module 中註冊。將被忽略。", module_name)
                else:
                    effective_module_names.append(module_name)
            if not effective_module_names and requested_active_modules: # All requested were invalid
                logger.warning("指定的 active_modules (%s) 均未在 main_module 註冊, 無模組可執行。", requested_active_modules)
            elif not effective_module_names: # requested_active_modules was empty list
                 logger.warning("active_modules 列表為空, 無模組可執行。")


        final_module_weights: dict[str, float] = {
            name: 1.0 for name in effective_module_names
        }
        if requested_module_weights:
            for name, weight in requested_module_weights.items():
                if name in final_module_weights:
                    final_module_weights[name] = float(weight)
                else:
                    logger.warning(
                        "權重配置中的模組 '%s' 未在生效模組列表 (%s) 中, 其權重將被忽略。",
                        name, effective_module_names
                    )
        logger.info("生效模組 (Effective Modules): %s", effective_module_names)
        logger.info("最終模組權重 (Final Module Weights): %s", final_module_weights)
        return effective_module_names, final_module_weights

    def _fuse_scores(
        self,
        module_scores_map: dict[str, np.ndarray],
        module_weights_map: dict[str, float],
        rows: int,
        cols: int,
        request_id: str | None = "N/A_REQ_ID"
    ) -> np.ndarray:
        fused_scores = np.zeros((rows, cols), dtype=float)
        if not module_scores_map:
            logger.warning(f"RequestID: {request_id} - 沒有從模組獲取到任何評分, 融合結果將為零矩陣。")
            return fused_scores

        active_module_names_with_scores = list(module_scores_map.keys())
        logger.debug(f"RequestID: {request_id} - 開始融合 {len(active_module_names_with_scores)} 個模組的評分: {active_module_names_with_scores}")

        for module_name, scores_array in module_scores_map.items():
            weight = module_weights_map.get(module_name)
            if weight is None:
                logger.error(f"RequestID: {request_id} - 嚴重內部錯誤: 模組 '{module_name}' 在評分融合階段缺少權重。將使用預設值 1.0。")
                weight = 1.0

            if not isinstance(scores_array, np.ndarray) or scores_array.shape != (rows, cols):
                logger.error(
                    f"RequestID: {request_id} - 模組 '{module_name}' 的評分格式不符 (期望 {rows}x{cols} np.ndarray, "
                    f"得到 {type(scores_array)} {scores_array.shape if isinstance(scores_array, np.ndarray) else 'N/A'})。此模組分數將被忽略。"
                )
                continue
            logger.debug(f"RequestID: {request_id} - 融合模組 '{module_name}' 的評分 (權重: {weight:.2f}).")
            fused_scores += scores_array * weight

        if fused_scores.size == 0: # Should not happen if rows, cols > 0
            return fused_scores

        min_score_val = np.min(fused_scores)
        max_score_val = np.max(fused_scores)

        if max_score_val == min_score_val:
            normalized_fused_scores = np.zeros_like(fused_scores)
            if min_score_val != 0: # Avoid log if all zeros already
                logger.debug(f"RequestID: {request_id} - Fused scores are all identical ({min_score_val:.4f}), normalized to 0.0.")
        else:
            normalized_fused_scores = (fused_scores - min_score_val) / (max_score_val - min_score_val)
        
        logger.debug(f"RequestID: {request_id} - Fused scores normalized from range [{min_score_val:.4f}, {max_score_val:.4f}] to [0, 1] (approx, if not all identical).")
        return normalized_fused_scores

    def _get_top_n_for_pv(
        self,
        fused_scores_board: np.ndarray,
        board_state: list[list[int]],
        top_n: int,
        request_id: str | None = "N/A_REQ_ID"
    ) -> list[dict[str, Any]]:
        suggestions: list[dict[str, Any]] = []
        if fused_scores_board.size == 0: # Empty board
            logger.info(f"RequestID: {request_id} - Fused scores board is empty. 無法提供建議。")
            return []

        rows, cols = fused_scores_board.shape
        candidate_cells: list[tuple[float, int, int]] = []
        has_fillable_cells = False

        for r_idx in range(rows):
            for c_idx in range(cols):
                if board_state[r_idx][c_idx] == -1:
                    has_fillable_cells = True
                    candidate_cells.append((fused_scores_board[r_idx, c_idx], r_idx, c_idx))

        if not has_fillable_cells:
            logger.info(f"RequestID: {request_id} - 盤面上沒有值為 -1 的可填入格子。無法為此 proposed_value 提供建議。")
            return [] # Return empty list

        if not candidate_cells: # Should be covered by has_fillable_cells check above
            logger.info(f"RequestID: {request_id} - 候選格子列表為空 (可能所有格子都不是 -1)。")
            return [] # Return empty list

        candidate_cells.sort(key=lambda x: x[0], reverse=True)
        for score, r_pos, c_pos in candidate_cells[:top_n]:
            suggestions.append({
                'position': [r_pos, c_pos],
                'score': round(float(score), 6)
            })
        return suggestions

    def analyze_board(
        self,
        new_card: list[list[int]],
        proposed_values: list[int],
        active_modules: list[str] | None = None,
        module_weights: dict[str, float] | None = None,
        top_n: int | None = None,
        request_id_for_logging: str | None = None
    ) -> dict[str, Any]:
        """執行棋盤分析的核心方法。"""
        req_id = request_id_for_logging
        if req_id is None:
            req_id = f"analyzer-req-{random.randint(10000, 99999)}"
            logger.info(f"Generated temporary RequestID for logging: {req_id}")

        logger.info(
            f"RequestID: {req_id} - 接收分析請求: {len(proposed_values) if proposed_values else 0} 個候選值, "
            f"盤面尺寸 {len(new_card)}x{len(new_card[0]) if new_card and new_card[0] else 'empty'} (approx). "
            f"Active modules hint: {str(active_modules) if active_modules else 'ALL'}"
        )
        
        current_rows, current_cols = (len(new_card), len(new_card[0])) if new_card and new_card[0] else (0,0)

        try:
            rows, cols, validated_pvs, val_active_modules, val_module_weights, final_top_n = \
                self._validate_inputs(new_card, proposed_values, active_modules, module_weights, top_n)
            current_rows, current_cols = rows, cols # Update after validation
        except InvalidInputError as e:
            logger.error(f"RequestID: {req_id} - 輸入參數驗證失敗: {e}", exc_info=True)
            return {
                'error': f"Invalid input: {e}",
                'suggestions': {},
                'visualization': self._generate_error_visualization(current_rows, current_cols, f"Invalid Input: {e}"),
                'board_dimensions': {'rows': current_rows, 'cols': current_cols},
                'processed_params': {'request_id': req_id, 'error': True}
            }

        effective_modules, final_weights = self._get_effective_modules_and_weights(
            val_active_modules, val_module_weights
        )

        all_suggestions: dict[int | str, list[dict[str, Any]]] = {}
        all_fused_scores_for_pvs: dict[int | str, np.ndarray] = {}

        if not effective_modules:
            logger.warning(f"RequestID: {req_id} - 沒有任何生效的分析模組。分析將產生空建議和零分盤面。")
            for pv_val in validated_pvs:
                all_suggestions[pv_val] = []
                all_fused_scores_for_pvs[pv_val] = np.zeros((rows, cols) if rows > 0 and cols > 0 else (0,0), dtype=float)
        else:
            for pv_idx, pv_val in enumerate(validated_pvs):
                logger.info(f"RequestID: {req_id} - Analyzer: Processing PV {pv_val} ({pv_idx + 1}/{len(validated_pvs)})")
                module_scores_for_pv: dict[str, np.ndarray] = {}

                for module_name in effective_modules:
                    try:
                        logger.debug(f"RequestID: {req_id} - Analyzer: Calling module '{module_name}' for PV '{pv_val}'")
                        # Assuming new_card is already validated list[list[int]]
                        raw_scores = self.main_module.get_module_score(module_name, new_card, pv_val)

                        if raw_scores is None:
                            logger.warning(f"RequestID: {req_id} - Module '{module_name}' for PV '{pv_val}' returned None. Skipping.")
                            continue

                        scores_np = np.array(raw_scores, dtype=float)
                        if scores_np.shape != (rows, cols):
                            logger.error(
                                f"RequestID: {req_id} - Module '{module_name}' for PV '{pv_val}' returned incorrect score shape. "
                                f"Expected {rows}x{cols}, got {scores_np.shape}. Skipping this module's scores."
                            )
                            continue
                        
                        non_zero_count = np.count_nonzero(scores_np)
                        sum_of_scores = np.sum(scores_np)
                        min_val_s, max_val_s, mean_val_s = 0.0, 0.0, 0.0
                        if scores_np.size > 0:
                           min_val_s = np.min(scores_np)
                           max_val_s = np.max(scores_np)
                           mean_val_s = np.mean(scores_np)
                        else:
                            logger.warning(f"RequestID: {req_id} - Module '{module_name}' for PV '{pv_val}' returned an empty scores_np after shape check (unexpected).")


                        logger.info(
                            f"RequestID: {req_id} - PV: {pv_val} - Module: [{module_name}] - "
                            f"Raw scores stats: Shape={scores_np.shape}, Non-zero={non_zero_count}, "
                            f"Sum={sum_of_scores:.4f}, Min={min_val_s:.4f}, Max={max_val_s:.4f}, Mean={mean_val_s:.4f}"
                        )
                        if rows <= 5 and cols <= 5: # 盤面較小時打印完整分數
                            logger.debug(f"RequestID: {req_id} - PV: {pv_val} - Module: [{module_name}] - Raw scores board:\n{scores_np}")
                        elif scores_np.size > 0: # 盤面較大時打印片段
                            logger.debug(f"RequestID: {req_id} - PV: {pv_val} - Module: [{module_name}] - Raw scores board (first {min(3,rows)}x{min(3,cols)} snippet if available):\n{scores_np[:min(3,rows), :min(3,cols)]}")

                        module_scores_for_pv[module_name] = scores_np
                        logger.debug(f"RequestID: {req_id} - Analyzer: Successfully got scores from module '{module_name}' for PV '{pv_val}'")
                    except Exception as e_module:
                        logger.error(
                            f"RequestID: {req_id} - Analyzer: Error calling or processing scores from module '{module_name}' for PV '{pv_val}': {e_module}. "
                            "This module's scores will be skipped.",
                            exc_info=True
                        )
                
                if not module_scores_for_pv:
                    logger.warning(f"RequestID: {req_id} - PV: {pv_val} - No valid scores obtained from any module.")
                    fused_scores_pv = np.zeros((rows, cols), dtype=float)
                else:
                    fused_scores_pv = self._fuse_scores(module_scores_for_pv, final_weights, rows, cols, request_id=req_id)
                
                all_fused_scores_for_pvs[pv_val] = fused_scores_pv
                all_suggestions[pv_val] = self._get_top_n_for_pv(fused_scores_pv, new_card, final_top_n, request_id=req_id)
                logger.info(f"RequestID: {req_id} - PV: {pv_val} - Found {len(all_suggestions[pv_val])} suggestions (Top-{final_top_n}).")
        try:
            # Update cell size attribute for potential use in _fig_to_base64
            self._current_cell_size_inch_for_dpi = max(0.5, min(1.0, 10.0 / max(rows, cols, 1)))

            visualization_b64 = self._generate_visualization(
                new_card,
                validated_pvs,
                all_suggestions,
                all_fused_scores_for_pvs,
                final_top_n,
                request_id=req_id
            )
        except Exception as e_viz:
            logger.error(f"RequestID: {req_id} - 生成視覺化圖像時發生嚴重錯誤: {e_viz}", exc_info=True)
            visualization_b64 = self._generate_error_visualization(rows, cols, f"視覺化失敗: {type(e_viz).__name__}")

        return {
            'suggestions': all_suggestions,
            'visualization': visualization_b64,
            'board_dimensions': {'rows': rows, 'cols': cols},
            'processed_params': {
                'requested_top_n': top_n if top_n is not None else f"default ({self.default_top_n})",
                'actual_top_n': final_top_n,
                'requested_active_modules': val_active_modules if val_active_modules is not None else "ALL_REGISTERED",
                'effective_active_modules': effective_modules,
                'requested_module_weights': val_module_weights if val_module_weights is not None else "DEFAULT_ALL_1.0",
                'final_module_weights': final_weights,
                'request_id': req_id
            }
        }

    def _generate_error_visualization(self, rows: int, cols: int, error_message: str) -> str:
        try:
            # Use provided rows/cols, default if they are 0
            fig_width = max(cols * 0.5 if cols > 0 else 1, 5)
            fig_height = max(rows * 0.5 if rows > 0 else 1, 3)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.text(0.5, 0.5, f"錯誤:\n{error_message}",
                    ha='center', va='center', fontsize=10, color='red', wrap=True)
            ax.axis('off')
            img_base64 = self._fig_to_base64(fig)
            plt.close(fig)
            return img_base64
        except Exception as e:
            logger.error("生成錯誤提示圖像本身也失敗了: %s", e, exc_info=True)
            return "Error generating error visualization."


    def _generate_visualization(
        self,
        board_state: list[list[int]],
        proposed_values: list[int],
        all_suggestions: dict[int | str, list[dict[str, Any]]],
        all_fused_scores_for_pvs: dict[int | str, np.ndarray],
        top_n_suggestion_count: int,
        request_id: str | None = "N/A_REQ_ID"
    ) -> str:
        logger.debug(f"RequestID: {request_id} - Generating visualization...")
        rows = len(board_state)
        cols = len(board_state[0]) if rows > 0 else 0

        if rows == 0 or cols == 0:
            logger.warning(f"RequestID: {request_id} - 無法生成視覺化: 盤面為空。")
            return self._generate_error_visualization(0, 0, "盤面為空")

        cell_size_inch = max(0.5, min(1.0, 10.0 / max(rows, cols, 1)))
        self._current_cell_size_inch_for_dpi = cell_size_inch # Update for savefig

        fig_width = max(cols * cell_size_inch, 6)
        fig_height = max(rows * cell_size_inch, 4)
        if len(proposed_values) > 3:
            fig_width += 2 # Make space for legend if many PVs

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5) # Inverted y-axis for matrix plot
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.set_xticklabels([str(i+1) for i in np.arange(cols)], fontsize=max(6, cell_size_inch * 10))
        ax.set_yticklabels([str(i+1) for i in np.arange(rows)], fontsize=max(6, cell_size_inch * 10))
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xlabel("列 (Col)", fontsize=max(7, cell_size_inch * 12))
        ax.set_ylabel("行 (Row)", fontsize=max(7, cell_size_inch * 12))
        ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')

        heatmap_data = np.full((rows, cols), np.nan)
        first_pv_for_heatmap: int | str | None = None
        if proposed_values and proposed_values[0] in all_fused_scores_for_pvs:
            first_pv_for_heatmap = proposed_values[0]
            scores_for_first_pv = all_fused_scores_for_pvs[first_pv_for_heatmap]
            if scores_for_first_pv.shape == (rows,cols): # ensure correct shape before assigning
                for r_idx in range(rows):
                    for c_idx in range(cols):
                        if board_state[r_idx][c_idx] == -1:
                            heatmap_data[r_idx, c_idx] = scores_for_first_pv[r_idx, c_idx]
        
        if not np.all(np.isnan(heatmap_data)):
            cmap = plt.cm.viridis.copy() # Use a copy to avoid modifying global cmap
            cmap.set_bad(color='white', alpha=0)
            ax.imshow(heatmap_data, cmap=cmap, alpha=0.6, aspect='auto', vmin=0, vmax=1)

        suggestion_texts_on_cells: dict[tuple[int, int], list[str]] = {}
        cell_highlights: list[dict[str, Any]] = []

        for pv_idx, pv_val in enumerate(proposed_values):
            pv_color = self.PV_COLORS[pv_idx % len(self.PV_COLORS)]
            if pv_val in all_suggestions:
                top_n_to_display_on_graph = min(top_n_suggestion_count, 3)
                for rank_idx, suggestion in enumerate(all_suggestions[pv_val][:top_n_to_display_on_graph]):
                    r_pos, c_pos = suggestion['position']
                    rank = rank_idx + 1
                    text_for_cell = f"{pv_val}(R{rank})"
                    if (r_pos, c_pos) not in suggestion_texts_on_cells:
                        suggestion_texts_on_cells[(r_pos, c_pos)] = []
                    suggestion_texts_on_cells[(r_pos, c_pos)].append(text_for_cell)

                    rect_line_width = 2.0 if rank == 1 else (1.5 if rank == 2 else 1.0)
                    cell_highlights.append({
                        'coords': (c_pos - 0.5, r_pos - 0.5), 'width': 1, 'height': 1,
                        'linewidth': rect_line_width, 'edgecolor': pv_color,
                        'facecolor': mcolors.to_rgba(pv_color, alpha=0.10 if rank == 1 else 0.05)
                    })
        
        for highlight in cell_highlights:
            # rect_params = {k:v for k,v in highlight.items() if k != 'coords'} # Python 3.8+ dict union |
            rect_params = highlight.copy()
            coords = rect_params.pop('coords')
            rect = patches.Rectangle(xy=coords, **rect_params)
            ax.add_patch(rect)

        font_size_base = max(6, cell_size_inch * 10)
        for r_idx in range(rows):
            for c_idx in range(cols):
                cell_val = board_state[r_idx][c_idx]
                current_cell_texts_list: list[str] = []
                if cell_val != -1:
                    current_cell_texts_list.append(str(cell_val))
                else:
                    if (r_idx, c_idx) in suggestion_texts_on_cells:
                        current_cell_texts_list.extend(suggestion_texts_on_cells[(r_idx, c_idx)])
                    else:
                        current_cell_texts_list.append(".") # Placeholder for empty, no suggestion
                
                final_display_text = "\n".join(current_cell_texts_list)
                num_lines = final_display_text.count('\n') + 1
                dynamic_font_size = font_size_base / num_lines if num_lines > 1 else font_size_base
                # Further reduce if text is too wide for cell (approximate)
                dynamic_font_size = max(4, dynamic_font_size * min(1, (cell_size_inch * 10) / (len(final_display_text) / max(num_lines,1) + 1)))


                ax.text(c_idx, r_idx, final_display_text,
                        ha='center', va='center', fontsize=dynamic_font_size, color='black', wrap=True)

        pv_str_display = ", ".join(map(str, proposed_values)) if proposed_values else "無"
        title_str_display = f'盤面分析 ({rows}x{cols}) - 候選值: [{pv_str_display}]'
        if not any(sugg_list for sugg_list in all_suggestions.values()):
            title_str_display += "\n(盤面無-1格或模組未提供有效建議)"
        
        plt.title(title_str_display, fontsize=max(8, cell_size_inch * 14), pad=20)
        
        legend_elements: list[patches.Patch] = []
        if proposed_values and any(s for pv_suggs in all_suggestions.values() for s in pv_suggs):
            added_pvs_to_legend: set[int | str] = set()
            for pv_idx, pv_val_legend in enumerate(proposed_values):
                if pv_val_legend not in added_pvs_to_legend and any(s for s in all_suggestions.get(pv_val_legend, [])):
                    color = self.PV_COLORS[pv_idx % len(self.PV_COLORS)]
                    legend_elements.append(patches.Patch(facecolor=color, edgecolor=color, label=f'候選值 {pv_val_legend} 建議'))
                    added_pvs_to_legend.add(pv_val_legend)

        if legend_elements:
            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.03, 0.5),
                      fontsize=max(7, cell_size_inch * 10), title="圖例")
            plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust for legend
        else:
            plt.tight_layout()


        img_base64 = self._fig_to_base64(fig)
        plt.close(fig) # Ensure figure is closed
        return img_base64

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        buf = io.BytesIO()
        try:
            # Use the _current_cell_size_inch_for_dpi attribute set during visualization generation
            dpi = max(75, int(self._current_cell_size_inch_for_dpi * 100)) # Adjusted DPI calculation
            fig.savefig(buf, format='png', dpi=dpi)
        except Exception as e:
            logger.error("fig.savefig failed: %s", e, exc_info=True)
            # plt.close(fig) should be called by the caller if savefig fails mid-way
            raise VisualizationError(f"Failed to save figure to buffer: {e}") from e
        finally:
            # Ensure the figure is closed even if other parts of _fig_to_base64 fail after savefig
            # However, common practice is to close it in the calling function after _fig_to_base64 returns.
            # If savefig itself fails, fig might still be open.
            pass # Let caller handle plt.close(fig)

        buf.seek(0)
        img_base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        return img_base64_str

# --- Example Usage and Mocking (for testing and demonstration) ---
if __name__ == '__main__':
    class MockMainModuleImpl(MainModuleProtocol): # Explicitly implement protocol for clarity
        """
        一個Mock的main.py 模組,用於Analyzer 的測試和演示。
        此 Mock 模組自身不進行複雜分析,僅返回符合格式的隨機或預定分數。
        """
        def __init__(self):
            self.registered_modules: dict[str, Callable[[list[list[int]], int], np.ndarray]] = {
                "GM1_Random": self._gm1_random_scorer,
                "GM2_TargetTopLeft": self._gm2_target_top_left,
                "GM3_PV_Bonus": self._gm3_pv_bonus,
                "GM_ErrorModule": self._gm_error_module,
                "GM_WrongShape": self._gm_wrong_shape,
            }
            logger.info("MockMainModuleImpl initialized with modules: %s", list(self.registered_modules.keys()))

        def _gm1_random_scorer(self, board: list[list[int]], proposed_value: int) -> np.ndarray:
            rows, cols = (len(board), len(board[0])) if board and board[0] else (0, 0)
            logger.debug("[Mock GM1_Random] PV %d. Board %dx%d. Returning random scores.", proposed_value, rows, cols)
            if rows == 0 or cols == 0: return np.array([[]], dtype=float).reshape(0,0) # Correct empty 2D array
            return np.random.rand(rows, cols)

        def _gm2_target_top_left(self, board: list[list[int]], proposed_value: int) -> np.ndarray:
            rows, cols = (len(board), len(board[0])) if board and board[0] else (0, 0)
            scores = np.zeros((rows, cols))
            if rows > 0 and cols > 0:
                scores[0, 0] = 0.9
            if rows > 1 and cols > 1:
                scores[1, 1] = 0.7
            logger.debug("[Mock GM2_TargetTopLeft] PV %d. Board %dx%d. Prioritizing top-left.", proposed_value, rows, cols)
            return scores

        def _gm3_pv_bonus(self, board: list[list[int]], proposed_value: int) -> np.ndarray:
            rows, cols = (len(board), len(board[0])) if board and board[0] else (0, 0)
            scores = np.full((rows, cols), 0.1)
            if proposed_value == 5:
                if rows > 0 and cols > 0 and board[rows - 1][cols - 1] == -1:
                    scores[rows - 1, cols - 1] = 1.0
            logger.debug("[Mock GM3_PV_Bonus] PV %d. Board %dx%d. Bonus for PV 5 at bottom-right.", proposed_value, rows, cols)
            return scores

        def _gm_error_module(self, board: list[list[int]], proposed_value: int) -> np.ndarray:
            logger.debug("[Mock GM_ErrorModule] Intentionally raising ModuleExecutionError for PV %d.", proposed_value)
            raise ModuleExecutionError("GM_ErrorModule: Simulated controlled module failure.")

        def _gm_wrong_shape(self, board: list[list[int]], proposed_value: int) -> np.ndarray:
            rows, cols = (len(board), len(board[0])) if board and board[0] else (0, 0)
            logger.debug("[Mock GM_WrongShape] Returning scores with wrong shape for PV %d.", proposed_value)
            if rows == 0 or cols == 0: return np.array([[]], dtype=float).reshape(0,0)
            return np.random.rand(rows + 1, cols) # Incorrect shape

        def get_module_score(self, module_name: str, new_card: list[list[int]], pv: int) -> np.ndarray:
            if module_name not in self.registered_modules:
                raise ModuleNotFoundError(f"Mock module '{module_name}' not found during get_module_score call.")
            return self.registered_modules[module_name](new_card, pv)

    logger.info("--- analyzer.py Self-Test/Demonstration START ---")
    analyzer_instance = Analyzer(main_module=MockMainModuleImpl(), default_top_n=3)

    # Test Case 1
    board1: list[list[int]] = [[1, -1, 3, -1], [-1, 5, -1, 8], [9, 10, 11, -1]]
    pvs1: list[int] = [2, 4, 7]
    logger.info("\n--- Test Case 1: Standard Board (3x4), PVs: %s ---", pvs1)
    try:
        results1 = analyzer_instance.analyze_board(
            new_card=board1, proposed_values=pvs1,
            active_modules=["GM1_Random", "GM2_TargetTopLeft", "GM3_PV_Bonus"],
            module_weights={"GM2_TargetTopLeft": 2.0, "GM_NonExistent": 0.5},
            request_id_for_logging="test_req_001"
        )
        logger.info("Test Case 1 Processed Params: %s", results1.get('processed_params'))
        if 'suggestions' in results1:
            for pv_val, suggestions in results1['suggestions'].items():
                logger.info(f"  Suggestions for PV {pv_val}:")
                for sugg in suggestions: logger.info(f"    Pos: {sugg['position']}, Score: {sugg['score']:.4f}")
        
        if results1.get('visualization') and isinstance(results1['visualization'], str) and \
           not results1['visualization'].startswith('Error'):
            with open("analyzer_test_case_1.png", "wb") as f:
                f.write(base64.b64decode(results1['visualization']))
            logger.info("Test Case 1 Visualization: analyzer_test_case_1.png")
        else:
            logger.warning("Test Case 1 Visualization data missing or indicates error: %s", results1.get('visualization'))
    except AnalyzerError as e_test:
        logger.error("Test Case 1 FAILED: %s", e_test, exc_info=True)

    # Test Case 2
    board2: list[list[int]] = [[-1, -1], [-1, 2]]
    pvs2: list[int] = [1]
    logger.info("\n--- Test Case 2: Small Board (2x2), PVs: %s, with faulty modules ---", pvs2)
    try:
        results2 = analyzer_instance.analyze_board(
            new_card=board2, proposed_values=pvs2,
            active_modules=["GM1_Random", "GM_ErrorModule", "GM_WrongShape", "GM2_TargetTopLeft"],
            top_n=2, request_id_for_logging="test_req_002"
        )
        logger.info("Test Case 2 Processed Params: %s", results2.get('processed_params'))
        logger.info("Test Case 2 results (GM_ErrorModule, GM_WrongShape contributions should be gracefully ignored):")
        if 'suggestions' in results2:
            for pv_val, suggestions in results2['suggestions'].items():
                logger.info(f"  Suggestions for PV {pv_val}: {suggestions}")

        if results2.get('visualization') and isinstance(results2['visualization'], str) and \
           not results2['visualization'].startswith('Error'):
            with open("analyzer_test_case_2.png", "wb") as f:
                f.write(base64.b64decode(results2['visualization']))
            logger.info("Test Case 2 Visualization: analyzer_test_case_2.png")
        else:
            logger.warning("Test Case 2 Visualization data missing or indicates error: %s", results2.get('visualization'))
    except AnalyzerError as e_test:
        logger.error("Test Case 2 FAILED: %s", e_test, exc_info=True)

    # Test Case 3
    board3: list[list[int]] = [[1, 2], [3, 4]]
    pvs3: list[int] = [5]
    logger.info("\n--- Test Case 3: Full Board (2x2, no -1), PVs: %s ---", pvs3)
    try:
        results3 = analyzer_instance.analyze_board(new_card=board3,
                                                   proposed_values=pvs3, request_id_for_logging="test_req_003")
        logger.info("Test Case 3 Processed Params: %s", results3.get('processed_params'))
        logger.info("Test Case 3 results (expect no suggestions):")
        if 'suggestions' in results3:
            for pv_val, suggestions in results3['suggestions'].items():
                assert not suggestions, f"PV {pv_val} should have no suggestions on a full board."
                logger.info(f"  Suggestions for PV {pv_val}: {suggestions}")

        if results3.get('visualization') and isinstance(results3['visualization'], str) and \
           not results3['visualization'].startswith('Error'): # Corrected results to results3
            with open("analyzer_test_case_3.png", "wb") as f:
                f.write(base64.b64decode(results3['visualization'])) # Corrected results to results3
            logger.info("Test Case 3 Visualization: analyzer_test_case_3.png")
        else:
            logger.warning("Test Case 3 Visualization data missing or indicates error: %s", results3.get('visualization'))
    except AnalyzerError as e_test:
        logger.error("Test Case 3 FAILED: %s", e_test, exc_info=True)

    # Test Case 4
    board4: list[list[int]] = [[-1,-1],[-1,-1]]
    pvs4: list[int] = [10]
    logger.info("\n--- Test Case 4: No effective modules, PVs: %s ---", pvs4)
    try:
        results4 = analyzer_instance.analyze_board(
            new_card=board4, proposed_values=pvs4,
            active_modules=["NonExistent_A", "NonExistent_B"],
            request_id_for_logging="test_req_004"
        )
        logger.info("Test Case 4 Processed Params: %s", results4.get('processed_params'))
        logger.info("Test Case 4 results (expect no suggestions from modules, scores should be 0):")
        if 'suggestions' in results4:
            for pv_val, suggestions in results4['suggestions'].items():
                logger.info(f"  Suggestions for PV {pv_val}: {suggestions}")
                # if no modules ran, scores should be 0 leading to suggestions with score 0 if cells are -1
                # The check `assert sugg['score'] == 0.0` might be too strict if fused_scores are normalized differently for all zeros.
                # The current _fuse_scores normalizes all-identical scores to 0.0.
                for sugg_item in suggestions:
                    assert sugg_item['score'] == 0.0, "Scores should be 0.0 if no modules ran or all returned 0."

        if results4.get('visualization') and isinstance(results4['visualization'], str) and \
           not results4['visualization'].startswith('Error'): # Corrected results to results4
            with open("analyzer_test_case_4.png", "wb") as f:
                f.write(base64.b64decode(results4['visualization']))
            logger.info("Test Case 4 Visualization: analyzer_test_case_4.png")
        else:
            logger.warning("Test Case 4 Visualization data missing or indicates error: %s", results4.get('visualization'))
    except AnalyzerError as e_test:
        logger.error("Test Case 4 FAILED: %s", e_test, exc_info=True)

    # Test Case 5
    board5: list[list[int]] = []
    pvs5: list[int] = [1]
    logger.info("\n--- Test Case 5: Empty board input ---")
    try:
        results5 = analyzer_instance.analyze_board(new_card=board5,
                                                   proposed_values=pvs5, request_id_for_logging="test_req_005")
        # This case should raise InvalidInputError, so this part of try might not be reached
        logger.info("Test Case 5 Processed Params (if not error): %s", results5.get('processed_params'))
    except InvalidInputError as e_test:
        logger.info("Test Case 5 Correctly caught InvalidInputError for empty board: %s", e_test)
    except Exception as e_test:
        logger.error("Test Case 5 FAILED with unexpected error: %s", e_test, exc_info=True)

    # Test Case 6
    board6: list[list[int]] = [[]]
    pvs6: list[int] = [1]
    logger.info("\n--- Test Case 6: Board with empty row input ---")
    try:
        results6 = analyzer_instance.analyze_board(new_card=board6,
                                                   proposed_values=pvs6, request_id_for_logging="test_req_006")
        logger.info("Test Case 6 Processed Params (if not error): %s", results6.get('processed_params'))
    except InvalidInputError as e_test:
        logger.info("Test Case 6 Correctly caught InvalidInputError for board with empty row: %s", e_test)
    except Exception as e_test:
        logger.error("Test Case 6 FAILED with unexpected error: %s", e_test, exc_info=True)

    logger.info("\n--- analyzer.py Self-Test/Demonstration COMPLETE ---")