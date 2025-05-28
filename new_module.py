# new_module.py
# coding: utf-8

"""
new_module.py
=============

Extreme N-Dimensional Tensor Operations Module (PuzzleTensorOps)

This module provides a high-performance, N-dimensional array/tensor
manipulation class, `PuzzleTensorOps`, built on NumPy for extreme vectorization.
It's designed for demanding puzzle engines, AI applications, and any system
requiring efficient, multi-dimensional array operations.

The API is crafted to be NumPy-like for ease of use and is structured to
facilitate straightforward re-implementation of its core logic in lower-level
languages like C++ (for JNI/NDK on Android or Swift/Objective-C wrappers on iOS)
to achieve true native mobile performance.

This single file includes:
- The `PuzzleTensorOps` class.
- Self-contained (conceptual) unit tests using `unittest`.
- Self-contained (conceptual) performance benchmarks using `timeit`.
- A self-contained (conceptual) FastAPI demo for API serving.
- Comprehensive docstrings in both English and Chinese.

本模組提供一個高效能的 N 維陣列/張量操作類別 `PuzzleTensorOps`，
基於 NumPy 實現極致向量化。專為高要求的解謎引擎、AI 應用以及任何需要
高效多維陣列操作的系統而設計。

其 API 設計風格類似 NumPy，易於使用，且其結構有利於將核心邏輯直接
以 C++ 等底層語言重新實現（用於 Android 的 JNI/NDK 或 iOS 的 Swift/Objective-C 封裝），
以達到真正的原生行動平台極限效能。

此單一檔案包含：
- `PuzzleTensorOps` 類別。
- 使用 `unittest` 的內建（概念性）單元測試。
- 使用 `timeit` 的內建（概念性）效能基準測試。
- 內建（概念性）FastAPI 應用程式示範 API 服務。
- 完整的中英文 docstring 文件。

Version: 1.0.0
Author: AI Assistant (Conceptual Implementation)
Date: 2025-05-28
"""

import numpy as np
import timeit
import unittest
from typing import Tuple, Union, Callable, Any, Sequence, Optional, List, Dict, TypeVar

# --- Type Variable for PuzzleTensorOps ---
PTO = TypeVar('PTO', bound='PuzzleTensorOps')

# --- Module Level Constants ---
# Can be expanded as needed
DEFAULT_FLOAT_TYPE = np.float64
DEFAULT_INT_TYPE = np.int64
DEFAULT_BOOL_TYPE = bool

class PuzzleTensorOps:
    """
    PuzzleTensorOps - Extreme N-Dimensional Tensor Operations Class.
    Manages and performs optimized N-dimensional array operations using NumPy vectorization.

    PuzzleTensorOps - 極限 N 維張量運算類別。
    使用 NumPy 向量化來管理和執行優化的 N 維陣列操作。
    """

    def __init__(self, data: np.ndarray, copy_data: bool = True) -> None:
        """
        Initializes the PuzzleTensorOps instance.
        初始化 PuzzleTensorOps 實例。

        Parameters
        ----------
        data : np.ndarray
            The input N-dimensional NumPy array. It must have at least one dimension.
            輸入的 N 維 NumPy 陣列，必須至少有一個維度。
        copy_data : bool, optional
            If True (default), a deep copy of `data` is stored internally.
            If False, the internal grid will be a reference to `data`.
            若為 True (預設)，則內部儲存 `data` 的深拷貝。
            若為 False，則內部網格將參考 `data`。

        Raises
        ------
        TypeError
            If `data` is not a NumPy ndarray.
            若 `data` 不是 NumPy ndarray。
        ValueError
            If `data` is a 0-dimensional array (scalar).
            若 `data` 是 0 維陣列 (純量)。
        """
        if not isinstance(data, np.ndarray):
            msg_en = "Input `data` must be a NumPy ndarray."
            msg_zh = "輸入資料 `data` 必須是 NumPy ndarray 型態。"
            raise TypeError(f"{msg_en} / {msg_zh}")
        if data.ndim == 0:
            msg_en = "Input `data` must be at least 1-dimensional; 0-d arrays (scalars) are not supported."
            msg_zh = "輸入資料 `data` 必須至少是一維；不支援 0 維陣列 (純量)。"
            raise ValueError(f"{msg_en} / {msg_zh}")

        self._grid: np.ndarray = data.copy() if copy_data else data
        self._last_op_duration: Optional[float] = None # For simple timing introspection

    @property
    def grid_view(self) -> np.ndarray:
        """
        Provides a read-only view of the internal N-dimensional array.
        提供內部 N 維陣列的唯讀視圖。

        Returns
        -------
        np.ndarray
            A view of the internal N-dimensional array.
            內部 N 維陣列的一個視圖。
        """
        return self._grid.view()

    def get_copy(self) -> np.ndarray:
        """
        Returns a deep copy of the internal N-dimensional array.
        返回內部 N 維陣列的一個完整深拷貝。

        Returns
        -------
        np.ndarray
            A deep copy of the internal array.
            內部陣列的深拷貝。

        Performance
        -----------
        Leverages NumPy's `copy()` method, which is C-optimized.
        利用 NumPy 的 `copy()` 方法，該方法已 C 語言優化。
        """
        start_time = timeit.default_timer()
        copied_array = self._grid.copy()
        self._last_op_duration = timeit.default_timer() - start_time
        return copied_array

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        The shape of the internal N-dimensional array.
        內部 N 維陣列的形狀。
        """
        return self._grid.shape

    @property
    def ndim(self) -> int:
        """
        The number of dimensions of the internal array.
        內部陣列的維度數。
        """
        return self._grid.ndim

    @property
    def dtype(self) -> np.dtype:
        """
        The data type of the elements in the internal array.
        內部陣列元素的資料型態。
        """
        return self._grid.dtype

    @property
    def size(self) -> int:
        """
        The total number of elements in the internal array.
        內部陣列的總元素數量。
        """
        return self._grid.size
        
    @property
    def last_op_duration_ms(self) -> Optional[float]:
        """
        Duration of the last major operation in milliseconds, if recorded.
        最後一個主要操作的持續時間 (毫秒)，如果已記錄。
        """
        return self._last_op_duration * 1000 if self._last_op_duration is not None else None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype}, data=\n{self._grid}\n)"

    def get_slice(self,
                  slicing_object: Union[slice, int, Ellipsis, np.ndarray, Sequence[Union[slice, int, Ellipsis, np.ndarray]]]) -> np.ndarray:
        """
        Retrieves a sub-array (slice) from the internal grid.
        從內部網格中檢索子陣列 (切片)。

        Parameters
        ----------
        slicing_object : slice, int, Ellipsis, np.ndarray, or sequence thereof
            A valid NumPy slicing object.
            有效的 NumPy 切片物件。

        Returns
        -------
        np.ndarray
            The selected sub-array. May be a view or a copy per NumPy rules.
            選取的子陣列。可能是視圖或拷貝，遵循 NumPy 規則。

        Raises
        ------
        IndexError, TypeError
            If `slicing_object` is invalid.
            若切片物件無效。
        """
        start_time = timeit.default_timer()
        try:
            result = self._grid[slicing_object]
            self._last_op_duration = timeit.default_timer() - start_time
            return result
        except (IndexError, TypeError) as e:
            self._last_op_duration = timeit.default_timer() - start_time
            msg_en = f"Invalid slice object '{slicing_object}' for array with shape {self.shape}: {e}"
            msg_zh = f"對於形狀為 {self.shape} 的陣列，切片物件 '{slicing_object}' 無效：{e}"
            raise type(e)(f"{msg_en} / {msg_zh}") from e

    def set_slice(self: PTO,
                  slicing_object: Union[slice, int, Ellipsis, np.ndarray, Sequence[Union[slice, int, Ellipsis, np.ndarray]]],
                  values: Union[int, float, bool, complex, np.ndarray]) -> PTO:
        """
        Sets values in a specified slice of the internal grid (in-place).
        在內部網格的指定切片中設定值 (原地操作)。

        Parameters
        ----------
        slicing_object : slice, int, Ellipsis, np.ndarray, or sequence thereof
            A valid NumPy slicing object defining the target region.
            定義目標區域的有效 NumPy 切片物件。
        values : scalar or np.ndarray
            The value(s) to assign. Must be broadcastable to the slice's shape.
            要賦予的值。若為陣列，其形狀必須能廣播到切片的形狀。

        Returns
        -------
        PuzzleTensorOps
            Returns `self` for method chaining.
            返回 `self` 以便鏈式操作。
        """
        start_time = timeit.default_timer()
        try:
            self._grid[slicing_object] = values
            self._last_op_duration = timeit.default_timer() - start_time
            return self
        except (IndexError, ValueError, TypeError) as e:
            self._last_op_duration = timeit.default_timer() - start_time
            value_info_en = f"type={type(values)}"
            value_info_zh = f"型態={type(values)}"
            if isinstance(values, np.ndarray):
                value_info_en += f", shape={values.shape}, dtype={values.dtype}"
                value_info_zh += f", 形狀={values.shape}, 資料型態={values.dtype}"
            msg_en = f"Failed to set slice '{slicing_object}' with values ({value_info_en}) for array with shape {self.shape}: {e}"
            msg_zh = f"為形狀為 {self.shape} 的陣列設定切片 '{slicing_object}' 失敗 (值資訊: {value_info_zh})：{e}"
            raise type(e)(f"{msg_en} / {msg_zh}") from e

    def apply_elementwise(self: PTO,
                          operation: Callable[..., np.ndarray],
                          *args: Any,
                          target_self: bool = False,
                          **kwargs: Any) -> PTO:
        """
        Applies a NumPy ufunc or compatible vectorized function element-wise.
        逐元素應用 NumPy ufunc 或相容的向量化函數。

        Parameters
        ----------
        operation : Callable[..., np.ndarray]
            The vectorized function (e.g., `np.add`, `np.sqrt`).
            向量化函數 (例如 `np.add`, `np.sqrt`)。
        args : Any
            Additional positional arguments for `operation`.
            傳遞給 `operation` 的額外位置參數。
        target_self : bool, optional
            If True (default=False), performs operation in-place. Otherwise, returns a new instance.
            若為 True (預設=False)，則原地執行操作。否則返回新實例。
        kwargs : Any
            Additional keyword arguments for `operation`. `out` is handled specially.
            傳遞給 `operation` 的額外關鍵字參數。`out` 參數會被特殊處理。

        Returns
        -------
        PuzzleTensorOps
            A new instance with the result, or `self` if `target_self` is True.
            包含結果的新實例，或者若 `target_self` 為 True 則返回 `self`。
        """
        start_time = timeit.default_timer()
        if target_self:
            if 'out' in kwargs and kwargs['out'] is not self._grid:
                msg_en = "Explicit 'out' kwarg cannot be used with `target_self=True` unless it's the internal grid itself."
                msg_zh = "當 `target_self=True` 時，不可使用明確的 'out' 關鍵字參數，除非它就是內部網格本身。"
                raise ValueError(f"{msg_en} / {msg_zh}")
            try:
                kwargs['out'] = self._grid # Attempt to use out for in-place
                result = operation(self._grid, *args, **kwargs)
                # If ufunc doesn't modify in-place even with 'out' (e.g. type change), assign back
                if result is not None and result is not self._grid: 
                    if result.shape == self._grid.shape:
                        self._grid[:] = result
                    else: # Should not happen if 'out' works as expected
                        msg_en = "In-place operation resulted in a shape change, which is not supported for target_self=True via 'out'."
                        msg_zh = "原地操作導致形狀改變，這在 target_self=True 且透過 'out' 參數時不被支援。"
                        raise ValueError(f"{msg_en} / {msg_zh}")
                self._last_op_duration = timeit.default_timer() - start_time
                return self
            except Exception as e:
                self._last_op_duration = timeit.default_timer() - start_time
                msg_en = f"In-place element-wise operation '{getattr(operation, '__name__', str(operation))}' failed: {e}"
                msg_zh = f"原地逐元素操作 '{getattr(operation, '__name__', str(operation))}' 失敗：{e}"
                raise type(e)(f"{msg_en} / {msg_zh}") from e
        else:
            kwargs_copy = kwargs.copy()
            if 'out' in kwargs_copy: # Politely ignore 'out' if not targeting self
                del kwargs_copy['out']
            try:
                result_grid = operation(self._grid, *args, **kwargs_copy)
                new_pto = PuzzleTensorOps(result_grid, copy_data=False)
                new_pto._last_op_duration = timeit.default_timer() - start_time
                self._last_op_duration = new_pto._last_op_duration # Also set on original for consistency if needed
                return new_pto
            except Exception as e:
                self._last_op_duration = timeit.default_timer() - start_time
                msg_en = f"Element-wise operation '{getattr(operation, '__name__', str(operation))}' failed: {e}"
                msg_zh = f"逐元素操作 '{getattr(operation, '__name__', str(operation))}' 失敗：{e}"
                raise type(e)(f"{msg_en} / {msg_zh}") from e

    def apply_mask_and_get_values(self, mask: np.ndarray) -> np.ndarray:
        """
        Applies a boolean mask and returns the selected elements as a new 1D array.
        應用布林遮罩並以新的一維陣列返回選定元素。

        Parameters
        ----------
        mask : np.ndarray
            A boolean array, broadcastable to the internal grid's shape.
            布林陣列，其形狀必須能廣播到內部網格的形狀。

        Returns
        -------
        np.ndarray
            A new 1D array containing selected elements.
            包含所選元素的新一維陣列。
        """
        start_time = timeit.default_timer()
        if not isinstance(mask, np.ndarray):
            msg_en = f"Mask must be a NumPy ndarray; got {type(mask)}."
            msg_zh = f"遮罩 `mask` 必須是 NumPy ndarray；得到 {type(mask)}。"
            raise TypeError(f"{msg_en} / {msg_zh}")
        if mask.dtype != DEFAULT_BOOL_TYPE:
            msg_en = f"Mask dtype must be bool; got {mask.dtype}."
            msg_zh = f"遮罩 `mask` 的資料型態必須是 bool；得到 {mask.dtype}。"
            raise TypeError(f"{msg_en} / {msg_zh}")
        try:
            result = self._grid[mask]
            self._last_op_duration = timeit.default_timer() - start_time
            return result
        except IndexError as e:
            self._last_op_duration = timeit.default_timer() - start_time
            msg_en = f"Mask shape {mask.shape} cannot be broadcast to grid shape {self.shape}: {e}"
            msg_zh = f"遮罩形狀 {mask.shape} 無法廣播到網格形狀 {self.shape}：{e}"
            raise ValueError(f"{msg_en} / {msg_zh}") from e

    def get_coordinates_where(self,
                              condition_or_mask: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]
                             ) -> Tuple[np.ndarray, ...]:
        """
        Returns N-dimensional coordinates of elements satisfying a condition or mask.
        返回滿足條件或遮罩的元素的 N 維座標。

        Parameters
        ----------
        condition_or_mask : np.ndarray (bool) or Callable returning bool np.ndarray
            A boolean mask or a callable that produces one from self._grid.
            布林遮罩或一個從 self._grid 產生布林遮罩的可呼叫物件。

        Returns
        -------
        Tuple[np.ndarray, ...]
            Tuple of N 1D arrays (indices for each dimension), like `np.where()`.
            包含 N 個一維陣列的元組 (每個維度的索引)，類似 `np.where()` 的返回格式。
        """
        start_time = timeit.default_timer()
        mask_array: np.ndarray
        if callable(condition_or_mask):
            try:
                mask_array = condition_or_mask(self._grid)
            except Exception as e:
                self._last_op_duration = timeit.default_timer() - start_time
                msg_en = f"Callable `condition_or_mask` failed during execution: {e}"
                msg_zh = f"可呼叫物件 `condition_or_mask` 執行失敗：{e}"
                raise ValueError(f"{msg_en} / {msg_zh}") from e
            if not isinstance(mask_array, np.ndarray) or mask_array.dtype != DEFAULT_BOOL_TYPE:
                msg_en = "Callable `condition_or_mask` must return a boolean NumPy ndarray."
                msg_zh = "可呼叫物件 `condition_or_mask` 必須返回布林 NumPy ndarray。"
                raise TypeError(f"{msg_en} / {msg_zh}")
            if mask_array.shape != self.shape:
                msg_en = f"Mask returned by callable (shape {mask_array.shape}) does not match grid shape ({self.shape})."
                msg_zh = f"可呼叫物件返回的遮罩 (形狀 {mask_array.shape}) 與網格形狀 ({self.shape}) 不匹配。"
                raise ValueError(f"{msg_en} / {msg_zh}")
        elif isinstance(condition_or_mask, np.ndarray) and condition_or_mask.dtype == DEFAULT_BOOL_TYPE:
            mask_array = condition_or_mask
        else:
            msg_en = "Input `condition_or_mask` must be a boolean NumPy ndarray or a callable that returns one."
            msg_zh = "輸入 `condition_or_mask` 必須是布林 NumPy ndarray 或返回此類陣列的可呼叫物件。"
            raise TypeError(f"{msg_en} / {msg_zh}")
        try:
            result = np.where(mask_array)
            self._last_op_duration = timeit.default_timer() - start_time
            return result
        except ValueError as e: # Catches shape mismatches for mask_array if not pre-validated by callable checks
            self._last_op_duration = timeit.default_timer() - start_time
            msg_en = f"Mask shape {mask_array.shape} is incompatible with grid shape {self.shape} for np.where: {e}"
            msg_zh = f"遮罩形狀 {mask_array.shape} 與網格形狀 {self.shape} 不相容 (用於 np.where)：{e}"
            raise ValueError(f"{msg_en} / {msg_zh}") from e

    def count_true_along_axis(self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> Union[int, np.ndarray]:
        """
        Counts True elements along an axis. Assumes boolean or convertible grid.
        沿著指定軸計算 True 元素的數量。假設網格是布林型態或可轉換為布林型態。

        Parameters
        ----------
        axis : int, tuple of ints, or None, optional
            Axis or axes along which to count. If None, counts in flattened array.
            計數的軸。若為 None，則在扁平化陣列中計數。
        keepdims : bool, optional
            If True, reduced axes are left in result with size one. Default False.
            若為 True，被縮減的軸將保留在結果中，大小為 1。預設為 False。

        Returns
        -------
        int or np.ndarray
            Count result.
            計數結果。
        """
        start_time = timeit.default_timer()
        grid_to_sum = self._grid
        if self._grid.dtype != DEFAULT_BOOL_TYPE:
            grid_to_sum = self._grid.astype(DEFAULT_BOOL_TYPE) # Ensure boolean for sum to count True
        
        result = np.sum(grid_to_sum, axis=axis, keepdims=keepdims)
        self._last_op_duration = timeit.default_timer() - start_time
        return result

    # --- Puzzle Specific Operation Prototypes ---
    def update_candidates_on_placement_nd(self: PTO,
                                       candidates_grid_pto: PTO, # PuzzleTensorOps instance for candidates
                                       placed_value: int, # 1-indexed
                                       placed_coords: Tuple[int, ...],
                                       # For N-D, constraint propagation needs careful definition.
                                       # This example assumes "line-of-sight" constraints along each primary axis.
                                       # More complex constraints (like Sudoku blocks) would need different logic.
                                       ) -> PTO:
        """
        [Prototype] Updates an N-D candidate grid after a value is placed.
        Assumes candidates_grid_pto's last dimension flags candidates (0-indexed).
        [原型] 在放置一個值後更新 N 維候選數網格。
        假設 candidates_grid_pto 的最後一個維度標記候選數 (0 索引)。

        Parameters
        ----------
        candidates_grid_pto : PuzzleTensorOps
            PTO instance for candidate booleans. Shape `(*grid_shape, num_candidates)`.
            用於候選布林值的 PTO 實例。形狀為 `(*grid_shape, num_candidates)`。
        placed_value : int
            The 1-indexed value that was placed.
            被放置的 1 索引值。
        placed_coords : Tuple[int, ...]
            N-D coordinates of placement. Length must be `self.ndim`.
            放置位置的 N 維座標。長度必須等於 `self.ndim`。

        Returns
        -------
        PuzzleTensorOps
            A new PTO instance with the updated candidates grid.
            包含更新後候選數網格的新 PTO 實例。

        Raises
        ------
        ValueError
            If dimensions or coordinates are inconsistent.
            若維度或座不一致。
        """
        start_time = timeit.default_timer()
        if candidates_grid_pto.ndim != self.ndim + 1:
            msg_en = f"Candidates grid ndim ({candidates_grid_pto.ndim}) must be puzzle grid ndim ({self.ndim}) + 1."
            msg_zh = f"候選數網格維度 ({candidates_grid_pto.ndim}) 必須是謎題網格維度 ({self.ndim}) + 1。"
            raise ValueError(f"{msg_en} / {msg_zh}")
        if len(placed_coords) != self.ndim:
            msg_en = f"Length of placed_coords ({len(placed_coords)}) must match puzzle grid ndim ({self.ndim})."
            msg_zh = f"放置座標的長度 ({len(placed_coords)}) 必須與謎題網格維度 ({self.ndim}) 相符。"
            raise ValueError(f"{msg_en} / {msg_zh}")
        
        num_total_candidates = candidates_grid_pto.shape[-1]
        if not (0 < placed_value <= num_total_candidates):
            msg_en = f"placed_value {placed_value} is out of range for {num_total_candidates} candidates."
            msg_zh = f"放置值 {placed_value} 超出了 {num_total_candidates} 個候選數的範圍。"
            raise ValueError(f"{msg_en} / {msg_zh}")

        updated_candidates_arr = candidates_grid_pto.get_copy() # Operate on a copy
        candidate_idx_to_remove = placed_value - 1 # Convert 1-indexed to 0-indexed

        # 1. Clear all candidates from the cell where the value was placed
        cell_slice = list(placed_coords) + [slice(None)]
        updated_candidates_arr[tuple(cell_slice)] = False

        # 2. Remove `placed_value` as a candidate from all "lines of sight"
        #    (rows, columns, and other dimensional equivalents) passing through `placed_coords`.
        for i in range(self.ndim): # For each dimension of the main puzzle grid
            line_slice_parts = list(placed_coords) # Create a base for slicing
            line_slice_parts[i] = slice(None)      # Allow this dimension to vary (the "line")
            
            # This slice now selects all cells along the i-th dimension line
            # that passes through `placed_coords`. We need to update the
            # `candidate_idx_to_remove`-th candidate in the last dimension.
            full_line_candidate_slice = tuple(line_slice_parts + [candidate_idx_to_remove])
            updated_candidates_arr[full_line_candidate_slice] = False
        
        # (Optional) If puzzle logic requires: after clearing lines, re-set the placed value
        # as the *only* candidate in its cell, if that's the convention for solved cells
        # in the candidate grid. For now, step 1 (clearing all) is assumed for solved cell.
        # E.g.:
        # final_cell_candidate_slice = tuple(list(placed_coords) + [candidate_idx_to_remove])
        # updated_candidates_arr[final_cell_candidate_slice] = True 

        new_pto = PuzzleTensorOps(updated_candidates_arr, copy_data=False)
        new_pto._last_op_duration = timeit.default_timer() - start_time
        self._last_op_duration = new_pto._last_op_duration # Also set on original
        return new_pto

    @staticmethod
    def from_array_list(arrays: List[np.ndarray], axis: int = 0) -> PTO:
        """
        Creates a PuzzleTensorOps instance by stacking a list of NumPy arrays.
        透過堆疊 NumPy 陣列列表來建立 PuzzleTensorOps 實例。

        Parameters
        ----------
        arrays : List[np.ndarray]
            List of NumPy arrays to stack. They must have compatible shapes for stacking.
            要堆疊的 NumPy 陣列列表。它們必須具有相容的堆疊形狀。
        axis : int, optional
            The axis along which the arrays will be stacked. Default is 0.
            堆疊陣列的軸。預設為 0。

        Returns
        -------
        PuzzleTensorOps
            A new instance containing the stacked array.
            包含堆疊後陣列的新實例。
        """
        if not arrays:
            msg_en = "Input `arrays` list cannot be empty."
            msg_zh = "輸入 `arrays` 列表不可為空。"
            raise ValueError(f"{msg_en} / {msg_zh}")
        try:
            stacked_array = np.stack(arrays, axis=axis)
            return PuzzleTensorOps(stacked_array, copy_data=False) # np.stack creates a new array
        except Exception as e:
            msg_en = f"Failed to stack arrays: {e}"
            msg_zh = f"堆疊陣列失敗：{e}"
            raise ValueError(f"{msg_en} / {msg_zh}") from e


# --- Conceptual Inline Unit Tests using unittest ---
class TestPuzzleTensorOps(unittest.TestCase):
    """
    Unit tests for the PuzzleTensorOps class.
    PuzzleTensorOps 類別的單元測試。
    """
    def setUp(self):
        """ Test fixture setup. / 測試固定裝置設定。"""
        self.data_2d = np.array([[1, 2, 3], [4, 5, 6]], dtype=DEFAULT_INT_TYPE)
        self.pto_2d = PuzzleTensorOps(self.data_2d.copy()) # Ensure fresh copy for each test

        self.data_3d = np.arange(24, dtype=DEFAULT_FLOAT_TYPE).reshape((2, 3, 4))
        self.pto_3d = PuzzleTensorOps(self.data_3d.copy())
        
        self.bool_data = np.array([[True, False], [True, True]])
        self.pto_bool = PuzzleTensorOps(self.bool_data.copy())

    def test_initialization_and_properties(self):
        """ Test basic initialization and properties. / 測試基本初始化和屬性。"""
        self.assertTrue(np.array_equal(self.pto_2d._grid, self.data_2d))
        self.assertEqual(self.pto_2d.shape, (2, 3))
        self.assertEqual(self.pto_2d.ndim, 2)
        self.assertEqual(self.pto_2d.dtype, DEFAULT_INT_TYPE)
        self.assertEqual(self.pto_2d.size, 6)

        pto_no_copy = PuzzleTensorOps(self.data_2d, copy_data=False)
        self.assertIs(pto_no_copy._grid, self.data_2d) # Checks identity

        with self.assertRaisesRegex(TypeError, "Input `data` must be a NumPy ndarray."):
            PuzzleTensorOps([1,2,3]) # type: ignore
        with self.assertRaisesRegex(ValueError, "Input `data` must be at least 1-dimensional"):
            PuzzleTensorOps(np.array(5)) # 0-D

    def test_get_copy(self):
        """ Test array copying. / 測試陣列複製。"""
        copy_arr = self.pto_2d.get_copy()
        self.assertTrue(np.array_equal(copy_arr, self.data_2d))
        self.assertIsNot(copy_arr, self.pto_2d._grid) # Should be a different object
        copy_arr[0, 0] = 99
        self.assertEqual(self.pto_2d._grid[0, 0], 1) # Original should be unchanged

    def test_get_and_set_slice(self):
        """ Test slicing and slice assignment. / 測試切片和切片賦值。"""
        sub_array = self.pto_3d.get_slice((0, slice(1, None), slice(None, None, 2)))
        expected_sub = self.data_3d[0, 1:, ::2]
        self.assertTrue(np.array_equal(sub_array, expected_sub))

        self.pto_3d.set_slice((0, 0, 0), 100.0)
        self.assertEqual(self.pto_3d._grid[0, 0, 0], 100.0)
        
        new_row = np.array([[[-1, -2, -3, -4]]], dtype=DEFAULT_FLOAT_TYPE) # Shape (1,1,4) for broadcasting
        self.pto_3d.set_slice((1, 0, slice(None)), new_row) # Set first row of second plane
        self.assertTrue(np.array_equal(self.pto_3d._grid[1,0,:], new_row.ravel()))

        with self.assertRaises(IndexError):
            self.pto_2d.get_slice((5, 5))
        with self.assertRaises(ValueError): # NumPy might raise ValueError for incompatible shape assignment
            self.pto_2d.set_slice((0,0), np.array([10,20]))


    def test_apply_elementwise(self):
        """ Test element-wise operations. / 測試逐元素操作。"""
        pto_added = self.pto_2d.apply_elementwise(np.add, 5)
        self.assertTrue(np.array_equal(pto_added._grid, self.data_2d + 5))
        self.assertIsNot(pto_added._grid, self.pto_2d._grid) # Should be new instance

        self.pto_2d.apply_elementwise(np.multiply, 2, target_self=True)
        self.assertTrue(np.array_equal(self.pto_2d._grid, self.data_2d * 2))

    def test_apply_mask_and_get_values(self):
        """ Test masking operations. / 測試遮罩操作。"""
        mask = np.array([[True, False, True], [False, False, True]])
        selected = self.pto_2d.apply_mask_and_get_values(mask)
        self.assertTrue(np.array_equal(selected, np.array([1, 3, 6])))
        
        with self.assertRaises(TypeError):
            self.pto_2d.apply_mask_and_get_values(np.array([1,0,1])) # Not boolean
        with self.assertRaises(ValueError):
            self.pto_2d.apply_mask_and_get_values(np.array([True])) # Wrong shape

    def test_get_coordinates_where(self):
        """ Test coordinate finding with conditions. / 測試條件座標查找。"""
        coords = self.pto_2d.get_coordinates_where(lambda x: x % 2 == 0)
        # data_2d is [[1,2,3],[4,5,6]] -> even are (0,1)=2, (1,0)=4, (1,2)=6
        self.assertTrue(np.array_equal(coords[0], np.array([0, 1, 1]))) # row indices
        self.assertTrue(np.array_equal(coords[1], np.array([1, 0, 2]))) # col indices

        mask = self.pto_2d._grid > 3
        coords_from_mask = self.pto_2d.get_coordinates_where(mask)
        self.assertTrue(np.array_equal(coords_from_mask[0], np.array([1, 1, 1])))
        self.assertTrue(np.array_equal(coords_from_mask[1], np.array([0, 1, 2])))
        
    def test_count_true_along_axis(self):
        """ Test counting true elements. / 測試計數 True 元素。"""
        self.assertEqual(self.pto_bool.count_true_along_axis(), 3)
        self.assertTrue(np.array_equal(self.pto_bool.count_true_along_axis(axis=0), np.array([2, 1])))
        self.assertTrue(np.array_equal(self.pto_bool.count_true_along_axis(axis=1), np.array([1, 2])))

    def test_update_candidates_on_placement_nd(self):
        """ Test candidate update logic. / 測試候選數更新邏輯。"""
        # Main grid (2x2 puzzle)
        puzzle_data = np.zeros((2,2), dtype=DEFAULT_INT_TYPE)
        pto_puzzle = PuzzleTensorOps(puzzle_data)

        # Candidate grid (2x2, 4 possible numbers: 1,2,3,4)
        # Shape: (2, 2, 4), all initially True
        candidates_data = np.full((2, 2, 4), True, dtype=DEFAULT_BOOL_TYPE)
        pto_candidates = PuzzleTensorOps(candidates_data)

        # Place number 1 (value) at (0,0) (coords)
        updated_pto_candidates = pto_puzzle.update_candidates_on_placement_nd(
            pto_candidates, 
            placed_value=1, 
            placed_coords=(0,0)
        )
        
        # Expected:
        # Cell (0,0) should have all candidates False.
        # Number 1 (index 0) should be False in row 0 and col 0 for other cells.
        
        # Check cell (0,0)
        self.assertFalse(updated_pto_candidates._grid[0,0,:].any())

        # Check effect on other cells in row 0 for candidate 1 (idx 0)
        # (0,1) should not have candidate 1
        self.assertFalse(updated_pto_candidates._grid[0,1,0]) 
        # (0,1) candidates 2,3,4 should still be True
        self.assertTrue(updated_pto_candidates._grid[0,1,1:].all()) 

        # Check effect on other cells in col 0 for candidate 1 (idx 0)
        # (1,0) should not have candidate 1
        self.assertFalse(updated_pto_candidates._grid[1,0,0])
        # (1,0) candidates 2,3,4 should still be True
        self.assertTrue(updated_pto_candidates._grid[1,0,1:].all())

        # Cell (1,1) should still have candidate 1 (unless it's on same row/col - which it is not directly)
        # Oh, the current logic of update_candidates clears line-of-sight.
        # (0,0) is placed. Line 0 (row 0) has value 1 removed. Line 1 (col 0) has value 1 removed.
        # So updated_pto_candidates._grid[0,:,0] should be all False
        # And updated_pto_candidates._grid[:,0,0] should be all False
        self.assertFalse(updated_pto_candidates._grid[0,:,0].any()) # All of candidate 1 in row 0 is false
        self.assertFalse(updated_pto_candidates._grid[:,0,0].any()) # All of candidate 1 in col 0 is false

        # Cell (1,1) should still have all its candidates initially (except if it was on a removed line, which it wasn't for value 1 from (0,0))
        # So (1,1) should have [F,T,T,T] if only value 1 from lines passing (0,0) affected it.
        # But if (0,0) is on a line with (1,1) somehow (not in 2D grid case directly), then it could be affected.
        # The current line-of-sight logic is simple. Let's verify (1,1) for candidate 1.
        self.assertTrue(updated_pto_candidates._grid[1,1,0]) # Candidate 1 should still be True for (1,1)

    def test_from_array_list(self):
        """ Test creation from list of arrays. / 測試從陣列列表創建。"""
        arr1 = np.array([[1,2],[3,4]])
        arr2 = np.array([[5,6],[7,8]])
        pto_stacked_axis0 = PuzzleTensorOps.from_array_list([arr1, arr2], axis=0)
        self.assertEqual(pto_stacked_axis0.shape, (2,2,2))
        self.assertTrue(np.array_equal(pto_stacked_axis0._grid, np.stack([arr1,arr2], axis=0)))

        pto_stacked_axis1 = PuzzleTensorOps.from_array_list([arr1, arr2], axis=1)
        self.assertEqual(pto_stacked_axis1.shape, (2,2,2)) # (2, N, 2)
        self.assertTrue(np.array_equal(pto_stacked_axis1._grid, np.stack([arr1,arr2], axis=1)))
        
        with self.assertRaises(ValueError):
            PuzzleTensorOps.from_array_list([])


# --- Conceptual Inline Performance Benchmarks using timeit ---
def run_puzzle_tensor_ops_benchmarks(shapes_to_test: Optional[List[Tuple[int,... ]]]=None, number=100, repeat=3):
    """
    Runs conceptual performance benchmarks for PuzzleTensorOps.
    執行 PuzzleTensorOps 的概念性效能基準測試。
    """
    print("\n--- PuzzleTensorOps Performance Benchmarks ---")
    if shapes_to_test is None:
        shapes_to_test = [(10,10), (100,100), (50,50,10)] # Default shapes

    results = []

    for shape in shapes_to_test:
        data = np.random.rand(*shape)
        pto = PuzzleTensorOps(data, copy_data=False) # Avoid copy in setup

        # 1. get_copy
        t = min(timeit.Timer(lambda: pto.get_copy()).repeat(repeat=repeat, number=number)) / number
        results.append({"op": "get_copy", "shape": shape, "time_s": t, "elements": pto.size})

        # 2. get_slice (first hyperplane/row)
        slice_obj = tuple([0] + [slice(None)] * (pto.ndim - 1)) if pto.ndim > 0 else slice(None)
        # Pre-calculate slice size for throughput
        try:
            slice_example = pto.get_slice(slice_obj)
            elements_in_slice = slice_example.size
        except: # Handle cases like 0-dim slice from 1-dim array
            elements_in_slice = 0 if pto.ndim > 0 else 1

        t = min(timeit.Timer(lambda: pto.get_slice(slice_obj)).repeat(repeat=repeat, number=number)) / number
        results.append({"op": "get_slice", "shape": shape, "time_s": t, "elements": elements_in_slice})

        # 3. apply_mask_and_get_values (50% density)
        mask = np.random.choice([True, False], size=shape, p=[0.5, 0.5])
        # Pre-calculate selected elements for throughput
        selected_elements_count = np.sum(mask)
        t = min(timeit.Timer(lambda: pto.apply_mask_and_get_values(mask)).repeat(repeat=repeat, number=max(1,number//10))) / number
        results.append({"op": "apply_mask (50%)", "shape": shape, "time_s": t, "elements": selected_elements_count})

        # 4. get_coordinates_where (approx 50% condition)
        # For numeric types only
        if np.issubdtype(pto.dtype, np.number) and pto.size > 0:
            mean_val = np.mean(data) # Use original data for mean
            condition = lambda x: x > mean_val
            # Pre-calculate num coords for throughput
            num_coords = len(np.where(condition(data))[0]) # Length of first dim indices array
            t = min(timeit.Timer(lambda: pto.get_coordinates_where(condition)).repeat(repeat=repeat, number=max(1,number//10))) / number
            results.append({"op": "get_coords_where", "shape": shape, "time_s": t, "elements": num_coords})
        
        # 5. Python Loop comparison for a conceptual operation (e.g., count non-zero)
        if pto.size > 0: # Avoid issues with empty arrays for loop version
            def python_count_nonzero_loop(arr_nd):
                count = 0
                # This is a conceptual N-D loop, real one would be more complex or use flat iterator
                for val in arr_nd.flat: # Iterate over flattened array
                    if val != 0:
                        count += 1
                return count
            
            t_loop = min(timeit.Timer(lambda: python_count_nonzero_loop(data)).repeat(repeat=repeat, number=max(1,number//10))) / number
            
            # PTO equivalent: count_true_along_axis(None) if data is boolean after (data != 0)
            pto_count_op = lambda: pto.apply_elementwise(np.not_equal, 0).count_true_along_axis(axis=None)
            t_pto_count = min(timeit.Timer(pto_count_op).repeat(repeat=repeat, number=max(1,number//10))) / number
            
            results.append({"op": "count_nonzero (PTO)", "shape": shape, "time_s": t_pto_count, "elements": pto.size})
            results.append({"op": "count_nonzero (PyLoop)", "shape": shape, "time_s": t_loop, "elements": pto.size})
            if t_pto_count > 0: # Avoid division by zero
                speedup = t_loop / t_pto_count
                print(f"    Shape {shape} count_nonzero: PTO={t_pto_count:.3e}s, Loop={t_loop:.3e}s, Speedup={speedup:.1f}x")


    print("\nBenchmark Results (time_s is average time per operation in seconds, elements is relevant element count for op):")
    # For prettier printing, pandas would be nice here.
    # Convert to pandas DataFrame for display and CSV export
    try:
        import pandas as pd
        df_results = pd.DataFrame(results)
        df_results["throughput_M_elements_s"] = df_results["elements"] / df_results["time_s"] / 1_000_000
        print(df_results.to_string())
        # Save to CSV (conceptual, real path would be better)
        csv_path = "new_module_benchmark_results.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"\nBenchmark results saved to: {csv_path}")

    except ImportError:
        print("Pandas not installed. Printing raw results:")
        for res in results:
            print(res)
    
    return results


# --- Conceptual Inline FastAPI Demo ---
# To run this demo:
# 1. Install FastAPI and Uvicorn: `pip install fastapi uvicorn`
# 2. Save this file as `new_module.py`
# 3. Run from terminal: `uvicorn new_module:app --reload`
# 4. Open your browser to `http://127.0.0.1:8000/docs` for API interaction.

# This part should ideally be guarded by `if __name__ == "__main__":` and only
# when a specific argument is passed, or be in a separate file.
# For the single-file requirement, it's included here conceptually.

_HAS_FASTAPI = False
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, conlist
    _HAS_FASTAPI = True
except ImportError:
    # print("FastAPI or Pydantic not installed. API demo server will not be available.")
    # To avoid runtime error if FastAPI is not installed when module is imported for other uses
    class FastAPI: pass 
    class BaseModel: pass
    def HTTPException(*args, **kwargs): pass
    def conlist(*args, **kwargs): return list


# Only define app if FastAPI is available
if _HAS_FASTAPI:
    app = FastAPI(
        title="PuzzleTensorOps API Demo",
        description="A conceptual FastAPI server demonstrating PuzzleTensorOps capabilities.",
        version="1.0.0"
    )

    # Global PTO instance for demo (in real app, manage state appropriately)
    # Initialize with some default data
    _default_demo_data = np.array([[1,2,3],[4,5,6]])
    _demo_pto = PuzzleTensorOps(_default_demo_data)


    class TensorInput(BaseModel):
        data: List[List[Union[int, float]]] # Simple 2D for demo
        copy_data: Optional[bool] = True

    class SliceInput(BaseModel):
        # Pydantic doesn't directly support tuple of slices easily,
        # so we might need a custom parser or a simpler representation for demo.
        # For simplicity, let's assume a string representation that we parse,
        # or specific slice parameters.
        # E.g., row_slice_start: Optional[int], row_slice_stop: Optional[int], ...
        # Here, we'll just take a list of lists for a simple sub-array for set_slice
        slicing_object_repr: str # e.g., "0, slice(1,None)"
        values: Optional[List[List[Union[int, float]]]] = None # For set_slice

    class ElementwiseOpInput(BaseModel):
        operation: str # e.g., "add", "multiply", "sqrt"
        operand: Union[float, int, List[List[Union[int, float]]]]
        target_self: Optional[bool] = False

    @app.post("/tensor/create", summary="Create or re-initialize the demo tensor.")
    async def create_tensor(tensor_input: TensorInput):
        """
        Re-initializes the global demo tensor with new data.
        使用新數據重新初始化全域演示張量。
        """
        global _demo_pto
        try:
            new_data = np.array(tensor_input.data)
            _demo_pto = PuzzleTensorOps(new_data, copy_data=tensor_input.copy_data)
            return {"message": "Tensor re-initialized successfully.", "shape": _demo_pto.shape, "dtype": str(_demo_pto.dtype)}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error creating tensor: {e}")

    @app.get("/tensor/view", summary="View the current demo tensor.")
    async def view_tensor():
        """
        Returns the current state of the global demo tensor.
        返回全域演示張量的當前狀態。
        """
        return {"shape": _demo_pto.shape, "dtype": str(_demo_pto.dtype), "data": _demo_pto.grid_view.tolist()}

    @app.post("/tensor/slice", summary="Get or set a slice of the demo tensor.")
    async def tensor_slice_operation(slice_input: SliceInput):
        """
        Get a slice or set values in a slice.
        To get a slice, provide `slicing_object_repr`.
        To set a slice, also provide `values`.
        獲取切片或在切片中設定值。
        若要獲取切片，請提供 `slicing_object_repr`。
        若要設定切片，同時提供 `values`。
        """
        try:
            # VERY basic parsing for demo. Real app needs robust slice parsing.
            # Example: "0,:" or "slice(None),0"
            # For a robust solution, ast.literal_eval or a dedicated parser is needed.
            # This demo version will be extremely limited.
            # Let's simplify: expect string like "r_start:r_stop,c_start:c_stop" for 2D
            parts = slice_input.slicing_object_repr.split(',')
            slices = []
            for part in parts:
                if ':' in part:
                    start, stop = map(lambda x: int(x) if x else None, part.split(':', 1))
                    slices.append(slice(start, stop))
                elif part == "...":
                    slices.append(...)
                else:
                    slices.append(int(part))
            slicing_obj = tuple(slices)

            if slice_input.values is not None:
                values_arr = np.array(slice_input.values)
                _demo_pto.set_slice(slicing_obj, values_arr)
                return {"message": "Slice set successfully.", "new_data": _demo_pto.grid_view.tolist()}
            else:
                result_slice = _demo_pto.get_slice(slicing_obj)
                return {"slice_data": result_slice.tolist(), "slice_shape": result_slice.shape}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error in slice operation: {str(e)}")

    @app.post("/tensor/elementwise", summary="Apply an element-wise operation.")
    async def tensor_elementwise_op(op_input: ElementwiseOpInput):
        """
        Applies np.add, np.subtract, np.multiply, np.divide, np.sqrt.
        應用 np.add, np.subtract, np.multiply, np.divide, np.sqrt。
        """
        op_map = {
            "add": np.add, "subtract": np.subtract, "multiply": np.multiply,
            "divide": np.divide, "sqrt": np.sqrt
        }
        if op_input.operation not in op_map:
            raise HTTPException(status_code=400, detail=f"Unsupported operation: {op_input.operation}")
        
        operation_func = op_map[op_input.operation]
        operand_val = np.array(op_input.operand) if isinstance(op_input.operand, list) else op_input.operand
        
        try:
            # If target_self, modify global _demo_pto
            if op_input.target_self:
                _demo_pto.apply_elementwise(operation_func, operand_val, target_self=True)
                return {"message": f"Operation '{op_input.operation}' applied in-place.", "new_data": _demo_pto.grid_view.tolist()}
            else:
                result_pto = _demo_pto.apply_elementwise(operation_func, operand_val, target_self=False)
                return {"result_data": result_pto.grid_view.tolist(), "result_shape": result_pto.shape}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error in elementwise operation: {str(e)}")

# --- Main execution for tests and benchmarks if run as script ---
if __name__ == "__main__":
    print(">>> Running new_module.py directly. Executing conceptual tests and benchmarks... <<<")
    
    print("\n--- Conceptual Unit Tests ---")
    # This will run all TestPuzzleTensorOps methods
    # In a real setup, you'd run `pytest tests/`
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPuzzleTensorOps))
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)
    
    if test_result.wasSuccessful():
        print("\n>>> ALL CONCEPTUAL UNIT TESTS PASSED <<<")
    else:
        print("\n>>> SOME CONCEPTUAL UNIT TESTS FAILED <<<")

    print("\n--- Conceptual Performance Benchmarks ---")
    # For more fine-grained control, you might pass specific shapes or ops
    run_puzzle_tensor_ops_benchmarks(shapes_to_test=[(50,50),(20,30,5)], number=50, repeat=2)

    print("\n--- FastAPI Demo Server (Conceptual) ---")
    print("If FastAPI and Uvicorn are installed, you can run the API demo server with:")
    print("  uvicorn new_module:app --reload --port 8000")
    print("Then open http://127.0.0.1:8000/docs in your browser.")
    print("Note: For this single-file version, the FastAPI app might run if imported.")
    print("In a production setup, the app definition and Uvicorn command would be separate.")
    print("\n>>> Direct execution finished. <<<")

