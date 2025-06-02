# new_module.py

import numpy as np
import math
from typing import Tuple, List, Optional, Dict, Any
from collections import deque

try:
    import numba
    from numba import njit
except ImportError:
    numba = None

# 导入所有 GM 模块函数
from brain1 import (
    EXT_GM1_Proximity_Vec,
    EXT_GM2_Heterogeneity_Vec,
    EXT_GM3_PotentialField_Vec,
    BaseModuleConfig as GM1_3_Config,
)
from brain2 import (
    EXT_GM4_Spatial_Auto_Corr_Vec,
    EXT_GM5_Line_Completion_Vec,
    EXT_GM6_Symmetry_Potential_Vec,
    EXT_GM7_Numeric_Gaps_Vec,
    EXT_GM8_Edge_Affinity_Vec,
    EXT_GM9_Center_Control_Vec,
    EXT_GM10_BlockingValue_Vec,
    EXT_GM11_PairCorrelation_Vec,
    EXT_GM12_IslandAnalysis_Vec,
    BaseModuleConfig as GM4_12_Config,
)
from brain3 import (
    EXT_GM13_Sequence_Diversity_Vec,
    EXT_GM14_Risk_Assessment_Vec,
    EXT_GM15_Information_Gain_Vec,
    EXT_GM16_Harmonic_Centrality_Vec,
    EXT_GM17_Local_Entropy_Vec,
    EXT_GM18_RL_Value_Estimation_Vec,
    EXT_GM19_SkipPattern_Vec,
    EXT_GM20_SkipPattern_Confidence_Vec,
    EXT_GM21_ClusterBalance_Vec,
    EXT_GM22_CoOccurrence_Vec,
    EXT_GM23_MotifDetection_Vec,
    EXT_GM24_TemporalCoherence_Vec,
    EXT_GM25_StrategicDepth_Vec,
    EXT_GM26_ContextualFlexibility_Vec,
    BaseModuleConfig as GM13_26_Config,
)


class PuzzleTensorOps:
    """
    通用 N 维张量操作模块，结合 NumPy 和可选的 Numba 加速，用于拼图引擎和高性能计算。
    """

    def __init__(self, tensor: np.ndarray):
        """
        tensor: 原始二维刮卡盘面，dtype=int，空格标记为 -1
        """
        self.tensor = tensor
        self.rows, self.cols = tensor.shape

    def get_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回两个 shape=(rows, cols) 的 ndarray，分别表示行索引矩阵和列索引矩阵
        """
        return np.indices(self.tensor.shape)

    def get_positions(self, mask: np.ndarray) -> np.ndarray:
        """
        mask: bool 型 ndarray，True 表示要收集的位置
        返回 shape=(N,2) ndarray，每行是 (r, c) 坐标
        """
        coords = np.stack(np.where(mask), axis=1)
        return coords.astype(int)

    def manhattan_distance_matrix(
        self,
        rr: np.ndarray,
        cc: np.ndarray,
        r_rep: np.ndarray,
        c_rep: np.ndarray,
    ) -> np.ndarray:
        """
        计算 shape=(rows, cols, N) 的曼哈顿距离张量
        [i,j,k] = |i - r_rep[k]| + |j - c_rep[k]|
        rr, cc: np.indices 返回的行/列坐标矩阵，shape=(rows, cols)
        r_rep, c_rep: 一维数组，表示已揭露或目标位置的坐标列表
        """
        rr_exp = rr[..., None]        # (rows, cols, 1)
        cc_exp = cc[..., None]
        r_rep_exp = r_rep[None, None, :]  # (1, 1, N)
        c_rep_exp = c_rep[None, None, :]
        return np.abs(rr_exp - r_rep_exp) + np.abs(cc_exp - c_rep_exp)

    def sum_along_axis(self, tensor: np.ndarray, axis: int) -> np.ndarray:
        """
        对 ndarray 在指定 axis 上求和
        """
        return np.sum(tensor, axis=axis)

    def connected_component_sizes(self, mask: np.ndarray) -> np.ndarray:
        """
        计算二维 mask 下的连通分量大小（只考虑 4 邻域）。
        返回 shape=(rows, cols) ndarray，若 mask[i,j] 为 True，则值为对应连通簇的大小；否则为 0。
        如果安装了 numba，会调用 JIT 加速版本；否则，用纯 Python 实现。
        """
        if numba is not None:
            return self._ccs_numba(mask)
        else:
            return self._ccs_python(mask)

    def _ccs_python(self, mask: np.ndarray) -> np.ndarray:
        rows, cols = self.rows, self.cols
        sizes = np.zeros((rows, cols), dtype=int)
        visited = np.zeros((rows, cols), dtype=bool)

        for i in range(rows):
            for j in range(cols):
                if mask[i, j] and not visited[i, j]:
                    queue = deque([(i, j)])
                    component = [(i, j)]
                    visited[i, j] = True
                    while queue:
                        r, c = queue.popleft()
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if (
                                0 <= nr < rows
                                and 0 <= nc < cols
                                and mask[nr, nc]
                                and not visited[nr, nc]
                            ):
                                visited[nr, nc] = True
                                component.append((nr, nc))
                                queue.append((nr, nc))
                    comp_size = len(component)
                    for (r0, c0) in component:
                        sizes[r0, c0] = comp_size
        return sizes

    def _ccs_numba(self, mask: np.ndarray) -> np.ndarray:
        """
        numba 加速的连通分量。由于 numba 不支持动态列表，这里用固定大小队列模拟 BFS。
        """
        rows, cols = self.rows, self.cols

        @njit
        def bfs_all(mask_np):
            sizes_np = np.zeros((rows, cols), dtype=np.int64)
            visited_np = np.zeros((rows, cols), dtype=np.bool_)
            for i0 in range(rows):
                for j0 in range(cols):
                    if mask_np[i0, j0] and not visited_np[i0, j0]:
                        queue_r = np.empty(rows * cols, dtype=np.int64)
                        queue_c = np.empty(rows * cols, dtype=np.int64)
                        head = 0
                        tail = 0
                        queue_r[tail] = i0
                        queue_c[tail] = j0
                        visited_np[i0, j0] = True
                        tail += 1
                        comp_size = 0
                        while head < tail:
                            r = queue_r[head]
                            c = queue_c[head]
                            head += 1
                            comp_size += 1
                            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                                nr = r + dr
                                nc = c + dc
                                if (
                                    0 <= nr < rows
                                    and 0 <= nc < cols
                                    and mask_np[nr, nc]
                                    and not visited_np[nr, nc]
                                ):
                                    visited_np[nr, nc] = True
                                    queue_r[tail] = nr
                                    queue_c[tail] = nc
                                    tail += 1
                        # 第二次遍历组件，将 comp_size 写入 sizes_np
                        head2 = 0
                        tail2 = 0
                        queue_r2 = np.empty(rows * cols, dtype=np.int64)
                        queue_c2 = np.empty(rows * cols, dtype=np.int64)
                        visited2 = np.zeros((rows, cols), dtype=np.bool_)
                        queue_r2[tail2] = i0
                        queue_c2[tail2] = j0
                        visited2[i0, j0] = True
                        tail2 += 1
                        while head2 < tail2:
                            r2 = queue_r2[head2]
                            c2 = queue_c2[head2]
                            head2 += 1
                            sizes_np[r2, c2] = comp_size
                            for dr2, dc2 in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                                nr2 = r2 + dr2
                                nc2 = c2 + dc2
                                if (
                                    0 <= nr2 < rows
                                    and 0 <= nc2 < cols
                                    and mask_np[nr2, nc2]
                                    and not visited2[nr2, nc2]
                                ):
                                    visited2[nr2, nc2] = True
                                    queue_r2[tail2] = nr2
                                    queue_c2[tail2] = nc2
                                    tail2 += 1
            return sizes_np

        return bfs_all(mask)

    def local_entropy(self, radius: int = 1) -> np.ndarray:
        """
        计算以每个空格为中心、半径为 radius 邻域内的香农熵，返回 shape=(rows, cols) 的 ndarray。
        已填格位置返回 0。
        """
        rows, cols = self.rows, self.cols
        ent = np.zeros((rows, cols), dtype=float)
        for r in range(rows):
            for c in range(cols):
                if self.tensor[r, c] != -1:
                    continue
                values = []
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            val = self.tensor[nr, nc]
                            if val != -1:
                                values.append(val)
                if not values:
                    ent[r, c] = 0.0
                else:
                    counts = {}
                    total = float(len(values))
                    for v in values:
                        counts[v] = counts.get(v, 0) + 1
                    e = 0.0
                    for cnt in counts.values():
                        p = cnt / total
                        e -= p * math.log2(p) if p > 0 else 0.0
                    ent[r, c] = e
        return ent

    def score_full_board(
        self,
        configs: Dict[str, Any],
        request_id: Optional[str] = "in-tensor"
    ) -> np.ndarray:
        """
        对整张盘面使用 GM1–GM26 全模块打分，返回 shape=(rows, cols) 的合并归一后分数矩阵。
        configs: {
            'GM1': BaseModuleConfig(...),
            'GM2': BaseModuleConfig(...),
            ...
            'GM26': BaseModuleConfig(...)
        }
        """
        rows, cols = self.rows, self.cols
        combined = np.zeros((rows, cols), dtype=float)

        # 定义模块顺序及对应函数
        modules = [
            ("GM1", EXT_GM1_Proximity_Vec),
            ("GM2", EXT_GM2_Heterogeneity_Vec),
            ("GM3", EXT_GM3_PotentialField_Vec),
            ("GM4", EXT_GM4_Spatial_Auto_Corr_Vec),
            ("GM5", EXT_GM5_Line_Completion_Vec),
            ("GM6", EXT_GM6_Symmetry_Potential_Vec),
            ("GM7", EXT_GM7_Numeric_Gaps_Vec),
            ("GM8", EXT_GM8_Edge_Affinity_Vec),
            ("GM9", EXT_GM9_Center_Control_Vec),
            ("GM10", EXT_GM10_BlockingValue_Vec),
            ("GM11", EXT_GM11_PairCorrelation_Vec),
            ("GM12", EXT_GM12_IslandAnalysis_Vec),
            ("GM13", EXT_GM13_Sequence_Diversity_Vec),
            ("GM14", EXT_GM14_Risk_Assessment_Vec),
            ("GM15", EXT_GM15_Information_Gain_Vec),
            ("GM16", EXT_GM16_Harmonic_Centrality_Vec),
            ("GM17", EXT_GM17_Local_Entropy_Vec),
            ("GM18", EXT_GM18_RL_Value_Estimation_Vec),
            ("GM19", EXT_GM19_SkipPattern_Vec),
            ("GM20", EXT_GM20_SkipPattern_Confidence_Vec),
            ("GM21", EXT_GM21_ClusterBalance_Vec),
            ("GM22", EXT_GM22_CoOccurrence_Vec),
            ("GM23", EXT_GM23_MotifDetection_Vec),
            ("GM24", EXT_GM24_TemporalCoherence_Vec),
            ("GM25", EXT_GM25_StrategicDepth_Vec),
            ("GM26", EXT_GM26_ContextualFlexibility_Vec),
        ]

        # 依次调用各模块，将各自分数累加到 combined
        for name, func in modules:
            cfg = configs.get(name)
            try:
                partial = func(self.tensor, cfg, request_id)
                if not isinstance(partial, np.ndarray) or partial.shape != (rows, cols):
                    raise ValueError(f"{name} 返回形状错误: {partial.shape}")
            except Exception:
                partial = np.zeros((rows, cols), dtype=float)
            combined += partial

        # 对所有空格位置做全图归一化
        mask_empty = (self.tensor == -1)
        if np.any(mask_empty):
            mn = float(np.min(combined[mask_empty]))
            mx = float(np.max(combined[mask_empty]))
            if not math.isclose(mn, mx):
                normalized = (combined - mn) / (mx - mn)
            else:
                normalized = np.zeros_like(combined)
            combined = np.where(mask_empty, normalized, 0.0)
        else:
            combined[:] = 0.0

        return combined