# main.py (所有 22 個模組 analyze() 已補齊具體邏輯)
import random
import math # For math.hypot, math.fabs
import numpy as np
from typing import List, Dict, Tuple, Any

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

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        """
        分析給定盤面位置並回傳一個分數。
        此方法應由子類別覆寫以實現具體的評分邏輯。
        (此基礎版本不應再被直接呼叫，因為所有子類都將覆寫它)
        """
        print(f"警告: 基礎 LogicModule.analyze 被呼叫 (模組: {self.module_id})。應由子類覆寫。")
        return 0.0 # 預設為0，表示無特定邏輯

    def __repr__(self) -> str:
        return f"<LogicModule module_id='{self.module_id}' name='{self.name}'>"

class BoardInput:
    """
    代表盤面輸入的資料結構。
    """
    def __init__(self, grid: List[List[Any]]):
        if not grid or not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
            raise ValueError("盤面必須是一個非空的二維列表。")
        row_lengths = [len(row) for row in grid]
        if not row_lengths: # 空盤面檢查
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
# 2. 特定模組實現 (全部 22 個模組)
# -----------------------------------------------------------------------------

class A2(LogicModule):
    def __init__(self):
        super().__init__(
            module_id="A2",
            name="Alpha Module 2 (Proximity Scorer)",
            description="Scores based on proximity to '1' tiles (higher if adjacent or self is '1')."
        )

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 評估目標格子自身是否為'1'或是否與'1'相鄰。
        score = 0.1
        is_one_itself = False
        is_adjacent_to_one = False
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)

        if not rows or not cols: return 0.0 # 空盤面處理

        try:
            if board_state[position_row][position_col] == 1:
                is_one_itself = True
        except (TypeError, ValueError, IndexError): pass

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = position_row + dr, position_col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                try:
                    if board_state[nr][nc] == 1:
                        is_adjacent_to_one = True; break
                except (TypeError, ValueError): pass
        
        if is_adjacent_to_one: score = 0.9
        if is_one_itself: score = max(score, 0.5)
        return score

class M3(LogicModule):
    def __init__(self):
        super().__init__(
            module_id="M3",
            name="Mega Module 3 (Neighborhood Counter)",
            description="Scores based on count of '1's in 3x3 neighborhood."
        )

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 計算目標格子3x3鄰域內(含自身)'1'的數量佔比。
        count_of_ones = 0
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols: return 0.0

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = position_row + dr, position_col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    try:
                        if board_state[nr][nc] == 1: count_of_ones += 1
                    except (TypeError, ValueError): pass
        
        max_possible_ones = 9.0
        return count_of_ones / max_possible_ones if max_possible_ones > 0 else 0.0

class D3(LogicModule):
    def __init__(self):
        super().__init__(module_id="D3", name="Delta Module 3 (Center Proximity)", description="Scores higher if closer to the board center.")

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 評估格子與盤面中心的距離，越近分數越高。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols: return 0.0

        center_r, center_c = (rows - 1) / 2.0, (cols - 1) / 2.0
        distance = math.hypot(position_row - center_r, position_col - center_c)
        max_distance = math.hypot(center_r, center_c) # Distance from corner to center
        
        if max_distance == 0: return 1.0 # Single cell board
        score = 1.0 - (distance / max_distance)
        return max(0.0, min(1.0, score)) # Ensure score is [0,1]

class F10(LogicModule):
    def __init__(self):
        super().__init__(module_id="F10", name="Feature Module 10 (Edge/Corner Bonus)", description="Scores for being on an edge or corner.")

    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 格子若在邊緣則加分，在角落則加更多分。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols: return 0.0
        if rows == 1 and cols == 1: return 1.0 # Single cell is a corner

        is_edge = position_row == 0 or position_row == rows - 1 or \
                  position_col == 0 or position_col == cols - 1
        is_corner = (position_row == 0 and position_col == 0) or \
                    (position_row == 0 and position_col == cols - 1) or \
                    (position_row == rows - 1 and position_col == 0) or \
                    (position_row == rows - 1 and position_col == cols - 1)

        if is_corner: return 1.0
        if is_edge: return 0.6
        return 0.1

# --- GM Modules (GM1 to GM18) ---

class GM1(LogicModule):
    def __init__(self): super().__init__("GM1", "Generated Module 1 (Row Occupancy - Ones)", "Proportion of '1's in the cell's row.")
    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 評估目標格子所在行中'1'的佔比。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols or not (0 <= position_row < rows): return 0.0
        
        row_content = board_state[position_row]
        ones_in_row = sum(1 for cell in row_content if cell == 1)
        return ones_in_row / cols if cols > 0 else 0.0

class GM2(LogicModule):
    def __init__(self): super().__init__("GM2", "Generated Module 2 (Column Occupancy - Ones)", "Proportion of '1's in the cell's column.")
    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 評估目標格子所在列中'1'的佔比。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols or not (0 <= position_col < cols): return 0.0

        ones_in_col = sum(1 for r in range(rows) if board_state[r][position_col] == 1)
        return ones_in_col / rows if rows > 0 else 0.0

class GM3(LogicModule):
    def __init__(self): super().__init__("GM3", "Generated Module 3 (Neighborhood Emptiness - Zeros)", "Proportion of '0's in 3x3 neighborhood (excluding self).")
    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 評估目標格子3x3鄰域(不含自身)中'0'(空格)的密度。
        zeros_count = 0
        neighbors_count = 0
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols: return 0.0

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue # Exclude self
                nr, nc = position_row + dr, position_col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors_count += 1
                    try:
                        if board_state[nr][nc] == 0: zeros_count += 1
                    except (TypeError, ValueError): pass # Ignore non-comparable cells
        return zeros_count / neighbors_count if neighbors_count > 0 else 0.0

class GM4(LogicModule):
    def __init__(self): super().__init__("GM4", "Generated Module 4 (Neighborhood Sum of 1s)", "Sum of '1's in 3x3 neighborhood (scaled).")
    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 計算3x3鄰域內(含自身)'1'的總和，並正規化。
        # This is similar to M3 but M3 calculates proportion. GM4 will be sum / 9. So effectively same as M3.
        # Let's change GM4 to sum of numerical values (if not just 0 or 1) or average value.
        # For 0/1 case: Average value of 3x3 neighborhood.
        current_sum = 0
        count = 0
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols: return 0.0

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = position_row + dr, position_col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    count +=1
                    try: # Assuming cells are numeric, primarily 0 or 1 for these tests
                        if isinstance(board_state[nr][nc], (int, float)):
                           current_sum += board_state[nr][nc]
                    except TypeError: pass
        return (current_sum / count) if count > 0 else 0.0 # Average value in neighborhood

class GM5(LogicModule):
    def __init__(self): super().__init__("GM5", "Generated Module 5 (Horizontal Alternation)", "Favors 0-X-0 or 1-X-1 if X is different.")
    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 檢查水平方向是否形成交替模式 (如0-cell-0, 1-cell-1)。cell值與鄰居不同時加分。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols: return 0.0
        
        score = 0.5 # Base score
        cell_val = board_state[position_row][position_col]
        
        left_val, right_val = None, None
        if position_col > 0: left_val = board_state[position_row][position_col-1]
        if position_col < cols - 1: right_val = board_state[position_row][position_col+1]

        try:
            if left_val is not None and left_val != cell_val: score += 0.25
            elif left_val is None: score +=0.1 # edge case bonus
            if right_val is not None and right_val != cell_val: score += 0.25
            elif right_val is None: score +=0.1 # edge case bonus
        except TypeError: # Handle non-comparable types gracefully
            return 0.1 
        return min(1.0, score)

class GM6(LogicModule):
    def __init__(self): super().__init__("GM6", "Generated Module 6 (Vertical Alternation)", "Favors X-cell-Y pattern if cell is different from X and Y vertically.")
    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 檢查垂直方向是否形成交替模式。cell值與鄰居不同時加分。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols: return 0.0

        score = 0.5
        cell_val = board_state[position_row][position_col]

        up_val, down_val = None, None
        if position_row > 0: up_val = board_state[position_row-1][position_col]
        if position_row < rows - 1: down_val = board_state[position_row+1][position_col]

        try:
            if up_val is not None and up_val != cell_val: score += 0.25
            elif up_val is None: score += 0.1
            if down_val is not None and down_val != cell_val: score += 0.25
            elif down_val is None: score += 0.1
        except TypeError:
            return 0.1
        return min(1.0, score)

class GM7(LogicModule):
    def __init__(self): super().__init__("GM7", "Generated Module 7 (Isolation Score - for '1')", "If cell is '1', scores higher if surrounded by '0's (4-connectivity).")
    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 若目標格子為'1'，則其周圍(上下左右)的'0'越多，分數越高。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols: return 0.0

        cell_val = board_state[position_row][position_col]
        if cell_val != 1: return 0.1 # Only scores '1's

        surrounding_zeros = 0
        neighbor_count = 0
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = position_row + dr, position_col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbor_count +=1
                try:
                    if board_state[nr][nc] == 0: surrounding_zeros += 1
                except TypeError: pass
        
        return surrounding_zeros / neighbor_count if neighbor_count > 0 else 0.0


class GM8(LogicModule):
    def __init__(self): super().__init__("GM8", "Generated Module 8 (Main Diagonal Focus)", "If on main diagonal, score by '1's on main diagonal; else low score.")
    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 若格子在主對角線上，則基於主對角線上'1'的密度評分，否則低分。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols or rows != cols: return 0.1 # Only for square boards or give low score

        if position_row != position_col: return 0.1 # Not on main diagonal

        ones_on_diag = 0
        for i in range(rows):
            try:
                if board_state[i][i] == 1: ones_on_diag +=1
            except TypeError: pass
        return ones_on_diag / rows if rows > 0 else 0.0

class GM9(LogicModule):
    def __init__(self): super().__init__("GM9", "Generated Module 9 (Anti-Diagonal Focus)", "If on anti-diagonal, score by '1's on anti-diagonal; else low score.")
    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 若格子在反對角線上，則基於反對角線上'1'的密度評分，否則低分。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols or rows != cols: return 0.1 # Only for square boards or give low score

        if position_row + position_col != rows - 1: return 0.1 # Not on anti-diagonal

        ones_on_anti_diag = 0
        for i in range(rows):
            try:
                if board_state[i][rows - 1 - i] == 1: ones_on_anti_diag +=1
            except TypeError: pass
        return ones_on_anti_diag / rows if rows > 0 else 0.0

class GM10(LogicModule):
    def __init__(self): super().__init__("GM10", "Generated Module 10 (L-Shape Pattern of 1s)", "Scores if cell (as '1') forms L-shape with two other '1's.")
    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 檢查目標格子(假設為'1')是否能與其他'1'形成'L'型圖案。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols: return 0.0
        # This module scores based on the cell itself being 1 and forming an L
        # If the cell is not 1, it cannot form an L starting with itself as 1.
        # However, scoring modules usually evaluate a cell *if* a certain value (e.g. 1) is placed there.
        # For simplicity here, we check if board_state[pr][pc] IS 1.
        # Or, we can assume the cell *is* 1 for the purpose of this pattern check.
        # Let's assume board_state[pr][pc] is the value to check.

        if board_state[position_row][position_col] != 1: return 0.0 # Only consider if cell is already 1

        # Potential L-shape corners relative to (r,c) as the "joint" or an "arm end"
        # Let (r,c) be the joint of the L
        l_patterns = [
            [(1,0), (0,1)], [(-1,0), (0,1)], # L pointing down-right, up-right
            [(1,0), (0,-1)], [(-1,0), (0,-1)], # L pointing down-left, up-left
        ]
        score = 0.0
        for p_arm1, p_arm2 in l_patterns:
            r1, c1 = position_row + p_arm1[0], position_col + p_arm1[1]
            r2, c2 = position_row + p_arm2[0], position_col + p_arm2[1]
            
            val1, val2 = None, None
            if 0 <= r1 < rows and 0 <= c1 < cols: val1 = board_state[r1][c1]
            if 0 <= r2 < rows and 0 <= c2 < cols: val2 = board_state[r2][c2]

            if val1 == 1 and val2 == 1:
                score = 1.0; break 
        return score


class GM11(LogicModule):
    def __init__(self): super().__init__("GM11", "Generated Module 11 (Row Parity Match for '1')", "Score if cell is '1' and row index is even.")
    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 若目標格子為'1'且其行索引為偶數，則給予高分。
        if board_state[position_row][position_col] == 1 and position_row % 2 == 0:
            return 0.9
        elif board_state[position_row][position_col] == 1 and position_row % 2 != 0:
            return 0.3
        return 0.1

class GM12(LogicModule):
    def __init__(self): super().__init__("GM12", "Generated Module 12 (Column Parity Match for '1')", "Score if cell is '1' and col index is odd.")
    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 若目標格子為'1'且其列索引為奇數，則給予高分。
        if board_state[position_row][position_col] == 1 and position_col % 2 != 0:
            return 0.9
        elif board_state[position_row][position_col] == 1 and position_col % 2 == 0:
            return 0.3
        return 0.1

class GM13(LogicModule):
    def __init__(self): super().__init__("GM13", "Generated Module 13 (Quadrant Density of 1s)", "Density of '1's in cell's quadrant.")
    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 評估目標格子所在象限中'1'的密度。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols: return 0.0

        mid_r, mid_c = rows / 2, cols / 2
        
        q_r_start, q_r_end = (0, math.ceil(mid_r)) if position_row < mid_r else (math.floor(mid_r), rows)
        q_c_start, q_c_end = (0, math.ceil(mid_c)) if position_col < mid_c else (math.floor(mid_c), cols)

        ones_in_quadrant = 0
        quadrant_size = 0
        for r in range(int(q_r_start), int(q_r_end)):
            for c in range(int(q_c_start), int(q_c_end)):
                if 0 <= r < rows and 0 <= c < cols: # Ensure within bounds
                    quadrant_size +=1
                    try:
                        if board_state[r][c] == 1: ones_in_quadrant +=1
                    except TypeError: pass
        return ones_in_quadrant / quadrant_size if quadrant_size > 0 else 0.0


class GM14(LogicModule):
    def __init__(self): super().__init__("GM14", "Generated Module 14 (Checkerboard Adherence for '1')", "If cell is '1', scores how well it fits a checkerboard pattern.")
    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 若目標格子為'1'，評估其是否符合(row+col)為奇數的棋盤格模式。
        # (row+col) is odd for '1's, (row+col) is even for '0's, or vice versa.
        # Let's say '1's should be on (row+col) % 2 != 0 (for a typical black square start at 0,0)
        if board_state[position_row][position_col] == 1:
            return 0.9 if (position_row + position_col) % 2 != 0 else 0.2
        return 0.1 # if cell is not '1'


class GM15(LogicModule):
    def __init__(self): super().__init__("GM15", "Generated Module 15 (Distance to Nearest '1')", "Inverse distance to nearest '1'. High if self is '1'.")
    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 分數與最近的'1'的曼哈頓距離成反比；若自身為'1'則最高分。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols: return 0.0

        if board_state[position_row][position_col] == 1: return 1.0

        min_dist = float('inf')
        found_one = False
        for r in range(rows):
            for c in range(cols):
                if board_state[r][c] == 1:
                    found_one = True
                    dist = abs(position_row - r) + abs(position_col - c)
                    if dist < min_dist:
                        min_dist = dist
        
        if not found_one: return 0.0 # No '1's on board
        if min_dist == float('inf') : return 0.0 # Should not happen if found_one is true
        
        # Max possible distance is rows + cols. Scale score.
        max_possible_dist = rows + cols 
        return 1.0 - (min_dist / max_possible_dist) if max_possible_dist > 0 else 0.0

class GM16(LogicModule):
    def __init__(self): super().__init__("GM16", "Generated Module 16 (Row Symmetry for '1')", "If cell is '1', scores symmetry of its row.")
    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 若目標格子為'1'，評估其所在行是否水平對稱。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols: return 0.0
        
        # This module scores based on the cell being '1' and its row having symmetry
        if board_state[position_row][position_col] != 1: return 0.1

        current_row_values = board_state[position_row]
        is_symmetric = True
        for i in range(cols // 2):
            try:
                if current_row_values[i] != current_row_values[cols - 1 - i]:
                    is_symmetric = False; break
            except TypeError: is_symmetric = False; break # Non-comparable
        return 0.9 if is_symmetric else 0.2

class GM17(LogicModule):
    def __init__(self): super().__init__("GM17", "Generated Module 17 (Column Symmetry for '1')", "If cell is '1', scores symmetry of its column.")
    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 若目標格子為'1'，評估其所在列是否垂直對稱。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols: return 0.0

        if board_state[position_row][position_col] != 1: return 0.1

        is_symmetric = True
        for i in range(rows // 2):
            try:
                if board_state[i][position_col] != board_state[rows - 1 - i][position_col]:
                    is_symmetric = False; break
            except TypeError: is_symmetric = False; break
        return 0.9 if is_symmetric else 0.2

class GM18(LogicModule):
    def __init__(self): super().__init__("GM18", "Generated Module 18 (2x2 Subgrid Sum of 1s)", "Sum of '1's in 2x2 subgrid starting at cell (if possible), scaled.")
    def analyze(self, board_state: List[List[Any]], position_row: int, position_col: int) -> float:
        # 專長分流原則: 計算以目標格子為左上角的2x2子網格中'1'的數量佔比。
        rows, cols = len(board_state), len(board_state[0]) if board_state else (0,0)
        if not rows or not cols: return 0.0

        sum_2x2 = 0
        count_2x2 = 0
        # Check 2x2 starting from (position_row, position_col)
        for r_offset in range(2):
            for c_offset in range(2):
                r, c = position_row + r_offset, position_col + c_offset
                if 0 <= r < rows and 0 <= c < cols:
                    count_2x2 += 1
                    try:
                        if board_state[r][c] == 1: sum_2x2 +=1
                    except TypeError: pass
        
        if count_2x2 < 4 : return 0.1 # Not a full 2x2 grid possible from this cell as top-left (or partially off board)
        return sum_2x2 / 4.0 # Max sum is 4 for '1's

# -----------------------------------------------------------------------------
# 3. 模듈註冊與全局權重
# -----------------------------------------------------------------------------
REGISTERED_MODULES: List[LogicModule] = [
    A2(), M3(), D3(), F10(),
    GM1(), GM2(), GM3(), GM4(), GM5(), GM6(), GM7(), GM8(), GM9(),
    GM10(), GM11(), GM12(), GM13(), GM14(), GM15(), GM16(), GM17(), GM18()
]

print(f"Registered {len(REGISTERED_MODULES)} modules:")
for mod in REGISTERED_MODULES:
    print(f" - {mod.module_id}: {mod.name}")

GLOBAL_MODULE_WEIGHTS: Dict[str, float] = {module.module_id: 1.0 for module in REGISTERED_MODULES}
if "A2" in GLOBAL_MODULE_WEIGHTS: GLOBAL_MODULE_WEIGHTS["A2"] = 2.0
if "M3" in GLOBAL_MODULE_WEIGHTS: GLOBAL_MODULE_WEIGHTS["M3"] = 1.5
if "D3" in GLOBAL_MODULE_WEIGHTS: GLOBAL_MODULE_WEIGHTS["D3"] = 0.5 # Example weight

# -----------------------------------------------------------------------------
# 4. 核心處理邏輯 (與之前版本相同)
# -----------------------------------------------------------------------------

def process_board(board_input: BoardInput, modules: List[LogicModule]) -> Dict[Tuple[int, int], Dict[str, float]]:
    all_cell_scores: Dict[Tuple[int, int], Dict[str, float]] = {}
    if not board_input.grid: return all_cell_scores # Handle empty board
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
            if mod_id in module_all_scores: # Check if module_id is expected
                 module_all_scores[mod_id].append(score)

    for mod_id_key, scores_list in module_all_scores.items():
        if not scores_list:
            for cell_pos_norm in normalized_scores_by_cell: # Ensure all cells have this module_id
                normalized_scores_by_cell[cell_pos_norm][mod_id_key] = 0.0
            continue

        if method == 'min-max':
            min_score, max_score = min(scores_list), max(scores_list)
            for cell_pos_norm in normalized_scores_by_cell:
                raw_score = module_scores_by_cell.get(cell_pos_norm, {}).get(mod_id_key)
                if raw_score is not None:
                    if (max_score - min_score) == 0:
                        normalized_scores_by_cell[cell_pos_norm][mod_id_key] = 0.0 if min_score == 0 else 0.5
                    else:
                        normalized_scores_by_cell[cell_pos_norm][mod_id_key] = (raw_score - min_score) / (max_score - min_score)
                else: # If a cell didn't have this module's score originally
                    normalized_scores_by_cell[cell_pos_norm][mod_id_key] = 0.0 # Default to 0 after normalization
        elif method == 'z-score':
            mean_score, std_score = float(np.mean(scores_list)), float(np.std(scores_list))
            for cell_pos_norm in normalized_scores_by_cell:
                raw_score = module_scores_by_cell.get(cell_pos_norm, {}).get(mod_id_key)
                if raw_score is not None:
                    if std_score == 0:
                        normalized_scores_by_cell[cell_pos_norm][mod_id_key] = 0.0
                    else:
                        normalized_scores_by_cell[cell_pos_norm][mod_id_key] = (raw_score - mean_score) / std_score
                else:
                    normalized_scores_by_cell[cell_pos_norm][mod_id_key] = 0.0
        else: # 'none' or unknown
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
            effective_weights[mod_id_in_score] = 1.0

    for cell_pos, mod_scores in scores_to_fuse_input.items():
        weighted_sum, sum_of_weights = 0.0, 0.0
        if not mod_scores:
            fused_scores_output[cell_pos] = 0.0; continue

        for module_id, norm_score in mod_scores.items():
            weight = effective_weights.get(module_id, 1.0)
            weighted_sum += norm_score * weight
            sum_of_weights += weight
        
        fused_scores_output[cell_pos] = (weighted_sum / sum_of_weights) if sum_of_weights != 0 else 0.0
    return fused_scores_output

def simple_fuse_scores(
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
# 5. 主執行區塊 (依照文件各 Section 進行)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    test_board_1_data = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    test_board_1 = BoardInput(grid=test_board_1_data)

    print("\n===== 第 1 節：模組類別骨架的自動化生成 =====")
    print("此版本中所有模組定義和實例化已整合到 main.py。")
    print(f"已註冊 {len(REGISTERED_MODULES)} 個模組。應為 22 個。")
    if len(REGISTERED_MODULES) != 22:
        print(f"警告: 實際註冊模組數 {len(REGISTERED_MODULES)} 不符預期!")

    print("\n===== 第 2 節：初始系統整合與最小盤面測試 =====")
    print(f"\n處理最小測試盤面 (test_board_1) 使用 {len(REGISTERED_MODULES)} 個模組 (具體邏輯)...")
    test_board_1.display()
    
    raw_scores_board1_sec2 = process_board(test_board_1, REGISTERED_MODULES)
    fused_scores_board1_simple = simple_fuse_scores(raw_scores_board1_sec2)

    print("\n最小測試盤面 (test_board_1) 的初步融合分數 (簡單平均):")
    print("| 格子座標 (列, 行) | 初步融合分數 (簡單平均) |")
    print("|---|---|")
    for r_idx in range(test_board_1.rows):
        for c_idx in range(test_board_1.cols):
            pos = (r_idx, c_idx)
            score_val = fused_scores_board1_simple.get(pos, float('nan'))
            print(f"| ({r_idx},{c_idx}) | {score_val:.4f} |")
    # 額外印出各模組原始分數以觀察差異
    print("\n部分模組在 test_board_1 (0,0) 的原始分數 (僅供參考，確認差異性):")
    if (0,0) in raw_scores_board1_sec2:
        for mod_id, score in list(raw_scores_board1_sec2[(0,0)].items())[:5]: # Show first 5
            print(f"  Mod {mod_id}: {score:.4f}")
    print("\n")

    print("\n===== 第 3 節：實作特定專長評分邏輯 =====")
    print(f"\n所有模組 (A2, M3 及 GM1-18 等) 現已使用具體邏輯。")
    test_board_1.display()

    # raw_scores_board1_mixed_logic = process_board(test_board_1, REGISTERED_MODULES) # Re-use from sec2
    fused_scores_board1_mixed_simple = simple_fuse_scores(raw_scores_board1_sec2) 

    print("\n使用全部具體邏輯模組的初步融合分數 (簡單平均):")
    print("以及 A2, M3, GM1, GM3 的原始分數 (範例):")
    print("| 格子 (R,C) | A2    | M3    | GM1   | GM3   | Fused (簡單平均) |")
    print("|---|-------|-------|-------|-------|-----------------|")
    for r_idx in range(test_board_1.rows):
        for c_idx in range(test_board_1.cols):
            pos = (r_idx, c_idx)
            s_a2 = raw_scores_board1_sec2.get(pos, {}).get("A2", float('nan'))
            s_m3 = raw_scores_board1_sec2.get(pos, {}).get("M3", float('nan'))
            s_gm1 = raw_scores_board1_sec2.get(pos, {}).get("GM1", float('nan'))
            s_gm3 = raw_scores_board1_sec2.get(pos, {}).get("GM3", float('nan'))
            fused_s = fused_scores_board1_mixed_simple.get(pos, float('nan'))
            print(f"| ({r_idx},{c_idx}) | {s_a2:.2f}  | {s_m3:.2f}  | {s_gm1:.2f}  | {s_gm3:.2f}  | {fused_s:.4f}        |")
    print("\n")

    print("\n===== 第 4 節：優化分數融合流程 =====")
    print("\nGLOBAL_MODULE_WEIGHTS (部分範例):")
    for mod_id_gw, weight_gw in list(GLOBAL_MODULE_WEIGHTS.items())[:4]: # Show first 4
         print(f"  {mod_id_gw}: {weight_gw}")
    print("  ... (其餘模組權重按GLOBAL_MODULE_WEIGHTS中的定義)")

    print(f"\n處理 test_board_1，使用 Min-Max 正規化和加權平均:")
    test_board_1.display()
    
    fused_scores_b1_adv_minmax, norm_scores_b1_minmax = get_final_scores_for_board(
        test_board_1, REGISTERED_MODULES, GLOBAL_MODULE_WEIGHTS, normalization_method='min-max'
    )

    print("\n使用 Min-Max 正規化和加權平均後的融合分數 (test_board_1):")
    print("| 格子 (R,C) | Norm_A2 | Norm_M3 | Norm_GM1 | Fused (Min-Max, Weighted) |")
    print("|---|---------|---------|----------|--------------------------|")
    for r_idx in range(test_board_1.rows):
        for c_idx in range(test_board_1.cols):
            pos = (r_idx, c_idx)
            final_s = fused_scores_b1_adv_minmax.get(pos, float('nan'))
            a2_n_s = norm_scores_b1_minmax.get(pos, {}).get("A2", float('nan'))
            m3_n_s = norm_scores_b1_minmax.get(pos, {}).get("M3", float('nan'))
            gm1_n_s = norm_scores_b1_minmax.get(pos, {}).get("GM1", float('nan'))
            print(f"| ({r_idx},{c_idx}) | {a2_n_s:.2f}    | {m3_n_s:.2f}    | {gm1_n_s:.2f}     | {final_s:.4f}                   |")

    print(f"\n處理 test_board_1，使用 Z-Score 正規化和加權平均:")
    fused_scores_b1_adv_zscore, norm_scores_b1_zscore = get_final_scores_for_board(
        test_board_1, REGISTERED_MODULES, GLOBAL_MODULE_WEIGHTS, normalization_method='z-score'
    )
    print("\n使用 Z-Score 正規化和加權平均後的融合分數 (test_board_1):")
    print("| 格子 (R,C) | Norm_A2 (Z) | Norm_M3 (Z) | Norm_GM1 (Z)| Fused (Z-Score, Weighted) |")
    print("|---|-------------|-------------|-------------|---------------------------|")
    for r_idx in range(test_board_1.rows):
        for c_idx in range(test_board_1.cols):
            pos = (r_idx, c_idx)
            final_s_z = fused_scores_b1_adv_zscore.get(pos, float('nan'))
            a2_n_s_z = norm_scores_b1_zscore.get(pos, {}).get("A2", float('nan'))
            m3_n_s_z = norm_scores_b1_zscore.get(pos, {}).get("M3", float('nan'))
            gm1_n_s_z = norm_scores_b1_zscore.get(pos, {}).get("GM1", float('nan'))
            print(f"| ({r_idx},{c_idx}) | {a2_n_s_z:+.2f}       | {m3_n_s_z:+.2f}       | {gm1_n_s_z:+.2f}      | {final_s_z:.4f}                    |")
    print("\n")

    print("\n===== 第 5 節：多樣化測試情境的綜合驗證 =====")
    test_board_2_data = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    test_board_2 = BoardInput(grid=test_board_2_data)
    test_board_3_data = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    test_board_3 = BoardInput(grid=test_board_3_data)
    
    test_boards = {
        "Board 1 (Minimal)": test_board_1, "Board 2 (A2 Favored)": test_board_2, "Board 3 (M3 Favored/Complex)": test_board_3
    }
    results_all_boards: Dict[str, Dict[Tuple[int, int], float]] = {}
    normalized_module_scores_all_boards: Dict[str, Dict[Tuple[int, int], Dict[str, float]]] = {}

    for board_name, board_obj in test_boards.items():
        print(f"\n--- 處理 {board_name} ---")
        board_obj.display()
        fused_scores, normalized_scores = get_final_scores_for_board(
            board_obj, REGISTERED_MODULES, GLOBAL_MODULE_WEIGHTS, normalization_method='min-max'
        )
        results_all_boards[board_name] = fused_scores
        normalized_module_scores_all_boards[board_name] = normalized_scores

        print(f"\n{board_name} 的融合分數 (Min-Max, 加權) 及部分正規化模組分數:")
        print("| 格子 (R,C) | Norm_A2 | Norm_M3 | Norm_GM1 | Fused (Min-Max, Weighted) |")
        print("|---|---------|---------|----------|--------------------------|")
        for r_idx in range(board_obj.rows):
            for c_idx in range(board_obj.cols):
                pos = (r_idx, c_idx)
                final_s_5 = fused_scores.get(pos, float('nan'))
                a2_n_s_5 = normalized_scores.get(pos, {}).get("A2", float('nan'))
                m3_n_s_5 = normalized_scores.get(pos, {}).get("M3", float('nan'))
                gm1_n_s_5 = normalized_scores.get(pos, {}).get("GM1", float('nan'))
                print(f"| ({r_idx},{c_idx}) | {a2_n_s_5:.2f}    | {m3_n_s_5:.2f}    | {gm1_n_s_5:.2f}     | {final_s_5:.4f}                   |")

    print("\nAPI 輸出範例 (Board 2 的融合分數):")
    example_api_output_board2 = results_all_boards.get("Board 2 (A2 Favored)", {})
    if example_api_output_board2:
        for pos_api, score_api in example_api_output_board2.items(): print(f"  Cell {pos_api}: {score_api:.4f}")
    else: print("  (Board 2 結果未找到)")

    print("\n表 5：跨多樣化測試盤面的代表性格子融合分數比較 (Min-Max 正規化, 加權平均)")
    print("| 格子座標 (列, 行) | 融合分數 (盤面 1) | 融合分數 (盤面 2) | 融合分數 (盤面 3) |")
    print("|---|-------------------|-------------------|-------------------|")
    for r_ex, c_ex in [(0,0), (1,1), (2,2)]: # Example coordinates
        pos_ex = (r_ex, c_ex)
        s1 = results_all_boards.get("Board 1 (Minimal)", {}).get(pos_ex, float('nan'))
        s2 = results_all_boards.get("Board 2 (A2 Favored)", {}).get(pos_ex, float('nan'))
        s3 = results_all_boards.get("Board 3 (M3 Favored/Complex)", {}).get(pos_ex, float('nan'))
        print(f"| ({r_ex},{c_ex}) | {s1: .4f}            | {s2: .4f}            | {s3: .4f}            |")
    print("(註：實際數值會因所有模組的具體邏輯而定，應觀察到分數差異。)\n")

    print("\n===== 第 6 節：最終程式碼結構、註釋與執行指南 =====")
    print("程式碼結構：所有內容已整合至此單一 main.py 檔案。")
    print("註釋與文檔字串已加入。")
    print("執行指南：")
    print("1. 確認 Python 版本 (建議 3.8+)。")
    print("2. 安裝必要函式庫: pip install numpy (如果尚未安裝)。")
    print("3. 將此完整程式碼儲存為 main.py。")
    print("4. 執行主程式: python main.py")
    print("5. 預期輸出將依照 Section 2 至 5 的內容逐步顯示，展示各模組對不同盤面特徵的評分差異。")

    print("\n結論：**所有模組 analyze() 已補齊** 具體的、有專長分流能力的邏輯。")
    print("後續建議：可進一步細化各模組邏輯、調整權重、擴充更複雜的測試盤面進行驗證。")

