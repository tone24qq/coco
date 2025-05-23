from celery import Celery
import numpy as np
from ortools.sat.python import cp_model
import os

app = Celery('tasks', broker='redis://localhost:6379/0')

def tensor_flow_score_vec_all(grid: np.ndarray) -> np.ndarray:
    # 範例空實作，正式用時請與 main.py 同步規則改寫
    return np.zeros(grid.shape, dtype=float)

def mem_score(r, c, v, legal_values):
    return 0.0  # 可自行擴充記憶體分數

def build_and_solve_cp_vec(grid: np.ndarray, candidates, legal_values):
    model = cp_model.CpModel()
    rows, cols = grid.shape
    x = [model.NewBoolVar(f"x_{i}") for i in range(len(candidates))]
    model.Add(sum(x) == 1)

    for i, (r, c, v) in enumerate(candidates):
        if v not in legal_values:
            model.Add(x[i] == 0)
    for i in range(len(candidates)):
        vi = candidates[i][2]
        for j in range(i+1, len(candidates)):
            vj = candidates[j][2]
            if vi == vj:
                model.Add(x[i] + x[j] <= 1)

    scores = tensor_flow_score_vec_all(grid)
    weights = []
    tensor_scores = []

    for i, (r, c, v) in enumerate(candidates):
        score = scores[r, c]
        score += 5.0 * mem_score(r, c, v, legal_values)
        tensor_scores.append(scores[r, c])
        weights.append(int(score * 1000))

    obj = sum(x[i] * weights[i] for i in range(len(candidates)))
    model.Maximize(obj)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0.5
    solver.parameters.num_search_workers = os.cpu_count() or 1
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return []

    best = []
    for i in range(len(candidates)):
        if solver.Value(x[i]):
            best.append((candidates[i][0], candidates[i][1], candidates[i][2], weights[i] / 1000.0, tensor_scores[i]))
    return best

@app.task
def solve_task(grid_list, candidates_list, legal_values_list):
    grid = np.array(grid_list, dtype=int)
    candidates = [tuple(x) for x in candidates_list]
    legal_values = set(legal_values_list)
    return build_and_solve_cp_vec(grid, candidates, legal_values)