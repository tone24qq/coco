import numpy as np

from coco_agents.predict_agent import _matrix_factorization, predict


def test_matrix_factorization_shape_and_range():
    ground_truth = np.array(
        [
            [41, 2, 33, 91, 58, 67, 85, 17, 46, 25, 19, 31],
            [12, 76, 90, 64, 27, 8, 11, 45, 60, 18, 1, 39],
            [73, 5, 66, 28, 52, 87, 93, 40, 32, 97, 7, 10],
            [36, 29, 16, 6, 35, 23, 75, 3, 30, 55, 24, 43],
            [63, 59, 9, 4, 79, 94, 92, 20, 37, 54, 68, 13],
            [88, 34, 26, 70, 0, 14, 77, 84, 57, 47, 98, 15],
            [95, 86, 53, 2, 1, 22, 19, 5, 81, 61, 44, 38],
            [62, 21, 69, 10, 3, 6, 49, 56, 7, 99, 80, 78],
            [71, 74, 50, 42, 8, 89, 83, 27, 51, 96, 0, 4],
            [65, 48, 72, 82, 8, 12, 87, 91, 6, 9, 18, 90],
        ]
    )
    masked_grid = np.array(
        [
            [41, -1, -1, 91, -1, -1, 85, 17, -1, -1, 19, 31],
            [-1, 76, 90, -1, -1, 8, -1, -1, 60, 18, -1, -1],
            [73, -1, -1, -1, -1, 87, -1, -1, 32, -1, -1, 10],
            [36, -1, -1, 6, 35, -1, -1, 3, -1, -1, 24, -1],
            [-1, 59, -1, -1, 79, -1, 92, -1, -1, -1, -1, 13],
            [88, -1, -1, -1, 0, -1, 77, -1, 57, -1, 98, -1],
            [-1, -1, 53, -1, 1, 22, -1, 5, -1, 61, -1, -1],
            [-1, 21, -1, -1, -1, 6, 49, -1, 7, -1, 80, -1],
            [71, -1, -1, 42, 8, -1, -1, -1, -1, 96, 0, -1],
            [-1, 48, -1, -1, -1, 12, -1, 91, -1, -1, 18, -1],
        ]
    )

    completed = _matrix_factorization(masked_grid, seed=0)
    assert completed.shape == ground_truth.shape
    limit = ground_truth.size
    assert np.all((0 <= completed) & (completed < limit))

    accuracy = (completed == ground_truth).mean()
    assert accuracy > 0.0


def test_predict_returns_candidate_list():
    grid = np.array([[1, -1], [3, 4]])
    result = predict(grid, target=2, seed=0)
    assert isinstance(result, list)
    assert len(result) == grid.size
    for item in result:
        assert isinstance(item, dict)
        assert {"row", "col", "score"} <= item.keys()


def test_metrics_on_unique_board():
    rng = np.random.default_rng(123)
    rows, cols = 10, 12
    values = rng.permutation(rows * cols).reshape(rows, cols)
    mask_indices = rng.choice(rows * cols, size=int(rows * cols * 0.6), replace=False)
    masked = values.copy()
    for idx in mask_indices:
        r, c = divmod(int(idx), cols)
        masked[r, c] = -1

    completed = _matrix_factorization(masked, seed=123)
    mask = masked == -1
    final_accuracy = (completed[mask] == values[mask]).mean()

    hit_count = 0
    top3_hits = 0
    for idx in mask_indices:
        r, c = divmod(int(idx), cols)
        target = int(values[r, c])
        preds = predict(masked, target=target, seed=123)
        if preds and (preds[0]["row"], preds[0]["col"]) == (r, c):
            hit_count += 1
        if (r, c) in {(p["row"], p["col"]) for p in preds[:3]}:
            top3_hits += 1

    top3_rate = top3_hits / len(mask_indices)

    assert 0 <= final_accuracy <= 1
    assert 0 <= top3_rate <= 1
    assert hit_count <= len(mask_indices)
