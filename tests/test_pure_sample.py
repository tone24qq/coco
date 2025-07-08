import json
import zipfile

import analyzer


def test_pure_sample_branch(tmp_path):
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    samples = tmp_path / "samples"
    samples.mkdir()
    b1 = {"rows": 2, "cols": 2, "grid": [[1, 2], [3, 4]]}
    b2 = {"rows": 2, "cols": 2, "grid": [[1, 3], [2, 4]]}
    with zipfile.ZipFile(samples / "s.zip", "w") as zf:
        zf.writestr("a.json", json.dumps(b1))
        zf.writestr("b.json", json.dumps(b2))

    grid = [[1, 2], [3, -1]]
    res = analyzer.predict_scratch_card(grid, target_num=4, history_dir=str(samples))
    assert res["mode"] == "sample_only"
    assert res["strategy"] == "pure_sample+global"
    assert res["predictions"][0]["row"] == 1
    assert res["predictions"][0]["col"] == 1
    assert abs(res["predictions"][0]["score"] - 1.0) < 1e-6


def test_pure_sample_neighbor_ranking(tmp_path):
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    samples = tmp_path / "samples"
    samples.mkdir()
    board = {"rows": 3, "cols": 3, "grid": [[1, 2, 2], [3, 4, 4], [5, 6, 7]]}
    with zipfile.ZipFile(samples / "s.zip", "w") as zf:
        zf.writestr("b.json", json.dumps(board))

    grid = [[1, -1, -1], [3, 4, -1], [5, 6, 7]]
    res = analyzer.predict_scratch_card(
        grid, target_num=2, history_dir=str(samples), top_n=2
    )
    preds = res["predictions"]
    assert preds[0]["row"] == 0 and preds[0]["col"] == 1
    assert preds[1]["row"] == 0 and preds[1]["col"] == 2


def test_pure_sample_final_score_weighting(tmp_path):
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    samples = tmp_path / "samples"
    samples.mkdir()
    board = {"rows": 2, "cols": 2, "grid": [[1, 2], [3, 4]]}
    with zipfile.ZipFile(samples / "s.zip", "w") as zf:
        zf.writestr("b.json", json.dumps(board))

    grid = [[1, -1], [3, -1]]
    res = analyzer.predict_scratch_card(
        grid, target_num=4, history_dir=str(samples), top_n=2
    )
    preds = res["predictions"]
    assert preds[0]["row"] == 1 and preds[0]["col"] == 1
    assert preds[1]["row"] == 0 and preds[1]["col"] == 1
    assert abs(preds[0]["score"] - 1.0) < 1e-6
    assert abs(preds[1]["score"] - 0.8) < 1e-6


def test_neighbor_relaxed_matching(tmp_path):
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    samples = tmp_path / "samples"
    samples.mkdir()
    board = {"rows": 3, "cols": 3, "grid": [[9, 2, 3], [4, 5, 6], [7, 8, 1]]}
    with zipfile.ZipFile(samples / "s.zip", "w") as zf:
        zf.writestr("b.json", json.dumps(board))

    grid = [[1, 2, -1], [4, 5, -1], [7, 8, -1]]
    res = analyzer.predict_scratch_card(grid, target_num=3, history_dir=str(samples))
    assert res["mode"] == "sample_only"
    assert res["strategy"] == "pure_sample+global"
    pred = res["predictions"][0]
    assert pred["row"] == 0 and pred["col"] == 2
