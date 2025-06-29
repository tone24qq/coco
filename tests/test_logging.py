import json
import logging
import zipfile

import numpy as np

import analyzer


def test_iter_sample_jsons_logging(tmp_path, caplog):
    samples = tmp_path / "samples"
    samples.mkdir()
    data = {"rows": 2, "cols": 2, "grid": [[1, 2], [3, 4]]}
    with zipfile.ZipFile(samples / "a.zip", "w") as zf:
        zf.writestr("d.json", json.dumps(data))
    with caplog.at_level(logging.INFO):
        list(analyzer.iter_sample_jsons(str(samples)))
    # 這裡改成抓實際出現的 log 關鍵字
    assert any("Loaded a.zip" in r.message for r in caplog.records)


def test_top3_logging(caplog):
    grid = [[-1, -1], [-1, -1]]
    with caplog.at_level(logging.INFO):
        analyzer.predict_scratch_card(grid, iterations=2, global_iter=1, focus_iter=1)
    # 這行不動，如果你 log 真的有 "prob_map top3"
    assert any("prob_map top3" in r.message for r in caplog.records)


def test_generate_full_boards_debug(caplog, make_grid):
    grid = np.array(make_grid(2, 2))
    rng = np.random.default_rng(0)
    with caplog.at_level(logging.DEBUG):
        analyzer.generate_full_boards(
            2,
            2,
            1,
            rng,
            ("random_entropy",),
            np.array([1.0]),
            grid,
        )
    assert any("generate_full_boards" in r.message for r in caplog.records)


def test_rank_cells_debug(caplog, tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    data = {"rows": 2, "cols": 2, "grid": [[1, 2], [3, 4]]}
    with zipfile.ZipFile(samples / "d.zip", "w") as zf:
        zf.writestr("a.json", json.dumps(data))
    cube = analyzer.compute_global_distribution(str(samples), 2, 2)
    grid = np.array([[-1, -1], [3, 4]])
    with caplog.at_level(logging.DEBUG):
        analyzer.rank_cells_by_prior_and_modules(
            grid,
            cube,
            ["EXT_Q1_ProximityEntropy_Vec"],
            [1.0],
            target_num=1,
        )
    assert any("rank_cells_by_prior_and_modules" in r.message for r in caplog.records)
