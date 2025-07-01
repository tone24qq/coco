import logging

import analyzer


def test_iter_sample_jsons_logging(caplog):
    with caplog.at_level(logging.INFO):
        analyzer.compute_position_probabilities("", 2, 2, n_synth=10, seed=0)
    assert any("generated" in r.message for r in caplog.records)


def test_top3_logging(caplog):
    grid = [[-1, -1], [-1, -1]]
    with caplog.at_level(logging.INFO):
        analyzer.predict_scratch_card(grid, iterations=2, global_iter=1, focus_iter=1)
    # 這行不動，如果你 log 真的有 "prob_map top3"
    assert any("prob_map top3" in r.message for r in caplog.records)
