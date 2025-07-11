import logging

import numpy as np

import analyzer


def test_iter_sample_jsons_logging(tmp_path, caplog):
    samples = tmp_path / "samples"
    samples.mkdir()
    boards = np.array([[[1, 2], [3, 4]]], dtype=np.int8)
    np.savez(samples / "a.npz", boards=boards)
    with caplog.at_level(logging.INFO):
        list(analyzer.iter_sample_jsons(str(samples)))
    # 這裡改成抓實際出現的 log 關鍵字
    assert any("已載入 a.npz" in r.message for r in caplog.records)


def test_top3_logging(tmp_path, caplog):
    out_npz = tmp_path / "out_npz"
    out_npz.mkdir()
    freq = np.ones((2, 2, 5))
    np.savez(out_npz / "global_pos_freq_2x2.npz", freq=freq)
    analyzer._GLOBAL_POS_FREQ_CACHE.clear()
    analyzer.compute_position_probabilities.cache_clear()
    analyzer.load_global_pos_freq_npz((2, 2), out_npz)

    grid = [[-1, -1], [-1, -1]]
    with caplog.at_level(logging.INFO):
        result = analyzer.predict_scratch_card(
            grid,
            iterations=2,
            global_iter=1,
            focus_iter=1,
            target_num=1,
        )
    assert result["predictions"]
    assert any("匹配到" in r.message for r in caplog.records)
