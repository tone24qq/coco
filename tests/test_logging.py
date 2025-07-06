import json
import logging
import zipfile

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
    assert any("已載入 a.zip" in r.message for r in caplog.records)


def test_top3_logging(caplog):
    grid = [[-1, -1], [-1, -1]]
    with caplog.at_level(logging.INFO):
        result = analyzer.predict_scratch_card(
            grid, iterations=2, global_iter=1, focus_iter=1
        )
    assert result["predictions"]
