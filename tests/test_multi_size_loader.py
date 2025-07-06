import json
import logging
import zipfile

import analyzer


def test_iter_sample_jsons_multi_size(tmp_path, caplog):
    samples = tmp_path / "samples"
    samples.mkdir()
    data = {
        "2x2": [[[1, 2], [3, 4]], [[4, 3], [2, 1]]],
        "invalid": 123,
        "3x3": [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2], [3, 4, 5]]],
    }
    with zipfile.ZipFile(samples / "m.zip", "w") as zf:
        zf.writestr("boards.json", json.dumps(data))

    with caplog.at_level(logging.WARNING):
        items = list(analyzer.iter_sample_jsons(str(samples)))
    assert len(items) == 3
    assert any("invalid" in r.message for r in caplog.records)
