import json
import zipfile

import analyzer


def test_pure_sample_branch(tmp_path):
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
    assert res["strategy"] == "pure_sample"
    assert res["predictions"][0]["row"] == 1
    assert res["predictions"][0]["col"] == 1
    assert abs(res["predictions"][0]["score"] - 1.0) < 1e-6
