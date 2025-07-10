import logging

import numpy as np

import analyzer


def test_iter_sample_jsons_multi_size(tmp_path, caplog):
    samples = tmp_path / "samples"
    samples.mkdir()
    np.savez(samples / "a.npz", boards=np.array([[[1, 2], [3, 4]], [[4, 3], [2, 1]]]))
    np.savez(samples / "b.npz", boards=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))
    np.savez(samples / "bad.npz", foo=np.array([1, 2, 3]))

    with caplog.at_level(logging.WARNING):
        items = list(analyzer.iter_sample_jsons(str(samples)))
    assert len(items) == 3
    assert any("boards missing" in r.message for r in caplog.records)


def test_iter_sample_jsons_list_only(tmp_path, caplog):
    samples = tmp_path / "samples"
    samples.mkdir()
    np.savez(
        samples / "list.npz", boards=np.array([[[1, 2], [3, 4]], [[4, 3], [2, 1]]])
    )

    with caplog.at_level(logging.WARNING):
        items = list(analyzer.iter_sample_jsons(str(samples)))

    assert len(items) == 2
    assert all(i["rows"] == 2 and i["cols"] == 2 for i in items)
