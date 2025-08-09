from train import find_datasets


def test_find_datasets(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    good = data_dir / "4x5.json"
    good.write_text("{}")
    (data_dir / "bad.json").write_text("{}")

    datasets = find_datasets(data_dir)
    assert datasets == [(4, 5, good)]
