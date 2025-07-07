import json
import zipfile

import numpy as np
import analyzer


def test_load_and_filter_samples(tmp_path):
    # 准备测试目录和压缩样本
    samples = tmp_path / "samples"
    samples.mkdir()
    board = [[1, 2], [3, 4]]
    with zipfile.ZipFile(samples / "z.zip", "w") as zf:
        zf.writestr("b.json", json.dumps({"rows": 2, "cols": 2, "grid": board}))

    # 清空缓存并加载
    analyzer._SAMPLE_CACHE.clear()
    loaded = analyzer._load_samples_for_shape(str(samples), 2, 2)

    # 新增：處理 grid_np 並調用 probability_heatmap
    BLANK_VAL = -1
    grid_np = np.asarray(board, dtype=object)
    grid_np = np.where(grid_np == BLANK_VAL, BLANK_VAL, grid_np).astype(np.int64)
    rows, cols = grid_np.shape

    try:
        _ = analyzer.probability_heatmap(
            grid_np,
            sample_gamma=0.0,  # 默認值
            history_dir=str(samples)
        )
    except Exception:
        pass

    blanks = [tuple(b) for b in np.argwhere(grid_np == BLANK_VAL)]

    # 验证加载结果是列表，且只有一个样本
    assert isinstance(loaded, list)
    assert len(loaded) == 1

    arr, filename = loaded[0]
    # 验证文件名
    assert filename == "z.zip"
    # 验证数组类型、shape 和内容
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == int
    assert arr.shape == (2, 2)
    assert np.array_equal(arr, np.array(board, dtype=int))