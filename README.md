# coco bingo analyzer

## Dynamic Cluster Weighting

`_adaptive_weights()` 現在支援動態 cluster 權重：

- `cluster_weight = 0.07 + 0.02 * min(cluster_score, 5)`
- `cluster_score` 由最近視窗內 `interval_cluster + tail_cluster + consecutive_cluster` 計算後再除以 window 大小。

這可在 cluster 爆發期提高 `cluster_pattern` 權重，平穩期則維持較保守比例。


## Six Statistical Vectors

`predict_next()` 新增六個統計向量並納入加權計分：

- `sum_range`: 近 800 期和值直方圖眾數區間（±1 bin）
- `odd_even_balance`: 奇偶與大小（1-40 / 41-80）平衡補償
- `delta_pattern`: 前 10 熱號差值模式
- `skip_heat`: skip=0 與 skip=1-5 熱冷補強
- `prime_boost`: 質數號碼加權
- `compression_boost`: 三期內區間壓縮（zone <= 4）補強

以上權重由 `ScoreWeights.as_dict()` 統一正規化，並與既有 dynamic cluster 權重機制共同運作。
