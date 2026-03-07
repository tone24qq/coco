# coco bingo analyzer

## Dynamic Cluster Weighting

`_adaptive_weights()` 現在支援動態 cluster 權重：

- `cluster_weight = 0.07 + 0.02 * min(cluster_score, 5)`
- `cluster_score` 由最近視窗內 `interval_cluster + tail_cluster + consecutive_cluster` 計算後再除以 window 大小。

這可在 cluster 爆發期提高 `cluster_pattern` 權重，平穩期則維持較保守比例。
