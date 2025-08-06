from collections import Counter, defaultdict
import numpy as np

def build_neighbor_stats(boards: np.ndarray,
                         targets: np.ndarray,
                         rows: int,
                         cols: int) -> dict[int, Counter]:
    """
    boards  : (N, rows*cols) 完整盤 (無 -1)
    targets : (N,)           每盤的 target 值
    回傳    : {t: Counter(鄰居數字 → 次數)}
    """
    stats = defaultdict(Counter)
    N = boards.shape[0]

    for i in range(N):
        b   = boards[i].reshape(rows, cols)
        tgt = targets[i]
        # 所有 t 的座標
        for r, c in zip(*np.where(b == tgt)):
            neigh = np.concatenate([b[r, :], b[:, c]])  # 同行 + 同列
            for n in neigh:
                if n != tgt:                # 自己別算
                    stats[tgt][int(n)] += 1
    return stats