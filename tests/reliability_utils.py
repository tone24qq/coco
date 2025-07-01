import numpy as np
from modules import generate_unique_grid
from analyzer import probability_heatmap
from tqdm import tqdm

def run_infinite_test(
    min_size=4, max_size=20, mask_ratio=0.5,
    max_iters=1_000_000, seed=42, log_every=1_000
):
    """
    无限压力测试：命令行执行 python -m tests.reliability_utils run_infinite_test
    """
    rng = np.random.default_rng(seed)
    hits = total = 0
    for i in tqdm(range(1, max_iters + 1), desc="Stress Testing"):
        # …（跟之前的 run_infinite_test 一样）…
        pass

def run_until_converged(
    min_size=4, max_size=20, mask_ratio=0.5,
    batch_size=200, delta=0.02, z=1.96, seed=0
):
    """
    批次收敛测试：被 pytest 的 test_accuracy_converges 调用
    """
    rng = np.random.default_rng(seed)
    hits = total = 0
    max_batches = 50
    for batch in range(1, max_batches + 1):
        # …（同之前）…
        pass
    return hits / total, z * ((hits/total)*(1-hits/total)/total)**0.5, total