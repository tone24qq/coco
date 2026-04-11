from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference_config import DEFAULT_CONFIG_PATH  # noqa: E402
from src.inference_service import _run_inference_detailed, compact_top10_response  # noqa: E402


def _make_board(rows: int, cols: int) -> list[list[int]]:
    n = 1
    board = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append(n)
            n += 1
        board.append(row)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 3 == 0:
                board[r][c] = -1
    return board


def _pick_target(board: list[list[int]]) -> int:
    rows = len(board)
    cols = len(board[0])
    n_total = rows * cols
    opened = {v for row in board for v in row if v != -1}
    for x in range(1, n_total + 1):
        if x not in opened:
            return x
    return 1


def _run_once_detailed(
    board: list[list[int]],
    target: int,
    fast_enabled: bool,
    pairwise_enabled: bool,
    runtime_mode: str,
) -> dict:
    module_settings = {
        "logic_rule": {"fast_enabled": fast_enabled},
        "prior_model": {"fast_enabled": fast_enabled},
        "directional_consistency": {"fast_enabled": fast_enabled},
        "line_consistency": {"fast_enabled": fast_enabled},
        "global_assignment_prior": {
            "assignment_mode": "greedy" if runtime_mode == "fast" else "exact",
            "top_m_candidates": 4 if runtime_mode == "fast" else 8,
            "exact_max_candidates": 20,
        },
        "pairwise_conditional_consistency": {
            "runtime_mode": runtime_mode,
            "candidate_top_n": 8,
            "global_assignment_mode": "greedy",
            "global_assignment_top_m_candidates": 4,
        },
    }
    module_weights = None
    if not pairwise_enabled:
        module_weights = {
            "logic_rule": 0.24,
            "pattern_model": 0.16,
            "prior_model": 0.12,
            "directional_consistency": 0.2,
            "line_consistency": 0.18,
            "global_assignment_prior": 0.1,
        }
    detailed = _run_inference_detailed(
            board=board,
            target_number=target,
            source="benchmark",
            module_settings=module_settings,
            module_weights=module_weights,
            apply_reranker_stage=False,
        )
    detailed["metadata"]["runtime_mode"] = runtime_mode
    return {"compact": compact_top10_response(detailed), "detailed": detailed}


def benchmark_case(
    board: list[list[int]],
    target: int,
    rounds: int,
    fast_enabled: bool,
    pairwise_enabled: bool,
    runtime_mode: str,
) -> tuple[float, dict]:
    _run_once_detailed(
        board,
        target,
        fast_enabled=fast_enabled,
        pairwise_enabled=pairwise_enabled,
        runtime_mode=runtime_mode,
    )
    start = time.perf_counter()
    out = {}
    for _ in range(rounds):
        out = _run_once_detailed(
            board,
            target,
            fast_enabled=fast_enabled,
            pairwise_enabled=pairwise_enabled,
            runtime_mode=runtime_mode,
        )
    elapsed = (time.perf_counter() - start) / rounds
    return elapsed, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()

    small = _make_board(4, 5)
    large = _make_board(8, 10)

    t_small = _pick_target(small)
    t_large = _pick_target(large)

    small_baseline, out_small_a = benchmark_case(
        deepcopy(small), t_small, args.rounds, fast_enabled=False, pairwise_enabled=True, runtime_mode="full"
    )
    small_fast, out_small_b = benchmark_case(
        deepcopy(small), t_small, args.rounds, fast_enabled=True, pairwise_enabled=True, runtime_mode="fast"
    )
    large_baseline, out_large_a = benchmark_case(
        deepcopy(large), t_large, args.rounds, fast_enabled=False, pairwise_enabled=True, runtime_mode="full"
    )
    large_fast, out_large_b = benchmark_case(
        deepcopy(large), t_large, args.rounds, fast_enabled=True, pairwise_enabled=True, runtime_mode="fast"
    )
    small_pair_off, _ = benchmark_case(
        deepcopy(small), t_small, args.rounds, fast_enabled=True, pairwise_enabled=False, runtime_mode="fast"
    )
    large_pair_off, _ = benchmark_case(
        deepcopy(large), t_large, args.rounds, fast_enabled=True, pairwise_enabled=False, runtime_mode="fast"
    )

    def same_rank(a: dict, b: dict) -> tuple[bool, bool, float]:
        ac = a["compact"]["top10"]
        bc = b["compact"]["top10"]
        top1 = ac[0] == bc[0]
        top10 = ac == bc
        aset = {(x["row"], x["col"]) for x in ac}
        bset = {(x["row"], x["col"]) for x in bc}
        overlap = len(aset & bset) / max(len(aset | bset), 1)
        return top1, top10, overlap

    s1, s10, so = same_rank(out_small_a, out_small_b)
    l1, l10, lo = same_rank(out_large_a, out_large_b)

    print(f"config={DEFAULT_CONFIG_PATH}")
    print(f"small_baseline_avg_sec={small_baseline:.6f}")
    print(f"small_fast_avg_sec={small_fast:.6f}")
    print(f"small_speedup={small_baseline / max(small_fast, 1e-12):.3f}x")
    print(f"small_top1_same={int(s1)} small_top10_same={int(s10)} small_top10_overlap={so:.3f}")
    print(f"small_pairwise_off_avg_sec={small_pair_off:.6f}")
    print(f"large_baseline_avg_sec={large_baseline:.6f}")
    print(f"large_fast_avg_sec={large_fast:.6f}")
    print(f"large_speedup={large_baseline / max(large_fast, 1e-12):.3f}x")
    print(f"large_top1_same={int(l1)} large_top10_same={int(l10)} large_top10_overlap={lo:.3f}")
    print(f"large_pairwise_off_avg_sec={large_pair_off:.6f}")


if __name__ == "__main__":
    main()
