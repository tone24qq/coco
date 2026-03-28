from __future__ import annotations

import itertools
import math
from collections import defaultdict

from .config import AppConfig, DEFAULT_CONFIG


class PredictError(RuntimeError):
    """Raised when prediction cannot be produced."""


def predict_top3(
    past_draws: list[list[int]],
    latest_period: int,
    config: AppConfig = DEFAULT_CONFIG,
) -> dict[str, object]:
    if len(past_draws) < config.recent_draws_count:
        raise PredictError(
            f"Need >= {config.recent_draws_count} draws, got {len(past_draws)}"
        )

    skips = {i: 0 for i in range(1, 81)}
    streaks = {i: 0 for i in range(1, 81)}

    for draw in past_draws:
        draw_set = set(draw)
        for num in range(1, 81):
            if num in draw_set:
                skips[num] = 0
                streaks[num] += 1
            else:
                skips[num] += 1
                streaks[num] = 0

    pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for draw in past_draws:
        for pair in itertools.combinations(sorted(draw), 2):
            pair_counts[pair] += 1

    kill_zone = sorted(
        num
        for num in range(1, 81)
        if skips[num] >= config.skip_kill_threshold
        or streaks[num] >= config.streak_kill_threshold
    )
    valid_pool = sorted(set(range(1, 81)) - set(kill_zone))

    if len(valid_pool) < 3:
        raise PredictError(
            "Valid number pool below 3 after applying kill-zone"
        )

    weights = config.score_weights
    candidates: list[dict[str, object]] = []

    for triplet in itertools.combinations(valid_pool, 3):
        score = 0
        n1, n2, n3 = triplet

        streak_nums = [
            n
            for n in triplet
            if config.streak_min <= streaks[n] <= config.streak_max
        ]
        if len(streak_nums) == 1:
            s_val = streaks[streak_nums[0]]
            if s_val == 1:
                score += 20
            elif s_val == 2:
                score += 10
            elif s_val == 3:
                score += 5
        elif len(streak_nums) == 2:
            score += 5
        elif len(streak_nums) == 3:
            score -= 20

        warm_nums = [
            n
            for n in triplet
            if config.warm_skip_min <= skips[n] <= config.warm_skip_max
        ]
        if len(warm_nums) == 1:
            w_val = skips[warm_nums[0]]
            if w_val == 3:
                score += 20
            elif w_val == 4:
                score += 10
            elif w_val == 5:
                score += 5
        elif len(warm_nums) == 2:
            score += 5
        elif len(warm_nums) == 3:
            score -= 20

        tails = [n % 10 for n in triplet]
        unique_tails = len(set(tails))
        if unique_tails == 2:
            score += weights.tail_perfect
        elif unique_tails == 1:
            score += weights.tail_bad

        odds = sum(1 for n in triplet if n % 2 != 0)
        if odds in (1, 2):
            score += weights.parity_balance

        smalls = sum(1 for n in triplet if n <= 40)
        if smalls in (1, 2):
            score += weights.size_balance

        tens = [n // 10 for n in triplet]
        if len(set(tens)) == 3:
            score += weights.dispersion

        pair1 = pair_counts[(n1, n2)]
        pair2 = pair_counts[(n2, n3)]
        pair3 = pair_counts[(n1, n3)]
        momentum_score = min(15, (pair1 + pair2 + pair3))
        score += momentum_score

        if score >= config.min_score_threshold:
            candidates.append(
                {"triplet": list(triplet), "score": score}
            )

    if not candidates:
        raise PredictError("No combinations exceed min_score_threshold")

    candidates.sort(key=lambda c: int(c["score"]), reverse=True)
    top3 = [entry["triplet"] for entry in candidates[:3]]

    return {
        "target_period": latest_period + 1,
        "latest_period": latest_period,
        "top3": top3,
        "kill_zone": kill_zone,
        "metadata": {
            "analyzed_draws": len(past_draws),
            "valid_pool_size": len(valid_pool),
            "total_combinations": math.comb(len(valid_pool), 3),
            "qualified_combinations": len(candidates),
            "min_score_threshold": config.min_score_threshold,
        },
    }
