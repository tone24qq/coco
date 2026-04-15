from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.whole_board_features import compute_board_state_features, euclidean

Board = List[List[int]]


@dataclass
class SizeClassProfile:
    size_class: str
    rows: int
    cols: int
    feature_means: Dict[str, float]
    feature_stds: Dict[str, float]


PROFILE_FEATURE_KEYS = (
    "tail_entropy",
    "same_tail_adjacency_rate",
    "same_decade_proximity_rate",
    "consecutive_neighbor_rate",
    "row_known_entropy",
    "col_known_entropy",
    "edge_center_balance",
)


def _flatten_board_features(board: Board) -> Dict[str, float]:
    feats = compute_board_state_features(board, target_number=1)
    return {k: float(feats.get(k, 0.0)) for k in PROFILE_FEATURE_KEYS}


def fit_profiles(real_boards: List[Dict[str, object]]) -> Dict[str, SizeClassProfile]:
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for row in real_boards:
        size_class = str(row["size_class"])
        grid = row["grid"]
        grouped.setdefault(size_class, []).append(_flatten_board_features(grid))

    profiles: Dict[str, SizeClassProfile] = {}
    for size_class, features in grouped.items():
        if not features:
            continue
        rows = int(size_class.split("x")[0])
        cols = int(size_class.split("x")[1])
        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}
        for key in PROFILE_FEATURE_KEYS:
            vals = [f[key] for f in features]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(len(vals), 1)
            means[key] = float(mean)
            stds[key] = float(max(var ** 0.5, 1e-6))
        profiles[size_class] = SizeClassProfile(
            size_class=size_class,
            rows=rows,
            cols=cols,
            feature_means=means,
            feature_stds=stds,
        )
    return profiles


def realism_score(board: Board, profile: SizeClassProfile) -> float:
    feats = _flatten_board_features(board)
    z = [(feats[k] - profile.feature_means[k]) / profile.feature_stds[k] for k in PROFILE_FEATURE_KEYS]
    distance = euclidean(z)
    return float(1.0 / (1.0 + distance))


def _random_swap(board: Board, rng: random.Random) -> Board:
    rows, cols = len(board), len(board[0])
    c = [row[:] for row in board]
    a = rng.randrange(rows * cols)
    b = rng.randrange(rows * cols)
    while b == a:
        b = rng.randrange(rows * cols)
    ar, ac = divmod(a, cols)
    br, bc = divmod(b, cols)
    c[ar][ac], c[br][bc] = c[br][bc], c[ar][ac]
    return c


def generate_synthetic_from_seed(
    seed_board: Board,
    profile: SizeClassProfile,
    num_samples: int,
    rng: random.Random,
    proposals_per_sample: int = 300,
    min_realism: float = 0.22,
) -> List[Tuple[Board, float]]:
    accepted: List[Tuple[Board, float]] = []
    for _ in range(num_samples):
        current = [row[:] for row in seed_board]
        current_score = realism_score(current, profile)
        for _step in range(proposals_per_sample):
            candidate = _random_swap(current, rng)
            cand_score = realism_score(candidate, profile)
            if cand_score >= current_score or rng.random() < 0.05:
                current = candidate
                current_score = cand_score
        if current_score >= min_realism:
            accepted.append((current, current_score))
    return accepted
