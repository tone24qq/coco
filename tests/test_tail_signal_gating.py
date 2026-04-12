from __future__ import annotations

from src.inference_service import aggregate_candidate_scores
from src.inference_config import load_module_settings
from src.scoring_modules import LocalArithmeticRelationModule, PatternModelModule
from src.vector_modules import tail_analyzer_vectorized


def test_weak_same_tail_scatter_should_abstain() -> None:
    board = [
        [11, -1, 28, -1],
        [-1, 35, -1, 42],
        [53, -1, 64, -1],
        [-1, 77, -1, 89],
    ]
    cells = [(0, 1), (1, 0), (1, 2), (2, 1)]
    target = 21
    pattern = PatternModelModule().score(board, cells, target)
    tail_scores = tail_analyzer_vectorized(board, cells, target, window_size=3)
    assert all(pattern.informative_cells[cell] == 0.0 for cell in cells)
    assert all(abs(pattern.scores[cell] - 0.5) < 1e-9 for cell in cells)
    assert all(tail_scores[cell] <= 0.55 for cell in cells)


def test_strong_local_tail_signal_should_still_help() -> None:
    board = [
        [11, 22, 31],
        [41, -1, 51],
        [61, 72, 81],
    ]
    cells = [(1, 1)]
    target = 21
    pattern = PatternModelModule().score(board, cells, target)
    tail_scores = tail_analyzer_vectorized(board, cells, target, window_size=3)
    assert pattern.informative_cells[(1, 1)] == 1.0
    assert 0.52 <= pattern.scores[(1, 1)] <= 0.72
    assert 0.50 < tail_scores[(1, 1)] <= 0.72


def test_local_arithmetic_prefers_near_value_over_same_tail_only() -> None:
    board = [
        [19, -1, 39],
        [18, -1, 20],
        [59, -1, 79],
    ]
    cells = [(0, 1), (1, 1), (2, 1)]
    target = 19
    module = LocalArithmeticRelationModule()
    res = module.score(board, cells, target)
    assert res.scores[(1, 1)] > res.scores[(0, 1)]
    assert res.scores[(1, 1)] > res.scores[(2, 1)]


def test_committee_weighting_reduces_tail_only_dominance() -> None:
    candidates = [
        {
            "cell": (0, 0),
            "module_scores": {"structural_consistency": 0.90, "tail_analyzer": 0.52},
            "module_details": {},
            "module_informative": {"structural_consistency": 1.0, "tail_analyzer": 1.0},
        },
        {
            "cell": (0, 1),
            "module_scores": {"structural_consistency": 0.55, "tail_analyzer": 0.72},
            "module_details": {},
            "module_informative": {"structural_consistency": 1.0, "tail_analyzer": 1.0},
        },
    ]
    aggregate_candidate_scores(
        candidates,
        {"structural_consistency": 0.26, "tail_analyzer": 0.03},
        {
            "type": "committee_weighted_sum",
            "weighting_mode": "yaml_normalized",
            "allow_abstain": True,
            "preserve_diagnostics": True,
            "diagnostics": {
                "include_vote_features": True,
                "include_rank_features": True,
                "include_score_features": True,
            },
            "judge": {"enabled": False, "use_for_primary_ranking": False},
        },
    )
    ranked = sorted(candidates, key=lambda x: x["final_rank_position"])
    assert ranked[0]["cell"] == (0, 0)


def test_neighborhood_default_seed_neighbor_families_exclude_same_tail() -> None:
    cfg = load_module_settings()
    nb = cfg["neighborhood_association"]
    assert "same_tail" not in nb["enabled_seed_families"]
    assert "same_tail" not in nb["enabled_neighbor_families"]
