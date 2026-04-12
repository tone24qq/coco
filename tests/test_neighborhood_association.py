from __future__ import annotations

from src.inference_service import build_cell_candidates, score_candidates
from src.neighborhood_association import NeighborhoodAssociationModule
from src.scoring_modules import build_modules


def test_no_seed_returns_neutral_scores() -> None:
    board = [
        [1, -1, 3],
        [4, -1, 6],
        [8, -1, 9],
    ]
    unopened = [(0, 1), (1, 1), (2, 1)]
    module = NeighborhoodAssociationModule(
        enabled_seed_families=["same_tail"],
        near_value_deltas=[1],
        neutral_score_when_no_seed=0.5,
    )
    out = module.score(board, unopened, target_number=27)
    assert all(abs(out.scores[cell] - 0.5) < 1e-9 for cell in unopened)
    assert all(out.details[cell]["no_seed_fallback_used"] == 1.0 for cell in unopened)


def test_same_tail_seed_affects_scores() -> None:
    board = [
        [19, 39, -1],
        [9, 29, -1],
        [11, 22, 35],
    ]
    unopened = [(0, 2), (1, 2)]
    with_tail = NeighborhoodAssociationModule(
        enabled_seed_families=["same_tail"],
        enabled_neighbor_families=["same_tail"],
        near_value_deltas=[1, 2, 10, 20],
        neighbor_value_deltas=[1, 2, 10, 20],
        radius=1,
        use_diagonal=True,
    ).score(board, unopened, target_number=29)
    without_tail = NeighborhoodAssociationModule(
        enabled_seed_families=["same_decade"],
        enabled_neighbor_families=["same_decade"],
        radius=1,
        use_diagonal=True,
    ).score(board, unopened, target_number=29)
    assert with_tail.scores != without_tail.scores


def test_same_decade_seed_affects_scores() -> None:
    board = [
        [72, 75, -1],
        [81, 78, -1],
        [64, 33, 55],
    ]
    unopened = [(0, 2), (1, 2)]
    with_decade = NeighborhoodAssociationModule(
        enabled_seed_families=["same_decade"],
        enabled_neighbor_families=["same_decade"],
        near_value_deltas=[1, 2],
        neighbor_value_deltas=[1, 2],
        radius=1,
        use_diagonal=True,
    ).score(board, unopened, target_number=79)
    without_decade = NeighborhoodAssociationModule(
        enabled_seed_families=["same_tail"],
        enabled_neighbor_families=["same_tail"],
        near_value_deltas=[1, 2],
        neighbor_value_deltas=[1, 2],
        radius=1,
        use_diagonal=True,
    ).score(board, unopened, target_number=79)
    assert with_decade.scores != without_decade.scores


def test_near_value_relation_affects_scores() -> None:
    board = [
        [88, 97, -1],
        [78, 65, -1],
        [12, 34, 56],
    ]
    unopened = [(0, 2), (1, 2)]
    with_near = NeighborhoodAssociationModule(
        enabled_seed_families=["near_value"],
        enabled_neighbor_families=["near_value"],
        near_value_deltas=[1, 2, 10, 20],
        neighbor_value_deltas=[1, 2, 10, 20],
        radius=1,
        use_diagonal=True,
    ).score(board, unopened, target_number=98)
    without_near = NeighborhoodAssociationModule(
        enabled_seed_families=["same_tail"],
        enabled_neighbor_families=["same_tail"],
        near_value_deltas=[1, 2, 10, 20],
        neighbor_value_deltas=[1, 2, 10, 20],
        radius=1,
        use_diagonal=True,
    ).score(board, unopened, target_number=98)
    assert with_near.scores != without_near.scores


def test_details_fields_present() -> None:
    board = [
        [19, 29, -1],
        [39, 49, -1],
        [12, 14, 16],
    ]
    unopened = [(0, 2), (1, 2)]
    out = NeighborhoodAssociationModule().score(board, unopened, target_number=29)
    required = {
        "seed_count",
        "effective_seed_count",
        "candidate_neighbor_count",
        "matched_seed_similarity_mean",
        "matched_seed_similarity_max",
        "same_decade_support",
        "same_tail_support",
        "near_value_support",
        "used_radius",
        "used_diagonal",
        "no_seed_fallback_used",
    }
    assert required.issubset(out.details[(0, 2)].keys())


def test_module_registered_in_factory() -> None:
    modules = build_modules({"neighborhood_association": {"radius": 2}})
    assert "neighborhood_association" in modules


def test_yaml_settings_are_respected() -> None:
    board = [
        [19, 29, 39, -1],
        [8, 18, 28, -1],
        [7, 17, 27, 37],
    ]
    unopened = [(0, 3), (1, 3)]
    module_r1 = NeighborhoodAssociationModule(radius=1, use_diagonal=False, neighbor_value_deltas=[1])
    module_r2 = NeighborhoodAssociationModule(radius=2, use_diagonal=True, neighbor_value_deltas=[1, 10, 20])
    out1 = module_r1.score(board, unopened, target_number=29)
    out2 = module_r2.score(board, unopened, target_number=29)
    assert out1.scores[(0, 3)] != out2.scores[(0, 3)]


def test_module_works_through_mainline_score_candidates() -> None:
    board = [
        [19, 29, -1],
        [9, 39, -1],
    ]
    candidates = build_cell_candidates([(0, 2), (1, 2)])
    scored, _, _ = score_candidates(
        board,
        candidates,
        target_number=29,
        module_weights={"neighborhood_association": 1.0},
        module_settings={
            "neighborhood_association": {
                "enabled_seed_families": ["same_tail", "same_decade", "near_value"],
                "enabled_neighbor_families": ["same_tail", "same_decade", "near_value"],
            }
        },
    )
    assert "neighborhood_association" in scored[0]["module_scores"]
