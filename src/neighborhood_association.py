from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
import math

Board = List[List[int]]
Cell = Tuple[int, int]


@dataclass
class ModuleScoreResult:
    scores: Dict[Cell, float]
    explanation: str
    details: Dict[Cell, Dict[str, float]]
    informative_cells: Dict[Cell, float]


@dataclass
class SeedInfo:
    row: int
    col: int
    value: int
    matched_families: List[str]


@dataclass
class NeighborhoodProfile:
    feature_values: Dict[str, float]
    neighbor_count: int
    total_weight: float


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def same_decade_family(a: int, b: int) -> bool:
    return (a - 1) // 10 == (b - 1) // 10


def same_tail_family(a: int, b: int) -> bool:
    return a % 10 == b % 10


def near_value_family(a: int, b: int, deltas: Sequence[int]) -> bool:
    return abs(a - b) in deltas


def _distance(dr: int, dc: int, use_diagonal: bool) -> int:
    if use_diagonal:
        return max(abs(dr), abs(dc))
    return abs(dr) + abs(dc)


def _iter_neighbors(
    board: Board,
    row: int,
    col: int,
    radius: int,
    use_diagonal: bool,
) -> Iterable[Tuple[int, int, int]]:
    rows, cols = len(board), len(board[0])
    for rr in range(max(0, row - radius), min(rows, row + radius + 1)):
        for cc in range(max(0, col - radius), min(cols, col + radius + 1)):
            if rr == row and cc == col:
                continue
            dr = rr - row
            dc = cc - col
            dist = _distance(dr, dc, use_diagonal)
            if dist <= 0 or dist > radius:
                continue
            value = board[rr][cc]
            if value == -1:
                continue
            yield rr, cc, value


def match_seed_family(
    opened_value: int,
    target_number: int,
    enabled_seed_families: Sequence[str],
    deltas: Sequence[int],
) -> List[str]:
    matched: List[str] = []
    for family in enabled_seed_families:
        if family == "same_decade" and same_decade_family(opened_value, target_number):
            matched.append(family)
        elif family == "same_tail" and same_tail_family(opened_value, target_number):
            matched.append(family)
        elif family == "near_value" and near_value_family(opened_value, target_number, deltas):
            matched.append(family)
        elif family == "exact_value" and opened_value == target_number:
            matched.append(family)
        elif family == "custom_relation":
            continue
    return matched


def neighbor_relation_features(
    neighbor_value: int,
    target_number: int,
    enabled_neighbor_families: Sequence[str],
    deltas: Sequence[int],
) -> Dict[str, float]:
    features: Dict[str, float] = {
        "same_decade_support": 0.0,
        "same_tail_support": 0.0,
        "near_value_support": 0.0,
        "exact_value_support": 0.0,
        "numeric_closeness": 1.0 / (1.0 + abs(neighbor_value - target_number)),
    }
    if "same_decade" in enabled_neighbor_families:
        features["same_decade_support"] = 1.0 if same_decade_family(neighbor_value, target_number) else 0.0
    if "same_tail" in enabled_neighbor_families:
        features["same_tail_support"] = 1.0 if same_tail_family(neighbor_value, target_number) else 0.0
    if "near_value" in enabled_neighbor_families:
        is_near = near_value_family(neighbor_value, target_number, deltas)
        features["near_value_support"] = 1.0 if is_near else 0.0
        for d in deltas:
            features[f"near_value_delta_{int(d)}"] = 1.0 if abs(neighbor_value - target_number) == int(d) else 0.0
    if "exact_value" in enabled_neighbor_families:
        features["exact_value_support"] = 1.0 if neighbor_value == target_number else 0.0
    return features


def _build_profile(
    board: Board,
    row: int,
    col: int,
    target_number: int,
    radius: int,
    use_diagonal: bool,
    decay_by_distance: bool,
    distance_decay_power: float,
    enabled_neighbor_families: Sequence[str],
    neighbor_value_deltas: Sequence[int],
) -> NeighborhoodProfile:
    weighted_sums: Dict[str, float] = {}
    total_weight = 0.0
    neighbor_count = 0

    for rr, cc, value in _iter_neighbors(board, row, col, radius, use_diagonal):
        dist = _distance(rr - row, cc - col, use_diagonal)
        weight = 1.0
        if decay_by_distance:
            weight = 1.0 / max(float(dist) ** max(distance_decay_power, 1e-6), 1e-6)
        feats = neighbor_relation_features(value, target_number, enabled_neighbor_families, neighbor_value_deltas)
        for key, val in feats.items():
            weighted_sums[key] = weighted_sums.get(key, 0.0) + weight * float(val)
        total_weight += weight
        neighbor_count += 1

    if total_weight <= 0:
        return NeighborhoodProfile(feature_values={}, neighbor_count=0, total_weight=0.0)

    return NeighborhoodProfile(
        feature_values={k: v / total_weight for k, v in weighted_sums.items()},
        neighbor_count=neighbor_count,
        total_weight=total_weight,
    )


def _profile_similarity(seed_profile: NeighborhoodProfile, candidate_profile: NeighborhoodProfile) -> float:
    if seed_profile.total_weight <= 0 and candidate_profile.total_weight <= 0:
        return 0.45
    if seed_profile.total_weight <= 0 or candidate_profile.total_weight <= 0:
        return 0.35

    keys = sorted(set(seed_profile.feature_values.keys()) | set(candidate_profile.feature_values.keys()))
    if not keys:
        return 0.45
    mean_abs_diff = sum(
        abs(seed_profile.feature_values.get(k, 0.0) - candidate_profile.feature_values.get(k, 0.0)) for k in keys
    ) / len(keys)
    return _clip(1.0 - mean_abs_diff)


def _board_profile(
    board: Board,
    target_number: int,
    enabled_neighbor_families: Sequence[str],
    neighbor_value_deltas: Sequence[int],
) -> NeighborhoodProfile:
    weighted_sums: Dict[str, float] = {}
    total_weight = 0.0
    neighbor_count = 0
    for r, row in enumerate(board):
        for c, value in enumerate(row):
            if value == -1:
                continue
            feats = neighbor_relation_features(value, target_number, enabled_neighbor_families, neighbor_value_deltas)
            for key, val in feats.items():
                weighted_sums[key] = weighted_sums.get(key, 0.0) + float(val)
            total_weight += 1.0
            neighbor_count += 1
    if total_weight <= 0:
        return NeighborhoodProfile(feature_values={}, neighbor_count=0, total_weight=0.0)
    return NeighborhoodProfile(
        feature_values={k: v / total_weight for k, v in weighted_sums.items()},
        neighbor_count=neighbor_count,
        total_weight=total_weight,
    )


def _line_profile(
    board: Board,
    row: int,
    col: int,
    target_number: int,
    enabled_neighbor_families: Sequence[str],
    neighbor_value_deltas: Sequence[int],
    axis: str,
) -> NeighborhoodProfile:
    weighted_sums: Dict[str, float] = {}
    total_weight = 0.0
    neighbor_count = 0
    if axis == "row":
        iterator = [(row, cc) for cc in range(len(board[0])) if cc != col]
    else:
        iterator = [(rr, col) for rr in range(len(board)) if rr != row]
    for rr, cc in iterator:
        value = board[rr][cc]
        if value == -1:
            continue
        feats = neighbor_relation_features(value, target_number, enabled_neighbor_families, neighbor_value_deltas)
        for key, val in feats.items():
            weighted_sums[key] = weighted_sums.get(key, 0.0) + float(val)
        total_weight += 1.0
        neighbor_count += 1
    if total_weight <= 0:
        return NeighborhoodProfile(feature_values={}, neighbor_count=0, total_weight=0.0)
    return NeighborhoodProfile(
        feature_values={k: v / total_weight for k, v in weighted_sums.items()},
        neighbor_count=neighbor_count,
        total_weight=total_weight,
    )


class NeighborhoodAssociationModule:
    name = "neighborhood_association"

    def __init__(
        self,
        radius: int = 1,
        use_diagonal: bool = True,
        min_seed_count: int = 1,
        decay_by_distance: bool = True,
        distance_decay_power: float = 1.0,
        enabled_seed_families: Sequence[str] = ("same_decade", "same_tail", "near_value"),
        near_value_deltas: Sequence[int] = (1, 2, 10, 20),
        enabled_neighbor_families: Sequence[str] = ("same_decade", "same_tail", "near_value"),
        neighbor_value_deltas: Sequence[int] = (1, 2, 10, 20),
        score_mode: str = "weighted_pattern_overlap",
        seed_aggregation: str = "mean",
        candidate_aggregation: str = "mean",
        neutral_score_when_no_seed: float = 0.5,
        floor_score: float = 0.0,
        ceil_score: float = 1.0,
        relation_source: str = "heuristic_family_profile_v1",
    ) -> None:
        if score_mode != "weighted_pattern_overlap":
            raise ValueError(f"Unsupported score_mode: {score_mode}")
        if seed_aggregation not in {"mean", "max"}:
            raise ValueError(f"Unsupported seed_aggregation: {seed_aggregation}")
        if candidate_aggregation not in {"mean"}:
            raise ValueError(f"Unsupported candidate_aggregation: {candidate_aggregation}")

        self.radius = max(1, int(radius))
        self.use_diagonal = bool(use_diagonal)
        self.min_seed_count = max(1, int(min_seed_count))
        self.decay_by_distance = bool(decay_by_distance)
        self.distance_decay_power = float(distance_decay_power)
        self.enabled_seed_families = [str(x) for x in enabled_seed_families]
        self.near_value_deltas = sorted({abs(int(x)) for x in near_value_deltas if int(x) > 0})
        self.enabled_neighbor_families = [str(x) for x in enabled_neighbor_families]
        self.neighbor_value_deltas = sorted({abs(int(x)) for x in neighbor_value_deltas if int(x) > 0})
        self.seed_aggregation = seed_aggregation
        self.neutral_score_when_no_seed = float(neutral_score_when_no_seed)
        self.floor_score = float(floor_score)
        self.ceil_score = float(ceil_score)
        self.relation_source = relation_source

    def _find_seeds(self, board: Board, target_number: int) -> List[SeedInfo]:
        seeds: List[SeedInfo] = []
        for r, row in enumerate(board):
            for c, value in enumerate(row):
                if value == -1:
                    continue
                matched = match_seed_family(value, target_number, self.enabled_seed_families, self.near_value_deltas)
                if matched:
                    seeds.append(SeedInfo(row=r, col=c, value=value, matched_families=matched))
        return seeds

    def score(self, board: Board, unopened_cells: List[Cell], target_number: int) -> ModuleScoreResult:
        if not unopened_cells:
            return ModuleScoreResult(
                {},
                "neighborhood_association: no unopened cells",
                details={},
                informative_cells={},
            )

        seeds = self._find_seeds(board, target_number)
        details: Dict[Cell, Dict[str, float]] = {}
        scores: Dict[Cell, float] = {}
        informative_cells: Dict[Cell, float] = {}

        if len(seeds) < self.min_seed_count:
            for cell in unopened_cells:
                scores[cell] = _clip(self.neutral_score_when_no_seed, self.floor_score, self.ceil_score)
                details[cell] = {
                    "seed_count": float(len(seeds)),
                    "effective_seed_count": 0.0,
                    "candidate_neighbor_count": 0.0,
                    "matched_seed_similarity_mean": 0.0,
                    "matched_seed_similarity_max": 0.0,
                    "same_decade_support": 0.0,
                    "same_tail_support": 0.0,
                    "near_value_support": 0.0,
                    "used_radius": float(self.radius),
                    "used_diagonal": float(self.use_diagonal),
                    "no_seed_fallback_used": 1.0,
                    "abstain_flag": 1.0,
                    "top_seed_row": -1.0,
                    "top_seed_col": -1.0,
                    "top_seed_value": -1.0,
                    "available_support_count": 0.0,
                    "normalized_support": 0.0,
                    "local_support": 0.0,
                    "row_support": 0.0,
                    "col_support": 0.0,
                    "global_support": 0.0,
                    "coverage_ratio": 0.0,
                    "zone_type": -1.0,
                    "raw_score_before_normalization": float(self.neutral_score_when_no_seed),
                    "bias_corrected_score": float(self.neutral_score_when_no_seed),
                }
                informative_cells[cell] = 0.0
            return ModuleScoreResult(
                scores,
                "neighborhood_association: 無足夠 seed，回退中性分數",
                details=details,
                informative_cells=informative_cells,
            )

        seed_profiles: List[Tuple[SeedInfo, NeighborhoodProfile, NeighborhoodProfile, NeighborhoodProfile]] = []
        board_prof = _board_profile(
            board, target_number, self.enabled_neighbor_families, self.neighbor_value_deltas
        )
        for seed in seeds:
            local_profile = _build_profile(
                board,
                seed.row,
                seed.col,
                target_number,
                self.radius,
                self.use_diagonal,
                self.decay_by_distance,
                self.distance_decay_power,
                self.enabled_neighbor_families,
                self.neighbor_value_deltas,
            )
            row_profile = _line_profile(
                board,
                seed.row,
                seed.col,
                target_number,
                self.enabled_neighbor_families,
                self.neighbor_value_deltas,
                axis="row",
            )
            col_profile = _line_profile(
                board,
                seed.row,
                seed.col,
                target_number,
                self.enabled_neighbor_families,
                self.neighbor_value_deltas,
                axis="col",
            )
            seed_profiles.append((seed, local_profile, row_profile, col_profile))

        effective_seed_count = sum(1 for _, p, _, _ in seed_profiles if p.neighbor_count > 0)

        for cell in unopened_cells:
            candidate_local = _build_profile(
                board,
                cell[0],
                cell[1],
                target_number,
                self.radius,
                self.use_diagonal,
                self.decay_by_distance,
                self.distance_decay_power,
                self.enabled_neighbor_families,
                self.neighbor_value_deltas,
            )
            candidate_row = _line_profile(
                board,
                cell[0],
                cell[1],
                target_number,
                self.enabled_neighbor_families,
                self.neighbor_value_deltas,
                axis="row",
            )
            candidate_col = _line_profile(
                board,
                cell[0],
                cell[1],
                target_number,
                self.enabled_neighbor_families,
                self.neighbor_value_deltas,
                axis="col",
            )
            local_sims: List[Tuple[float, SeedInfo]] = [
                (_profile_similarity(seed_local, candidate_local), seed) for seed, seed_local, _, _ in seed_profiles
            ]
            row_sims = [_profile_similarity(seed_row, candidate_row) for _, _, seed_row, _ in seed_profiles]
            col_sims = [_profile_similarity(seed_col, candidate_col) for _, _, _, seed_col in seed_profiles]
            global_sim = _profile_similarity(board_prof, candidate_local)
            only_scores = [x[0] for x in local_sims]
            sim_mean = sum(only_scores) / max(len(only_scores), 1)
            sim_max = max(only_scores) if only_scores else 0.0
            row_mean = sum(row_sims) / max(len(row_sims), 1)
            col_mean = sum(col_sims) / max(len(col_sims), 1)
            local_mean = sim_max if self.seed_aggregation == "max" else sim_mean
            final_score = 0.40 * local_mean + 0.20 * row_mean + 0.20 * col_mean + 0.20 * global_sim
            final_score = _clip(final_score, self.floor_score, self.ceil_score)
            scores[cell] = final_score

            top_sim, top_seed = max(local_sims, key=lambda x: x[0]) if local_sims else (0.0, None)
            rows, cols = len(board), len(board[0])
            r, c = cell
            if r in (0, rows - 1) and c in (0, cols - 1):
                zone_type = "corner"
            elif r in (0, rows - 1) or c in (0, cols - 1):
                zone_type = "edge"
            else:
                zone_type = "center"
            local_h = min(rows - 1, r + self.radius) - max(0, r - self.radius) + 1
            local_w = min(cols - 1, c + self.radius) - max(0, c - self.radius) + 1
            local_avail = float(local_h * local_w - 1)
            local_support = float(candidate_local.neighbor_count) / max(local_avail, 1.0)
            row_support = float(candidate_row.neighbor_count) / max(float(cols - 1), 1.0)
            col_support = float(candidate_col.neighbor_count) / max(float(rows - 1), 1.0)
            global_support = float(board_prof.neighbor_count) / max(float(rows * cols), 1.0)
            normalized_support = (local_support + row_support + col_support + global_support) / 4.0
            confidence = math.sqrt(max(normalized_support, 0.0))
            details[cell] = {
                "seed_count": float(len(seeds)),
                "effective_seed_count": float(effective_seed_count),
                "candidate_neighbor_count": float(candidate_local.neighbor_count),
                "matched_seed_similarity_mean": float(sim_mean),
                "matched_seed_similarity_max": float(sim_max),
                "row_similarity_mean": float(row_mean),
                "col_similarity_mean": float(col_mean),
                "global_similarity": float(global_sim),
                "same_decade_support": float(candidate_local.feature_values.get("same_decade_support", 0.0)),
                "same_tail_support": float(candidate_local.feature_values.get("same_tail_support", 0.0)),
                "near_value_support": float(candidate_local.feature_values.get("near_value_support", 0.0)),
                "used_radius": float(self.radius),
                "used_diagonal": float(self.use_diagonal),
                "no_seed_fallback_used": 0.0,
                "abstain_flag": 0.0,
                "available_support_count": float(local_avail + (rows - 1) + (cols - 1) + rows * cols),
                "normalized_support": float(normalized_support),
                "local_support": float(local_support),
                "row_support": float(row_support),
                "col_support": float(col_support),
                "global_support": float(global_support),
                "coverage_ratio": float(normalized_support),
                "zone_type": 0.0 if zone_type == "corner" else (1.0 if zone_type == "edge" else 2.0),
                "raw_score_before_normalization": float(final_score),
                "bias_corrected_score": float(final_score),
                "top_seed_row": float(top_seed.row + 1) if top_seed else -1.0,
                "top_seed_col": float(top_seed.col + 1) if top_seed else -1.0,
                "top_seed_value": float(top_seed.value) if top_seed else -1.0,
                "top_seed_similarity": float(top_sim),
            }
            informative_cells[cell] = float(confidence)
        return ModuleScoreResult(
            scores,
            "neighborhood_association: 以 target 關聯 family 的局部鄰域共現支持度評分",
            details=details,
            informative_cells=informative_cells,
        )
