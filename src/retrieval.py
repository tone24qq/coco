from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt

import numpy as np

from src.utils import DrawRecord


@dataclass(frozen=True)
class RetrievalWeights:
    aligned_overlap: float = 0.30
    recency_aligned_overlap: float = 0.20
    freq_similarity: float = 0.15
    profile_similarity: float = 0.10
    exact_draw_match: float = 0.15
    same_day_progress_bonus: float = 0.10

    @classmethod
    def from_mapping(cls, values: dict[str, float] | None) -> "RetrievalWeights":
        values = values or {}
        return cls(
            aligned_overlap=float(values.get("aligned_overlap", 0.30)),
            recency_aligned_overlap=float(values.get("recency_aligned_overlap", 0.20)),
            freq_similarity=float(values.get("freq_similarity", 0.15)),
            profile_similarity=float(values.get("profile_similarity", 0.10)),
            exact_draw_match=float(values.get("exact_draw_match", 0.15)),
            same_day_progress_bonus=float(values.get("same_day_progress_bonus", 0.10)),
        )


@dataclass
class RetrievalMatch:
    end_issue: str
    similarity: float
    next_draw_numbers: tuple[int, ...]
    same_day_progress: bool
    exact_draw_match_count: int
    exact_window_match: bool


def _jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter / union) if union else 0.0


def _number_freq(window: list[DrawRecord]) -> list[float]:
    freq = [0.0] * 80
    for row in window:
        for n in row.numbers:
            freq[n - 1] += 1.0
    total = max(1.0, 20.0 * len(window))
    return [x / total for x in freq]


def _freq_similarity(a: list[float], b: list[float]) -> float:
    dist = sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return 1.0 / (1.0 + dist)


def _draw_profile(draw: DrawRecord) -> tuple[float, float, float, float, float]:
    nums = draw.numbers
    odd = sum(1 for n in nums if n % 2 == 1) / 20.0
    big = sum(1 for n in nums if n > 40) / 20.0
    z1 = sum(1 for n in nums if 1 <= n <= 20) / 20.0
    z2 = sum(1 for n in nums if 21 <= n <= 40) / 20.0
    z3 = sum(1 for n in nums if 41 <= n <= 60) / 20.0
    z4 = sum(1 for n in nums if 61 <= n <= 80) / 20.0
    zone_var = ((z1 - 0.25) ** 2 + (z2 - 0.25) ** 2 + (z3 - 0.25) ** 2 + (z4 - 0.25) ** 2) ** 0.5
    total = sum(nums) / (80.0 * 20.0)
    span = (max(nums) - min(nums)) / 79.0
    consecutive = sum(1 for x, y in zip(nums, nums[1:]) if y - x == 1) / 19.0
    return odd, big, zone_var, total, span + consecutive


def _profile_similarity(target: list[DrawRecord], candidate: list[DrawRecord]) -> float:
    t_prof = [_draw_profile(r) for r in target]
    c_prof = [_draw_profile(r) for r in candidate]
    sims: list[float] = []
    for tp, cp in zip(t_prof, c_prof):
        dist = sqrt(sum((a - b) ** 2 for a, b in zip(tp, cp)))
        sims.append(1.0 / (1.0 + dist))
    return float(sum(sims) / len(sims)) if sims else 0.0


class SimilarWindowRetriever:
    def __init__(
        self,
        top_k: int = 20,
        weights: RetrievalWeights | None = None,
        require_same_length_window: bool = True,
        prefer_same_day_progress: bool = True,
        coarse_multiplier: int = 8,
    ) -> None:
        self.top_k = top_k
        self.weights = weights or RetrievalWeights()
        self.require_same_length_window = require_same_length_window
        self.prefer_same_day_progress = prefer_same_day_progress
        self.coarse_multiplier = max(1, int(coarse_multiplier))

    def _aligned_overlap(self, target: list[DrawRecord], candidate: list[DrawRecord]) -> tuple[float, float, int, bool]:
        per_draw = [_jaccard(t.numbers, c.numbers) for t, c in zip(target, candidate)]
        if not per_draw:
            return 0.0, 0.0, 0, False
        aligned = float(sum(per_draw) / len(per_draw))
        weights = [(i + 1) / len(per_draw) for i in range(len(per_draw))]
        weighted = sum(v * w for v, w in zip(per_draw, weights)) / sum(weights)
        exact_count = sum(1 for t, c in zip(target, candidate) if t.numbers == c.numbers)
        exact_window = exact_count == len(per_draw)
        return aligned, weighted, exact_count, exact_window

    def _query_legacy_python(
        self, history: list[DrawRecord], target_window: list[DrawRecord], day_issue_index: int
    ) -> list[RetrievalMatch]:
        n = len(target_window)
        if n <= 0:
            return []
        if self.require_same_length_window and len(history) <= n:
            return []

        target_freq = _number_freq(target_window)
        matches: list[RetrievalMatch] = []
        max_end = len(history) - 2
        for end_idx in range(n - 1, max_end + 1):
            candidate = history[end_idx - n + 1 : end_idx + 1]
            if len(candidate) != n:
                continue
            next_row = history[end_idx + 1]
            aligned, recency_aligned, exact_count, exact_window = self._aligned_overlap(target_window, candidate)
            freq_sim = _freq_similarity(target_freq, _number_freq(candidate))
            profile_sim = _profile_similarity(target_window, candidate)
            same_progress = bool(self.prefer_same_day_progress and history[end_idx].day_issue_index == day_issue_index)
            progress_bonus = 1.0 if same_progress else 0.0
            similarity = (
                self.weights.aligned_overlap * aligned
                + self.weights.recency_aligned_overlap * recency_aligned
                + self.weights.freq_similarity * freq_sim
                + self.weights.profile_similarity * profile_sim
                + self.weights.exact_draw_match * (exact_count / max(1, n))
                + self.weights.same_day_progress_bonus * progress_bonus
            )
            matches.append(
                RetrievalMatch(
                    end_issue=history[end_idx].issue,
                    similarity=similarity,
                    next_draw_numbers=next_row.numbers,
                    same_day_progress=same_progress,
                    exact_draw_match_count=exact_count,
                    exact_window_match=exact_window,
                )
            )
        matches.sort(key=lambda x: x.similarity, reverse=True)
        return matches[: self.top_k]

    @staticmethod
    def _history_indicator_matrix(history: list[DrawRecord]) -> np.ndarray:
        mat = np.zeros((len(history), 80), dtype=np.float64)
        for i, row in enumerate(history):
            nums = np.asarray(row.numbers, dtype=np.int64) - 1
            mat[i, nums] = 1.0
        return mat

    @staticmethod
    def _draw_profile_matrix(history: list[DrawRecord]) -> np.ndarray:
        out = np.zeros((len(history), 5), dtype=np.float64)
        for i, row in enumerate(history):
            nums = np.asarray(row.numbers, dtype=np.int64)
            odd = float(np.mean((nums % 2) == 1))
            big = float(np.mean(nums > 40))
            z1 = float(np.mean((1 <= nums) & (nums <= 20)))
            z2 = float(np.mean((21 <= nums) & (nums <= 40)))
            z3 = float(np.mean((41 <= nums) & (nums <= 60)))
            z4 = float(np.mean((61 <= nums) & (nums <= 80)))
            zone_var = float(((z1 - 0.25) ** 2 + (z2 - 0.25) ** 2 + (z3 - 0.25) ** 2 + (z4 - 0.25) ** 2) ** 0.5)
            total = float(np.sum(nums) / (80.0 * 20.0))
            span = float((int(np.max(nums)) - int(np.min(nums))) / 79.0)
            consecutive = float(np.sum(np.diff(nums) == 1) / 19.0)
            out[i] = np.array([odd, big, zone_var, total, span + consecutive], dtype=np.float64)
        return out

    def query(self, history: list[DrawRecord], target_window: list[DrawRecord], day_issue_index: int) -> list[RetrievalMatch]:
        n = len(target_window)
        if n <= 0:
            return []
        if self.require_same_length_window and len(history) <= n:
            return []
        max_end = len(history) - 2
        if max_end < n - 1:
            return []

        history_mat = self._history_indicator_matrix(history)
        target_mat = self._history_indicator_matrix(target_window)
        history_prof = self._draw_profile_matrix(history)
        target_prof = self._draw_profile_matrix(target_window)
        target_freq = np.sum(target_mat, axis=0) / max(1.0, 20.0 * n)
        target_set = np.array([set(row.numbers) for row in target_window], dtype=object)

        ends = np.arange(n - 1, max_end + 1, dtype=np.int64)
        starts = ends - n + 1
        win_count = len(ends)

        aligned_overlap = np.zeros(win_count, dtype=np.float64)
        recency_overlap = np.zeros(win_count, dtype=np.float64)
        exact_count = np.zeros(win_count, dtype=np.int64)
        freq_sim = np.zeros(win_count, dtype=np.float64)
        profile_sim = np.zeros(win_count, dtype=np.float64)
        same_progress = np.zeros(win_count, dtype=np.bool_)

        history_prefix = np.concatenate([np.zeros((1, 80), dtype=np.float64), np.cumsum(history_mat, axis=0)], axis=0)
        freq_windows = (history_prefix[starts + n] - history_prefix[starts]) / max(1.0, 20.0 * n)
        coarse_dist = np.sqrt(np.sum((freq_windows - target_freq[None, :]) ** 2, axis=1))
        coarse_freq_sim = 1.0 / (1.0 + coarse_dist)

        day_progress_arr = np.asarray([history[int(st + n - 1)].day_issue_index for st in starts], dtype=np.int64)
        coarse_progress = (day_progress_arr == int(day_issue_index)).astype(np.float64) if self.prefer_same_day_progress else np.zeros(win_count, dtype=np.float64)
        coarse_score = (
            self.weights.freq_similarity * coarse_freq_sim
            + self.weights.same_day_progress_bonus * coarse_progress
        )

        coarse_k = min(win_count, max(self.top_k * self.coarse_multiplier, self.top_k))
        coarse_idx = np.argsort(-coarse_score, kind="mergesort")[:coarse_k]

        recency_weights = np.arange(1, n + 1, dtype=np.float64)
        recency_weights = recency_weights / np.sum(recency_weights)

        for i in coarse_idx.tolist():
            st = int(starts[i])
            cand_block = history_mat[st : st + n]
            inter = np.sum(cand_block * target_mat, axis=1)
            union = np.sum(cand_block + target_mat - cand_block * target_mat, axis=1)
            jacc = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            aligned_overlap[i] = float(np.mean(jacc))
            recency_overlap[i] = float(np.sum(jacc * recency_weights))

            matches_draw = 0
            for j, tset in enumerate(target_set):
                if set(history[st + j].numbers) == tset and history[st + j].numbers == target_window[j].numbers:
                    matches_draw += 1
            exact_count[i] = matches_draw

            freq_sim[i] = float(coarse_freq_sim[i])

            cand_prof = history_prof[st : st + n]
            p_dist = np.sqrt(np.sum((target_prof - cand_prof) ** 2, axis=1))
            profile_sim[i] = float(np.mean(1.0 / (1.0 + p_dist)))
            same_progress[i] = bool(self.prefer_same_day_progress and history[st + n - 1].day_issue_index == day_issue_index)

        similarity = (
            self.weights.aligned_overlap * aligned_overlap
            + self.weights.recency_aligned_overlap * recency_overlap
            + self.weights.freq_similarity * freq_sim
            + self.weights.profile_similarity * profile_sim
            + self.weights.exact_draw_match * (exact_count.astype(np.float64) / max(1, n))
            + self.weights.same_day_progress_bonus * same_progress.astype(np.float64)
        )

        k = min(self.top_k, win_count)
        if k <= 0:
            return []
        top_idx = np.argsort(-similarity, kind="mergesort")[:k]

        out: list[RetrievalMatch] = []
        for idx in top_idx.tolist():
            end_idx = int(ends[idx])
            out.append(
                RetrievalMatch(
                    end_issue=history[end_idx].issue,
                    similarity=float(similarity[idx]),
                    next_draw_numbers=history[end_idx + 1].numbers,
                    same_day_progress=bool(same_progress[idx]),
                    exact_draw_match_count=int(exact_count[idx]),
                    exact_window_match=bool(exact_count[idx] == n),
                )
            )
        return out


def retrieval_features(matches: list[RetrievalMatch], candidate: int, context_n: int) -> dict[str, float]:
    if not matches:
        return {
            "retrieval_match_count_topk": 0.0,
            "retrieval_similarity_mean": 0.0,
            "retrieval_similarity_max": 0.0,
            "retrieval_exact_window_match_count": 0.0,
            "retrieval_exact_draw_match_count_mean": 0.0,
            "retrieval_next_draw_vote_count": 0.0,
            "retrieval_next_draw_weighted_vote": 0.0,
            "retrieval_next_draw_posterior": 0.0,
            "retrieval_same_day_progress_bonus": 0.0,
            "retrieval_top1_hit_flag": 0.0,
            "retrieval_top3_hit_flag": 0.0,
            "retrieval_dynamic_context_n": float(context_n),
        }
    similarities = [m.similarity for m in matches]
    total_weight = sum(similarities)
    vote_count = sum(1 for m in matches if candidate in m.next_draw_numbers)
    weighted_vote = sum(m.similarity for m in matches if candidate in m.next_draw_numbers)
    return {
        "retrieval_match_count_topk": float(len(matches)),
        "retrieval_similarity_mean": float(sum(similarities) / len(similarities)),
        "retrieval_similarity_max": float(max(similarities)),
        "retrieval_exact_window_match_count": float(sum(1 for m in matches if m.exact_window_match)),
        "retrieval_exact_draw_match_count_mean": float(sum(m.exact_draw_match_count for m in matches) / len(matches)),
        "retrieval_next_draw_vote_count": float(vote_count),
        "retrieval_next_draw_weighted_vote": float(weighted_vote),
        "retrieval_next_draw_posterior": float(weighted_vote / total_weight) if total_weight else 0.0,
        "retrieval_same_day_progress_bonus": float(sum(1 for m in matches if m.same_day_progress) / len(matches)),
        "retrieval_top1_hit_flag": 1.0 if candidate in matches[0].next_draw_numbers else 0.0,
        "retrieval_top3_hit_flag": 1.0 if any(candidate in m.next_draw_numbers for m in matches[:3]) else 0.0,
        "retrieval_dynamic_context_n": float(context_n),
    }


def retrieval_features_frame(matches: list[RetrievalMatch], context_n: int) -> dict[int, dict[str, float]]:
    if not matches:
        base = retrieval_features(matches, 1, context_n)
        return {cand: dict(base) for cand in range(1, 81)}

    similarities = np.asarray([m.similarity for m in matches], dtype=np.float64)
    next_hit = np.zeros((len(matches), 80), dtype=np.float64)
    for i, m in enumerate(matches):
        next_hit[i, np.asarray(m.next_draw_numbers, dtype=np.int64) - 1] = 1.0
    vote_count = np.sum(next_hit, axis=0)
    weighted_vote = np.sum(next_hit * similarities[:, None], axis=0)
    total_weight = float(np.sum(similarities))
    top1 = next_hit[0]
    top3 = np.max(next_hit[: min(3, len(matches))], axis=0)
    match_count = float(len(matches))
    sim_mean = float(np.mean(similarities))
    sim_max = float(np.max(similarities))
    exact_window = float(sum(1 for m in matches if m.exact_window_match))
    exact_draw_mean = float(np.mean([m.exact_draw_match_count for m in matches]))
    same_day_bonus = float(np.mean([1.0 if m.same_day_progress else 0.0 for m in matches]))
    out: dict[int, dict[str, float]] = {}
    for cand in range(1, 81):
        idx = cand - 1
        row = {
            "retrieval_match_count_topk": match_count,
            "retrieval_similarity_mean": sim_mean,
            "retrieval_similarity_max": sim_max,
            "retrieval_exact_window_match_count": exact_window,
            "retrieval_exact_draw_match_count_mean": exact_draw_mean,
            "retrieval_next_draw_vote_count": float(vote_count[idx]),
            "retrieval_next_draw_weighted_vote": float(weighted_vote[idx]),
            "retrieval_next_draw_posterior": float(weighted_vote[idx] / total_weight) if total_weight else 0.0,
            "retrieval_same_day_progress_bonus": same_day_bonus,
            "retrieval_top1_hit_flag": float(top1[idx]),
            "retrieval_top3_hit_flag": float(top3[idx]),
            "retrieval_dynamic_context_n": float(context_n),
        }
        out[cand] = row
    return out


def aggregate_next_draw_votes(matches: list[RetrievalMatch]) -> dict[int, float]:
    votes: Counter[int] = Counter()
    for m in matches:
        for n in m.next_draw_numbers:
            votes[n] += m.similarity
    return {k: float(v) for k, v in votes.items()}
