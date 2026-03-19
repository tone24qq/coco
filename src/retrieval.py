from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from src.utils import DrawRecord


@dataclass
class RetrievalMatch:
    end_issue: str
    similarity: float
    next_draw_numbers: tuple[int, ...]
    same_day_progress: bool


class SimilarWindowRetriever:
    def __init__(self, window_size: int = 100, top_k: int = 20, progress_weight: float = 0.2) -> None:
        self.window_size = window_size
        self.top_k = top_k
        self.progress_weight = progress_weight

    @staticmethod
    def _freq_vector(rows: list[DrawRecord]) -> list[float]:
        freq = [0.0] * 80
        for row in rows:
            for n in row.numbers:
                freq[n - 1] += 1.0
        total = max(1.0, len(rows) * 20.0)
        return [v / total for v in freq]

    @staticmethod
    def _distance(v1: list[float], v2: list[float]) -> float:
        return sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    def query(
        self,
        history: list[DrawRecord],
        target_window: list[DrawRecord],
        day_issue_index: int,
    ) -> list[RetrievalMatch]:
        if len(target_window) < self.window_size:
            return []
        target_vec = self._freq_vector(target_window[-self.window_size :])
        matches: list[RetrievalMatch] = []
        for end_idx in range(self.window_size - 1, len(history) - 1):
            candidate = history[end_idx - self.window_size + 1 : end_idx + 1]
            next_row = history[end_idx + 1]
            cand_vec = self._freq_vector(candidate)
            dist = self._distance(target_vec, cand_vec)
            similarity = 1.0 / (1.0 + dist)
            same_progress = history[end_idx].day_issue_index == day_issue_index
            if same_progress:
                similarity += self.progress_weight
            matches.append(
                RetrievalMatch(
                    end_issue=history[end_idx].issue,
                    similarity=similarity,
                    next_draw_numbers=next_row.numbers,
                    same_day_progress=same_progress,
                )
            )
        matches.sort(key=lambda x: x.similarity, reverse=True)
        return matches[: self.top_k]


def retrieval_features(matches: list[RetrievalMatch], candidate: int) -> dict[str, float]:
    if not matches:
        return {
            "retrieval_match_count_topk": 0.0,
            "retrieval_similarity_mean": 0.0,
            "retrieval_similarity_max": 0.0,
            "retrieval_weighted_hit_score": 0.0,
            "retrieval_weighted_next_draw_posterior": 0.0,
            "retrieval_next_draw_vote_count": 0.0,
            "retrieval_next_draw_weighted_rank": 0.0,
            "retrieval_same_day_progress_bonus": 0.0,
            "retrieval_exact_progress_match_flag": 0.0,
        }
    total_weight = sum(m.similarity for m in matches)
    weighted_hits = sum(m.similarity for m in matches if candidate in m.next_draw_numbers)
    vote_count = sum(1.0 for m in matches if candidate in m.next_draw_numbers)
    same_progress = sum(1.0 for m in matches if m.same_day_progress)
    return {
        "retrieval_match_count_topk": float(len(matches)),
        "retrieval_similarity_mean": total_weight / len(matches),
        "retrieval_similarity_max": max(m.similarity for m in matches),
        "retrieval_weighted_hit_score": weighted_hits,
        "retrieval_weighted_next_draw_posterior": weighted_hits / total_weight if total_weight else 0.0,
        "retrieval_next_draw_vote_count": vote_count,
        "retrieval_next_draw_weighted_rank": weighted_hits / (1.0 + vote_count),
        "retrieval_same_day_progress_bonus": same_progress / len(matches),
        "retrieval_exact_progress_match_flag": 1.0 if any(m.same_day_progress for m in matches) else 0.0,
    }
