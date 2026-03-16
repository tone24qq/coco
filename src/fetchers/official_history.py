from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OfficialHistoryFetcher:
    base_url: str = "https://www.taiwanlottery.com/lotto/history/history_result/"

    def fetch_by_month(self, year: int, month: int) -> list[dict]:
        return []
