from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OfficialCurrentFetcher:
    schedule_url: str = "https://www.taiwanlottery.com/run_lottery/schedule/"

    def fetch_latest(self) -> list[dict]:
        return []
