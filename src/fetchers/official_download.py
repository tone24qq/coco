from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OfficialDownloadFetcher:
    """Placeholder fetcher for official result_download yearly files.

    Local-first pipeline only calls this when local audit detects missing years/issues.
    """

    base_url: str = "https://www.taiwanlottery.com/"

    def fetch_year(self, year: int) -> list[dict]:
        # Keep interface for repair path; network implementation can be expanded safely.
        return []
