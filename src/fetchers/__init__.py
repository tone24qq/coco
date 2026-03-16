from src.fetchers.auzo_bingo import (
    BingoDrawFetcher,
    DrawRecord,
    FetchDrawsError,
    build_recent_draws,
)
from src.fetchers.official_current import OfficialCurrentFetcher
from src.fetchers.official_download import OfficialDownloadFetcher
from src.fetchers.official_history import OfficialHistoryFetcher
from src.fetchers.source_consensus import (
    build_fetch_health_report,
    compare_draw_records,
    detect_missing_issues,
    detect_source_conflict,
)
from src.fetchers.winwin_bingo import parse_winwin_bingo_rows

__all__ = [
    "BingoDrawFetcher",
    "DrawRecord",
    "FetchDrawsError",
    "build_recent_draws",
    "parse_winwin_bingo_rows",
    "OfficialDownloadFetcher",
    "OfficialHistoryFetcher",
    "OfficialCurrentFetcher",
    "compare_draw_records",
    "detect_missing_issues",
    "detect_source_conflict",
    "build_fetch_health_report",
]
