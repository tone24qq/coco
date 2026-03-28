from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta

import requests
from bs4 import BeautifulSoup

from .config import AppConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class FetchError(RuntimeError):
    """Raised when upstream fetch/parse fails."""


def parse_draws_from_html(
    html: str,
) -> list[tuple[int, list[int]]]:
    soup = BeautifulSoup(html, "html.parser")
    draws: set[tuple[int, tuple[int, ...]]] = set()

    for tr in soup.find_all("tr"):
        text_blocks = [
            td.get_text(strip=True)
            for td in tr.find_all(["td", "th"])
        ]
        period = None
        nums: list[int] = []

        for block in text_blocks:
            clean_str = re.sub(r"\D", "", block)
            if not clean_str:
                continue

            val = int(clean_str)
            if val > 100000000:
                period = val
            elif 1 <= val <= 80:
                nums.append(val)

        unique_nums = tuple(sorted(set(nums)))
        if period and len(unique_nums) == 20:
            draws.add((period, unique_nums))

    return sorted([(p, list(n)) for p, n in draws], key=lambda x: x[0])


def parse_draws_from_json(payload: str) -> list[tuple[int, list[int]]]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise FetchError(f"Invalid upstream JSON payload: {exc}") from exc

    draws: list[tuple[int, list[int]]] = []
    if not isinstance(data, list):
        raise FetchError("Unexpected upstream JSON format (expected list)")

    for row in data:
        if not isinstance(row, dict):
            continue
        period_text = str(row.get("No", "")).strip()
        numbers_text = str(row.get("BigShowOrder", "")).strip()
        if not period_text.isdigit() or not numbers_text:
            continue
        numbers = [int(x) for x in re.findall(r"\d+", numbers_text)]
        unique = sorted(set(numbers))
        if len(unique) == 20:
            draws.append((int(period_text), unique))

    return sorted(draws, key=lambda x: x[0])


def fetch_latest_draws(
    config: AppConfig = DEFAULT_CONFIG,
) -> tuple[list[list[int]], int]:
    logger.info(
        "Fetching latest draws from %s",
        config.source_url,
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        ),
        "Referer": "https://www.google.com/",
    }

    endpoint = f"{config.source_url.rstrip('/')}/GetBingoData"
    draw_map: dict[int, list[int]] = {}

    for delta in range(config.history_lookback_days):
        day = date.today() - timedelta(days=delta)
        try:
            response = requests.get(
                endpoint,
                params={"date": day.isoformat()},
                headers=headers,
                timeout=config.request_timeout,
            )
            response.raise_for_status()
            day_draws = parse_draws_from_json(response.text)
            for period, nums in day_draws:
                draw_map[period] = nums
            if len(draw_map) >= config.recent_draws_count:
                break
        except (requests.RequestException, FetchError) as exc:
            logger.warning(
                "JSON fetch failed for %s: %s",
                day.isoformat(),
                exc,
            )

    draws = sorted(draw_map.items(), key=lambda x: x[0])

    if not draws:
        try:
            page = requests.get(
                config.source_url,
                headers=headers,
                timeout=config.request_timeout,
            )
            page.raise_for_status()
            draws = parse_draws_from_html(page.text)
        except requests.RequestException as exc:
            raise FetchError(f"Failed to fetch upstream data: {exc}") from exc

    if not draws:
        raise FetchError("No valid draws parsed from upstream source")

    recent_draws = draws[-config.recent_draws_count:]
    if len(recent_draws) < config.recent_draws_count:
        raise FetchError(
            "Not enough draws to compute prediction "
            f"(got={len(recent_draws)}, need={config.recent_draws_count})"
        )

    latest_period = recent_draws[-1][0]
    return [d[1] for d in recent_draws], latest_period
