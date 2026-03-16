from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from urllib import error, request

AUZO_ANALYSIS_URL = "https://lotto.auzo.tw/analyes.php?lotto=keno&action={action}"
AUZO_RI_URL = "https://lotto.auzo.tw/RI.php"
AUZO_RJ_URL = "https://lotto.auzo.tw/RJ.php"


@dataclass
class AuzoConfig:
    timeout_seconds: float = 2.0
    ttl_seconds: int = 60


_CACHE: dict[str, tuple[float, dict]] = {}


def _http_get(url: str, timeout: float) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    req = request.Request(url, headers=headers)
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _extract_issue_ids(html: str, limit: int = 50) -> list[int]:
    found = [int(x) for x in re.findall(r"\b(\d{9})\b", html)]
    uniq: list[int] = []
    seen = set()
    for x in found:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
        if len(uniq) >= limit:
            break
    return uniq


def _extract_float_features(html: str, limit: int = 20) -> list[float]:
    out: list[float] = []
    for token in re.findall(r"-?\d+(?:\.\d+)?", html):
        try:
            val = float(token)
        except ValueError:
            continue
        if abs(val) > 1e9:
            continue
        out.append(val)
        if len(out) >= limit:
            break
    return out


def fetch_auzo_external_analysis(config: AuzoConfig | None = None) -> dict:
    cfg = config or AuzoConfig()
    cache_key = f"{cfg.timeout_seconds}:{cfg.ttl_seconds}"
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and (now - cached[0]) <= cfg.ttl_seconds:
        return {**cached[1], "cache_hit": True}

    actions = [
        "comprehensive",
        "locations",
        "shape_oe",
        "shape_bs",
        "shape_po",
        "sumvalue",
        "span",
        "average",
        "total_reduce_mantissa",
        "max_min_mantissa",
    ]
    try:
        action_payload = {}
        for action in actions:
            html = _http_get(
                AUZO_ANALYSIS_URL.format(action=action), cfg.timeout_seconds
            )
            action_payload[action] = {
                "issues_hint": _extract_issue_ids(html, limit=20),
                "numeric_preview": _extract_float_features(html, limit=20),
            }

        ri_html = _http_get(AUZO_RI_URL, cfg.timeout_seconds)
        rj_html = _http_get(AUZO_RJ_URL, cfg.timeout_seconds)
        payload = {
            "external_status": "ok",
            "provider": "auzo",
            "external_analysis": {
                "actions": action_payload,
                "RI": {
                    "issues_hint": _extract_issue_ids(ri_html, limit=30),
                    "numeric_preview": _extract_float_features(ri_html, limit=30),
                },
                "RJ": {
                    "issues_hint": _extract_issue_ids(rj_html, limit=30),
                    "numeric_preview": _extract_float_features(rj_html, limit=30),
                },
            },
            "cache_hit": False,
        }
    except (error.URLError, TimeoutError, OSError, ValueError) as exc:
        payload = {
            "external_status": "degraded",
            "provider": "auzo",
            "external_analysis": {},
            "error": str(exc),
            "cache_hit": False,
        }

    _CACHE[cache_key] = (now, payload)
    return payload


def dump_external_analysis_json(path: str, config: AuzoConfig | None = None) -> dict:
    payload = fetch_auzo_external_analysis(config=config)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload
