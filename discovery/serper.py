"""Serper (Google SERP) discovery.

This is the only discovery channel in v1. Exa was declined; Reddit's API is deferred.

Two billing facts shape the interface:

* **num > 10 costs 2 credits, not 1.** Depth is therefore a per-query argument rather
  than a global default -- cheap exploratory branches stay at 10 and only the leaves
  worth the money go deeper. A global `num=20` silently doubles the run's bill.
* **Credits expire after 6 months.** Not something the code can act on, but it means
  buying a large pack "to be safe" is a real loss, not just deferred spend.

The ``after:`` operator passes through to Google and is the only freshness lever
available on LinkedIn -- and the only one Quora structurally cannot provide, since it
exposes no question creation date to logged-out clients.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import httpx

from core.config import settings
from core.urls import PLATFORM_LINKEDIN, PLATFORM_QUORA, PLATFORM_REDDIT

log = logging.getLogger(__name__)

ENDPOINT = "https://google.serper.dev/search"

# LinkedIn is scoped to /posts specifically: it is the only anonymously readable
# route. Scoping to linkedin.com alone would return profile and company URLs that
# canonicalization then refuses -- paying for results we cannot fetch.
SITE_FILTERS = {
    PLATFORM_REDDIT: "site:reddit.com",
    PLATFORM_QUORA: "site:quora.com",
    PLATFORM_LINKEDIN: "site:linkedin.com/posts",
}


@dataclass
class SerpResult:
    url: str
    title: str | None
    snippet: str | None


class SerperError(RuntimeError):
    pass


def build_query(q: str, platform: str, freshness_months: int | None = None) -> str:
    site = SITE_FILTERS.get(platform)
    parts = [site, q.strip()] if site else [q.strip()]
    if freshness_months:
        cutoff = date.today() - timedelta(days=int(freshness_months * 30.44))
        parts.append(f"after:{cutoff.isoformat()}")
    return " ".join(p for p in parts if p)


def credits_for(depth: int) -> int:
    """Serper bills 2 credits for any query requesting more than 10 results."""
    return 2 if depth > 10 else 1


def search(q: str, platform: str, depth: int = 10, timeout: float = 10.0) -> list[SerpResult]:
    """Run one SERP query. Raises on failure so dramatiq's retry middleware can act.

    Deliberately does not catch exceptions: swallowing them here is what made the
    spec's retries dead code. Let it propagate; the actor's on_retry_exhausted handles
    the terminal case.
    """
    key = settings().serper_api_key
    if not key:
        raise SerperError("SERPER_API_KEY is not set")

    query = build_query(q, platform, settings().freshness_months)
    response = httpx.post(
        ENDPOINT,
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
        json={"q": query, "num": depth},
        timeout=timeout,
    )
    if response.status_code == 429:
        raise SerperError("serper rate limited")
    response.raise_for_status()
    return _parse(response.json())


def _parse(payload: dict[str, Any]) -> list[SerpResult]:
    results: list[SerpResult] = []
    for item in payload.get("organic", []) or []:
        link = item.get("link")
        if not link:
            continue
        results.append(
            SerpResult(url=link, title=item.get("title"), snippet=item.get("snippet"))
        )
    return results
