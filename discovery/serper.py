"""Serper (Google SERP) discovery -- the primary provider.

Billing facts worth keeping in mind (they shape compile_queries, not this module):

* **num > 10 costs 2 credits, not 1** -- depth is a per-query knob so cheap branches
  stay at 10.
* **Credits expire after 6 months** -- buying a large pack "to be safe" is a real loss.

Error handling classifies by whether a retry could ever help, so the dispatcher and
the actor can treat a bad key differently from a rate limit -- see discovery/common.py.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from core.config import settings
from discovery.common import (
    DiscoveryError,
    DiscoveryPermanentError,
    SerpResult,
    build_query,
    describe,
)

log = logging.getLogger(__name__)

ENDPOINT = "https://google.serper.dev/search"


def credits_for(depth: int) -> int:
    """Serper bills 2 credits for any query requesting more than 10 results."""
    return 2 if depth > 10 else 1


def search(q: str, platform: str, depth: int = 10, timeout: float = 10.0) -> list[SerpResult]:
    """Run one Serper query, or raise a classified DiscoveryError/DiscoveryPermanentError."""
    key = settings().serper_api_key
    if not key:
        raise DiscoveryPermanentError("SERPER_API_KEY is not set")

    query = build_query(q, platform, settings().freshness_months)
    response = httpx.post(
        ENDPOINT,
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
        json={"q": query, "num": depth},
        timeout=timeout,
    )

    code = response.status_code
    if code == 429:
        raise DiscoveryError(f"serper rate limited -- {describe(response)}")  # retryable
    if 400 <= code < 500:
        # Bad key, forbidden, bad request. Surface Serper's own message so
        # "403: Unauthorized." (invalid key) is obvious, not a generic 403.
        raise DiscoveryPermanentError(f"serper rejected the request -- {describe(response)}")
    response.raise_for_status()  # 5xx (and anything else) -> retryable HTTPStatusError
    return _parse(response.json())


def _parse(payload: dict[str, Any]) -> list[SerpResult]:
    results: list[SerpResult] = []
    for item in payload.get("organic", []) or []:
        link = item.get("link")
        if not link:
            continue
        results.append(SerpResult(url=link, title=item.get("title"), snippet=item.get("snippet")))
    return results
