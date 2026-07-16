"""SerpApi (Google SERP) discovery -- the fallback provider.

Same job as Serper, different vendor: an HTTP GET to ``serpapi.com/search`` with the
key as a query param, returning ``organic_results``. Called via httpx (no SDK) to
keep it consistent with the Serper provider and dependency-free.

SerpApi has one quirk worth handling: a 200 response can still carry an ``error``
field. Some of those are just "no results for this query" (return empty); others are
account/credit problems (permanent) -- classified below.
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

ENDPOINT = "https://serpapi.com/search"

# Substrings in a 200-body error that mean "this will not fix itself on retry":
# a bad key or an exhausted plan. Anything else on a 200 is treated as empty results.
_PERMANENT_ERROR_MARKERS = ("api key", "run out of searches", "credit", "plan", "account", "unauthorized")


def search(q: str, platform: str, depth: int = 10, timeout: float = 15.0) -> list[SerpResult]:
    """Run one SerpApi query, or raise a classified DiscoveryError/DiscoveryPermanentError."""
    key = settings().serpapi_api_key
    if not key:
        raise DiscoveryPermanentError("SERPAPI_API_KEY is not set")

    query = build_query(q, platform, settings().freshness_months)
    response = httpx.get(
        ENDPOINT,
        params={"engine": "google", "q": query, "num": depth, "api_key": key},
        timeout=timeout,
    )

    code = response.status_code
    if code == 429:
        raise DiscoveryError(f"serpapi rate limited -- {describe(response)}")  # retryable
    if 400 <= code < 500:
        # 401 "Invalid API key", 400 bad request, etc. Retrying will not help.
        raise DiscoveryPermanentError(f"serpapi rejected the request -- {describe(response)}")
    response.raise_for_status()  # 5xx -> retryable HTTPStatusError

    payload = response.json()
    error = payload.get("error")
    if error:
        low = str(error).lower()
        if any(marker in low for marker in _PERMANENT_ERROR_MARKERS):
            raise DiscoveryPermanentError(f"serpapi: {str(error)[:200]}")
        # e.g. "Google hasn't returned any results for this query." -- just empty, not a failure.
        return []
    return _parse(payload)


def _parse(payload: dict[str, Any]) -> list[SerpResult]:
    results: list[SerpResult] = []
    for item in payload.get("organic_results", []) or []:
        link = item.get("link")
        if not link:
            continue
        results.append(SerpResult(url=link, title=item.get("title"), snippet=item.get("snippet")))
    return results
