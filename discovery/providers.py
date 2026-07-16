"""Discovery dispatcher: Serper primary, SerpApi fallback.

Mirrors llm/providers.py -- try the primary provider, fall through to the next
configured one on any failure. Set DISCOVERY_PROVIDER=serpapi to flip the order.

Failure classification is the one subtlety. Each provider raises either a permanent
error (bad key) or a retryable one (rate limit, 5xx). After trying every configured
provider, the *combined* result is:

* **retryable** if any single provider failed retryably -- a later dramatiq retry of
  the whole query might get through once that rate limit clears, even if the other
  provider's key is permanently bad.
* **permanent** only if every provider failed permanently (all keys bad) -- there is
  nothing to wait for, so the actor fast-fails.

The individual exceptions and SerpResult are re-exported here so callers import one
module.
"""

from __future__ import annotations

import logging

from core.config import settings
from discovery import serper, serpapi
from discovery.common import (  # noqa: F401  -- re-exported for callers
    DiscoveryError,
    DiscoveryPermanentError,
    SerpResult,
    build_query,
)

log = logging.getLogger(__name__)

PROVIDER_NAMES = ("serper", "serpapi")


def _run(provider: str, q: str, platform: str, depth: int) -> list[SerpResult]:
    return {"serper": serper.search, "serpapi": serpapi.search}[provider](q, platform, depth)


def _configured(provider: str) -> bool:
    s = settings()
    if provider == "serper":
        return bool(s.serper_api_key)
    if provider == "serpapi":
        return bool(s.serpapi_api_key)
    return False


def provider_order() -> list[str]:
    primary = settings().discovery_provider
    if primary not in PROVIDER_NAMES:
        primary = "serper"
    return [primary] + [p for p in PROVIDER_NAMES if p != primary]


def search(q: str, platform: str, depth: int = 10) -> list[SerpResult]:
    """Run one query through the provider chain. Returns the first success."""
    order = provider_order()
    attempted: list[str] = []
    last_retryable: Exception | None = None
    last_permanent: Exception | None = None

    for provider in order:
        if not _configured(provider):
            log.info("discovery: skipping %s (no api key)", provider)
            continue
        attempted.append(provider)
        try:
            results = _run(provider, q, platform, depth)
        except DiscoveryPermanentError as exc:
            last_permanent = exc
            log.warning("discovery: %s permanently failed (%s); trying next", provider, exc)
            continue
        except Exception as exc:  # noqa: BLE001 -- retryable: DiscoveryError, httpx errors, etc.
            last_retryable = exc
            log.warning("discovery: %s failed (%s); trying next", provider, exc)
            continue
        if provider != order[0]:
            log.warning("discovery: served by fallback provider %s", provider)
        return results

    if not attempted:
        raise DiscoveryPermanentError(
            "no discovery provider configured -- set SERPER_API_KEY (primary) or SERPAPI_API_KEY (fallback)"
        )
    # A rate limit or 5xx anywhere means a retry is worth it; only all-permanent is fast-fail.
    if last_retryable is not None:
        raise DiscoveryError(f"all discovery providers failed (retryable) {attempted}: {last_retryable}")
    raise DiscoveryPermanentError(f"all discovery providers failed {attempted}: {last_permanent}")
