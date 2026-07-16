"""Shared discovery types and helpers, used by every SERP provider.

Kept in its own leaf module (not in providers.py) so serper.py and serpapi.py can
import these without a circular import back through the dispatcher.

Both providers query Google, so the query builder (``site:`` scoping + ``after:``
freshness) and the result shape are identical -- only the endpoint, auth, and
response-field names differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import httpx

from core.urls import PLATFORM_LINKEDIN, PLATFORM_QUORA, PLATFORM_REDDIT

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


class DiscoveryError(RuntimeError):
    """A retryable failure -- rate limit, 5xx, timeout. Propagate it so dramatiq retries."""


class DiscoveryPermanentError(RuntimeError):
    """A non-retryable failure -- a bad key, forbidden/malformed request, or nothing configured.

    Retrying a 403 four times per query never succeeds; it just floods the logs. The
    actor catches this and fails the query immediately instead.
    """


def build_query(q: str, platform: str, freshness_months: int | None = None) -> str:
    """The Google query string. ``site:`` scope + optional ``after:`` freshness.

    Both Serper and SerpApi proxy Google and honour these operators, so the query is
    built once here regardless of which vendor serves it.
    """
    site = SITE_FILTERS.get(platform)
    parts = [site, q.strip()] if site else [q.strip()]
    if freshness_months:
        cutoff = date.today() - timedelta(days=int(freshness_months * 30.44))
        parts.append(f"after:{cutoff.isoformat()}")
    return " ".join(p for p in parts if p)


def describe(response: httpx.Response) -> str:
    """The provider's own error message, e.g. '403: Unauthorized.' or '401: Invalid API key'.

    Surfacing it is what turns a generic 'Client error 403' into an actionable cause.
    Serper puts it in ``message``, SerpApi in ``error`` -- check both.
    """
    message: Any
    try:
        body = response.json()
        message = body.get("message") or body.get("error") or response.text
    except Exception:
        message = response.text
    return f"{response.status_code}: {str(message)[:200]}"
