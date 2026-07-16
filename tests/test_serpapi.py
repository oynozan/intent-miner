"""SerpApi (fallback discovery provider) error classification.

Same retryable-vs-permanent contract as Serper, plus SerpApi's quirk: a 200 response
can carry an ``error`` field -- a bad key/exhausted plan is permanent, "no results"
is just empty.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from discovery import serpapi
from discovery.common import DiscoveryError, DiscoveryPermanentError


@pytest.fixture(autouse=True)
def _serpapi_key(monkeypatch: pytest.MonkeyPatch):
    from core.config import settings

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    settings.cache_clear()
    yield
    settings.cache_clear()


ENDPOINT = "https://serpapi.com/search"


def test_missing_key_is_permanent(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.config import settings

    monkeypatch.setenv("SERPAPI_API_KEY", "")
    settings.cache_clear()
    with pytest.raises(DiscoveryPermanentError, match="not set"):
        serpapi.search("q", "quora")


@respx.mock
def test_401_invalid_key_is_permanent() -> None:
    respx.get(ENDPOINT).respond(401, json={"error": "Invalid API key. Your API key should be here: ..."})
    with pytest.raises(DiscoveryPermanentError) as exc:
        serpapi.search("q", "quora")
    assert "Invalid API key" in str(exc.value)


@respx.mock
def test_429_is_retryable() -> None:
    respx.get(ENDPOINT).respond(429, json={"error": "Your account has exceeded the rate limit"})
    with pytest.raises(DiscoveryError):
        serpapi.search("q", "quora")


@respx.mock
def test_500_is_retryable() -> None:
    respx.get(ENDPOINT).respond(500, text="server error")
    with pytest.raises(httpx.HTTPStatusError):
        serpapi.search("q", "quora")


@respx.mock
def test_200_parses_organic_results() -> None:
    respx.get(ENDPOINT).respond(
        200,
        json={
            "search_metadata": {"status": "Success"},
            "organic_results": [
                {"position": 1, "link": "https://quora.com/a", "title": "A", "snippet": "sa"},
                {"position": 2, "link": "https://quora.com/b", "title": "B", "snippet": "sb"},
                {"title": "no link -> skipped"},
            ],
        },
    )
    results = serpapi.search("q", "quora")
    assert [r.url for r in results] == ["https://quora.com/a", "https://quora.com/b"]


@respx.mock
def test_200_no_results_error_is_empty_not_a_failure() -> None:
    """SerpApi surfaces 'no results' as an error string on a 200. That is not a failure
    -- it is an empty result set for that query."""
    respx.get(ENDPOINT).respond(
        200, json={"error": "Google hasn't returned any results for this query."}
    )
    assert serpapi.search("q", "quora") == []


@respx.mock
def test_200_with_credit_error_is_permanent() -> None:
    """An exhausted plan on a 200 body must be permanent -- retrying won't add credits."""
    respx.get(ENDPOINT).respond(200, json={"error": "You have run out of searches on your plan."})
    with pytest.raises(DiscoveryPermanentError):
        serpapi.search("q", "quora")
