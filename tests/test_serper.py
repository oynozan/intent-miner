"""Serper error classification.

The regression this locks: a 403 "Unauthorized" (invalid key) is a PERMANENT failure.
Retrying it four times per query -- which is what raise_for_status() + max_retries=3
did -- floods the logs and never succeeds. Only 429/5xx/timeouts are retryable.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from discovery import serper
from discovery.serper import SerperError, SerperPermanentError


@pytest.fixture(autouse=True)
def _serper_key(monkeypatch: pytest.MonkeyPatch):
    from core.config import settings

    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    settings.cache_clear()
    yield
    settings.cache_clear()


ENDPOINT = "https://google.serper.dev/search"


@respx.mock
def test_403_unauthorized_is_permanent_with_the_real_message() -> None:
    """The exact failure the user hit: an invalid key. Must be permanent, and must
    carry Serper's own 'Unauthorized.' so the cause is obvious, not a generic 403."""
    respx.post(ENDPOINT).respond(403, json={"message": "Unauthorized.", "statusCode": 403})
    with pytest.raises(SerperPermanentError) as exc:
        serper.search("q", "quora")
    assert "Unauthorized." in str(exc.value)
    assert "403" in str(exc.value)
    assert not isinstance(exc.value, SerperError), "must not be classified as retryable"


def test_missing_key_is_permanent_not_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing key will not appear on retry."""
    from core.config import settings

    monkeypatch.setenv("SERPER_API_KEY", "")
    settings.cache_clear()
    with pytest.raises(SerperPermanentError, match="not set"):
        serper.search("q", "quora")


@respx.mock
def test_429_rate_limit_is_retryable() -> None:
    respx.post(ENDPOINT).respond(429, json={"message": "Too many requests"})
    with pytest.raises(SerperError) as exc:
        serper.search("q", "quora")
    assert not isinstance(exc.value, SerperPermanentError), "rate limit must stay retryable"


@respx.mock
def test_400_bad_request_is_permanent() -> None:
    respx.post(ENDPOINT).respond(400, json={"message": "Bad request"})
    with pytest.raises(SerperPermanentError):
        serper.search("q", "quora")


@respx.mock
def test_500_server_error_is_retryable() -> None:
    """5xx is transient -- it must NOT be permanent, so the retry machinery gets it."""
    respx.post(ENDPOINT).respond(500, text="upstream error")
    with pytest.raises(httpx.HTTPStatusError):
        serper.search("q", "quora")


@respx.mock
def test_200_parses_organic_results() -> None:
    respx.post(ENDPOINT).respond(
        200,
        json={
            "organic": [
                {"link": "https://quora.com/a", "title": "A", "snippet": "sa"},
                {"link": "https://quora.com/b", "title": "B", "snippet": "sb"},
                {"title": "no link -> skipped"},
            ]
        },
    )
    results = serper.search("q", "quora")
    assert [r.url for r in results] == ["https://quora.com/a", "https://quora.com/b"]
    assert results[0].title == "A"


@respx.mock
def test_permanent_error_survives_a_non_json_body() -> None:
    """Some 403s come back as HTML/plain text; _describe must not blow up on that."""
    respx.post(ENDPOINT).respond(403, text="<html>Forbidden</html>")
    with pytest.raises(SerperPermanentError):
        serper.search("q", "quora")
