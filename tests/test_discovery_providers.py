"""Discovery dispatcher: Serper primary, SerpApi fallback.

The provider-level callers are monkeypatched, so these tests exercise ordering, the
fallback, and the combined retryable-vs-permanent classification -- not either vendor.
"""

from __future__ import annotations

import pytest

from discovery import providers, serpapi, serper
from discovery.common import DiscoveryError, DiscoveryPermanentError, SerpResult


@pytest.fixture(autouse=True)
def _reset_settings():
    from core.config import settings

    settings.cache_clear()
    yield
    settings.cache_clear()


def _both_keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERPER_API_KEY", "s")
    monkeypatch.setenv("SERPAPI_API_KEY", "a")
    monkeypatch.setenv("DISCOVERY_PROVIDER", "serper")


def _ok(url: str):
    return lambda q, p, d: [SerpResult(url=url, title="t", snippet="s")]


def _raise(exc):
    def _f(q, p, d):
        raise exc
    return _f


def test_serper_is_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    _both_keyed(monkeypatch)
    called: list[str] = []
    monkeypatch.setattr(serper, "search", lambda q, p, d: called.append("serper") or _ok("https://x/1")(q, p, d))
    monkeypatch.setattr(serpapi, "search", lambda q, p, d: called.append("serpapi") or _ok("https://x/2")(q, p, d))

    results = providers.search("q", "quora")
    assert [r.url for r in results] == ["https://x/1"]
    assert called == ["serper"], "serpapi must not be called when serper succeeds"


def test_falls_back_to_serpapi_when_serper_permanently_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """The user's exact scenario: Serper key is bad (403). SerpApi should cover it."""
    _both_keyed(monkeypatch)
    monkeypatch.setattr(serper, "search", _raise(DiscoveryPermanentError("serper 403: Unauthorized.")))
    monkeypatch.setattr(serpapi, "search", _ok("https://serpapi/win"))

    results = providers.search("q", "quora")
    assert [r.url for r in results] == ["https://serpapi/win"]


def test_falls_back_when_serper_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.setenv("SERPAPI_API_KEY", "a")
    monkeypatch.setenv("DISCOVERY_PROVIDER", "serper")
    monkeypatch.setattr(serper, "search", _raise(AssertionError("serper must not run without a key")))
    monkeypatch.setattr(serpapi, "search", _ok("https://serpapi/only"))

    assert [r.url for r in providers.search("q", "quora")] == ["https://serpapi/only"]


def test_discovery_provider_env_flips_the_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERPER_API_KEY", "s")
    monkeypatch.setenv("SERPAPI_API_KEY", "a")
    monkeypatch.setenv("DISCOVERY_PROVIDER", "serpapi")
    called: list[str] = []
    monkeypatch.setattr(serper, "search", lambda q, p, d: called.append("serper") or _ok("https://x/1")(q, p, d))
    monkeypatch.setattr(serpapi, "search", lambda q, p, d: called.append("serpapi") or _ok("https://x/2")(q, p, d))

    results = providers.search("q", "quora")
    assert [r.url for r in results] == ["https://x/2"]
    assert called == ["serpapi"], "DISCOVERY_PROVIDER=serpapi must make serpapi primary"


def test_all_permanent_is_permanent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both keys bad -> nothing to wait for -> fast-fail (permanent)."""
    _both_keyed(monkeypatch)
    monkeypatch.setattr(serper, "search", _raise(DiscoveryPermanentError("serper bad key")))
    monkeypatch.setattr(serpapi, "search", _raise(DiscoveryPermanentError("serpapi bad key")))

    with pytest.raises(DiscoveryPermanentError, match="all discovery providers failed"):
        providers.search("q", "quora")


def test_one_retryable_makes_the_whole_thing_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serper permanently bad but SerpApi merely rate-limited -> retry the whole query;
    a later attempt might get through SerpApi once the limit clears."""
    _both_keyed(monkeypatch)
    monkeypatch.setattr(serper, "search", _raise(DiscoveryPermanentError("serper bad key")))
    monkeypatch.setattr(serpapi, "search", _raise(DiscoveryError("serpapi rate limited")))

    with pytest.raises(DiscoveryError) as exc:
        providers.search("q", "quora")
    assert not isinstance(exc.value, DiscoveryPermanentError)


def test_no_provider_configured_is_permanent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    with pytest.raises(DiscoveryPermanentError, match="no discovery provider configured"):
        providers.search("q", "quora")
