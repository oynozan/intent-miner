"""LinkedIn fetch actor: throttle retry, terminal empty, and the circuit breaker.

The behaviours here are the fix for "why no LinkedIn links": a burst-throttled 200
(gated page, no post data) must retry rather than be recorded as empty, a genuinely
text-less post must be terminal, and persistent throttling must trip a breaker instead
of hammering a throttled IP for the whole run.

Uses the real Redis (for the breaker) and monkeypatches the network fetch + the DB write.
"""

from __future__ import annotations

import os
import uuid

import pytest
from redis import Redis
from redis.exceptions import RedisError

from pipeline import actors
from scrape import linkedin

# See tests/test_limits.py: host is configurable too, so the suite runs both from the
# host (127.0.0.1:6380) and from inside a container (redis:6379).
REDIS_HOST = os.environ.get("TEST_REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("TEST_REDIS_PORT", "6380"))

_REAL_POST_LDJSON = (
    '<html><script type="application/ld+json">'
    '{"@type":"SocialMediaPosting","articleBody":"Looking for a tool to cut out video backgrounds",'
    '"datePublished":"2026-03-01T10:00:00Z","commentCount":3}'
    "</script></html>"
)
# A real throttled page: LARGE (so is_authwalled's <10KB check misses it) and carrying
# *some* ld+json (an Organization node, so the "ld+json not in html" check misses it too)
# but NO SocialMediaPosting node. This is the exact shape that slipped through as "empty"
# during the run, and the shape had_posting_ldjson is designed to catch.
_GATED_PAGE = (
    '<html><head><script type="application/ld+json">{"@type":"Organization","name":"LinkedIn"}</script></head>'
    "<body>" + ("Sign in to view this post. " * 600) + "</body></html>"
)


@pytest.fixture(autouse=True)
def _redis_up():
    try:
        Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=3).ping()
    except RedisError as exc:
        pytest.fail(f"redis unavailable on {REDIS_HOST}:{REDIS_PORT} ({exc})")


@pytest.fixture(autouse=True)
def _no_jitter(monkeypatch: pytest.MonkeyPatch):
    from core.config import settings

    monkeypatch.setenv("LINKEDIN_FETCH_JITTER_MS", "0")
    monkeypatch.setenv("LINKEDIN_THROTTLE_BREAKER", "3")
    settings.cache_clear()
    yield
    settings.cache_clear()


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def _save(candidate_id, body, author, posted_at, engagement, raw_key, status, **kw):
        calls.append({"candidate_id": candidate_id, "body": body, "status": status, **kw})

    monkeypatch.setattr(actors.repo, "save_fetched", _save)
    return calls


@pytest.fixture
def run_id() -> str:
    rid = f"litest-{uuid.uuid4()}"
    yield rid
    Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True).delete(f"linkedin_throttle:{rid}")


def _mock_fetch(monkeypatch: pytest.MonkeyPatch, html: str, status: int = 200) -> None:
    monkeypatch.setattr(linkedin, "fetch", lambda url, **k: (status, html, {}, url))


def test_real_post_is_saved_ok(monkeypatch, saved, run_id) -> None:
    _mock_fetch(monkeypatch, _REAL_POST_LDJSON)
    actors._fetch_linkedin(run_id, "c1", "https://linkedin.com/posts/x")
    assert saved and saved[0]["status"] == "ok"
    assert "cut out video backgrounds" in saved[0]["body"]


def test_throttled_gated_page_retries_and_bumps_breaker(monkeypatch, saved, run_id) -> None:
    """A gated 200 (no ld+json) must RAISE (so it retries), not be saved as empty."""
    _mock_fetch(monkeypatch, _GATED_PAGE)
    with pytest.raises(RuntimeError, match="throttled"):
        actors._fetch_linkedin(run_id, "c1", "https://linkedin.com/posts/x")
    assert saved == [], "a throttle must not be recorded as a terminal result"
    assert actors._bump_linkedin_throttle(run_id) >= 2, "throttle events must accumulate"


def test_genuinely_textless_post_is_terminal_empty(monkeypatch, saved, run_id) -> None:
    """A post with the ld+json node but empty articleBody is a real empty post -- no retry."""
    html = '<html><script type="application/ld+json">{"@type":"SocialMediaPosting","articleBody":""}</script></html>'
    _mock_fetch(monkeypatch, html)
    actors._fetch_linkedin(run_id, "c1", "https://linkedin.com/posts/x")  # must not raise
    assert saved and saved[0]["status"] == "empty"


def test_circuit_breaker_skips_once_throttling_is_persistent(monkeypatch, saved, run_id) -> None:
    """After the breaker threshold, LinkedIn fetches short-circuit to 'skipped' without
    even hitting the network -- no hammering a throttled IP."""
    # Trip the breaker (threshold is 3 via the fixture).
    for _ in range(3):
        actors._bump_linkedin_throttle(run_id)

    fetched: list[str] = []
    monkeypatch.setattr(linkedin, "fetch", lambda url, **k: fetched.append(url) or (200, _REAL_POST_LDJSON, {}, url))

    actors._fetch_linkedin(run_id, "c1", "https://linkedin.com/posts/x")

    assert fetched == [], "breaker must skip the network entirely when open"
    assert saved and saved[0]["status"] == "skipped"
    assert "circuit breaker" in (saved[0].get("error") or "")


def test_authwall_bumps_breaker_and_raises(monkeypatch, saved, run_id) -> None:
    _mock_fetch(monkeypatch, "<html>tiny</html>", status=999)
    with pytest.raises(RuntimeError, match="authwall"):
        actors._fetch_linkedin(run_id, "c1", "https://linkedin.com/posts/x")
    assert saved == []
