"""Reddit enrichment: the payload mapping, and the guard that keeps refusals out of the DB.

Two things are being locked here.

**The mapping.** ``ups`` and ``num_comments`` must land in *different* columns. Upvotes
are reach and feed ``engagement``, which multiplies the score up; comments are saturation
and feed ``answers_total``, which the scorer reads as ``existing_answer_count`` when
deciding ``actionable``. Summing them into one number would make a busy thread
simultaneously more attractive and more crowded, which is incoherent.

**The guard.** ``is_gated`` must reject on *shape*, not on status codes. This is a
regression test for a bug that already shipped once on Quora: ``is_challenge``
enumerated 403 and let 429 through, so 29 error pages were stored with
``fetch_status='ok'`` and ``body='Error 429 (Too Many Requests)'``. Nothing downstream
caught it -- the prefilter's semantic kill happened to drop them, which is luck, not a
guard. Reddit's refusal is a ~190KB HTML page and would parse into a plausible row.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

from pipeline import actors
from scrape import reddit

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
POST = (FIXTURES / "reddit_post.json").read_text(encoding="utf-8")
GATED = (FIXTURES / "reddit_gated.html").read_text(encoding="utf-8")


# --- the guard ---------------------------------------------------------------------

def test_gated_html_is_rejected() -> None:
    """The captured 403. A real refusal, not a synthesised one."""
    assert reddit.is_gated(403, GATED) is True


def test_a_200_carrying_html_is_still_gated() -> None:
    """The exact shape of the Quora incident: the status says success and the body is an
    error page. Trusting the code alone is what stored 29 poisoned rows."""
    assert reddit.is_gated(200, GATED) is True
    assert reddit.is_gated(200, "<!DOCTYPE html><html>whatever</html>") is True


def test_real_payload_is_not_gated() -> None:
    assert reddit.is_gated(200, POST) is False


def test_empty_body_is_gated() -> None:
    """A truncated or empty response must not slip through as an empty post."""
    assert reddit.is_gated(200, "") is True
    assert reddit.is_gated(200, "   ") is True


# --- the mapping -------------------------------------------------------------------

def test_parses_the_four_mapped_fields() -> None:
    page = reddit.parse(POST, "https://www.reddit.com/comments/1i9by4t/")
    assert page.ups and page.ups > 0
    assert page.num_comments and page.num_comments > 0
    assert page.posted_at is not None and page.posted_at.tzinfo is not None
    assert page.title


def test_body_is_title_plus_selftext() -> None:
    page = reddit.parse(POST, "u")
    assert page.title in page.body
    if page.selftext:
        assert page.selftext in page.body


def test_link_post_without_selftext_still_has_a_body() -> None:
    """Link posts carry an empty selftext. Degrading to the title is still no worse than
    the SERP row this replaces, so it must not read as an empty fetch."""
    payload = json.loads(POST)
    payload[0]["data"]["children"][0]["data"]["selftext"] = ""
    page = reddit.parse(json.dumps(payload), "u")
    assert page.body == page.title
    assert page.body


def test_unexpected_shape_degrades_instead_of_raising() -> None:
    """A worker must not die because Reddit changed its envelope. is_gated is the guard;
    the parser's job is to come back empty, not to explode."""
    for junk in ("{}", "[]", '{"data": null}', '[{"data": {"children": []}}]'):
        page = reddit.parse(junk, "u")
        assert page.body == ""


def test_post_id_handles_both_url_forms() -> None:
    """SERP rows arrive in both shapes; the short one is what discovery stores."""
    assert reddit.post_id("https://reddit.com/comments/1i9by4t") == "1i9by4t"
    assert reddit.post_id(
        "https://www.reddit.com/r/Wordpress/comments/1i9by4t/some_slug/") == "1i9by4t"
    assert reddit.post_id("https://www.reddit.com/r/Wordpress/") is None
    assert reddit.json_url("https://reddit.com/comments/1i9by4t") == \
        "https://www.reddit.com/comments/1i9by4t/.json"


# --- the actor path ----------------------------------------------------------------

@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def _save(candidate_id, body, author, posted_at, engagement, raw_key, status, **kw):
        calls.append({"body": body, "status": status, "engagement": engagement,
                      "posted_at": posted_at, **kw})

    monkeypatch.setattr(actors.repo, "save_fetched", _save)
    return calls


def _candidate() -> dict:
    return {"id": "c1", "platform": "reddit",
            "url": "https://reddit.com/comments/1i9by4t",
            "title": "SERP title", "snippet": "SERP snippet"}


def test_enriched_row_splits_ups_from_comments(monkeypatch, saved) -> None:
    """The whole point of the mapping: engagement gets upvotes, answers_total gets
    comments, and they are never the same number."""
    monkeypatch.setattr(reddit, "jar", lambda stale=None: {"c": "1"})
    monkeypatch.setattr(reddit, "fetch", lambda url, cookies, **kw: (200, POST))

    actors._fetch_reddit("c1", _candidate())

    assert len(saved) == 1
    row, expected = saved[0], reddit.parse(POST, "u")
    assert row["status"] == "ok"
    assert row["engagement"] == expected.ups
    assert row["answers_total"] == expected.num_comments
    assert row["posted_at"] is not None, "real recency must replace the 0.5 default"
    assert "SERP snippet" not in row["body"], "enriched rows use the post, not the SERP row"


def test_gated_response_falls_back_to_the_serp_row(monkeypatch, saved) -> None:
    """The safety property. A stale jar or a Reddit change must cost detail, never a
    candidate -- reddit supplies most of this pipeline's leads."""
    monkeypatch.setattr(reddit, "jar", lambda stale=None: {"c": "1"})
    monkeypatch.setattr(reddit, "fetch", lambda url, cookies, **kw: (403, GATED))

    actors._fetch_reddit("c1", _candidate())

    assert len(saved) == 1
    assert saved[0]["status"] == "ok"
    assert "SERP title" in saved[0]["body"] and "SERP snippet" in saved[0]["body"]
    assert saved[0]["posted_at"] is None


def test_a_gated_response_asks_for_a_replacement_jar_once(monkeypatch, saved) -> None:
    """A dead jar is the overwhelmingly likely cause, so retry once -- but only once, or
    a Reddit-wide outage costs a browser launch per URL. The second ask must name the jar
    that just failed, which is how a burst of gated fetches shares one replacement."""
    asked: list[dict | None] = []
    monkeypatch.setattr(reddit, "jar",
                        lambda stale=None: (asked.append(stale) or {"c": "1"}))
    monkeypatch.setattr(reddit, "fetch", lambda url, cookies, **kw: (403, GATED))

    actors._fetch_reddit("c1", _candidate())

    assert asked == [None, {"c": "1"}], f"expected one replacement ask, got {asked}"


def test_second_attempt_succeeding_is_used(monkeypatch, saved) -> None:
    """The re-mint has to actually be able to rescue the fetch, not just run."""
    calls = {"n": 0}

    def _fetch(url, cookies, **kw):
        calls["n"] += 1
        return (403, GATED) if calls["n"] == 1 else (200, POST)

    monkeypatch.setattr(reddit, "jar", lambda stale=None: {"c": "1"})
    monkeypatch.setattr(reddit, "fetch", _fetch)

    actors._fetch_reddit("c1", _candidate())

    assert saved[0]["posted_at"] is not None, "the retry's payload should have been used"


def test_network_error_falls_back_rather_than_raising(monkeypatch, saved) -> None:
    """_fetch_reddit sits on dramatiq's retry path. Raising would burn the message's
    retry budget for a candidate we can still serve from the SERP row."""
    def _boom(*a, **k):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(reddit, "jar", lambda stale=None: {"c": "1"})
    monkeypatch.setattr(reddit, "fetch", _boom)

    actors._fetch_reddit("c1", _candidate())      # must not raise

    assert saved[0]["status"] == "ok"
    assert "SERP title" in saved[0]["body"]


def test_non_post_url_never_fetches(monkeypatch, saved) -> None:
    """Subreddit and profile URLs turn up in SERP results and have no .json post."""
    monkeypatch.setattr(reddit, "jar",
                        lambda stale=None: pytest.fail("must not mint for a non-post URL"))
    candidate = _candidate() | {"url": "https://www.reddit.com/r/Wordpress/"}

    actors._fetch_reddit("c1", candidate)

    assert saved[0]["status"] == "ok"
    assert "SERP title" in saved[0]["body"]


# --- minting -------------------------------------------------------------------------

def test_mint_runs_the_browser_in_a_child_process(monkeypatch) -> None:
    """Playwright's sync API needs a main thread. In a dramatiq worker thread it dies
    with [Errno 9] Bad file descriptor, which is silent -- the fetch just falls back."""
    seen = {}

    def _run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout='{"loid": "x"}', stderr="")

    monkeypatch.setattr(reddit.subprocess, "run", _run)

    assert reddit.mint_jar() == {"loid": "x"}
    assert seen["cmd"][1:] == ["-m", "scrape.reddit"]


def test_a_failed_child_raises_rather_than_returning_an_empty_jar(monkeypatch) -> None:
    """An empty jar would be cached for the full TTL and gate every fetch behind it."""
    monkeypatch.setattr(
        reddit.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom"))

    with pytest.raises(RuntimeError, match="boom"):
        reddit.mint_jar()


# --- jar sharing ---------------------------------------------------------------------

class _FakeRedis:
    """Just enough Redis for jar(): get/setex over a dict, no expiry simulation."""

    def __init__(self, **seed: bytes) -> None:
        self.store: dict[str, bytes] = dict(seed)

    def get(self, key):                     # noqa: D102
        return self.store.get(key)

    def setex(self, key, _ttl, value):      # noqa: D102
        self.store[key] = value


@pytest.fixture
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> _FakeRedis:
    from core import limits

    fake = _FakeRedis()
    monkeypatch.setattr(limits, "_redis", fake)
    return fake


def test_a_cold_jar_is_minted_once_for_the_whole_thread_pool(monkeypatch, fake_redis) -> None:
    """Every reddit fetch in a run is enqueued at once. Without the lock plus the
    re-check inside it, each worker thread launches its own chromium on a cold jar."""
    import threading
    import time

    mints = []
    start = threading.Barrier(6)

    def _mint():
        time.sleep(0.05)                    # a real launch is ~7s; this is the same race
        mints.append(1)
        return {"loid": "x"}

    def _worker():
        start.wait(timeout=5)               # all six ask for the jar at the same moment
        reddit.jar()

    monkeypatch.setattr(reddit, "mint_jar", _mint)
    threads = [threading.Thread(target=_worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(mints) == 1, f"expected one browser launch, got {len(mints)}"
    assert json.loads(fake_redis.store[reddit._JAR_KEY]) == {"loid": "x"}


def test_a_burst_of_gated_fetches_shares_one_replacement(monkeypatch, fake_redis) -> None:
    """A jar dies of volume, so when it dies every in-flight fetch gates within a second
    or two of the others. All of them naming the same dead jar must cost one launch."""
    mints = []
    monkeypatch.setattr(reddit, "mint_jar", lambda: (mints.append(1) or {"loid": str(len(mints))}))

    dead = {"loid": "dead"}
    fake_redis.store[reddit._JAR_KEY] = json.dumps(dead)
    replacements = [reddit.jar(stale=dead) for _ in range(6)]

    assert len(mints) == 1, f"expected one replacement for the burst, got {len(mints)}"
    assert all(r == {"loid": "1"} for r in replacements), "all six must get the new jar"


def test_the_next_jar_to_die_is_replaced_too(monkeypatch, fake_redis) -> None:
    """~30 fetches later the replacement dies in turn. A time-based grace window would
    suppress this second mint; naming the dead jar cannot."""
    mints = []
    monkeypatch.setattr(reddit, "mint_jar", lambda: (mints.append(1) or {"loid": str(len(mints))}))

    first = reddit.jar(stale={"loid": "dead"})
    second = reddit.jar(stale=first)

    assert len(mints) == 2 and first != second


def test_a_live_jar_is_never_replaced(monkeypatch, fake_redis) -> None:
    """The common path: nothing gated, so nothing mints."""
    fake_redis.store[reddit._JAR_KEY] = json.dumps({"loid": "live"})
    monkeypatch.setattr(reddit, "mint_jar",
                        lambda: pytest.fail("must not mint while the jar works"))

    assert reddit.jar() == {"loid": "live"}
    assert reddit.jar(stale={"loid": "some other dead jar"}) == {"loid": "live"}


def test_an_unreadable_cached_jar_is_replaced(monkeypatch, fake_redis) -> None:
    fake_redis.store[reddit._JAR_KEY] = b"not json"
    monkeypatch.setattr(reddit, "mint_jar", lambda: {"loid": "x"})

    assert reddit.jar() == {"loid": "x"}


def test_reddit_fetch_is_paced() -> None:
    """Unpaced concurrent fetching is what turned Quora into 541 rejections and zero
    successes. Reddit's ceiling is unmeasured, so it is paced from the start."""
    from core import limits

    assert "reddit.com" in limits.LIMITS
    windows = {w for _, w in limits.LIMITS["reddit.com"]}
    assert len(windows) > 1, "a per-second rate alone cannot express a longer quota"
