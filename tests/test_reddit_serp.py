"""``_from_serp``: the discovery row used as the content. Quora's path, Reddit's fallback.

Reddit lived here permanently on the finding that its anonymous surfaces are all gated
(200 + "wait for verification" on HTML even under Chrome TLS impersonation, 403 on
old.reddit and every .json variant). That was measured with one client and generalised
too far -- the gate is state, and a cookie jar clears it -- so reddit now enriches via
``_fetch_reddit`` and lands here only when that fails (see tests/test_reddit_json.py).

The invariant these tests lock is unchanged and is what makes the fallback safe: this
function turns title + snippet into a body **without touching the network** and
**without ever raising** -- a raise here would put a no-op on dramatiq's retry path
forever, since no request means no transient failure to recover from.
"""

from __future__ import annotations

import pytest

from pipeline import actors


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def _save(candidate_id, body, author, posted_at, engagement, raw_key, status, **kw):
        calls.append({"body": body, "status": status, "engagement": engagement, **kw})

    monkeypatch.setattr(actors.repo, "save_fetched", _save)
    return calls


def _candidate(title: str | None, snippet: str | None) -> dict:
    return {"id": "c1", "platform": "reddit", "url": "https://reddit.com/comments/1nhh66z",
            "title": title, "snippet": snippet}


def test_reddit_is_an_enabled_platform() -> None:
    """The expand prompt and LEAF_SCHEMA have always emitted reddit queries; leaving
    reddit out of PLATFORMS silently discarded every one of them."""
    assert "reddit" in actors.PLATFORMS


def test_title_and_snippet_become_the_body(saved) -> None:
    actors._from_serp("c1", _candidate(
        "What are the best AI Tools for Writing better SEO Content?",
        "Need suggestions for AI tools that helps in writing better SEO Content.",
    ))
    assert len(saved) == 1
    assert saved[0]["status"] == "ok"
    assert "best AI Tools for Writing" in saved[0]["body"]
    assert "Need suggestions" in saved[0]["body"]


def test_title_alone_is_enough(saved) -> None:
    """The title is the intent statement on its own -- a missing snippet is not a failure."""
    actors._from_serp("c1", _candidate("Best AI SEO content generator?", None))
    assert saved[0]["status"] == "ok"
    assert saved[0]["body"] == "Best AI SEO content generator?"


def test_no_text_at_all_is_empty_not_ok(saved) -> None:
    actors._from_serp("c1", _candidate(None, None))
    assert saved[0]["status"] == "empty"
    assert saved[0]["body"] is None


def test_never_raises_so_it_never_retries(saved) -> None:
    """No network call means no transient failure. Raising would retry a pure function
    three times and land on fetch_failed for no reason."""
    for candidate in (_candidate(None, None), _candidate("", ""), _candidate("t", None)):
        actors._from_serp("c1", candidate)  # must not raise
    assert len(saved) == 3


def test_fetch_candidate_routes_reddit_to_the_enricher(monkeypatch, saved) -> None:
    """End-to-end through the actor body: reddit must reach _fetch_reddit, not the
    'no fetcher' branch and not Quora's SERP-only branch."""
    monkeypatch.setattr(actors.repo, "get_candidate",
                        lambda cid: _candidate("Best AI SEO content generator?", "snippet here"))
    monkeypatch.setattr(actors, "_arrive_fetch", lambda run_id, cid: None)

    seen: list[str] = []
    monkeypatch.setattr(actors, "_fetch_reddit", lambda cid, cand: seen.append(cid))

    actors.fetch_candidate("run1", "c1")

    assert seen == ["c1"]
    assert not saved, "routing to the enricher must not also write a SERP row"


def test_fetch_candidate_still_routes_quora_to_serp(monkeypatch, saved) -> None:
    """Quora's fetch was removed for measured reasons and must not be revived by the
    reddit split -- both platforms shared one branch before it."""
    candidate = {"id": "c1", "platform": "quora", "url": "https://quora.com/x",
                 "title": "How do I stop AI text sounding generic?", "snippet": None}
    monkeypatch.setattr(actors.repo, "get_candidate", lambda cid: candidate)
    monkeypatch.setattr(actors, "_arrive_fetch", lambda run_id, cid: None)
    monkeypatch.setattr(actors, "_fetch_reddit",
                        lambda *a: pytest.fail("quora must not reach the reddit enricher"))

    actors.fetch_candidate("run1", "c1")

    assert saved and saved[0]["status"] == "ok"
    assert saved[0]["body"] == "How do I stop AI text sounding generic?"
