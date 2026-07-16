"""run_query fast-fail wiring, and the /urls endpoint's repo query.

The fast-fail tests are pure unit tests (no infra) -- they prove run_query routes a
permanent error to _fail_query without raising (no retry) and lets a retryable error
propagate. The repo tests need the compose Postgres.
"""

from __future__ import annotations

import uuid

import pytest
from psycopg import OperationalError

from discovery import providers as discovery
from discovery.common import DiscoveryError, DiscoveryPermanentError
from pipeline import actors, repo


# --- run_query classification (no infra) ---

def _fake_query(qid: str) -> dict:
    return {"id": qid, "run_id": "run1", "q": "x", "platform": "quora", "depth": 10}


def test_run_query_fast_fails_on_permanent_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """When every discovery provider is permanently bad, go straight to _fail_query --
    no raise, so no retry storm."""
    monkeypatch.setattr(repo, "get_query", _fake_query)

    def raise_permanent(*a, **k):
        raise DiscoveryPermanentError("all discovery providers failed -- 403: Unauthorized.")

    monkeypatch.setattr(discovery, "search", raise_permanent)
    calls: list[tuple] = []
    monkeypatch.setattr(actors, "_fail_query", lambda r, q, e: calls.append((r, q, e)))

    actors.run_query.fn("run1", "q1")  # must NOT raise -> no retry

    assert calls, "permanent error must be routed to _fail_query"
    assert "Unauthorized." in calls[0][2]


def test_run_query_releases_barrier_when_query_vanished(monkeypatch: pytest.MonkeyPatch) -> None:
    """A vanished query row must still arrive at the barrier, or the run hangs forever."""
    monkeypatch.setattr(repo, "get_query", lambda qid: None)
    arrivals: list[tuple] = []
    monkeypatch.setattr(actors, "arrive", lambda run_id, stage, party, then: arrivals.append((run_id, stage, party)))

    actors.run_query.fn("run1", "gone")

    assert arrivals == [("run1", "discover", "gone")], "vanished query must release its barrier slot"


def test_run_query_propagates_retryable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rate limit / 5xx must propagate so the retry middleware handles it -- not fast-fail."""
    monkeypatch.setattr(repo, "get_query", _fake_query)

    def raise_retryable(*a, **k):
        raise DiscoveryError("all discovery providers failed (retryable)")

    monkeypatch.setattr(discovery, "search", raise_retryable)
    monkeypatch.setattr(actors, "_fail_query", lambda *a: pytest.fail("retryable error must not fast-fail"))

    with pytest.raises(DiscoveryError):
        actors.run_query.fn("run1", "q1")


# --- result_urls repo query (needs Postgres) ---

@pytest.fixture
def run_with_scores():
    """Build a run with two leaves, three candidates, and scores that exercise dedupe,
    ordering, and the min_score gate. Yields (run_id, node_a_id, node_b_id)."""
    try:
        run_id = repo.create_run("a video background removal api described in prose form", None)
    except OperationalError as exc:
        pytest.skip(f"postgres unavailable ({exc}) -- run `docker compose up -d postgres`")

    node_a = repo.insert_node(run_id, None, 2, "leaf", "leaf A", "pain A")
    node_b = repo.insert_node(run_id, None, 2, "leaf", "leaf B", "pain B")

    # url1 matches both leaves; its best score (0.8) is what should rank it.
    c1 = repo.upsert_candidate(run_id, "https://quora.com/1", uuid.uuid4().hex, "quora", "t1", "s1")
    c2 = repo.upsert_candidate(run_id, "https://quora.com/2", uuid.uuid4().hex, "quora", "t2", "s2")
    c3 = repo.upsert_candidate(run_id, "https://quora.com/3", uuid.uuid4().hex, "quora", "t3", "s3")

    def score(cid, nid, final):
        return {
            "candidate_id": cid, "node_id": nid, "is_seeking": True, "pain_match": 80,
            "icp_match": 50, "actionable": True, "reason": "r", "reply_angle": "a",
            "model": "test", "final": final,
        }

    repo.save_scores([
        score(c1, node_a, 0.8),
        score(c1, node_b, 0.5),   # same url, weaker leaf -> must dedupe to 0.8
        score(c2, node_a, 0.3),
        score(c3, node_a, 0.0),   # gated out (ad / dead thread)
    ])
    yield run_id, node_a, node_b


def test_result_urls_dedupes_and_ranks_by_best_score(run_with_scores) -> None:
    run_id, _, _ = run_with_scores
    urls = repo.result_urls(run_id, min_score=0.0)
    assert urls == [
        "https://quora.com/1",   # 0.8 (deduped from its two leaves)
        "https://quora.com/2",   # 0.3
        "https://quora.com/3",   # 0.0
    ]
    assert urls.count("https://quora.com/1") == 1, "a url matching two leaves must appear once"


def test_result_urls_min_score_excludes_gated_out_posts(run_with_scores) -> None:
    run_id, _, _ = run_with_scores
    urls = repo.result_urls(run_id, min_score=0.01)
    assert "https://quora.com/3" not in urls, "the 0.0-scored post must be filtered"
    assert urls == ["https://quora.com/1", "https://quora.com/2"]


def test_result_urls_node_filter(run_with_scores) -> None:
    run_id, _, node_b = run_with_scores
    urls = repo.result_urls(run_id, node_id=node_b)
    assert urls == ["https://quora.com/1"], "only url1 is scored against leaf B"
