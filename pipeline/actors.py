"""The pipeline, as dramatiq actors.

Read ``pipeline/stages.py`` first -- it explains the actor shape every stage here
follows, and why the obvious alternatives (try/except around the body, arriving in a
finally) are broken.

Flow:

    POST /runs -> expand_tree -> compile_queries -> [fan-out] run_query
      -> barrier -> [fan-out] fetch_candidate -> barrier
      -> prefilter -> [fan-out] score_batch -> barrier -> finalize
"""

from __future__ import annotations

import hashlib
import logging
import math
from datetime import datetime, timezone

import dramatiq
import numpy as np

from core.broker import broker, redis_client  # noqa: F401  -- broker import sets the global broker
from core.config import settings
from core.urls import NotCanonicalizable, canonicalize, platform_of, url_hash
from llm import client as llm
from llm import embeddings
from pipeline import prefilter as pf
from pipeline import repo
from pipeline.stages import arrive, fan_out

log = logging.getLogger(__name__)

SCORE_BATCH_SIZE = 10
PLATFORMS = ("quora", "linkedin")  # reddit deferred: no obtainable credentials, see plan §7


# --- 1. expand ---------------------------------------------------------------------

@dramatiq.actor(queue_name="expand", max_retries=2, time_limit=600_000)
def start_run(run_id: str) -> None:
    """Expand the solution into a tree of pains, then compile and dispatch queries."""
    run = repo.get_run(run_id)
    if run is None:
        log.error("run %s not found", run_id)
        return

    repo.set_status(run_id, "expanding")
    tree = llm.expand_tree(run["input_text"], run["icp"])

    root_id = repo.insert_node(
        run_id, None, 0, "root", tree["root"]["label"], tree["root"]["description"]
    )

    leaf_rows: list[tuple[str, str]] = []  # (node_id, description) -- description is what we embed
    query_rows: list[dict] = []

    for branch in tree["branches"]:
        branch_id = repo.insert_node(
            run_id, root_id, 1, "branch", branch["label"], branch["description"]
        )
        for leaf in branch["leaves"]:
            leaf_id = repo.insert_node(
                run_id, branch_id, 2, "leaf", leaf["label"], leaf["description"],
                leaf.get("pain_phrases", []), leaf.get("negative_terms", []), leaf.get("icp_hint"),
            )
            leaf_rows.append((leaf_id, leaf["description"]))
            query_rows.extend(_compile_queries(run_id, leaf_id, leaf))

    # Embed leaf descriptions -- the sufferer's phrasing of the pain, which is what the
    # prefilter matches candidate posts against.
    if leaf_rows:
        vectors = embeddings.embed([d for _, d in leaf_rows], input_type="query")
        repo.set_node_embeddings(zip((nid for nid, _ in leaf_rows), vectors))

    query_ids = repo.insert_queries(query_rows)
    repo.merge_stats(run_id, {
        "leaves": len(leaf_rows),
        "queries_compiled": len(query_rows),
        "queries_after_dedupe": len(query_ids),
    })

    # Fail loud rather than silently truncate. A tree that overshoots the cap is a
    # prompt regression, and quietly dropping half the queries would hide it behind a
    # merely-mediocre run.
    if len(query_ids) > settings().max_queries_per_run:
        repo.set_status(run_id, "failed", finished=True)
        raise RuntimeError(
            f"run {run_id}: {len(query_ids)} queries exceeds cap {settings().max_queries_per_run}"
        )

    repo.set_status(run_id, "discovering")
    fan_out(
        run_id, "discover", query_ids,
        send=lambda qid: run_query.send(run_id, qid),
        on_empty=lambda: fan_out_fetch.send(run_id),
    )


def _compile_queries(run_id: str, leaf_id: str, leaf: dict) -> list[dict]:
    """Flatten a leaf's per-platform queries into rows.

    The LLM writes each platform's queries itself, in that platform's voice. We do not
    cross-product one query set across platforms -- Reddit phrasing is not LinkedIn
    phrasing, and pasting one across all three is how a SERP budget disappears for
    nothing.
    """
    rows: list[dict] = []
    for platform in PLATFORMS:
        for q in leaf.get("queries", {}).get(platform, []):
            q = q.strip()
            if not q:
                continue
            rows.append({
                "run_id": run_id,
                "node_id": leaf_id,
                "platform": platform,
                "channel": "serper",
                "q": q,
                "q_hash": hashlib.sha256(f"{platform}:{q}".encode()).hexdigest(),
                "depth": 10,  # 11+ bills 2 Serper credits; raise per-query only when it earns it
            })
    return rows


# --- 2. discover -------------------------------------------------------------------

@dramatiq.actor(queue_name="discover", max_retries=3, on_retry_exhausted="query_failed", time_limit=60_000)
def run_query(run_id: str, query_id: str) -> None:
    """Run one SERP query and insert its results as candidates.

    Two failure paths, and exactly one fires per query:

    * A retryable error (429, 5xx, timeout) propagates -- the Retries middleware backs
      off and eventually hands the terminal case to query_failed. No try/except around
      those, or retries become dead code.
    * A permanent error (bad key, forbidden, malformed request) is caught here and the
      query is failed immediately. Retrying a 403 four times just floods the logs and
      never succeeds. This does NOT re-raise, so no retry and query_failed does not
      also fire -- _fail_query owns the barrier release either way.
    """
    from discovery import providers as discovery

    query = repo.get_query(query_id)
    if query is None:
        # A vanished query is still a barrier party. Returning without arriving would
        # stall the discover barrier forever -- release the slot instead of hanging.
        log.error("query %s vanished; releasing its barrier slot", query_id)
        arrive(run_id, "discover", query_id, then=lambda: fan_out_fetch.send(run_id))
        return

    try:
        # Serper primary, SerpApi fallback -- see discovery/providers.py. A permanent
        # error means every configured provider rejected the request (bad keys), so
        # fail fast; retryable errors propagate to the retry middleware.
        results = discovery.search(query["q"], query["platform"], depth=query["depth"])
    except discovery.DiscoveryPermanentError as exc:
        _fail_query(run_id, query_id, str(exc))
        return

    inserted = 0
    for result in results:
        platform = platform_of(result.url)
        if platform != query["platform"]:
            continue  # SERP leakage: a site: query occasionally returns off-site results
        try:
            canonical = canonicalize(result.url)
        except NotCanonicalizable:
            # e.g. a LinkedIn /feed/update/ URL with no /posts/ slug. Fetching it would
            # 307 to the signup wall and score as an empty post -- drop it here.
            continue
        if repo.upsert_candidate(
            run_id, canonical, url_hash(result.url), platform, result.title, result.snippet
        ):
            inserted += 1

    repo.mark_query(query_id, "ok", hits=inserted)
    arrive(run_id, "discover", query_id, then=lambda: fan_out_fetch.send(run_id))


def _fail_query(run_id: str, query_id: str, error: str) -> None:
    """Mark a query failed and release its barrier slot. The single terminal path.

    Called from run_query's fast-fail (permanent errors) and from query_failed (retry
    exhausted). Only one fires per query, and arrive() is idempotent regardless.
    """
    repo.mark_query(query_id, "failed", error=error[:500])
    log.warning("run %s: query %s failed: %s", run_id, query_id, error[:200])
    arrive(run_id, "discover", query_id, then=lambda: fan_out_fetch.send(run_id))


@dramatiq.actor(queue_name="discover", max_retries=0)
def query_failed(message: dict, retry_info: dict) -> None:
    """Terminal path for a *retryable* query whose retries were exhausted.

    A failed query degrades recall for one leaf; it must not hang the run.
    """
    run_id, query_id = message["args"][0], message["args"][1]
    _fail_query(run_id, query_id, str(retry_info))


# --- 3. fetch ----------------------------------------------------------------------

@dramatiq.actor(queue_name="expand", max_retries=2)
def fan_out_fetch(run_id: str) -> None:
    repo.set_status(run_id, "fetching")
    candidates = repo.pending_candidates(run_id)
    # Record how discovery went. If bad discovery keys (Serper and its SerpApi fallback)
    # failed every query, this is where candidates_discovered: 0 gets its explanation
    # (queries_failed + a sample error) instead of leaving the user to dig through logs.
    repo.merge_stats(run_id, {"candidates_discovered": len(candidates), **repo.query_counts(run_id)})

    # Route by platform. LinkedIn goes to its own low-concurrency queue: LinkedIn
    # throttles an IP under crawl volume, and the shared `fetch` queue (2 x 16 threads)
    # fires it too fast -- a whole run's LinkedIn candidates came back as gated 200s with
    # no post body. Quora is fine at high concurrency (curl_cffi handles its gate).
    platform_by_id = {str(c["id"]): c["platform"] for c in candidates}

    def _send(cid: str) -> None:
        if platform_by_id.get(cid) == "linkedin":
            fetch_linkedin_candidate.send(run_id, cid)
        else:
            fetch_candidate.send(run_id, cid)

    fan_out(
        run_id, "fetch", [str(c["id"]) for c in candidates],
        send=_send,
        on_empty=lambda: run_prefilter.send(run_id),
    )


def _arrive_fetch(run_id: str, candidate_id: str) -> None:
    """Shared barrier arrival for both fetch queues. The fetch barrier holds every
    candidate id regardless of which queue processed it."""
    arrive(run_id, "fetch", candidate_id, then=lambda: run_prefilter.send(run_id))


@dramatiq.actor(queue_name="fetch", max_retries=3, on_retry_exhausted="fetch_failed", time_limit=120_000)
def fetch_candidate(run_id: str, candidate_id: str) -> None:
    """Fetch a non-LinkedIn candidate (Quora). Plain HTTP -- no browser in this pipeline."""
    candidate = repo.get_candidate(candidate_id)
    if candidate is None:
        _arrive_fetch(run_id, candidate_id)  # vanished, but still a barrier party
        return

    if candidate["platform"] == "quora":
        _fetch_quora(candidate_id, candidate["url"])
    else:
        repo.save_fetched(candidate_id, None, None, None, 0, None, "skipped",
                          error=f"no fetcher for {candidate['platform']}")

    _arrive_fetch(run_id, candidate_id)


@dramatiq.actor(queue_name="fetch_linkedin", max_retries=3, on_retry_exhausted="fetch_failed", time_limit=120_000)
def fetch_linkedin_candidate(run_id: str, candidate_id: str) -> None:
    """Fetch one LinkedIn post on the dedicated low-concurrency queue.

    Retries a *throttled* response (a gated 200 with no post data) so it recovers as the
    burst subsides, but records a genuinely text-less post as terminal. A per-run circuit
    breaker stops LinkedIn fetching entirely once throttling is clearly persistent.
    """
    candidate = repo.get_candidate(candidate_id)
    if candidate is None:
        _arrive_fetch(run_id, candidate_id)
        return
    _fetch_linkedin(run_id, candidate_id, candidate["url"])
    _arrive_fetch(run_id, candidate_id)


# --- LinkedIn throttle circuit breaker (per run, in Redis) -------------------------

def _linkedin_breaker_key(run_id: str) -> str:
    return f"linkedin_throttle:{run_id}"


def _linkedin_breaker_tripped(run_id: str) -> bool:
    count = redis_client.get(_linkedin_breaker_key(run_id))
    return count is not None and int(count) >= settings().linkedin_throttle_breaker


def _bump_linkedin_throttle(run_id: str) -> int:
    key = _linkedin_breaker_key(run_id)
    with redis_client.pipeline() as pipe:
        pipe.incr(key)
        pipe.expire(key, 3600)
        count, _ = pipe.execute()
    return int(count)


def _fetch_quora(candidate_id: str, url: str) -> None:
    from scrape import quora

    status, html, headers = quora.fetch(url)
    if quora.is_challenge(html, status, headers):
        # Raise so the retry middleware backs off. Quora's 403s are short-lived per-IP
        # penalty windows and are not URL-sticky -- the same URL succeeds minutes later.
        # Never tight-loop; that just burns the same window.
        raise RuntimeError(f"quora challenge ({status}) for {url}")

    page = quora.parse(html, url)
    if not page.body:
        repo.save_fetched(candidate_id, None, None, None, 0, None, "empty")
        return

    repo.save_fetched(
        candidate_id,
        body=page.embed_text,       # the question -- see scrape/quora.py on why not `body`
        author=None,                # deliberately not collected: rank the post, never the person
        posted_at=page.posted_at,
        engagement=page.engagement,
        raw_key=None,
        status="ok",
        answers_seen=page.answers_seen,
        answers_total=page.answer_count,   # the completeness oracle + the saturation signal
    )


def _fetch_linkedin(run_id: str, candidate_id: str, url: str) -> None:
    import random
    import time

    from scrape import linkedin

    # Circuit breaker: once throttling is clearly persistent for this run, stop fetching
    # LinkedIn rather than hammering a throttled IP for every remaining candidate.
    if _linkedin_breaker_tripped(run_id):
        repo.save_fetched(candidate_id, None, None, None, 0, None, "skipped",
                          error="linkedin throttled (circuit breaker open)")
        return

    # Pace even the low-concurrency queue so it never becomes a tight burst.
    jitter = settings().linkedin_fetch_jitter_ms
    if jitter > 0:
        time.sleep(random.uniform(0, jitter / 1000))

    status, html, _headers, final_url = linkedin.fetch(url)
    if linkedin.is_authwalled(status, final_url, html):
        # 999 / auth-wall redirect: LinkedIn's verdict on the caller. Count it toward the
        # breaker and raise so the retry middleware backs off.
        _bump_linkedin_throttle(run_id)
        raise RuntimeError(f"linkedin authwall ({status}) for {url}")

    post = linkedin.parse(html, url)
    if not post.body:
        if post.looks_throttled:
            # 200 with no post ld+json node -> a gated/throttled page, not the post.
            # Count it and raise so it retries as the burst subsides; genuinely text-less
            # posts (posting node present, empty body) fall through to a terminal "empty".
            count = _bump_linkedin_throttle(run_id)
            raise RuntimeError(f"linkedin throttled (gated 200, breaker {count}) for {url}")
        repo.save_fetched(candidate_id, None, None, None, 0, None, "empty")
        return

    repo.save_fetched(
        candidate_id,
        body=post.body,
        author=None,                # never resolve the person -- see plan
        posted_at=post.posted_at,
        engagement=post.engagement,
        raw_key=None,
        status="ok" if not post.is_truncated else "ok_truncated",
    )


@dramatiq.actor(queue_name="fetch", max_retries=0)
def fetch_failed(message: dict, retry_info: dict) -> None:
    run_id, candidate_id = message["args"][0], message["args"][1]
    repo.save_fetched(candidate_id, None, None, None, 0, None, "failed", error=str(retry_info)[:500])
    arrive(run_id, "fetch", candidate_id, then=lambda: run_prefilter.send(run_id))


# --- 4. prefilter ------------------------------------------------------------------

@dramatiq.actor(queue_name="expand", max_retries=2, time_limit=600_000)
def run_prefilter(run_id: str) -> None:
    """Embed candidates, drop the weak ones, dispatch survivors to scoring.

    Runs in-process as one numpy matmul rather than in pgvector -- see
    pipeline/prefilter.py for why (GEMM vs per-pair is ~118x on this hardware, and a
    threshold gate cannot use an HNSW index anyway).
    """
    repo.set_status(run_id, "prefiltering")

    candidates = repo.fetched_candidates(run_id)
    leaf_rows = repo.leaves(run_id)
    if not candidates or not leaf_rows:
        repo.merge_stats(run_id, {"candidates_fetched": len(candidates), "pairs_surviving": 0})
        finalize.send(run_id)
        return

    vectors = embeddings.embed([c["body"] for c in candidates], input_type="document")
    repo.save_candidate_embeddings(run_id, [(str(c["id"]), v) for c, v in zip(candidates, vectors)])

    # Negative terms run before the cosine gate: they are a precision lever for
    # ambiguous pain phrases, and they are cheap.
    dropped = 0
    keep_idx: list[int] = []
    for i, cand in enumerate(candidates):
        if any(pf.drop_by_negative_terms(cand["body"], leaf["negative_terms"]) for leaf in leaf_rows):
            dropped += 1
            continue
        keep_idx.append(i)

    if not keep_idx:
        repo.merge_stats(run_id, {"dropped_by_negative_terms": dropped, "pairs_surviving": 0})
        finalize.send(run_id)
        return

    cand_matrix = np.asarray([vectors[i] for i in keep_idx], dtype=np.float32)
    leaf_matrix = np.asarray([_parse_vec(leaf["embedding"]) for leaf in leaf_rows], dtype=np.float32)

    pairs = pf.select(cand_matrix, leaf_matrix, keep_percentile=_keep_percentile())
    rows = [
        (str(candidates[keep_idx[p.candidate_index]]["id"]), str(leaf_rows[p.node_index]["id"]), p.cosine)
        for p in pairs
    ]
    repo.save_candidate_nodes(rows)
    repo.merge_stats(run_id, {
        "candidates_fetched": len(candidates),
        "dropped_by_negative_terms": dropped,
        "pairs_surviving": len(rows),
        "kill_rate": round(1 - (len({r[0] for r in rows}) / max(len(candidates), 1)), 4),
    })

    batches = [rows[i : i + SCORE_BATCH_SIZE] for i in range(0, len(rows), SCORE_BATCH_SIZE)]
    repo.set_status(run_id, "scoring")
    fan_out(
        run_id, "score", [str(i) for i in range(len(batches))],
        send=lambda idx: score_batch.send(run_id, int(idx)),
        on_empty=lambda: finalize.send(run_id),
    )


def _keep_percentile() -> float:
    """Read the calibrated percentile, defaulting to 0.15 until the sweep has run.

    A percentile rather than a raw cosine on purpose: absolute cosine values shift with
    model, input_type, dimension, and dtype, so a hardcoded cutoff silently means
    something different after any of those change.
    """
    from core.db import connection

    with connection() as conn:
        row = conn.execute("SELECT keep_percentile FROM prefilter_config WHERE id = 1").fetchone()
    return float(row["keep_percentile"]) if row else 0.15


def _parse_vec(value) -> list[float]:
    if isinstance(value, str):
        return [float(x) for x in value.strip("[]").split(",")]
    return list(value)


# --- 5. score ----------------------------------------------------------------------

@dramatiq.actor(queue_name="score", max_retries=3, on_retry_exhausted="score_failed", time_limit=180_000)
def score_batch(run_id: str, batch_index: int) -> None:
    import json

    all_pairs = repo.surviving_pairs(run_id)
    batch = all_pairs[batch_index * SCORE_BATCH_SIZE : (batch_index + 1) * SCORE_BATCH_SIZE]
    if not batch:
        arrive(run_id, "score", str(batch_index), then=lambda: finalize.send(run_id))
        return

    payload = json.dumps({
        "pairs": [
            {
                "pair_id": f"{p['candidate_id']}:{p['node_id']}",
                "leaf": {"label": p["node_label"], "pain": p["node_description"], "icp": p["icp_hint"]},
                "post": {
                    "platform": p["platform"],
                    "title": p["title"],
                    "text": (p["body"] or "")[:4_000],
                    "posted_at": p["posted_at"].isoformat() if p["posted_at"] else None,
                    # The scorer needs this for `actionable`: a question with 900 answers
                    # has no room left for a useful reply, however well the pain matches.
                    "existing_answer_count": p["answers_total"],
                },
            }
            for p in batch
        ]
    }, default=str)

    scored = llm.score_batch(payload)
    by_id = {s["pair_id"]: s for s in scored}

    rows = []
    for p in batch:
        pair_id = f"{p['candidate_id']}:{p['node_id']}"
        s = by_id.get(pair_id)
        if not s:
            log.warning("scorer omitted pair %s", pair_id)
            continue
        rows.append({
            "candidate_id": p["candidate_id"],
            "node_id": p["node_id"],
            "is_seeking": s["is_seeking"],
            "pain_match": s["pain_match"],
            "icp_match": s["icp_match"],
            "actionable": s["actionable"],
            "reason": s["reason"],
            "reply_angle": s["reply_angle"],
            "model": settings().score_model,
            "final": _final_score(s, p),
        })
    repo.save_scores(rows)
    arrive(run_id, "score", str(batch_index), then=lambda: finalize.send(run_id))


def _final_score(s: dict, pair: dict) -> float:
    """final = pain * icp * recency_decay * log1p(engagement)

    is_seeking and actionable are hard gates rather than factors. A vendor's ad and a
    dead thread are not weak leads to be ranked low -- they are not leads, and letting
    a strong pain_match drag them up the list is exactly the failure that makes a
    results page untrustworthy.
    """
    if not s["is_seeking"] or not s["actionable"]:
        return 0.0

    pain = s["pain_match"] / 100
    icp = s["icp_match"] / 100

    posted_at = pair.get("posted_at")
    if posted_at:
        age_days = (datetime.now(timezone.utc) - posted_at).days
        recency = math.exp(-max(age_days, 0) / settings().recency_half_life_days)
    else:
        # Unknown date -- common on Quora, which gives logged-out clients no question
        # creation time. Neutral-ish rather than 1.0: we cannot verify it is fresh.
        recency = 0.5

    engagement = math.log1p(max(pair.get("engagement") or 0, 0))
    return round(pain * icp * recency * (1 + engagement), 6)


@dramatiq.actor(queue_name="score", max_retries=0)
def score_failed(message: dict, retry_info: dict) -> None:
    run_id, batch_index = message["args"][0], message["args"][1]
    log.warning("run %s: score batch %s failed permanently", run_id, batch_index)
    arrive(run_id, "score", str(batch_index), then=lambda: finalize.send(run_id))


# --- 6. finalize -------------------------------------------------------------------

@dramatiq.actor(queue_name="expand", max_retries=2)
def finalize(run_id: str) -> None:
    from core.db import connection

    with connection() as conn:
        stats = conn.execute(
            "SELECT count(*) AS scored, "
            "       count(*) FILTER (WHERE s.final > 0) AS leads, "
            "       count(*) FILTER (WHERE NOT s.is_seeking) AS not_seeking, "
            "       count(*) FILTER (WHERE NOT s.actionable) AS not_actionable "
            "FROM scores s JOIN candidates c ON c.id = s.candidate_id WHERE c.run_id = %s",
            (run_id,),
        ).fetchone()

    repo.merge_stats(run_id, {
        "scored": stats["scored"],
        "leads": stats["leads"],
        "rejected_not_seeking": stats["not_seeking"],
        "rejected_not_actionable": stats["not_actionable"],
    })
    repo.set_status(run_id, "done", finished=True)
    log.info("run %s done: %s leads from %s scored", run_id, stats["leads"], stats["scored"])
