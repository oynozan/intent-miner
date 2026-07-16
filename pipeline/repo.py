"""Database access for the pipeline.

Plain SQL, no ORM. The queries here are few, shaped by the pipeline's stages, and the
interesting ones (ON CONFLICT dedupe, the run-scoped reads) are clearer as SQL than as
an ORM's idea of SQL.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Iterable, Sequence

from psycopg.types.json import Jsonb

from core.db import connection


# --- runs -------------------------------------------------------------------------

def create_run(input_text: str, icp: str | None) -> str:
    run_id = str(uuid.uuid4())
    with connection() as conn:
        conn.execute(
            "INSERT INTO runs (id, input_text, icp, status) VALUES (%s, %s, %s, 'pending')",
            (run_id, input_text, icp),
        )
    return run_id


def set_status(run_id: str, status: str, *, finished: bool = False) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE runs SET status = %s, finished_at = CASE WHEN %s THEN now() ELSE finished_at END "
            "WHERE id = %s",
            (status, finished, run_id),
        )


def merge_stats(run_id: str, patch: dict[str, Any]) -> None:
    """Shallow-merge into runs.stats. Stages write their own metrics independently, so
    a whole-object write would race and lose the other stages' numbers."""
    with connection() as conn:
        conn.execute("UPDATE runs SET stats = stats || %s WHERE id = %s", (Jsonb(patch), run_id))


def get_run(run_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        return conn.execute("SELECT * FROM runs WHERE id = %s", (run_id,)).fetchone()


# --- nodes ------------------------------------------------------------------------

def insert_node(
    run_id: str,
    parent_id: str | None,
    depth: int,
    kind: str,
    label: str,
    description: str,
    pain_phrases: Sequence[str] = (),
    negative_terms: Sequence[str] = (),
    icp_hint: str | None = None,
) -> str:
    node_id = str(uuid.uuid4())
    with connection() as conn:
        conn.execute(
            "INSERT INTO nodes (id, run_id, parent_id, depth, kind, label, description, "
            "pain_phrases, negative_terms, icp_hint) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (node_id, run_id, parent_id, depth, kind, label, description,
             list(pain_phrases), list(negative_terms), icp_hint),
        )
    return node_id


def set_node_embeddings(rows: Iterable[tuple[str, list[float]]]) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.executemany("UPDATE nodes SET embedding = %s WHERE id = %s",
                            [(_vec(v), nid) for nid, v in rows])


def leaves(run_id: str) -> list[dict[str, Any]]:
    with connection() as conn:
        return conn.execute(
            "SELECT id, label, description, pain_phrases, negative_terms, icp_hint, embedding "
            "FROM nodes WHERE run_id = %s AND kind = 'leaf' ORDER BY id",
            (run_id,),
        ).fetchall()


def tree(run_id: str) -> list[dict[str, Any]]:
    with connection() as conn:
        return conn.execute(
            "SELECT id, parent_id, depth, kind, label, description, pain_phrases, "
            "negative_terms, icp_hint FROM nodes WHERE run_id = %s ORDER BY depth, label",
            (run_id,),
        ).fetchall()


# --- queries ----------------------------------------------------------------------

def insert_queries(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Insert compiled queries, deduped on (run_id, channel, q_hash).

    Returns the ids that actually landed. Dedupe happens in the database rather than in
    Python because the uniqueness constraint is the real source of truth -- and the
    returned ids are the barrier's party list, so they must reflect what exists, not
    what we hoped to insert.
    """
    if not rows:
        return []
    ids: list[str] = []
    with connection() as conn:
        with conn.cursor() as cur:
            for row in rows:
                qid = str(uuid.uuid4())
                result = cur.execute(
                    "INSERT INTO queries (id, run_id, node_id, platform, channel, q, q_hash, depth) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT (run_id, channel, q_hash) DO NOTHING RETURNING id",
                    (qid, row["run_id"], row["node_id"], row["platform"], row["channel"],
                     row["q"], row["q_hash"], row.get("depth", 10)),
                ).fetchone()
                if result:
                    ids.append(str(result["id"]))
    return ids


def get_query(query_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        return conn.execute("SELECT * FROM queries WHERE id = %s", (query_id,)).fetchone()


def mark_query(query_id: str, status: str, hits: int = 0, error: str | None = None) -> None:
    with connection() as conn:
        conn.execute("UPDATE queries SET status = %s, hits = %s, error = %s WHERE id = %s",
                     (status, hits, error, query_id))


def query_counts(run_id: str) -> dict[str, Any]:
    """Per-status query tallies plus one sample error, for run stats.

    Surfacing this is how a bad Serper key stops being invisible: without it, the run
    just shows candidates_discovered: 0 and the cause is buried in worker logs.
    """
    with connection() as conn:
        row = conn.execute(
            "SELECT count(*) FILTER (WHERE status = 'ok') AS ok, "
            "       count(*) FILTER (WHERE status = 'failed') AS failed, "
            "       count(*) AS total, "
            "       (SELECT error FROM queries WHERE run_id = %s AND status = 'failed' "
            "        AND error IS NOT NULL LIMIT 1) AS sample_error "
            "FROM queries WHERE run_id = %s",
            (run_id, run_id),
        ).fetchone()
    return {
        "queries_ok": row["ok"],
        "queries_failed": row["failed"],
        "query_error_sample": (row["sample_error"] or None) if row["failed"] else None,
    }


# --- candidates -------------------------------------------------------------------

def upsert_candidate(
    run_id: str, url: str, url_hash: str, platform: str,
    title: str | None, snippet: str | None,
) -> str | None:
    """Insert a discovered URL. Returns its id, or None if already present this run.

    ON CONFLICT DO NOTHING is what makes run_query idempotent under dramatiq's
    at-least-once delivery: a redelivered query re-inserts the same URLs and changes
    nothing.
    """
    cid = str(uuid.uuid4())
    with connection() as conn:
        row = conn.execute(
            "INSERT INTO candidates (id, run_id, url, url_hash, platform, title, snippet) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (run_id, url_hash) DO NOTHING RETURNING id",
            (cid, run_id, url, url_hash, platform, title, snippet),
        ).fetchone()
    return str(row["id"]) if row else None


def pending_candidates(run_id: str) -> list[dict[str, Any]]:
    with connection() as conn:
        return conn.execute(
            "SELECT id, url, platform FROM candidates WHERE run_id = %s AND fetch_status = 'pending'",
            (run_id,),
        ).fetchall()


def get_candidate(candidate_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        return conn.execute("SELECT * FROM candidates WHERE id = %s", (candidate_id,)).fetchone()


def save_fetched(
    candidate_id: str, body: str | None, author: str | None, posted_at: datetime | None,
    engagement: int, raw_key: str | None, status: str,
    answers_seen: int | None = None, answers_total: int | None = None,
    error: str | None = None,
) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE candidates SET body=%s, author=%s, posted_at=%s, engagement=%s, raw_key=%s, "
            "fetch_status=%s, fetch_error=%s, answers_seen=%s, answers_total=%s WHERE id=%s",
            (body, author, posted_at, engagement, raw_key, status, error,
             answers_seen, answers_total, candidate_id),
        )


def fetched_candidates(run_id: str) -> list[dict[str, Any]]:
    with connection() as conn:
        return conn.execute(
            "SELECT id, url, platform, title, body, posted_at, engagement, answers_total "
            "FROM candidates WHERE run_id = %s AND fetch_status = 'ok' AND body IS NOT NULL "
            "ORDER BY id",
            (run_id,),
        ).fetchall()


def save_candidate_embeddings(run_id: str, rows: Sequence[tuple[str, list[float]]]) -> None:
    if not rows:
        return
    with connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO candidate_embeddings (candidate_id, run_id, embedding) VALUES (%s,%s,%s) "
                "ON CONFLICT (candidate_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                [(cid, run_id, _vec(v)) for cid, v in rows],
            )


def save_candidate_nodes(rows: Sequence[tuple[str, str, float]]) -> None:
    if not rows:
        return
    with connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO candidate_nodes (candidate_id, node_id, cosine) VALUES (%s,%s,%s) "
                "ON CONFLICT (candidate_id, node_id) DO UPDATE SET cosine = EXCLUDED.cosine",
                list(rows),
            )


def surviving_pairs(run_id: str) -> list[dict[str, Any]]:
    with connection() as conn:
        return conn.execute(
            "SELECT cn.candidate_id, cn.node_id, cn.cosine, c.url, c.platform, c.title, c.body, "
            "       c.posted_at, c.engagement, c.answers_total, "
            "       n.label AS node_label, n.description AS node_description, n.icp_hint "
            "FROM candidate_nodes cn "
            "JOIN candidates c ON c.id = cn.candidate_id "
            "JOIN nodes n ON n.id = cn.node_id "
            "WHERE c.run_id = %s ORDER BY cn.cosine DESC",
            (run_id,),
        ).fetchall()


# --- scores -----------------------------------------------------------------------

def save_scores(rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO scores (candidate_id, node_id, is_seeking, pain_match, icp_match, "
                "actionable, reason, reply_angle, model, final) "
                "VALUES (%(candidate_id)s,%(node_id)s,%(is_seeking)s,%(pain_match)s,%(icp_match)s,"
                "%(actionable)s,%(reason)s,%(reply_angle)s,%(model)s,%(final)s) "
                "ON CONFLICT (candidate_id, node_id) DO UPDATE SET "
                "is_seeking=EXCLUDED.is_seeking, pain_match=EXCLUDED.pain_match, "
                "icp_match=EXCLUDED.icp_match, actionable=EXCLUDED.actionable, "
                "reason=EXCLUDED.reason, reply_angle=EXCLUDED.reply_angle, final=EXCLUDED.final",
                list(rows),
            )


def results(run_id: str, node_id: str | None = None, min_score: float = 0.0) -> list[dict[str, Any]]:
    sql = (
        "SELECT s.final, s.is_seeking, s.pain_match, s.icp_match, s.actionable, s.reason, "
        "       s.reply_angle, c.url, c.platform, c.title, c.posted_at, c.engagement, "
        "       n.id AS node_id, n.label AS node_label, n.description AS node_description "
        "FROM scores s "
        "JOIN candidates c ON c.id = s.candidate_id "
        "JOIN nodes n ON n.id = s.node_id "
        "WHERE c.run_id = %s AND s.final >= %s"
    )
    params: list[Any] = [run_id, min_score]
    if node_id:
        sql += " AND s.node_id = %s"
        params.append(node_id)
    sql += " ORDER BY s.final DESC"
    with connection() as conn:
        return conn.execute(sql, params).fetchall()


def result_urls(run_id: str, node_id: str | None = None, min_score: float = 0.0) -> list[str]:
    """Distinct result URLs, ranked by best score, for the flat /urls endpoint.

    A single URL can match several leaves; here it appears once, ranked by its strongest
    match (MAX(final)). Note that gated-out posts (vendor ads, dead threads) score
    exactly 0.0, so a small positive min_score returns genuine leads only.
    """
    sql = (
        "SELECT c.url, MAX(s.final) AS score "
        "FROM scores s JOIN candidates c ON c.id = s.candidate_id "
        "WHERE c.run_id = %s AND s.final >= %s"
    )
    params: list[Any] = [run_id, min_score]
    if node_id:
        sql += " AND s.node_id = %s"
        params.append(node_id)
    sql += " GROUP BY c.url ORDER BY score DESC"
    with connection() as conn:
        return [row["url"] for row in conn.execute(sql, params).fetchall()]


def _vec(values: list[float]) -> str:
    """pgvector's text input format."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"
