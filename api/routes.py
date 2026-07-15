from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.schemas import CreateRun, LeafResults, Node, Result, RunCreated, RunStatus
from pipeline import repo

router = APIRouter()


@router.post("/runs", response_model=RunCreated, status_code=202)
def create_run(body: CreateRun) -> RunCreated:
    """Start a run. Returns immediately; the pipeline is asynchronous."""
    from pipeline.actors import start_run

    run_id = repo.create_run(body.input_text, body.icp)
    start_run.send(run_id)
    return RunCreated(run_id=run_id)


@router.get("/runs/{run_id}", response_model=RunStatus)
def get_run(run_id: str) -> RunStatus:
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return RunStatus(
        run_id=str(run["id"]),
        status=run["status"],
        stats=run["stats"],
        created_at=run["created_at"],
        finished_at=run["finished_at"],
    )


@router.get("/runs/{run_id}/tree", response_model=list[Node])
def get_tree(run_id: str) -> list[Node]:
    if not repo.get_run(run_id):
        raise HTTPException(404, "run not found")
    return [
        Node(
            id=str(n["id"]),
            parent_id=str(n["parent_id"]) if n["parent_id"] else None,
            depth=n["depth"],
            kind=n["kind"],
            label=n["label"],
            description=n["description"],
            pain_phrases=n["pain_phrases"] or [],
            negative_terms=n["negative_terms"] or [],
            icp_hint=n["icp_hint"],
        )
        for n in repo.tree(run_id)
    ]


@router.get("/runs/{run_id}/results", response_model=list[LeafResults])
def get_results(
    run_id: str,
    node_id: str | None = None,
    min_score: float = Query(default=0.0, ge=0.0),
) -> list[LeafResults]:
    """Ranked links, grouped by leaf.

    ``min_score`` defaults to 0.0 so nothing is hidden by default -- but note that
    anything the scorer judged a vendor post or a dead thread already scores exactly
    0.0, so `min_score` slightly above 0 is the useful filter.
    """
    if not repo.get_run(run_id):
        raise HTTPException(404, "run not found")

    rows = repo.results(run_id, node_id=node_id, min_score=min_score)

    groups: dict[str, LeafResults] = {}
    for row in rows:
        nid = str(row["node_id"])
        if nid not in groups:
            groups[nid] = LeafResults(
                node_id=nid,
                node_label=row["node_label"],
                node_description=row["node_description"],
                count=0,
                results=[],
            )
        groups[nid].results.append(
            Result(
                url=row["url"],
                platform=row["platform"],
                title=row["title"],
                posted_at=row["posted_at"],
                engagement=row["engagement"],
                final=row["final"],
                is_seeking=row["is_seeking"],
                pain_match=row["pain_match"],
                icp_match=row["icp_match"],
                actionable=row["actionable"],
                reason=row["reason"],
                reply_angle=row["reply_angle"],
                node_id=nid,
                node_label=row["node_label"],
            )
        )
        groups[nid].count += 1

    # Strongest leaf first: which branch converts is the finding, not the individual link.
    return sorted(
        groups.values(),
        key=lambda g: max((r.final for r in g.results), default=0.0),
        reverse=True,
    )
